"""Parquet over Range reads: the footer (row groups, per-column sizes, schema), one row group with column projection,
and rows as flat dictionaries. pyarrow is imported inside the functions that use it.

Columns are named by dotted leaf paths (`source_meta.file_meta.license_type`); a list column is named by its own path
(`source_meta.file_meta.detected_licenses`, not the parquet path with `.list.element`). A projection may also name a
struct prefix (`translation` reads `translation.en` and `translation.zh`).

Rows come out flat, `{leaf path: value}`, in file order. The text columns come out as bytes, read through a binary
view that does not validate UTF-8, so a document that is not valid UTF-8 reaches the `utf8_valid` filter instead of
failing the whole row group.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import CorporaError
from .hub import HubClient, RemoteFile
from .rangefile import RangeFile

FOOTER_BLOCK = 64 << 10
ROW_BATCH = 4096  # rows converted to Python objects at a time


@dataclass(frozen=True)
class RowGroupInfo:
    index: int
    num_rows: int
    columns: Mapping[str, tuple[int, int]]  # leaf path -> (compressed, uncompressed) bytes

    def size_of(self, columns: Sequence[str]) -> tuple[int, int]:
        """(compressed, uncompressed) bytes of every leaf under `columns` (leaves or struct prefixes)."""
        picked = [v for k, v in self.columns.items() if any(k == c or k.startswith(c + ".") for c in columns)]
        return sum(c for c, _ in picked), sum(u for _, u in picked)


@dataclass(frozen=True)
class FooterInfo:
    num_rows: int
    row_groups: tuple[RowGroupInfo, ...]
    schema: Mapping[str, str]  # leaf path -> arrow type (sources.ARROW_TYPES, or the arrow name for others)
    created_by: str

    def to_json(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "created_by": self.created_by,
            "schema": dict(self.schema),
            "row_groups": [
                {"index": rg.index, "num_rows": rg.num_rows, "columns": {k: list(v) for k, v in rg.columns.items()}}
                for rg in self.row_groups
            ],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> FooterInfo:
        groups = tuple(
            RowGroupInfo(g["index"], g["num_rows"], {k: (int(v[0]), int(v[1])) for k, v in g["columns"].items()})
            for g in data["row_groups"]
        )
        return cls(data["num_rows"], groups, dict(data["schema"]), data["created_by"])


def type_name(arrow_type) -> str:
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return f"list<{type_name(arrow_type.value_type)}>"
    names = {pa.float32(): "float", pa.float64(): "double", pa.int64(): "int64", pa.bool_(): "bool"}
    return names.get(arrow_type, str(arrow_type))


def leaf_types(schema) -> dict[str, str]:
    """`{dotted leaf path: type name}` of an arrow schema; structs are walked, lists are leaves."""
    import pyarrow as pa

    out: dict[str, str] = {}

    def walk(field, prefix: str) -> None:
        name = f"{prefix}{field.name}"
        if pa.types.is_struct(field.type):
            for i in range(field.type.num_fields):
                walk(field.type.field(i), name + ".")
        else:
            out[name] = type_name(field.type)

    for field in schema:
        walk(field, "")
    return out


def footer_info(metadata) -> FooterInfo:
    schema = leaf_types(metadata.schema.to_arrow_schema())
    leaves = sorted(schema, key=len, reverse=True)
    groups = []
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        sizes: dict[str, list[int]] = {}
        for j in range(rg.num_columns):
            chunk = rg.column(j)
            path = chunk.path_in_schema
            leaf = next((leaf for leaf in leaves if path == leaf or path.startswith(leaf + ".")), path)
            entry = sizes.setdefault(leaf, [0, 0])
            entry[0] += chunk.total_compressed_size
            entry[1] += chunk.total_uncompressed_size
        groups.append(RowGroupInfo(i, rg.num_rows, {k: (v[0], v[1]) for k, v in sizes.items()}))
    return FooterInfo(metadata.num_rows, tuple(groups), schema, metadata.created_by or "")


def read_metadata(f: RangeFile):
    """The pyarrow FileMetaData of the file behind `f` (reads only its tail)."""
    import pyarrow.parquet as pq

    try:
        return pq.ParquetFile(f).metadata
    except Exception as err:  # pyarrow raises its own error types for a corrupt or truncated footer
        raise CorporaError(f"{f.url.file.path}: cannot read the parquet footer: {err}") from None


def footer(f: RangeFile) -> FooterInfo:
    return footer_info(read_metadata(f))


def read_row_group(f: RangeFile, rg: int, columns: Sequence[str], *, metadata=None):
    """Row group `rg` with only `columns` (leaf paths or struct prefixes); refuses a row count that differs from
    the footer's."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(f, metadata=metadata)
    if not 0 <= rg < parquet.metadata.num_row_groups:
        raise CorporaError(f"{f.url.file.path}: no row group {rg} (the file has {parquet.metadata.num_row_groups})")
    table = parquet.read_row_group(rg, columns=list(columns), use_threads=False)
    expected = parquet.metadata.row_group(rg).num_rows
    if table.num_rows != expected:
        raise CorporaError(f"{f.url.file.path} row group {rg}: read {table.num_rows} rows, the footer says {expected}")
    return table


def _flat(table):
    while any(str(t).startswith("struct<") for t in table.schema.types):
        table = table.flatten()
    return table


def iter_rows(table, text_columns: Sequence[str], batch: int = ROW_BATCH) -> Iterator[dict[str, Any]]:
    """The rows of `table` as flat `{leaf path: value}` dictionaries in order; text columns as bytes."""
    import pyarrow as pa

    flat = _flat(table)
    missing = [c for c in text_columns if c not in flat.column_names]
    if missing:
        raise CorporaError(f"text columns {missing} are not in the table (columns: {flat.column_names})")
    for offset in range(0, flat.num_rows, batch):
        part = flat.slice(offset, batch)
        values = {}
        for name in part.column_names:
            column = part.column(name)
            if name in text_columns:
                kind = column.type
                if pa.types.is_string(kind):
                    column = pa.chunked_array([c.view(pa.binary()) for c in column.chunks], type=pa.binary())
                elif pa.types.is_large_string(kind):
                    column = pa.chunked_array([c.view(pa.large_binary()) for c in column.chunks], pa.large_binary())
            values[name] = column.to_pylist()
        for i in range(part.num_rows):
            yield {name: column[i] for name, column in values.items()}


class RemoteParquet:
    """One Hub parquet file: resolved once, its footer read through small blocks, row groups read through a
    RangeFile that re-resolves when the signed URL expires. `bytes_fetched` sums every request's body bytes."""

    def __init__(self, client: HubClient, remote: RemoteFile, *, block: int = 64 << 10):
        self.client = client
        self.remote = remote
        self.resolved = client.resolve(remote)
        self.file = RangeFile(
            self.resolved,
            block=block,
            refresh=lambda: client.resolve(remote),
            timeout=client.timeout,
            retries=client.retries,
            sleep=client.sleep,
        )
        self.metadata = read_metadata(self.file)
        self.footer = footer_info(self.metadata)

    @property
    def bytes_fetched(self) -> int:
        return self.file.bytes_fetched

    def read(self, rg: int, columns: Sequence[str]):
        return read_row_group(self.file, rg, columns, metadata=self.metadata)


class RemoteFiles:
    """Open RemoteParquet files keyed by path, the `size` most recently used kept: a file's footer is read once
    while it stays open (the UN PC footers are about 1.2 MB, its row groups about 160 KB [verified])."""

    def __init__(self, client: HubClient, size: int = 16):
        self.client = client
        self.size = size
        self.open: dict[str, RemoteParquet] = {}

    def get(self, remote: RemoteFile) -> tuple[RemoteParquet, int]:
        """(the open file, bytes fetched to open it now: 0 when it was already open)."""
        if remote.path in self.open:
            opened = self.open.pop(remote.path)
            self.open[remote.path] = opened
            return opened, 0
        opened = RemoteParquet(self.client, remote)
        self.open[remote.path] = opened
        while len(self.open) > self.size:
            self.open.pop(next(iter(self.open)))
        return opened, opened.bytes_fetched
