"""The control matrix of MS §7, one row per ID, with its status and (until a run fills it) the result "未运行".

Statuses (MS §7 "状态说明"): `minimal-primary` (主, a confirmatory arm of the minimal decisive set, MS §13),
`minimal-secondary` (次, a preregistered secondary arm of the set), `P1` (the next phase), `deferred` (延后, needs a
component that does not exist yet) and `G0` (a unit test of the G0 gate only: B7 = B6). T0 is the reference arm of
H_C (主（参照）) and counts as minimal-primary.

`python -m surogate_bilingual matrix [--json] [--results PATH ...]` prints the table. With result files
(`bilingual-result/1`, written by `eval-tasks`), each row shows the metrics recorded under its ID; every other row
says 未运行. Rows are data: nothing here runs a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

STATUSES = ("minimal-primary", "minimal-secondary", "P1", "deferred", "G0")
NOT_RUN = "未运行"


@dataclass(frozen=True)
class Row:
    id: str
    section: str  # "7.1" family I, "7.2" family II, "7.3" Bridge
    condition: str
    training: str  # new training the row needs ("—" when it only assembles)
    purpose: str
    status: str
    stage: str | None = None  # the examples/pilot_rows.toml stage that trains it, when one does

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"row {self.id}: unknown status {self.status!r}")


ROWS: tuple[Row, ...] = (
    # --- 7.1 family I (SwiGLU, f = 1728, seed S_I) ----------------------------------------------------------------
    Row("B0", "7.1", "种子 S_I", "—", "起点；保持参照", "minimal-secondary", "seed-I"),
    Row("B1", "7.1", "ZH 或 EN 专家单独运行", "ZH、EN 各 0.2B", "单专家能力；G2", "minimal-secondary", "zh-I, en-I"),
    Row("B2", "7.1", "一个双语专家，从种子 FFN 出发，在 D_zh ∪ D_en 上训练", "0.4B", "模块化代价参照（RQ3b）",
        "minimal-secondary", "b2-I"),
    Row("B3a", "7.1", "ZH 与 EN 的简单平均", "—", "同初始化合并基线", "minimal-secondary"),
    Row("B3b", "7.1", "task arithmetic（λ 在 dev 上从 5 点网格中选）", "—", "同初始化合并基线", "minimal-secondary"),
    Row("B3c", "7.1", "TIES（只作用于 FFN delta）", "—", "同初始化合并基线", "minimal-secondary"),
    Row("B3d", "7.1", "DARE", "—", "合并的补充", "P1"),
    Row("B3e", "7.1", "RegMean 或 Fisher", "—", "合并的补充", "P1"),
    Row("B3-rb", "7.1", "Git Re-Basin（解为恒等置换时标为“等同于 B3a”）", "—", "合并的补充", "P1"),
    Row("B4", "7.1", "Top-1 加 R_out 按段路由（oracle 标签路由参照）", "—", "切换代价（RQ3a）；T0", "minimal-secondary"),
    Row("B4-src", "7.1", "R_src（泄漏型 oracle）", "—", "路由上界", "minimal-secondary"),
    Row("B4-wrong", "7.1", "错误路由", "—", "路由下界", "minimal-secondary"),
    Row("B4-rand", "7.1", "随机路由", "—", "路由下界", "minimal-secondary"),
    Row("RtS", "7.1", "recall-then-switch，另设输出 token 数对齐的对照", "—", "参数式任务上最强的 Top-1 竞争者",
        "minimal-secondary"),
    Row("B5", "7.1", "学习式 Router（R-early / R-late）", "Router", "真实路由能力", "deferred"),
    Row("B6", "7.1", "双路输出混合，g = (½, ½)", "—", "族 I 对 H_A 的次要复制", "minimal-secondary"),
    Row("B7", "7.1", "用 B6 的 g 做块门控拼接", "—", "只作 G0 单元测试（B6 ≡ B7）", "G0"),
    Row("B10-I", "7.1", "两个半宽专家做 Top-2", "需要半宽种子", "硬预算线", "P1"),
    Row("B10-ch", "7.1", "54 块中选 27 块", "需要半宽种子", "硬预算线", "P1"),
    Row("B11a", "7.1", "Top-1 加零初始化 adapter（r = 128）", "adapter", "反驳“必须扩容”", "P1"),
    Row("B11b", "7.1", "Top-1 加表示对齐", "adapter", "反驳“必须扩容”", "P1"),
    Row("B12a", "7.1", "非分区、容量对齐的天花板：2f 宽 FFN，在 D_zh ∪ D_en 上同 token 训练", "0.4B",
        "切换与模块化代价上界；G1′ 天花板；超出预算", "minimal-secondary", "b12a-I"),
    Row("B12b", "7.1", "混合路由的 code-switched 窗口上联合训练 ZH 与 EN", "—", "独立训练的代价", "P1"),
    Row("LP-4", "7.1", "层局部专家：两端各 4 层为语言专家，中间 16 层用冻结的种子 FFN", "ZH、EN 各 0.2B",
        "语言特异性是否集中在首尾层", "minimal-secondary", "lp4-zh-I, lp4-en-I"),
    Row("FX", "7.1", "FlexOLMo 式协调训练", "2 × 0.2B", "“无协调”是否是弱点", "P1"),
    Row("ST", "7.1", "BTS 式两专家交替 stitch", "stitch 参数", "冻结专家之间学习连接的先例", "P1"),
    # --- 7.2 family II (the canonical slot operator, seed S_II) ---------------------------------------------------
    Row("B0'", "7.2", "种子 S_II", "0.8B", "族 II 起点", "minimal-secondary", "seed-II"),
    Row("B1'", "7.2", "ZH-II 或 EN-II 单独运行", "各 0.2B", "族 II 单专家；G2", "minimal-secondary", "zh-II, en-II"),
    Row("B4-II", "7.2", "Top-1 加 R_out 按段路由", "—", "H_A 的对照", "minimal-primary"),
    Row("B9-II", "7.2", "双路，先读出后混合，g = (½, ½)", "—", "H_A 的处理臂；H_B 的对照", "minimal-primary"),
    Row("B8", "7.2", "双路，先混合后读出，g = (½, ½)，与 B9-II 同一组检查点", "—", "H_B 的处理臂", "minimal-primary"),
    Row("B3-II", "7.2", "族 II 内逐张量平均", "—", "乘积空间合并 vs 逐张量合并", "minimal-secondary"),
    Row("B9-II+aff", "7.2", "B9 加每通道拟合的 offset/scale（dev 上拟合）", "仿射参数", "只能消掉曲率项的均值",
        "minimal-secondary"),
    Row("B8-lin", "7.2", "P_i 线性（MEO 式）", "一个种子加两个专家", "线性坍缩的对照", "P1"),
    Row("B10-II", "7.2", "n = 395 的双路", "需要单独的种子", "硬预算线", "P1"),
    # --- 7.3 Bridge (family I; ZH-I and EN-I frozen; variant (c)) -------------------------------------------------
    Row("T0", "7.3", "ZH+EN 原样组装（即 B4）", "0", "两专家基线（H_C 的参照）", "minimal-primary"),
    Row("T1'", "7.3", "按方向微调目标语言专家的副本，只在目标段上计损失", "0.1B", "H_C 的对照", "minimal-primary",
        "t1p-en-I, t1p-zh-I"),
    Row("T2", "7.3", "ZH+EN+Bridge：Bridge 从种子 FFN 初始化，只训练 Bridge", "0.1B", "H_C 的处理臂",
        "minimal-primary", "t2-I"),
    Row("T3'", "7.3", "第三个专家，在拆散配对的平行数据上训练", "0.1B", "隔离对齐信号", "minimal-secondary", "t3p-I"),
    Row("T3''", "7.3", "第三个专家，在等 token 的单语回放数据上训练", "0.1B（单语）", "隔离参数量",
        "minimal-secondary", "t3pp-I"),
    Row("T4", "7.3", "Bridge 与双路混合的组合", "—", "协同作用", "deferred"),
    Row("T5", "7.3", "Bridge 与 Top-2 的组合", "—", "协同作用", "deferred"),
)  # fmt: skip

ROW_IDS = tuple(r.id for r in ROWS)
if len(set(ROW_IDS)) != len(ROW_IDS):
    raise RuntimeError("duplicate matrix row ids")


def rows() -> tuple[Row, ...]:
    return ROWS


def read_results(paths: Iterable[Path]) -> list[dict]:
    """Every result row of the given `bilingual-result/1` files (or results.json under the given directories)."""
    from ..canonical import check_format

    out = []
    for path in paths:
        path = Path(path)
        for f in sorted(path.rglob("results.json")) if path.is_dir() else [path]:
            doc = json.loads(f.read_text("utf-8"))
            check_format(doc, "result")
            out.extend(doc["results"])
    return out


def table(results: Sequence[dict] = ()) -> list[dict]:
    """The matrix with a `result` column: the recorded metrics per row, or 未运行."""
    by_row: dict[str, list[str]] = {}
    for r in results:
        if r.get("status") == "done" and r.get("value") is not None:
            by_row.setdefault(r["row"], []).append(f"{r['metric']} r{r['replicate']} = {r['value']:.4f}")
    return [{**asdict(row), "result": "; ".join(by_row[row.id]) if row.id in by_row else NOT_RUN} for row in ROWS]


def markdown(entries: Sequence[dict]) -> str:
    lines = ["| ID | § | 条件 | 新训练 | 状态 | 训练阶段 | 结果 |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['id']} | {e['section']} | {e['condition']} | {e['training']} | {e['status']} | "
                     f"{e['stage'] or '—'} | {e['result']} |")  # fmt: skip
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual matrix", description="status of the MS §7 rows")
    ap.add_argument("--results", type=Path, nargs="*", default=[], help="bilingual-result/1 files or directories")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="the table is only printed in any case")
    args = ap.parse_args(list(argv))
    try:
        entries = table(read_results(args.results))
    except (OSError, ValueError, KeyError) as exc:
        print(f"matrix: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(entries, ensure_ascii=False, indent=1) if args.json else markdown(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
