"""Pinned corpus adapters for the fork's projects: UltraData (and UN PC) parquet shards on the Hugging Face Hub, read
by HTTP Range one row group at a time, filtered and exported to each project's corpus format with provenance.

A fork-owned leaf package, kept out of every wheel the fork ships. No project imports it: data flows only through
the files it writes (JSONL parts, a spec fragment, a provenance manifest). Importing it loads neither pyarrow nor any
`surogate*` module; the modules that read parquet import pyarrow inside functions. Layout, environment and commands:
fork/corpora/README.md.
"""

__version__ = "0.1.0.dev0"
