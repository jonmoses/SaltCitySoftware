"""tools — the composable single-purpose CLIs of the annotation pipeline.

Each module here does one transformation, reading and writing named artifacts in
a working directory, and exposes a `main(argv=None)` entry point so it runs as
`python -m tools.<name>`. The Makefile wires them into the full pipeline.
"""
