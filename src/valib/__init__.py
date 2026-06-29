"""valib — lifted pure logic shared by the composable annotation tools.

Each tool in `tools/` is a small single-purpose CLI; the reusable, side-effect-free
logic they share (GO DAG operations, metrics, ESM windowing, caches) lives here so
it can be unit-tested without network or GPU. See the plan and CLAUDE.md for the
code-style rules every function in this package follows.
"""
