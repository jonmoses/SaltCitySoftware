"""Viral protein functional-annotation pipeline.

SBIR DPA26BZ03-DV014, annotation stage. See docs/ for design and references.

Top-level import is intentionally cheap: heavy ML deps (torch, transformers) are
imported lazily inside the modules that need them, so `import viral_annotation`
works with only the core dependencies installed.
"""

__version__ = "0.0.1"
