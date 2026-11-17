"""Explicit adapters around pinned, unmodified author IDQL JAX sources.

Import ``core`` only in an environment with the optional IDQL dependencies.
"""

from .upstream import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY, source_receipt, verify_upstream

__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_REPOSITORY", "source_receipt", "verify_upstream"]
