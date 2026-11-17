"""Load byte-verified author IDQL modules without unrelated eager imports.

The vendored files are unmodified.  Their absolute ``jaxrl5`` imports are
resolved into a private namespace by a per-module import function.  Only the
``agents``, ``data`` and ``agents.ddpm_iql_simple`` aggregators are skipped:
they import unrelated
algorithms and datasets.  Every module actually required by the learner is
executed from its real upstream source; missing dependencies propagate normally.
No installed ``jaxrl5`` package is replaced in ``sys.modules``.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType


UPSTREAM_COMMIT = "7582571ce686c77b4ee0f592dd639ab8151295fa"
UPSTREAM_REPOSITORY = "https://github.com/philippe-eecs/IDQL"
UPSTREAM_MANIFEST_SHA256 = "fe478e2e6373f6c17d449d572f754e604448a1d06aae70f4040b13a586538769"
VENDOR_ROOT = Path(__file__).resolve().parent.parent / "_vendor" / "idql_official"
UPSTREAM_MANIFEST = VENDOR_ROOT / "UPSTREAM.json"
_NAMESPACE = "post_training._official_idql_7582571"
_LOCK = threading.RLock()
_SKIP_AGGREGATORS = {"agents", "data", "agents.ddpm_iql_simple"}
_ORIGINAL_IMPORT = builtins.__import__


def verify_upstream() -> dict:
    """Verify the fixed source manifest and every vendored tracked file."""
    raw = UPSTREAM_MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_MANIFEST_SHA256:
        raise ValueError("Official IDQL upstream manifest differs from pinned source")
    manifest = json.loads(raw)
    if manifest["commit"] != UPSTREAM_COMMIT or manifest["repository"] != UPSTREAM_REPOSITORY:
        raise ValueError("Official IDQL upstream identity differs")
    for relative, record in manifest["files"].items():
        path = VENDOR_ROOT / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(VENDOR_ROOT):
            raise ValueError(f"Invalid official IDQL source path: {relative}")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Official IDQL source verification failed: {relative}")
    for source in VENDOR_ROOT.rglob("*.py"):
        if source.relative_to(VENDOR_ROOT).as_posix() not in manifest["files"]:
            raise ValueError("Unpinned Python source found in official IDQL vendor directory")
    return manifest


def _upstream_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "jaxrl5" or name.startswith("jaxrl5.")):
        mapped = _NAMESPACE + name[len("jaxrl5"):]
        module = importlib.import_module(mapped)
        if fromlist:
            # Support Python's ``from package import module`` semantics while
            # retaining real errors for absent modules or attributes.
            for item in fromlist:
                if item != "*" and not hasattr(module, item) and hasattr(module, "__path__"):
                    candidate = f"{mapped}.{item}"
                    if importlib.util.find_spec(candidate) is not None:
                        importlib.import_module(candidate)
            return module
        return importlib.import_module(_NAMESPACE)
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Never execute a previously generated .pyc in place of verified source.
        return self.source_to_code(self.get_data(self.path), self.path)

    def exec_module(self, module: ModuleType) -> None:
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=_upstream_import)
        super().exec_module(module)


class _UpstreamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _NAMESPACE and not fullname.startswith(_NAMESPACE + "."):
            return None
        relative = fullname[len(_NAMESPACE):].lstrip(".")
        base = VENDOR_ROOT / "jaxrl5"
        if relative:
            base = base.joinpath(*relative.split("."))
        if base.is_dir():
            init = base / "__init__.py"
            if relative in _SKIP_AGGREGATORS:
                # A genuine namespace package over upstream's real directory,
                # rather than a replacement object for any algorithm or type.
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = [str(base)]
                return spec
            if init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname, init, loader=_SourceLoader(fullname, str(init)),
                    submodule_search_locations=[str(base)],
                )
        source = base.with_suffix(".py")
        if source.is_file():
            return importlib.util.spec_from_file_location(
                fullname, source, loader=_SourceLoader(fullname, str(source))
            )
        return None


_FINDER = _UpstreamFinder()


def load_upstream() -> ModuleType:
    """Return the actual current-main simple learner from pinned author files."""
    with _LOCK:
        verify_upstream()
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
        return importlib.import_module(
            f"{_NAMESPACE}.agents.ddpm_iql_simple.ddpm_iql_learner"
        )


def load_networks() -> ModuleType:
    load_upstream()
    return importlib.import_module(f"{_NAMESPACE}.networks")


def load_pixel_encoder() -> type:
    load_upstream()
    return importlib.import_module(
        f"{_NAMESPACE}.networks.encoders.d4pg_encoder"
    ).D4PGEncoder


def load_augmentations() -> ModuleType:
    """Return the real author DrQ crop implementation, without source edits."""
    load_upstream()
    return importlib.import_module(f"{_NAMESPACE}.agents.drq.augmentations")


def source_receipt() -> dict:
    manifest = verify_upstream()
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "manifest_sha256": UPSTREAM_MANIFEST_SHA256,
        "files": manifest["files"],
        "upstream_license": manifest["upstream_license"],
        "entrypoint": "jaxrl5.agents.ddpm_iql_simple.ddpm_iql_learner.DDPMIQLLearner",
        "loader": "private_namespace_original_source_skip_agents_data_and_simple_aggregators",
    }


# A descriptive alias for callers inspecting the original class and functions.
official_module = load_upstream
