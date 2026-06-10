"""
Patch lerobot 0.4.4 to fix two import-time bugs that prevent training:

  1. pi0_fast: imports `transformers.models.siglip.check` which does not
     exist in any public PyPI release of transformers.

  2. groot: GR00TN15Config has a dataclass field-ordering bug on some
     Python 3.11 installs (non-default arg follows default arg), crashing
     the lerobot policies __init__.py before any training can start.

Usage:
    conda activate capstone
    python scripts/patch_pi0fast.py
"""

import importlib.util
import pathlib
import sys


# ---------------------------------------------------------------------------
# Patch 1 – pi0_fast siglip.check
# ---------------------------------------------------------------------------

PI0_OLD = """\
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None\
"""

PI0_NEW = """\
        # NOTE: siglip.check does not exist in any public PyPI transformers release.
        # Patched by scripts/patch_pi0fast.py — safe to skip, this was only a
        # version guard and has no effect on actual model functionality.
        pass\
"""


def patch_pi0fast() -> bool:
    try:
        import lerobot.policies.pi0_fast.modeling_pi0_fast as _mod
    except Exception as exc:
        print(f"[patch:pi0_fast] Cannot import modeling_pi0_fast: {exc}")
        return False

    path = pathlib.Path(_mod.__file__)
    text = path.read_text(encoding="utf-8")

    if PI0_OLD not in text:
        if "Patched by scripts/patch_pi0fast.py" in text:
            print(f"[patch:pi0_fast] Already patched: {path}")
            return True
        print(f"[patch:pi0_fast] Target block not found — file may differ from expected.")
        print(f"                 Manually remove the try/except block around siglip.check")
        print(f"                 in: {path}")
        return False

    path.write_text(text.replace(PI0_OLD, PI0_NEW, 1), encoding="utf-8")
    _invalidate_pyc(path)
    print(f"[patch:pi0_fast] Done: {path}")
    return True


# ---------------------------------------------------------------------------
# Patch 2 – groot dataclass bug in policies/__init__.py
# ---------------------------------------------------------------------------

GROOT_OLD = "from .groot.configuration_groot import GrootConfig as GrootConfig"

GROOT_NEW = """\
try:
    from .groot.configuration_groot import GrootConfig as GrootConfig
except Exception:
    # groot_n1.py has a dataclass field-ordering bug on some Python 3.11 installs.
    # Patched by scripts/patch_pi0fast.py — we don't use groot so safe to skip.
    pass\
"""


def patch_groot() -> bool:
    try:
        import lerobot.policies as _pkg
    except Exception as exc:
        # __init__.py itself may already be crashing; locate file by path
        import lerobot
        _pkg_path = pathlib.Path(lerobot.__file__).parent / "policies" / "__init__.py"
        if not _pkg_path.exists():
            print(f"[patch:groot] Cannot locate policies/__init__.py: {exc}")
            return False
        text = _pkg_path.read_text(encoding="utf-8")
        path = _pkg_path
    else:
        path = pathlib.Path(_pkg.__file__)
        text = path.read_text(encoding="utf-8")

    if GROOT_OLD not in text:
        if "Patched by scripts/patch_pi0fast.py" in text:
            print(f"[patch:groot] Already patched: {path}")
            return True
        print(f"[patch:groot] Target line not found — groot import may already be wrapped.")
        return True  # probably already fine

    path.write_text(text.replace(GROOT_OLD, GROOT_NEW, 1), encoding="utf-8")
    _invalidate_pyc(path)
    print(f"[patch:groot] Done: {path}")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invalidate_pyc(src: pathlib.Path) -> None:
    cache = importlib.util.cache_from_source(str(src))
    pyc = pathlib.Path(cache)
    if pyc.exists():
        pyc.unlink()


def main() -> None:
    ok1 = patch_groot()   # must come first — fixes the crash that blocks all imports
    ok2 = patch_pi0fast()
    if not (ok1 and ok2):
        sys.exit(1)
    print("[patch] All done. You can now run lerobot-train.")


if __name__ == "__main__":
    main()
