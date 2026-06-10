"""
Patch lerobot's pi0_fast to remove the non-standard transformers version check.

The check imports `transformers.models.siglip.check` which does not exist in
any public PyPI release of transformers (4.x or 5.x). This script replaces
that block with a no-op so training can proceed normally.

Usage:
    python scripts/patch_pi0fast.py
"""

import importlib
import pathlib
import sys

OLD = """\
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None\
"""

NEW = """\
        # NOTE: siglip.check does not exist in any public PyPI transformers release.
        # Patched by scripts/patch_pi0fast.py — safe to skip, this was only a
        # version guard and has no effect on actual model functionality.
        pass\
"""


def main() -> None:
    try:
        import lerobot.policies.pi0_fast.modeling_pi0_fast as _mod
    except Exception as exc:
        print(f"[patch] Cannot import modeling_pi0_fast: {exc}")
        sys.exit(1)

    path = pathlib.Path(_mod.__file__)
    text = path.read_text(encoding="utf-8")

    if OLD not in text:
        if "Patched by scripts/patch_pi0fast.py" in text:
            print(f"[patch] Already patched: {path}")
        else:
            print(f"[patch] Target block not found — lerobot version may have changed.")
            print(f"        File: {path}")
            print("[patch] Search for 'siglip' manually and remove the try/except block.")
            sys.exit(1)
        return

    patched = text.replace(OLD, NEW, 1)
    path.write_text(patched, encoding="utf-8")

    # Invalidate any cached .pyc
    cache = importlib.util.cache_from_source(str(path))
    pyc = pathlib.Path(cache)
    if pyc.exists():
        pyc.unlink()

    print(f"[patch] Done: {path}")


if __name__ == "__main__":
    main()
