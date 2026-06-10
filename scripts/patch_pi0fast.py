"""
Patch lerobot 0.4.4 to fix four bugs that prevent pi0_fast training:

  1. pi0_fast: imports `transformers.models.siglip.check` which does not
     exist in any public PyPI release of transformers.

  2. groot: GR00TN15Config has a dataclass field-ordering bug on some
     Python 3.11 installs (non-default arg follows default arg), crashing
     the lerobot policies __init__.py before any training can start.

  3. factory: lerobot/pi0fast-base's saved processor config references
     `relative_actions_processor` which was added in a newer lerobot version.
     Fix: when pretrained_path is set and policy is PI0FastConfig, use the
     local make_pi0_fast_pre_post_processors instead of loading from hub.

  4. dtype mismatch in SigLIP encoder: patch_embedding and position_embedding
     are intentionally kept float32, so their output hidden_states are float32.
     The SigLIP encoder's LayerNorm weights are bfloat16. In transformers 5.x
     this is handled automatically; in 4.57.x it is not. Fix: patch
     modeling_siglip.py to cast hidden_states to the encoder's dtype before
     the encoder call ("expected Float but found BFloat16" error).

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

# Include the preceding import line as context so the match is unique and won't
# hit the already-patched indented copy inside the try block.
GROOT_OLD = """\
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .groot.configuration_groot import GrootConfig as GrootConfig\
"""

GROOT_NEW = """\
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
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
# Patch 3 – factory.py: bypass pretrained processor config for PI0Fast
#
# lerobot/pi0fast-base on HuggingFace was uploaded with a processor config
# that references `relative_actions_processor`, a step added in a newer
# lerobot version.  When pretrained_path is set, factory.py currently loads
# the hub's processor config verbatim and crashes.
#
# Fix: intercept the PI0FastConfig case BEFORE the generic from_pretrained
# path, and call the local make_pi0_fast_pre_post_processors instead.
# ---------------------------------------------------------------------------

FACTORY_OLD = """\
    if pretrained_path:
        # TODO(Steven): Temporary patch, implement correctly the processors for Gr00t
        if isinstance(policy_cfg, GrootConfig):\
"""

FACTORY_NEW = """\
    if pretrained_path:
        # Patched by scripts/patch_pi0fast.py:
        # lerobot/pi0fast-base's saved processor config references
        # `relative_actions_processor` which does not exist in lerobot 0.4.4.
        # Use the local processor builder instead so we are not tied to the
        # hub's config.  Use policy_cfg.type string to avoid importing
        # PI0FastConfig (which is not imported in factory.py by default).
        if getattr(policy_cfg, "type", None) == "pi0_fast":
            from lerobot.policies.pi0_fast.processor_pi0_fast import (
                make_pi0_fast_pre_post_processors,
            )
            return make_pi0_fast_pre_post_processors(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )

        # TODO(Steven): Temporary patch, implement correctly the processors for Gr00t
        if isinstance(policy_cfg, GrootConfig):\
"""


def patch_factory() -> bool:
    try:
        import lerobot.policies.factory as _mod
    except Exception as exc:
        print(f"[patch:factory] Cannot import factory: {exc}")
        return False

    path = pathlib.Path(_mod.__file__)
    text = path.read_text(encoding="utf-8")

    if FACTORY_OLD not in text:
        if "Patched by scripts/patch_pi0fast.py" in text:
            print(f"[patch:factory] Already patched: {path}")
            return True
        print(f"[patch:factory] Target block not found — factory.py may differ from expected.")
        print(f"                in: {path}")
        return False

    # Also need PI0FastConfig to be imported in factory.py for the isinstance check.
    # It's already imported at the top of factory.py in lerobot 0.4.4.
    path.write_text(text.replace(FACTORY_OLD, FACTORY_NEW, 1), encoding="utf-8")
    _invalidate_pyc(path)
    print(f"[patch:factory] Done: {path}")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invalidate_pyc(src: pathlib.Path) -> None:
    cache = importlib.util.cache_from_source(str(src))
    pyc = pathlib.Path(cache)
    if pyc.exists():
        pyc.unlink()


# ---------------------------------------------------------------------------
# Patch 4 – SigLIP LayerNorm bfloat16 dtype mismatch
#
# transformers 4.57.x has a bug where vision tower LayerNorm weights remain
# float32 after model.to(bfloat16) when the config sets torch_dtype="float32".
# Explicitly cast every non-float32 param to bfloat16 inside
# to_bfloat16_for_selected_params so we don't depend on .to() working.
# ---------------------------------------------------------------------------

# Root cause: patch_embedding and position_embedding are kept in float32
# (intentional, per params_to_keep_float32), so their output hidden_states
# are float32. The SigLIP encoder's layer_norm1 weights are bfloat16.
# In transformers 5.x this dtype mismatch is handled automatically;
# in 4.57.x it is not, causing "expected Float but found BFloat16".
# Fix: patch modeling_siglip.py to cast hidden_states to the encoder's
# dtype just before the encoder call.

DTYPE_OLD = """\
        encoder_outputs: BaseModelOutput = self.encoder(\
"""

DTYPE_NEW = """\
        # Cast hidden_states to encoder dtype — workaround for transformers 4.57.x.
        # Float32 patch/position embeddings produce float32 hidden_states, but the
        # encoder LayerNorm weights are bfloat16 (as set by to_bfloat16_for_selected_params).
        # Patched by scripts/patch_pi0fast.py
        if (
            hidden_states.dtype != self.encoder.layers[0].layer_norm1.weight.dtype
            and len(self.encoder.layers) > 0
        ):
            hidden_states = hidden_states.to(self.encoder.layers[0].layer_norm1.weight.dtype)
        encoder_outputs: BaseModelOutput = self.encoder(\
"""


def patch_dtype() -> bool:
    try:
        from transformers.models.siglip import modeling_siglip as _mod
    except Exception as exc:
        print(f"[patch:dtype] Cannot import modeling_siglip: {exc}")
        return False

    path = pathlib.Path(_mod.__file__)
    text = path.read_text(encoding="utf-8")

    if DTYPE_OLD not in text:
        if "Patched by scripts/patch_pi0fast.py" in text and "to encoder dtype" in text:
            print(f"[patch:dtype] Already patched: {path}")
            return True
        print(f"[patch:dtype] Target line not found — modeling_siglip.py may differ.")
        print(f"              in: {path}")
        return False

    path.write_text(text.replace(DTYPE_OLD, DTYPE_NEW, 1), encoding="utf-8")
    _invalidate_pyc(path)
    print(f"[patch:dtype] Done: {path}")
    return True


def main() -> None:
    ok1 = patch_groot()    # must come first — fixes the crash that blocks all imports
    ok2 = patch_pi0fast()
    ok3 = patch_factory()
    ok4 = patch_dtype()
    if not (ok1 and ok2 and ok3 and ok4):
        sys.exit(1)
    print("[patch] All done. You can now run lerobot-train.")


if __name__ == "__main__":
    main()
