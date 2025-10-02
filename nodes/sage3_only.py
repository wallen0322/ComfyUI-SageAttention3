import torch
import torch.nn.functional as F

try:
    from sageattn3.api import sageattn3_blackwell as _SAGE3_FN  # type: ignore
    _SAGE3_AVAILABLE = True
except Exception as e:
    _SAGE3_AVAILABLE = False
    _SAGE3_IMPORT_ERROR = e
    _SAGE3_FN = None

_ORIG_SDPA = None

def _sdpa_to_sage3(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # Comfy SDPA convention is [B, H, T, D]; sageattn3 also expects [B, H, L, D].
    if not (q.dtype == k.dtype == v.dtype):
        k = k.to(q.dtype)
        v = v.to(q.dtype)

    restore_fp32 = (q.dtype == torch.float32)
    q_use = q if not restore_fp32 else q.to(torch.float16)
    k_use = k if not restore_fp32 else k.to(torch.float16)
    v_use = v if not restore_fp32 else v.to(torch.float16)

    # Reduce memory: disable per-block mean unless explicitly needed.
    out = _SAGE3_FN(q_use, k_use, v_use, per_block_mean=False)

    if restore_fp32:
        out = out.to(torch.float32)
    return out

def _install_patch_strict():
    if not _SAGE3_AVAILABLE:
        raise RuntimeError(f"Sage3 (sageattn3) is not available: {_SAGE3_IMPORT_ERROR}")
    global _ORIG_SDPA
    if _ORIG_SDPA is None:
        _ORIG_SDPA = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _sdpa_to_sage3  # type: ignore

def _remove_patch():
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        F.scaled_dot_product_attention = _ORIG_SDPA  # type: ignore
        _ORIG_SDPA = None

class Sage3AttentionOnlySwitch:
    """
    Strict Sage3-only attention backend.
    Requires: `pip install sageattn3` (wheel with `sageattn3.api.sageattn3_blackwell`).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "print_backend": ("BOOLEAN", {"default": True, "help": "Print which backend is active."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "attention/Sage3"

    def apply(self, model, enable=True, print_backend=True):
        if enable:
            _install_patch_strict()
            setattr(model, "_sage3_enabled", True)
            if print_backend:
                print("[SageAttention3] Using: sageattn3.api.sageattn3_blackwell")
        else:
            _remove_patch()
            setattr(model, "_sage3_enabled", False)
            if print_backend:
                print("[SageAttention3] Disabled; restored torch.sdpa")
        return (model,)
