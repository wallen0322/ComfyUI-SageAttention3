# ComfyUI-SageAttention3
An experimental node

Strict **Sage 3 (Blackwell)** attention for ComfyUI official flows.

- Hot-patches `torch.nn.functional.scaled_dot_product_attention` to `sageattn3.api.sageattn3_blackwell`
- Prints which backend is active (toggleable)

## Install
1. Copy `ComfyUI-SageAttention3` into `ComfyUI/custom_nodes/`
2. Install: `pip install sageattn3`

## Use
Add **Attention: Sage 3 (Blackwell)** before samplers, pass `MODEL` through, set `enable=True`.
