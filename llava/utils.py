"""Minimal LLaVA utilities kept for SearchV / lmms-eval compatibility."""


def disable_torch_init():
    """Disable redundant torch default initialization to accelerate model creation."""
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
