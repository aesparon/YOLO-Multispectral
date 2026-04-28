import torch.nn as nn

def _iter_named_params(module: nn.Module):
    # Safer than module.named_parameters() if you ever wrap modules
    for name, p in module.named_parameters(recurse=True):
        yield name, p

def _collect_param_ids(module: nn.Module):
    return {id(p) for _, p in _iter_named_params(module)}

def _get_backbone_module(root_model: nn.Module):
    """
    Ultralytics models often expose:
      - model.backbone (preferred, per your goal)
      - or model.model.backbone (depending on wrapper)
    """
    if hasattr(root_model, "backbone") and isinstance(root_model.backbone, nn.Module):
        return root_model.backbone
    if hasattr(root_model, "model") and hasattr(root_model.model, "backbone") and isinstance(root_model.model.backbone, nn.Module):
        return root_model.model.backbone
    raise AttributeError("Could not find .backbone on model or model.model")

def _find_wnn_param_ids(root_model: nn.Module, wnn_class_names=("Wavelet", "WNN", "WaveletNN")):
    """
    Collect params belonging to wavelet/WNN modules by class name match.
    This avoids hard imports if your WNN lives in a custom module.
    """
    wnn_ids = set()
    for m in root_model.modules():
        cls = m.__class__.__name__
        if any(k.lower() in cls.lower() for k in wnn_class_names):
            wnn_ids |= _collect_param_ids(m)
    return wnn_ids

def tag_params_by_model_backbone(root_model: nn.Module, require_backbone=True):
    """
    Returns dict: param -> tag in {"backbone","head"}.
    WNN tagging is handled separately in grouping (wnn overrides backbone/head).
    """
    backbone_mod = _get_backbone_module(root_model)
    bb_ids = _collect_param_ids(backbone_mod)

    if require_backbone and not bb_ids:
        raise RuntimeError("Backbone param id set is empty - backbone detection failed.")

    tags = {}
    for _, p in _iter_named_params(root_model):
        tags[id(p)] = "backbone" if id(p) in bb_ids else "head"
    return tags

def build_param_groups(
    root_model: nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 1.0,
    wnn_lr_mult: float = 1.0,
    require_nonempty_wnn: bool = False,
):
    """
    Produces 3 param groups: backbone, head, wnn.
    WNN params are removed from backbone/head and placed into wnn group.
    """
    tags = tag_params_by_model_backbone(root_model)

    wnn_ids = _find_wnn_param_ids(root_model)

    backbone, head, wnn = [], [], []
    for name, p in _iter_named_params(root_model):
        if not p.requires_grad:
            continue
        pid = id(p)

        if pid in wnn_ids:
            wnn.append(p)
            continue

        if tags.get(pid) == "backbone":
            backbone.append(p)
        else:
            head.append(p)

    if require_nonempty_wnn and len(wnn) == 0:
        raise RuntimeError("Param grouping failed: WNN group is empty (wavelet_on=True expected wavelet modules).")

    if len(backbone) == 0:
        raise RuntimeError("Param grouping failed: backbone group is empty (backbone tagging failed).")
    if len(head) == 0:
        raise RuntimeError("Param grouping failed: head group is empty (unexpected).")

    groups = [
        {"params": backbone, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": head,     "lr": base_lr,                   "weight_decay": weight_decay},
    ]
    if len(wnn) > 0:
        groups.append({"params": wnn, "lr": base_lr * wnn_lr_mult, "weight_decay": weight_decay})

    return groups, {"backbone": len(backbone), "head": len(head), "wnn": len(wnn)}
