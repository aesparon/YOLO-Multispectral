# -*- coding: utf-8 -*-
"""
mod_pt_model_seg.py

- Patch YOLOv11-seg input conv weights to support multispectral (N>3) channels
- Supports multiple init strategies for extra channels and logs effective init mode
- Includes a small, safe patch for ultralytics/nn/tasks.py to avoid YAML-build crashes

Drop this file alongside run_transfer_vs_scratch_baseline.py (same folder).
"""
from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Optional: tasks.py patch (safe, idempotent)
# --------------------------------------------------------------------------------------
def patch_ultralytics_tasks_scale_and_nc(yolo_source_path: Union[str, Path]) -> None:
    """
    Patch <yolo_source_path>/ultralytics/nn/tasks.py to prevent:
      - UnboundLocalError: 'scale' referenced before assignment (parse_model)
      - KeyError: 'nc' during YAML-only model builds (self.yaml['nc'] missing)

    Idempotent: will not double-patch.
    """
    yolo_source_path = Path(yolo_source_path)
    tasks_py = yolo_source_path / "ultralytics" / "nn" / "tasks.py"
    if not tasks_py.exists():
        print(f"[WARN] Cannot find tasks.py at: {tasks_py}")
        return

    txt = tasks_py.read_text(encoding="utf-8", errors="ignore")

    # 1) Ensure parse_model defines `scale`
    if "YOLO-MultiSpectral patch: define scale default" not in txt:
        m = re.search(
            r"^\s*nc,\s*act,\s*scales\s*=\s*\(d\.get\(x\)\s*for\s*x\s*in\s*\(\"nc\",\s*\"activation\",\s*\"scales\"\)\)\s*$",
            txt,
            flags=re.M,
        )
        if m:
            patch = (
                "\n    # --- YOLO-MultiSpectral patch: define scale default ---\n"
                "    scale = (d.get('scale') or '') if isinstance(d, dict) else ''\n"
                "    # --- end patch ---\n"
            )
            txt = txt[:m.end()] + patch + txt[m.end():]
        else:
            print("[WARN] Could not find nc/act/scales unpack in parse_model(); scale patch not applied.")

    # 2) Ensure nc exists for YAML-only builds BEFORE self.names uses self.yaml['nc']
    if "YOLO-MultiSpectral patch: ensure nc exists for YAML-only builds" not in txt:
        m = re.search(r"^\s*self\.names\s*=\s*\{.*range\(self\.yaml\[[\"']nc[\"']\]\).*", txt, flags=re.M)
        if m:
            up_to = txt[:m.start()].splitlines()
            insert_line_idx = None
            for i in range(len(up_to) - 1, max(len(up_to) - 60, 0), -1):
                if "self.yaml" in up_to[i] and "=" in up_to[i] and "cfg" in up_to[i]:
                    insert_line_idx = i + 1
                    break
            if insert_line_idx is None:
                insert_line_idx = len(up_to)
            patch_lines = [
                "        # --- YOLO-MultiSpectral patch: ensure nc exists for YAML-only builds ---",
                "        # Some YAML-only construction paths may omit 'nc' (and nc arg may be None).",
                "        # Ensure a sane default to allow the model graph to build; training will override with dataset nc.",
                "        if isinstance(self.yaml, dict) and (self.yaml.get('nc', None) is None):",
                "            self.yaml['nc'] = int(nc) if nc is not None else 80",
                "        # --- end patch ---",
            ]
            new_lines = up_to[:insert_line_idx] + patch_lines + up_to[insert_line_idx:]
            txt = "\n".join(new_lines) + "\n" + "\n".join(txt[m.start():].splitlines())
        else:
            print("[WARN] Could not find self.names=...self.yaml['nc'] line; nc patch not applied.")

    tasks_py.write_text(txt, encoding="utf-8")
    print(f"[PATCH] tasks.py patched (scale + nc guards) -> {tasks_py}")

# --------------------------------------------------------------------------------------
# Conv patching helpers
# --------------------------------------------------------------------------------------
def _find_first_conv3(model: nn.Module) -> Optional[nn.Conv2d]:
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and getattr(m, "weight", None) is not None:
            if m.weight.ndim == 4 and m.weight.shape[1] == 3:
                return m
    return None

def _init_extra_channels(weight: torch.Tensor, new_channels: int, mode: str) -> Tuple[torch.Tensor, str]:
    """
    weight: (out_ch, 3, k, k)
    returns extra: (out_ch, new_channels, k, k), effective_mode
    """
    mode = (mode or "avg").lower()

    if new_channels <= 0:
        return weight[:, :0, :, :], "na"

    if mode == "avg":
        avg = weight[:, :3, :, :].mean(dim=1, keepdim=True)
        extra = avg.expand(-1, new_channels, -1, -1).contiguous()
        return extra, "avg"

    if mode == "zeros":
        extra = torch.zeros(weight.size(0), new_channels, *weight.shape[2:], device=weight.device, dtype=weight.dtype)
        return extra, "zeros"

    if mode == "random":
        std = float(weight.std().item()) if weight.numel() else 0.02
        extra = torch.randn(weight.size(0), new_channels, *weight.shape[2:], device=weight.device, dtype=weight.dtype) * std
        return extra, "random"

    if mode == "same":
        src = weight[:, :3, :, :]
        rep = src.repeat(1, math.ceil(new_channels / 3), 1, 1)[:, :new_channels, :, :].contiguous()
        return rep, "same"

    if mode == "copy_g":
        g = weight[:, 1:2, :, :]
        extra = g.expand(-1, new_channels, -1, -1).contiguous()
        return extra, "copy_g"

    if mode == "copy_r":
        r = weight[:, 0:1, :, :]
        extra = r.expand(-1, new_channels, -1, -1).contiguous()
        return extra, "copy_r"

    if mode == "copy_b":
        b = weight[:, 2:3, :, :]
        extra = b.expand(-1, new_channels, -1, -1).contiguous()
        return extra, "copy_b"

    if mode == "repeat_rgb":
        src = weight[:, :3, :, :]
        rep = src.repeat(1, math.ceil(new_channels / 3), 1, 1)[:, :new_channels, :, :].contiguous()
        return rep, "repeat_rgb"

    raise ValueError(f"Unknown channel_init_mode: {mode}")

def patch_yolo_seg_ckpt(
    input_ckpt: Union[str, Path],
    out_dir: Union[str, Path],
    in_channels: int,
    channel_init_mode: str = "avg",
    nc: Optional[int] = None,
    yolo_source_path: Optional[Union[str, Path]] = None,
    save_tag: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Load a YOLO checkpoint or YAML, patch the first 3-channel conv to in_channels, save to out_dir.

    Returns: (patched_ckpt_path, effective_init_mode)
    """
    input_ckpt = str(input_ckpt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if yolo_source_path is not None:
        patch_ultralytics_tasks_scale_and_nc(yolo_source_path)

    # Import YOLO lazily so sys.path injection can happen before this is called
    from ultralytics import YOLO  # type: ignore

    print(f"📦 Loading YOLO model: {input_ckpt}")




    # Tidy later
    base_name = os.path.basename(input_ckpt)
    model_base_wo_ext, extension = os.path.splitext(base_name)
    #input_ckpt = model_base


    #  Check mask or BB and qppend to file path for tracebility
    # detect yolo11x-seg. pt or yaml
    if model_base_wo_ext[-3:] == 'seg':
        bb_seg = 'seg'
    else:
        bb_seg = 'bb'



    if extension == '.yaml':

        # load yaml file
        model = YOLO(input_ckpt)
        # model = YOLO(model_yaml_path)
        # # Save the model to a .pt file
        save_yaml_ckpt =  model_base_wo_ext + '_yaml.pt'
        model.save(save_yaml_ckpt)

        no_tl_patch =  model_base_wo_ext + '_yaml_patch.pt'
        # with torch.serialization.safe_globals([SegmentationModel]):
        #     ckpt = torch.load(save_yaml_ckpt , map_location='cpu', weights_only=False)
        ckpt = torch.load(save_yaml_ckpt, map_location='cpu')
        
        # Extract raw model
        raw_model = ckpt['model']

        # Rebuild new checkpoint format 
        new_ckpt = {
            'model': raw_model,
            'epoch': 0,
            'best_fitness': 0.0,
            'ema': None,
            'updates': 0,
            'optimizer': None,
            'train_args': {
                'task': 'segment',          # maybe only upadte mod required
                #'imgsz': 640,
                #'batch': 16,
                #'epochs': 300,
                #'model': input_ckpt ,
                #'data': 'data.yaml',
                #'resume': False,
                #'device': 'cuda:0',
            },
            'date': ckpt.get('date', ''),
            'version': ckpt.get('version', ''),
            'license': ckpt.get('license', ''),
            'docs': ckpt.get('docs', '')
        }

        # Save new TL-style .pt file
        torch.save(new_ckpt, no_tl_patch)

        model = YOLO(no_tl_patch)

        # new modified model from YAML so no transfer learning
        #output_model_train = output_model_train  + '_no_TL_'  + bb_seg 
        #model_created = output_model_train + '/mod_model/' + model_base_wo_ext +  '_no_TL.pt'

        
    elif extension == '.pt' or extension == '':
        # if no extension default to .pt  
        model = YOLO(input_ckpt)
        #  pt model with weights -> trans_learn:
        # add for traceility
        #output_model_train = output_model_train  + '_TL_'  + bb_seg 
        #model_created = output_model_train + '/mod_model/'  + model_base_wo_ext  +  '_TL.pt'
    else:
        # update 
        raise Exception("base_model should have extension .pt or .yaml - if no extension defaults to .pt!")




    #model = YOLO(input_ckpt)





    conv = _find_first_conv3(model.model)
    if conv is None:
        raise RuntimeError("Could not find a 3-channel Conv2d to patch (expected first stem conv with in_ch=3).")

    with torch.no_grad():
        weight = conv.weight.data  # (out, 3, k, k)

        if in_channels == 3:
            effective = "na"
        elif in_channels > 3:
            new_channels = in_channels - 3
            extra, effective = _init_extra_channels(weight, new_channels, channel_init_mode)
            conv.weight = nn.Parameter(torch.cat([weight, extra], dim=1))
        else:
            conv.weight = nn.Parameter(weight[:, :in_channels, :, :].contiguous())
            effective = "crop"

        conv.in_channels = in_channels

    tag = save_tag or f"patched_ch{in_channels}_{effective}"
    out_path = out_dir / f"{Path(input_ckpt).stem}_{tag}.pt"
    model.save(str(out_path))

    # Sidecar metadata for provenance
    meta = {
        "input_ckpt": input_ckpt,
        "in_channels": int(in_channels),
        "requested_init_mode": str(channel_init_mode),
        "effective_init_mode": str(effective),
        "nc_arg": None if nc is None else int(nc),
    }
    (out_dir / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"🛠️ Patched input conv to {in_channels}ch using '{effective}' -> {out_path}")
    return str(out_path), effective
