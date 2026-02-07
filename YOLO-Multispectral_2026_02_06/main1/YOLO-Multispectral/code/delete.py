import torch
from pathlib import Path
import sys

# Paths
CODE_PATH = Path(__file__).resolve().parent
CODE_PARENT = CODE_PATH.parent

YOLO_SOURCE_PATH = CODE_PATH / "ultralytics_MS"
if str(YOLO_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE_PATH))

from ultralytics import YOLO

base_pt = r"D:\PD\Publications\Yolo_mod\github\YOLO-Multispectral_2026_01_29\test_notebook_v2_restore\YOLO-Multispectral\code\runs_transfer_vs_scratch_EXTENDED\rgb3_extended__pt__avg__CBAM__seed0\mod_model_extended\yolo11x-seg_EXT_cbam1_eca0_spec0_db0_gn0_wnnS0.pt"
cbam_pt = r"D:\PD\Publications\Yolo_mod\github\YOLO-Multispectral_2026_01_29\test_notebook_v2_restore\YOLO-Multispectral\code\runs_transfer_vs_scratch_EXTENDED\rgb3_extended__pt__avg__BASE__seed0\mod_model_extended\yolo11x-seg_EXT_cbam0_eca0_spec0_db0_gn0_wnnS0.pt"





m0 = YOLO(base_pt).model.eval()
m1 = YOLO(cbam_pt).model.eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    y0 = m0(x)
    y1 = m1(x)

def first_tensor(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, (list, tuple)):
        for t in o:
            if torch.is_tensor(t):
                return t
    if isinstance(o, dict):
        for v in o.values():
            if torch.is_tensor(v):
                return v
    return None

t0 = first_tensor(y0)
t1 = first_tensor(y1)

print("t0 shape:", tuple(t0.shape))
print("t1 shape:", tuple(t1.shape))
print("max abs diff:", (t0 - t1).abs().max().item())
