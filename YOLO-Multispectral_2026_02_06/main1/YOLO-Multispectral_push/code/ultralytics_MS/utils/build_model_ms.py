from ultralytics import YOLO

def build_yolo_ms_model(
    ckpt_path: str,
    data_yaml: str,
    wavelet_on: bool,
    ms_channels: int,
    patch_yolo_seg_ckpt,
    remap_first_conv_to_ms,
    apply_wavelet_patches,
    device: str = "cuda",
):
    """
    This is the ONLY place you should:
      1) load ckpt
      2) patch seg ckpt
      3) remap first conv
      4) wavelet patch
    Both experiment_runner.py and notebook must call this.
    """
    y = YOLO(ckpt_path)
    model = y.model  # torch.nn.Module

    # 1) patch_yolo_seg_ckpt (checkpoint/model surgery)
    patch_yolo_seg_ckpt(model)

    # 2) multispectral first conv remap
    remap_first_conv_to_ms(model, in_ch=ms_channels)

    # 3) wavelet patches
    if wavelet_on:
        apply_wavelet_patches(model)

    # final: move to device
    model.to(device)

    # optional: store flags for debugging parity
    model.ms_channels = ms_channels
    model.wavelet_on = wavelet_on
    model.data_yaml = data_yaml

    return y  # return YOLO wrapper to train/eval normally
