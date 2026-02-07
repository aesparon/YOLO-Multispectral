# YOLO Multispectral 🚀

<!--
YOLO-Multispectral: Modified Ultralytics YOLOv11 for 4+ band multispectral object detection and instance segmentation.
Keywords: YOLO multispectral, transfer learning, RGB+NIR, UAV imagery, weed detection, remote sensing, instance segmentation.
-->

![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/github/license/aesparon/YOLO-Multispectral)
![Issues](https://img.shields.io/github/issues/aesparon/YOLO-Multispectral)
![Stars](https://img.shields.io/github/stars/aesparon/YOLO-Multispectral)

---

## 📄 Paper & Reproducibility (RSL Submission)

This repository accompanies the paper:

**“Maintaining Transfer Learning in YOLOv11 for 4+-Band Multispectral Remote-Sensing Detection and Segmentation”**  
(*submitted to* **Remote Sensing Letters**)

### 🔒 Reproducible release (paper snapshot)
➡️ **Tagged release:**  
**`rsl-transfer-learning-v1.0`**  
https://github.com/aesparon/YOLO-Multispectral/releases/tag/rsl-transfer-learning-v1.0

This release contains:
- Exact code used for the RSL experiments
- Dataset configuration and training protocol
- Scripts used to generate reported results

> ⚠️ **Important:**  
> The RSL paper evaluates **transfer learning vs training from scratch only**.  
> Attention modules and architectural extensions are **future work** and are **not used** in the paper experiments.

---

## 🔍 What is YOLO Multispectral?

**YOLO Multispectral** is an open-source extension of **Ultralytics YOLOv11**, designed to support **4+ band multispectral remote-sensing imagery** (e.g. RGB + NIR, RedEdge) for **object detection and instance segmentation**.

The project focuses on **minimal architectural modification**, enabling controlled evaluation of:
- Transfer learning from RGB-pretrained weights
- Training-from-scratch for multispectral inputs
- UAV-based agricultural and ecological monitoring

---

## 🌿 Why Multispectral Object Detection?

Standard RGB models often overlook critical spectral cues related to vegetation structure and physiology.  
By incorporating **near-infrared (NIR)** and **red-edge (RE)** bands, multispectral models can improve discrimination under:

- Variable illumination
- Complex soil backgrounds
- Overlapping foliage
- Early growth-stage weeds

---

## 💡 Key Features (Paper-Relevant)

- ✅ YOLOv11-seg instance segmentation
- ✅ 4+ channel multispectral input (RGB, RGB+NIR+RE)
- ✅ Transfer learning from RGB-pretrained weights
- ✅ Training-from-scratch baselines
- ✅ TIFF / GeoTIFF input support
- ✅ UAV-scale high-resolution imagery

---

## 🧠 Architectural Extensions (Future Work)

The repository also contains **experimental modules** intended for future research and **not used in the RSL paper**:

| Module        | Purpose                           |
|---------------|-----------------------------------|
| CBAM          | Channel & spatial attention       |
| ECA           | Lightweight channel attention     |
| SpectralConv  | Band-specific spectral filtering  |
| DropBlock     | Spatial regularisation            |
| GroupNorm     | Robust normalisation              |

These components are under active development and will be evaluated in **subsequent studies**.

---

## 📊 Example Results (WeedsGalore)

<p align="center">
  <br>
  <em>
  Example RGB and RGB+NIR+RE instance segmentation results on the
  <a href="https://github.com/GFZ/weedsgalore">WeedsGalore dataset</a>.
  </em>
  
</p>




**Table 1. Dataset split statistics**

| Split      | Images | Crop Instances | Weed Instances | Total Instances |
|------------|-------:|---------------:|---------------:|----------------:|
| Train      |    104 |          1,461 |          6,512 |           7,973 |
| Validation |     26 |            451 |          1,808 |           2,259 |
| Test       |     26 |            257 |          1,711 |           1,968 |
| **Total**  | **156**|        **2,169**|       **10,031**|        **12,200** |




**Table S2. Effect of transfer learning on instance-segmentation performance (mask mAP50) for selected backbone modifications within the YOLO-Multispectral framework.**  
Results are reported separately for training from scratch and transfer-learning initialisation, using identical dataset splits, training schedules, optimisation parameters, and evaluation protocols as described in Section 3. Reported mAP50 values correspond to the best validation performance achieved during training, and the associated Epoch denotes the mean epoch at which this best performance occurred across runs. Early stopping was applied uniformly across all experiments using identical patience criteria. Apparent differences in training duration therefore reflect differences in optimisation stability rather than differences in training protocol.

| Bands       | Backbone modifications     | Scratch mAP50      | Scratch Epoch | Transfer mAP50     | Transfer Epoch |
|-------------|----------------------------|-------------------:|-------------:|-------------------:|---------------:|
| RGB         | None (baseline)            | 46.79 ± 0.56       |          467 | 51.16 ± 0.26       |            163 |
| RGB         | CBAM                       | <0.01              |            1 | 49.13 ± 0.25       |            344 |
| RGB         | CBAM + Stage-1 WNN         | <0.011             |           79 | 48.82 ± 3.10       |            424 |
| RGB         | Stage-1 WNN                | 46.75 ± 0.80       |          471 | 49.64 ± 0.00       |            200 |
| RGB         | ECA                        | 47.81 ± 0.85       |          389 | 51.95 ± 0.86       |            186 |
| RGB         | DropBlock                  | 46.81 ± 0.21       |          572 | 52.01 ± 0.64       |            452 |
| RGB         | GroupNorm                  | 38.84 ± 1.81       |          507 | 40.69 ± 0.00       |            523 |
| RGB         | Spectral                   | 29.66 ± 15.56      |          217 | 41.16 ± 3.68       |            538 |
| RGB+NIR+RE  | None (baseline)            | 51.57 ± 1.10       |          511 | 56.42 ± 0.00       |            237 |
| RGB+NIR+RE  | CBAM                       | <0.011             |            1 | 52.90 ± 1.00       |            348 |
| RGB+NIR+RE  | CBAM + Stage-1 WNN         | <0.011             |           35 | 52.73 ± 0.98       |            334 |
| RGB+NIR+RE  | Stage-1 WNN                | 50.81 ± 1.33       |          401 | 55.80 ± 0.00       |            118 |
| RGB+NIR+RE  | ECA                        | 51.09 ± 1.12       |          357 | 55.40 ± 0.82       |            280 |
| RGB+NIR+RE  | DropBlock                  | 50.55 ± 0.37       |          556 | 56.99 ± 0.29       |            368 |
| RGB+NIR+RE  | GroupNorm                  | 40.57 ± 1.09       |          517 | 43.61 ± 0.00       |            593 |
| RGB+NIR+RE  | Spectral                   | 28.17 ± 11.25      |           73 | 45.87 ± 4.06       |            413 |





---

## 🚀 Reproduce the Experiments

Detailed instructions are provided in:

- 📁 `paper/README.md` – paper context & experiment overview  
- 📁 `repro/README.md` – exact commands to reproduce results

### Google Colab
Run the multispectral YOLO demo in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb
)

---

## 📖 Citation

If you use this repository, please cite:

```bibtex
@misc{esparon_yolo_multispectral_2025,
  author = {Esparon, Andrew James},
  title  = {YOLO-Multispectral: Maintaining Transfer Learning in YOLOv11 for Multispectral Remote Sensing},
  year   = {2025},
  url    = {https://github.com/aesparon/YOLO-Multispectral},
}
