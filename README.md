# YOLO Multispectral 🚀

## Maintaining RGB-pretrained transfer learning in YOLOv11x-seg for multispectral remote-sensing instance segmentation

This repository accompanies the manuscript:

**Maintaining RGB-pretrained transfer learning in YOLOv11x-seg for multispectral remote-sensing instance segmentation**

Submitted to *Remote Sensing Letters* as a Method article.

---

## 📄 Paper & Reproducibility

This repository provides code, experiment configuration files, plotting scripts, result summaries and reproducibility material for a controlled multispectral YOLO transfer-learning study.

The submitted paper evaluates whether RGB-pretrained transfer learning remains effective when **YOLOv11x-seg** is adapted from RGB input to five-band multispectral input using the public **WeedsGalore** UAV dataset.

The main comparison uses four controlled experimental conditions:

- **RGB-scratch**
- **RGB-transfer**
- **MS5-scratch**
- **MS5-transfer**

The core finding is that **RGB-pretrained representations remain useful after multispectral input expansion**, and that multispectral input and RGB-pretrained transfer learning provide complementary benefits.

### 🔒 Reproducible release

A reproducible paper snapshot is provided as a tagged release:

➡️ **Release:** `rsl-transfer-learning-v1.0`

https://github.com/aesparon/YOLO-Multispectral/releases/tag/rsl-transfer-learning-v1.0

This release is intended to preserve the code, dataset configuration and scripts used for the Remote Sensing Letters submission.

---

## 🔍 What is YOLO Multispectral?

**YOLO Multispectral** is an open-source extension of Ultralytics YOLO designed to support multispectral remote-sensing imagery with more than three input channels.

The submitted paper focuses on **YOLOv11x-seg** and a five-band multispectral input configuration, referred to as **MS5**:

- RGB
- near-infrared \(NIR\)
- red-edge

The project is designed around a minimal and reproducible model adaptation:

> only the first YOLOv11x-seg input-stem convolution is expanded from 3 to 5 input channels, while the remaining backbone, neck and segmentation head are kept unchanged.

This design allows controlled testing of whether RGB-pretrained weights can still be useful when adapting a YOLO segmentation model to multispectral imagery.

---

## 🌿 Why Multispectral Object Detection?

Remote-sensing object detection and instance segmentation often require more than visible RGB information.

In vegetation, agriculture and ecological monitoring, additional spectral bands such as **NIR** and **red-edge** can provide useful information related to vegetation structure, contrast and condition.

Multispectral imagery can help when targets are difficult to separate using RGB alone, including scenes with:

- variable illumination;
- complex soil and plant backgrounds;
- dense or overlapping vegetation;
- early growth-stage weeds;
- visually similar crop and weed instances;
- small and irregular object boundaries.

However, most large pretrained computer-vision models are trained on RGB imagery. This raises an important practical question:

> Can RGB-pretrained YOLO representations still be useful after the input stem is expanded to accept multispectral imagery?

This repository addresses that question using a controlled scratch-versus-transfer experiment.

---

## 💡 Key Features \(Paper-Relevant\)

- ✅ YOLOv11x-seg instance segmentation
- ✅ Five-band multispectral input: RGB+NIR+red-edge \(MS5\)
- ✅ Minimal input-stem modification from 3 to 5 channels
- ✅ RGB-pretrained transfer learning retained for multispectral input
- ✅ Scratch-training baselines for both RGB and MS5
- ✅ Controlled comparison across matched dataset splits and training settings
- ✅ Five random seeds \(0–4\)
- ✅ Held-out test-set reporting as mean ± standard deviation
- ✅ Bounding-box and instance-mask metrics
- ✅ Supplementary CBAM, ECA and YOLOv26 compatibility checks
- ✅ Reproducibility material for the submitted manuscript

---

## 🧠 Main Research Question

Does RGB-pretrained transfer learning remain effective when YOLOv11x-seg is adapted from RGB to five-band multispectral input for remote-sensing object detection and instance segmentation?

---

## 🧩 Main Model Adaptation

The base model is **YOLOv11x-seg**.

For RGB experiments, the standard three-channel input configuration is used.

For MS5 experiments, only the first convolutional layer of the input stem is expanded from three to five channels.

The remaining model components are kept unchanged:

- backbone;
- neck;
- segmentation head.

### RGB-pretrained MS5 initialisation

For RGB-pretrained five-band models:

- weights for the original RGB channels are retained from the RGB-pretrained model;
- the added NIR and red-edge channel weights are initialised using the average of the pretrained RGB input-stem weights.

This keeps the transfer-learning comparison controlled and reproducible.

### Scratch initialisation

Scratch-trained models are randomly initialised using the standard non-pretrained YOLO initialisation pathway.

---

## 📦 Dataset

Experiments use the public **WeedsGalore** multispectral UAV dataset.

The dataset provides 600 × 600 pixel image tiles of RGB, NIR and red-edge imagery with semantic and instance annotations for crop and weed segmentation in maize fields.

For this study, annotations are grouped into two instance-segmentation classes:

- crop;
- weed.

The imagery has a ground sampling distance of approximately **2.5 mm per pixel**, giving each 600 × 600 tile an approximate ground footprint of **1.5 × 1.5 m**.

### Dataset split used in this study

| Split | Images | Crop instances | Weed instances | Total instances |
|---|---:|---:|---:|---:|
| Train | 104 | 1,461 | 6,512 | 7,973 |
| Validation | 26 | 451 | 1,808 | 2,259 |
| Test | 26 | 257 | 1,711 | 1,968 |
| **Total** | **156** | **2,169** | **10,031** | **12,200** |

---

## 🧪 Main Experimental Conditions

| Condition | Input | Initialisation | Description |
|---|---|---|---|
| RGB-scratch | RGB | Random | Standard RGB input, trained from scratch |
| RGB-transfer | RGB | RGB-pretrained | Standard RGB input with pretrained YOLOv11x-seg initialisation |
| MS5-scratch | RGB+NIR+red-edge | Random | Five-band input-stem adaptation, trained from scratch |
| MS5-transfer | RGB+NIR+red-edge | RGB-pretrained | Five-band input-stem adaptation with retained RGB-pretrained weights |

All four main experimental conditions use matched:

- dataset partitions;
- two-class crop/weed label structure;
- image size;
- batch size;
- augmentation settings;
- optimisation settings;
- validation procedure;
- early-stopping settings;
- evaluation protocol.

---

## 📏 Evaluation Metrics

YOLOv11x-seg jointly predicts:

- bounding boxes;
- instance masks.

Therefore, both bounding-box and mask metrics are reported.

The main reported metrics are:

- **mAP50\(B\)**: bounding-box mAP at IoU 0.50
- **mAP50-95\(B\)**: bounding-box mAP averaged over IoU 0.50–0.95
- **mAP50\(M\)**: mask mAP at IoU 0.50
- **mAP50-95\(M\)**: mask mAP averaged over IoU 0.50–0.95

Mask mAP50, **mAP50\(M\)**, is treated as the primary metric because the study focuses on instance segmentation.

All test-set values are reported as **mean ± standard deviation** across five seed-specific best-validation checkpoints.

---

## 📊 Main Test-Set Results

Held-out test-set performance of YOLOv11x-seg trained with instance mask annotations.

Metrics are reported on a 0–100 AP scale.

| Condition | mAP50\(B\) | mAP50-95\(B\) | mAP50\(M\) | mAP50-95\(M\) |
|---|---:|---:|---:|---:|
| **MS5-transfer** | **79.23 ± 0.87** | **52.54 ± 1.32** | **74.14 ± 0.56** | **34.25 ± 0.90** |
| RGB-transfer | 72.63 ± 0.84 | 45.25 ± 0.98 | 67.99 ± 1.16 | 28.44 ± 0.90 |
| MS5-scratch | 76.57 ± 0.66 | 48.38 ± 0.87 | 71.40 ± 0.66 | 31.60 ± 0.68 |
| RGB-scratch | 71.29 ± 0.71 | 42.72 ± 0.98 | 65.12 ± 0.57 | 25.38 ± 0.81 |

---

## 🔑 Key Findings

The best-performing configuration was **MS5-transfer**.

Compared with **RGB-transfer**, MS5-transfer improved:

- **mAP50\(B\)** by **6.6 percentage points**
- **mAP50\(M\)** by **6.2 percentage points**

Compared with **MS5-scratch**, MS5-transfer improved:

- **mAP50\(B\)** by **2.7 percentage points**
- **mAP50\(M\)** by **2.7 percentage points**

MS5-scratch also outperformed RGB-transfer on the held-out test set. This indicates that the additional NIR and red-edge bands provide useful task-specific information even without pretrained initialisation.

Overall, the results indicate that:

- multispectral input improves performance;
- RGB-pretrained transfer learning improves performance;
- the two benefits are complementary rather than mutually exclusive;
- RGB-pretrained representations remain useful after moderate multispectral input expansion.

---

## 📈 Validation-Curve Summary

Validation learning curves showed the same ordering for both bounding-box and mask mAP50:

1. MS5-transfer
2. MS5-scratch
3. RGB-transfer
4. RGB-scratch

Best averaged validation points were:

| Condition | Validation mAP50\(B\) | Validation mAP50\(M\) |
|---|---:|---:|
| MS5-transfer | 78.9 | 72.1 |
| MS5-scratch | 76.1 | 69.8 |
| RGB-transfer | 73.8 | 67.0 |
| RGB-scratch | 70.7 | 63.7 |

Transfer-learning runs reached strong validation performance earlier than scratch-trained runs, supporting their practical value for iterative remote-sensing model development.

---

## 🧠 Supplementary Compatibility Experiments

Supplementary experiments are included to test whether the main transfer-learning pattern persists when lightweight attention modules or updated YOLO implementation code are introduced.

These experiments are **not** the central benchmark contribution. They are included as compatibility checks.

### CBAM and ECA attention extensions

The supplementary attention experiments test selected lightweight feature-reweighting modules:

| Module | Description |
|---|---|
| CBAM | Convolutional Block Attention Module |
| ECA | Efficient Channel Attention |

The attention modules are inserted after selected internal convolutional feature blocks in the YOLO backbone/neck feature-extraction pathway.

The segmentation prediction head is not modified.

The supplementary results show that:

- ECA follows a similar transfer-learning pattern to the baseline configuration;
- CBAM reduces performance in this setting;
- scratch-trained CBAM fails to converge to a useful solution, with validation mask mAP50 remaining near zero.

These results support the practical conclusion that attention modules should be treated as optional refinements, not replacements for a strong pretrained baseline.

### YOLOv26 compatibility check

The same minimal five-band input-stem adaptation was also ported to an updated YOLOv26 segmentation codebase.

This compatibility check tests whether the input-stem modification can be reapplied as YOLO implementations evolve.

The updated implementation shows the same qualitative pattern as the main YOLOv11x-seg experiments, with RGB-pretrained initialisation remaining beneficial after multispectral input expansion.

---

## 📎 Supplementary Material

The supplementary material includes:

- **Supplementary Figure S1**: location of CBAM and ECA attention modifications in selected YOLO convolutional feature blocks.
- **Supplementary Figure S2**: validation mask mAP50 learning curves for CBAM and ECA attention-extension experiments.
- **Supplementary Figure S3**: validation bounding-box and mask mAP50 learning curves for the YOLOv26 implementation compatibility check.
- **Supplementary Table S1**: training and evaluation settings for the main YOLOv11x-seg experiments.

---

## ⚙️ Training and Evaluation Settings

| Setting | Value |
|---|---|
| Dataset | WeedsGalore |
| Split protocol | Official train, validation and test partitions |
| Classes | Two classes: crop and weed |
| Input configurations | RGB; MS5 |
| Model | YOLOv11x-seg |
| Multispectral modification | First convolutional input-stem layer expanded from 3 to 5 channels |
| Unchanged model components | Backbone, neck and segmentation head |
| Pretrained RGB channels | Retained from RGB-pretrained weights |
| Added NIR/red-edge channels | Initialised from the average of pretrained RGB stem weights |
| Scratch models | Random initialisation |
| Main experimental conditions | RGB-scratch; RGB-transfer; MS5-scratch; MS5-transfer |
| Random seeds | 0, 1, 2, 3, 4 |
| Maximum epochs | 500 |
| Early stopping patience | 50 epochs |
| Best checkpoint selection | Best validation mask mAP50 checkpoint for each seed |
| Primary validation metric | Mask mAP50 \[mAP50\(M\)\] |
| Reported test metrics | mAP50\(B\), mAP50-95\(B\), mAP50\(M\), mAP50-95\(M\) |
| Result summary | Mean ± standard deviation across five seed-specific best-validation checkpoints |
| Learning-curve averaging | Complete-case epoch-aligned mean across five seeds |
| Image size | 600 |
| Batch size | 8 |
| Hardware | NVIDIA RTX 5090 |
| Software | Modified Ultralytics YOLOv11x-seg source |
| Other settings | Default Ultralytics YOLOv11x-seg segmentation settings unless otherwise specified |

---

## 🚀 Reproduce the Experiments

Detailed reproduction material is provided in the repository.

Recommended starting points:

- `paper/README.md` — paper context and experiment overview
- `repro/README.md` — exact reproduction notes and commands
- `datasets/` — dataset configuration files
- `code/` — training, evaluation and plotting scripts
- `assets/` — figures and supporting material
- `docs/` — supplementary notes and additional documentation

### Google Colab
Run the multispectral YOLO demo in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/notebooks/rsl_transfer_learning_repro_V3.ipynb
)

---

## 📖 Citation

If you use this repository, please cite:

```bibtex
@misc{esparon_yolo_multispectral_2026,
  author = {Esparon, Andrew and Gautam, Deepak},
  title = {YOLO-Multispectral: RGB-pretrained transfer learning for multispectral remote-sensing instance segmentation},
  year = {2026},
  url = {https://github.com/aesparon/YOLO-Multispectral}
}