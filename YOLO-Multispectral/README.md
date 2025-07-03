## YOLO-MultiSpectral: Object Detection and Segmentation for Multispectral images.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Under%20Review-orange)]()
[![Stars](https://img.shields.io/github/stars/aesparon/YOLO-MultiSpectral.svg?style=social)](https://github.com/aesparon/YOLO-MultiSpectral/stargazers)
[![DOI](https://zenodo.org/badge/123456789.svg)](https://zenodo.org/badge/latestdoi/123456789)

> ***Object detection and segmentation in 4+ channel multispectral imagery, powered by enhanced YOLOv3-v12+ backbones.***

---

## 🔍 Overview

**YOLO-MultiSpectral** is a deep learning framework based on YOLO that supports multispectral imagery with 4 or more bands (e.g., RGB + NIR + RedEdge) for object detection and instance segmentation. Designed for UAV imagery, precision agriculture, and environmental monitoring, it integrates spatial-spectral attention mechanisms like **CBAM** and **ECA** to enhance accuracy and generalization.

---

## 🚀 Features
- ✅ 4+ band multispectral TIFF input support (current support for uint8 and soon to be modified for uint16)
- ✅ +10% mAP@50 gain over standard RGB images
- ✅ Ability to leverage transfer learning to MultiSpectral inputs
- ✅ Attention modules for spectral feature enhancement:
  - CBAM (Convolutional Block Attention Module)
  - ECA (Efficient Channel Attention) 
  - Spectral-aware Convolutions
  - DropBlock and GroupNorm
  - Safe gradient clipping and early stopping
- ✅ Optimized for QGIS workflows and UAV imagery
- ✅ Modular and open source architecture for easy integration and extension

---


## 🧪 Quick Start using Weeds-galore dataset 

This repo includes training support for the **Weeds-Galore** dataset:
- 5-band UAV imagery (R, G, B, NIR, RedEdge)
- Classes: *maize, amaranth, grass, quickweed, other*
- Achieved **+10% mAP@50** over YOLO (RGB)


## 🎥 Watch this for overview (UNDER CONSTUCTION)

[![Coming soon](https://img.shields.io/badge/Demo-YouTube-red)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

### Run Instantly on Google Colab
<a href="https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="height:30px;">
</a>


## 📁 Preliminary results

ADD EXCEL RESULTS










## 📁 Repository Structure
```
├── datasets/
│   └── weeds-galore/             # Optional example dataset
├── utils/
│   ├── patch_backbone_with_attention.py
│   └── mod_pt_model.py          # CBAM, ECA definitions
├── models/
├── train_evaluate.py            # Main training/evaluation logic
├── requirements.txt
├── README.md
├── CITATION.cff
└── LICENSE
```

---

## 📦 Local installation

### Requirements:
```bash
ADD STEPS HERE
```



---

## 🧩 Extending YOLO-MultiSpec

YOLO-MultiSpectral is modular by design. You can:
- 💡 Add new attention modules (e.g., SE, CBNet, Transformer blocks)
- 🔄 Swap backbone (e.g., ResNet, CSPDarknet, ConvNeXt)
- 🧠 Modify loss functions (e.g., Focal-EIoU, GIoU)
- 📊 Integrate NDVI/NDRE indices into the model input

We welcome contributions — see [CONTRIBUTING.md](CONTRIBUTING.md)!

---


---

## 📦 Citation
"If you use this software, please cite it as below.":
Currently under pre-publication review. Not yet submitted for Journal publication (RSL)
```bibtex
@misc{yolo-multispectral,
  author       = {Andrew Esparon},
  title        = {YOLO-MultiSpectral: A Deep Learning Framework for Multispectral Object Detection and Instance Segmentation},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/aesparon/YOLO-MultiSpectral}},
  doi          = {to be added upon publication}
}
```

---

## 📬 Contact
**Andrew Esparon**  
📧 andrew.esparon@cdu.edu.au  
🌐 [Charles Darwin University](https://www.cdu.edu.au)
🌐 [Office of the Supervising Scientist](https://www.dcceew.gov.au/science-research/supervising-scientist)

---

## 📝 License
AGPL-3.0 License – See [LICENSE](LICENSE)

---

## 🔖 Keywords
`Multispectral` · `YOLOv8` · `YOLOv11` · `YOLOv12` · `deep learning` · `weeds galore` · `CNN` · `Object Detection` · `Instance Segmentation` · `Remote Sensing AI` · `Agricultural Deep Learning` · `CBAM` · `5-channel Imagery` · `Precision Agriculture` · `Geospatial Deep Learning`
