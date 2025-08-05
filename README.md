# YOLO-Multispectral 🚀

<!--
  YOLO-Multispectral: Modified Ultralytics YOLOv8/v11 for 4+ multispectral object detection and segmentation.
  Keywords: YOLO multispectral, multispectral object detection, RGBN, RGB+NIR, YOLOv8, YOLOv11, drone imagery, vegetation segmentation, CBAM, ECA, geospatial deep learning, QGIS, remote sensing.
-->

![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/github/license/aesparon/YOLO-Multispectral)
![Issues](https://img.shields.io/github/issues/aesparon/YOLO-Multispectral)
![Stars](https://img.shields.io/github/stars/aesparon/YOLO-Multispectral)

---  

## 🔍 What is YOLO-Multispectral?

**YOLO-Multispectral** is an open-source extension of **Ultralytics YOLOv8/v11**, tailored to support **4+ band multispectral remote sensing imagery** (e.g., RGB + NIR, RedEdge) for **object detection** and **semantic segmentation**.  
It includes support for **geospatial metadata**, **TIFF inputs**, and **attention-enhanced backbones** (CBAM, ECA).

---

## 🌿 Why Multispectral Object Detection?

Standard RGB models often overlook critical spectral cues found in vegetation, soil, and water. By integrating **NIR** and **RedEdge** bands, this model provides superior accuracy for:
- Weed detection
- Crop health monitoring
- Ecological surveys
- Land cover classification

---

## 💡 Key Features

- ✅ TIFF multispectral input (4+ channels)
- ✅ Custom YOLOv8/v11/v12 models with 4+ input patching
- ✅ Transfer learning from RGB to multispectral weights
- ✅ SpectralConv, CBAM, ECA, DropBlock, and GroupNorm modules
- ✅ Mask + Box segmentation support (v8/v11); Box-only for v12
- ✅ GeoTIFF & tiled dataset compatibility (e.g., QGIS-ready outputs)


---

## 📊 Results

These are preliminary results applying YOLO-multispectral to weeds-galore public dataset [weeds-galore dataset](https://arxiv.org/abs/1807.06521) .
<p align="center">
  <img src="assets/figure_2_and 5_class.jpg" alt=" 2- class and 5-Class Detection Example" width="80%">
  <br>
  <em>Figure 1: Example results for RGB with and without transfer learning and RGB, red edge, and NIR multispectral weed segmentation ([Dataset weeds galore](https://github.com/GFZ/weedsgalore).). Refer Colbab rxample below to duplicate results</em>
</p>




## 🧠 Model Enhancements

| Module        | Purpose                           | Reference |
|---------------|-----------------------------------|-----------|
| CBAM          | Channel & Spatial Attention       | [CBAM 2018](https://arxiv.org/abs/1807.06521) |
| ECA           | Lightweight Channel Attention     | [ECA 2020](https://arxiv.org/abs/1910.03151) |
| SpectralConv  | Band-specific Spectral Filtering  | *Inspired by 3D CNNs for HSI* |
| DropBlock     | Spatial Dropout Regularization    | [DropBlock 2018](https://arxiv.org/abs/1810.12890) |
| GroupNorm     | Robust for Small Batches          | [GN 2018](https://arxiv.org/abs/1803.08494) |

---

## 📦 Dataset Support

- ✅ **MicaSense / RedEdge**: 5-10 band drone imagery
- ✅ **WeedsGalore**: RGB + NIR semantic segmentation
- ✅ **LLVIP**: Infrared and visible-light object detection
- ✅ Custom TIFF stacks with spectral metadata

---

## 🚀 Quick Links

### 📘 Example Notebook (in progress)
This notebook demonstrates how to to quickly use modified YOLO models for 5 channel input using weeds-galore dataset.

[Dataset weeds galore](https://github.com/GFZ/weedsgalore). 
RGB and RGBRN (red,green,blue,red edge, Near infra red)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb)

> 🛠️ **Colab demo coming soon (ETA: 11/7/2025)**

### 🎥 Demo Video (coming soon)

[![YouTube](https://img.shields.io/badge/Demo-YouTube-red)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

---

## 📖 Citation

If you use **YOLO-Multispectral** in your research, please cite:

```bibtex
@misc{yolo-multispectral,
  author = {Esparon, Andrew},
  title = {YOLO-Multispectral: Multispectral Object Detection and Segmentation using YOLOv8/v11},
  year = {2025},
  howpublished = {\url{https://github.com/aesparon/YOLO-Multispectral}},
}
