<div align="center">

# <img src="assets/stackmffv3_framework.jpg" alt="StackMFF V3" height="320" style="vertical-align: middle;"/> StackMFF V3

**General Multi-focus Image Fusion Network**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-StackMFF--V3-black.svg)](https://github.com/Xinzhe99/StackMFF-V3)

*Official PyTorch implementation for General Multi-focus Image Fusion Network*

</div>

## 📢 News

> [!NOTE]
> 🎉 **2025.11**: Our article has been submitted, so please stay tuned. The complete code and data will be added after it is accepted.
</div>

##  Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Downloads](#-downloads)
- [Usage](#-usage)
- [Citation](#-citation)

## 📖 Overview
Multi-focus image fusion is a computational imaging technique that overcomes the depth-of-field limitation of optical systems by integrating information from multiple focal planes into an all-in-focus image. Recently, learning-based multi-focus image fusion approaches have attracted growing attention. Among them, the StackMFF Series has progressively advanced the paradigm from image pairs to image stacks. Its first-generation model effectively mitigated error accumulation during fusion but failed to preserve the fidelity of the fused image. The second-generation model further incorporated an ordered-focus prior, providing an open-source solution whose performance rivals that of commercial software. Nevertheless, it assumes ideal inputs, requiring a well-ordered multi-focus image stack without defocused or invalid layers. To eliminate this constraint and enhance generality, we propose StackMFF V3, the first general multi-focus image fusion network featuring a redesigned architecture and training strategy. It first employs a Pyramid Fusion MLP to model long-range intra-layer dependencies and estimate layer-wise focus, followed by the proposed Pixel-wise Cross-layer Attention module, which efficiently captures cross-layer relations without relying on focus order. Finally, we formulate focus map generation as a pixel-wise multi-class classification task, directly predicting the focus map used to synthesize the fused image. Extensive experiments demonstrate that StackMFF V3 is currently the most versatile and comprehensive model, achieving state-of-the-art performance across diverse benchmarks and real-world applications.

<div align="center">
<img src="assets/stackmffv3_framework.jpg" width="800px"/>
<p>Overview of StackMFF-V3 Framework</p>
</div>

## ✨ Highlights

🌟 Presents the first general multi-focus image fusion network.
🔑 Reformulates stack fusion as a pixel-wise multi-class classification task.
🛠️ Employs an MLP-based backbone for intra-layer focus estimation with global context.
🎯 Proposes a \textit{Pixel-wise Cross-layer Attention} module for inter-layer modeling.
🏆 Provides an open-source solution that outperforms commercial software at low cost.
 
## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Xinzhe99/StackMFF-V3.git
cd StackMFF-V3
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n stackmffv3 python=3.8
conda activate stackmffv3
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Data Preparation

We provide the test datasets used in our paper for research purposes. These datasets were used to evaluate the performance of our proposed method and compare with other state-of-the-art approaches:
- Mobile_Depth
- Middlebury
- FlyingThings3D
- Road_MF

## 📥 Downloads

| Resource | Link | Code | Description |
|----------|------|------|-------------|
| 🗂️ **Test Datasets** | [![Download](https://img.shields.io/badge/Download-4CAF50?style=flat-square)](https://pan.baidu.com/s/1VbdYvN5_H8X08wLgyww0iw?pwd=cite) | `cite` | Complete evaluation datasets |
| 📊 **Benchmark Results** | [![Download](https://img.shields.io/badge/Download-FF9800?style=flat-square)](https://pan.baidu.com/s/1E1jMPWQH9QHmmjlD06Tw4w?pwd=cite) | `cite` | Fusion results from all methods |

These are the exact datasets used in our quantitative evaluation and computational efficiency analysis. After downloading, please organize the datasets following the structure described in the [Predict Dataset](#predict-dataset) section.

The `make_datasets` folder contains all the necessary code for processing and splitting the training datasets:

- `ADE/1_extract.py`: Extracts and organizes images from the [ADE20K](https://hyper.ai/cn/datasets/5212) dataset
- `DUTS/filter.py`: Filters out images with uniform backgrounds from the [DUTS](https://hyper.ai/cn/datasets/16458) dataset
- `DIODE/extract_from_ori.py`: Processes and converts images from the [DIODE](https://hyper.ai/cn/datasets/19918) dataset
- `NYU V2 Depth/`: Processing the original [NYU V2 Depth](https://hyper.ai/cn/datasets/5376) dataset
  - `1_crop_nyu_v2.py`: Crops RGB and depth images to remove boundary artifacts
  - `2_nyu_depth_norm.py`: Normalizes depth maps to a standard range
  - `3_split.py`: Splits the dataset into training and testing sets
- `Cityscapes/1_move.py`: Reorganizes the [Cityscapes](https://hyper.ai/cn/datasets/5205) dataset into a flattened structure
- `make_dataset.py`: Generates multi-focus image stacks using depth maps

For depth maps, except for the NYU Depth V2 dataset which uses its own depth maps, all other depth maps are obtained through inference using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

## 💻 Usage

The pre-trained model weights file [`stackmffv3_star.pth` and `stackmffv3_star.pth`](https://pan.baidu.com/s/1Xhs-VBV3ZVfAmNFTM5306g?pwd=cite) should be placed in the `weights` directory.

### Predict Single Stack

```bash
python predict_one_stack_star.py \
    --model_path weights/stackmffv3_star.pth \
    --input_dir path/to/input/stack \
    --output_dir path/to/output
```

### Predict Dataset

For batch testing multiple datasets, organize your test data as follows:

```
test_datasets/
├── Mobile_Depth/
│   └── dof_stack/
│       ├── scene1/
│       │   ├── 1.png
│       │   ├── 2.png
│       │   └── ...
│       └── scene2/
│           ├── 1.png
│           ├── 2.png
│           └── ...
├── Middlebury/
│   └── dof_stack/
│       ├── scene1/
│       └── scene2/
├── FlyingThings3D/
│   └── dof_stack/
├── Road-MF/
│   └── dof_stack/
└── NYU-V2/
    └── dof_stack/
```

Each dataset folder (e.g., Mobile_Depth, Middlebury) should contain a `dof_stack` subfolder with multiple scene folders. Each scene folder contains the multi-focus image stack numbered sequentially.

### Predict Image Pair Datasets

For processing image pair datasets with A/B folder structure, use the `predict_pair_datasets_star.py` script. This script processes each dataset independently, similar to `predict_datasets_star.py`, and is specifically designed for datasets where images are organized as pairs in separate 'A' and 'B' subfolders.

Organize your image pair datasets as follows:

```
test_root/
├── dataset1/
│   ├── A/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── B/
│       ├── 1.png
│       ├── 2.png
│       └── ...
├── dataset2/
│   ├── A/
│   └── B/
└── dataset3/
    ├── A/
    └── B/


### Some training details
- **Case 1 (multi-GPU training):**  
  - **OS:** Ubuntu 
  - **GPU:** 2 × NVIDIA RTX A6000 
  - **Hyperparameters:** original settings  
  - **Training time:** ~3.54 h per epoch

## 📚 Citation

If you use this project in your research, please cite our papers:

<details>
<summary>📋 BibTeX</summary>

```bibtex
@article{xie2025stackmffv2,
title = {One-shot multi-focus image stack fusion via focal depth regression},
journal = {Engineering Applications of Artificial Intelligence},
volume = {162},
pages = {112667},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.112667},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625026983},
author = {Xinzhe Xie and Buyu Guo and Shuangyan He and Yanzhen Gu and Yanjun Li and Peiliang Li},
keywords = {Multi-focus image fusion, Focus measure, Computational photography, Image stack processing},
abstract = {Multi-focus image fusion is a vital computational imaging technique for applications that require an extended depth of field, including medical imaging, microscopy, professional photography, and autonomous driving. While existing methods excel at fusing image pairs, they often suffer from error accumulation that leads to quality degradation, as well as computational inefficiency when applied to large image stacks. To address these challenges, we introduce a one-shot fusion framework that reframes image-stack fusion as a focal-plane depth regression problem. The framework comprises three key stages: intra-layer focus estimation, inter-layer focus estimation, and focus map regression. By employing a differentiable soft regression strategy and using depth maps as proxy supervisory signals, our method enables end-to-end training without requiring manual focus map annotations. Comprehensive experiments on five public datasets demonstrate that our approach achieves state-of-the-art performance with minimal computational overhead. The resulting efficiency and scalability make the proposed framework a compelling solution for real-time deployment in resource-constrained environments and lay the groundwork for broader practical adoption of multi-focus image fusion. The code is available at https://github.com/Xinzhe99/StackMFF-V2.}
}
@article{xie2025stackmff,
  title={StackMFF: end-to-end multi-focus image stack fusion network},
  author={Xie, Xinzhe and Qingyan, Jiang and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={503},
  year={2025},
  publisher={Springer}
}

```

</details>

## 🙏 Acknowledgments

TBD
⭐ If you find this project helpful, please give it a star!
</div>
