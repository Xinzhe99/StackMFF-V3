# Source attribution and data conditions

The MIT license in this repository covers our code. It does not replace the terms attached to datasets, third-party implementations, commercial software, or image-derived benchmark outputs. No third-party model weights, original training datasets or commercial applications are redistributed in the comparison archives.

## Evaluation images and derived outputs

The archives contain fused outputs produced for the research comparison. The source images and references must be obtained from their original providers or the [StackMFF V2 benchmark](https://github.com/Xinzhe99/StackMFF-V2#-downloads). Preserve the original attribution and applicable conditions when using or redistributing derived images.

| Source | Attribution and conditions |
|---|---|
| Mobile Depth | Suwajanakorn et al., *Depth from Focus with Your Mobile Phone*, CVPR 2015; the focal-stack version is distributed with the StackMFF benchmark. |
| Middlebury | Scharstein et al., *High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth*, GCPR 2014; [official data page](https://vision.middlebury.edu/stereo/data/). |
| FlyingThings3D | Mayer et al., *A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation*, CVPR 2016. The [provider's terms](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#terms) restrict the data to research and prohibit commercial use; they require citation. Fused outputs derived from these data are not offered as unrestricted MIT assets. |
| Road-MF | The focal-stack benchmark distributed with the StackMFF series; retain the attribution in the benchmark and the earlier papers. |

The code license is not a warranty that the image data are licensed for a proposed commercial application. Original dataset rights remain with their owners.

## Real optical stacks

- **CCCID:** Onishi et al., F1000Research 2026, DOI [10.12688/f1000research.179164.1](https://doi.org/10.12688/f1000research.179164.1). The evaluated files are from [Zenodo record 18904734, version 1.0.0](https://zenodo.org/records/18904734), whose record metadata specifies **CC BY-NC 4.0**. Retain creator attribution, identify registration/fusion as modifications, and observe the noncommercial condition. A newer data version does not silently replace this pinned cohort.
- **WBC:** Park et al., Scientific Data 2024, DOI [10.1038/s41597-024-03938-1](https://doi.org/10.1038/s41597-024-03938-1). The [data record, Figshare 26781052 v1](https://doi.org/10.6084/m9.figshare.26781052.v1), specifies **CC0**. The data license is distinct from the article's publication license.
- **Camera bursts:** Araujo, Ponce and Mairal, [*Towards Real-World Focus Stacking with Deep Learning*](https://arxiv.org/abs/2311.17846), with data linked from their [official repository](https://github.com/araujoalexandre/FocusStackingDataset). Its README does not state a separate data-redistribution license. We retain that source attribution and do not offer the image-derived research results as unrestricted MIT assets; obtain appropriate permission from the original creators for additional redistribution or uses outside inspecting this research. Original camera bursts are not included in our archives.

All real-stack result images are derived from the selected source stacks by shared ECC registration followed by the named fusion method. They are not new raw acquisitions or all-in-focus ground truth. Label/split records refer to the public image identifiers; they are not patient-level validation claims.

## Implementation and metric sources

- Intra-image window attention follows [Swin Transformer](https://github.com/microsoft/Swin-Transformer), MIT license, Copyright (c) 2021 Microsoft. The network here is adapted for variable-length focal stacks and cross-image attention.
- SSIM follows Wang et al. and the `ssim_index.m` reference; VIF follows the LIVE `vifp_mscale.m` implementation; mutual information follows `mutual_info.m`; gradient transfer follows Xydeas and Petrović; MEF-SSIM follows Ma et al. Function-level references are documented in `evaluation/metrics_lib.py`.
- The wordmark retains Noto Sans glyph outlines. Noto Sans is licensed under the [SIL Open Font License 1.1](https://openfontlicense.org/open-font-license-official-text/). The geometric icon is our own vector artwork.
- Baseline source repositories, evaluated commits, runtime accommodations and training-versus-optimization status are recorded in the manuscript supplement and benchmark documentation. Their licenses apply to their original code and weights.

## Training sources

NYU Depth V2, DUTS, DIODE, Cityscapes and ADE20K have their own dataset terms. Estimated depth maps use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), whose model variants have different licenses. This release contains our trained checkpoint, not those training datasets or third-party depth-model checkpoints.

## Swin Transformer license text

```text
MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```
