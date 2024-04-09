<div align="center">
<img src="assets/logo_512.png" width="200">
<h1>MambaAD </h1>
<h3>MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection</h3>

[Haoyang He<sup>1#</sup>](https://scholar.google.com/citations?hl=zh-CN&user=8NfQv1sAAAAJ),
[Yuhu Bai<sup>1#</sup>](https://scholar.google.com/citations?hl=zh-CN&user=ucCvgooAAAAJ),
[Jiangning Zhang<sup>2*</sup>](https://zhangzjn.github.io),
[Qingdong He<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=gUJWww0AAAAJ),
[Hongxu Chen<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=uFT3YfMAAAAJ),

[Zhenye Gan<sup>2</sup>](https://scholar.google.com/citations?user=fa4NkScAAAAJ&hl=zh-CN),
[Chengjie Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ),
[Xiangtai Li<sup>3</sup>](https://lxtgh.github.io/),
[Guanzhong Tian<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=0q-7PI4AAAAJ),
[Lei Xie<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=7ZZ_-m0AAAAJ)

(#Equal contribution, *Project Leader)

<sup>1</sup>College of Control Science and Engineering, Zhejiang University, 
<sup>2</sup>Youtu Lab, Tencent,
<sup>3</sup>Nanyang Technological University, Singapore

[[`Paper`]()] 
[[`Project Page`](https://lewandofskee.github.io/projects/MambaAD/)]

Our MambaAD is based on [ADer](https://github.com/zhangzjn/ADer). We will release the full code within a week.

</div>

## Abstract
Recent advancements in anomaly detection have seen the efficacy of CNN- and transformer-based approaches. However, CNNs struggle with long-range dependencies, while transformers are burdened by quadratic computational complexity. Mamba-based models, with their superior long-range modeling and linear efficiency, have garnered substantial attention. This study pioneers the application of Mamba to multi-class unsupervised anomaly detection, presenting MambaAD, which consists of a pre-trained encoder and a Mamba decoder featuring Locality-Enhanced State Space (LSS) modules at multi-scales. The proposed LSS module, integrating parallel cascaded (Hybrid State Space) HSS blocks and multi-kernel convolutions operations, effectively captures both long-range and local information. The HSS block, utilizing (Hybrid Scanning) HS encoders, encodes feature maps into five scanning methods and eight directions, thereby strengthening global connections through the (State Space Model) SSM. The use of Hilbert scanning and eight directions significantly improves feature sequence modeling. Comprehensive experiments on six diverse anomaly detection datasets and seven metrics demonstrate SoTA performance, substantiating the method's effectiveness.

## Overview
<p align="center">
  <img src="assets/mambaad.png" alt="accuracy" width="80%">
</p>

**Hybrid Scanning**

<p align="center">
  <img src="assets/hybrid_scan.png" alt="arch" width="80%">
</p>

## Main results
<p align="center">
  <img src="assets/all_result.png" alt="arch" width="80%">
</p>

## Citation
If you find this code useful, don't forget to star the repo and cite the paper:
```

```
## Acknowledgements
We thank the great works [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet) and [ADer](https://github.com/zhangzjn/ADer) for providing assistance for our research.

