# DenseFish-v13

**DenseFish-v13: A Symmetry-Aware NMS-Free YOLOv13-Mamba Framework for Dense Underwater Fish Detection and Bio-Kinematic Behavior Recognition**

DenseFish-v13 is a research-oriented intelligent underwater image-processing framework designed for dense aquaculture scenarios. It addresses the coupled challenges of extreme fish occlusion, aeration-induced visual noise, turbidity, motion blur, and behavior-oriented monitoring in high-density underwater environments.

The core idea of DenseFish-v13 is to reformulate dense underwater fish recognition as a **symmetry–asymmetry modeling problem**: fish bodies usually exhibit quasi-symmetric contours and quasi-periodic scale textures, while occlusion, bubbles, turbidity, and motion blur introduce asymmetric visual disturbances. DenseFish-v13 integrates frequency-domain feature refinement, global state-space modeling, NMS-free dense-instance separation, and trajectory-level behavior interpretation into a unified framework.

---

## Highlights

- **Symmetry-Preserving Bio-Harmonic Frequency Gate (B-HFG)**  
  Suppresses broadband aeration-bubble noise in the frequency domain while preserving structured biological textures such as fish-scale and contour-related responses.

- **YOLOv13-Mamba Backbone**  
  Introduces Visual State Space modeling into the YOLOv13 detection pipeline to recover partially occluded fish structures through long-range contextual modeling with linear computational complexity.

- **Asymmetry-Aware Density Repulsion Loss**  
  Enhances NMS-free dense matching by penalizing excessive latent similarity between highly overlapping fish instances, reducing merged boxes and missed detections in extreme-density scenes.

- **Bio-Kinematic Behavior Head**  
  Converts frame-level detections into short trajectories and extracts velocity- and turning-based descriptors for behavior-state recognition, including Normal, Feeding, and Hypoxia-related floating states.

- **Edge-Oriented Deployment**  
  Designed for real-time inference on embedded platforms such as NVIDIA Jetson Orin NX using TensorRT FP16 acceleration.

---

## Framework Overview

The DenseFish-v13 pipeline consists of four main stages:

1. **Spectral Feature Refinement**  
   Underwater images are processed by the Bio-Harmonic Frequency Gate to reduce aeration-induced noise and preserve biologically meaningful structures.

2. **Global Occlusion-Aware Feature Extraction**  
   The YOLOv13-Mamba backbone models long-range dependencies to infer partially visible fish from surrounding contextual cues.

3. **NMS-Free Dense Instance Separation**  
   A bipartite matching-based detection head is trained with Density-Aware Repulsion Loss to separate highly overlapping fish instances.

4. **Bio-Kinematic Behavior Recognition**  
   Detection results are associated across frames to form trajectories, from which motion descriptors are extracted for behavior-state classification.

## Dataset Availability

The complete datasets used in this study are not directly stored in this repository due to their large file size.

This repository provides lightweight demo subsets for quick code verification:

- `sample-pond-dataset.zip`
- `sample-salmon-dataset.zip`

The full datasets can be downloaded from:

- Pond Fish Detection Dataset: https://data.mendeley.com/datasets/7w45jx35hd/1
- Healthy and Loser Salmon Dataset: https://data.mendeley.com/datasets/rvrt4zs969/1