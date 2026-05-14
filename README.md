# DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving

<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
    <a href='https://arxiv.org/abs/2601.01528'><img src='https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red'></a>
    <a href='https://drivinggen-bench.github.io/'><img src='https://img.shields.io/badge/DrivingGen-Website-green?logo=googlechrome&logoColor=green'></a>
    <a href='https://huggingface.co/datasets/yangzhou99/DrivingGen'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow?logo=huggingface&logoColor=yellow'></a>
</div>


> #### [DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](https://arxiv.org/abs/2601.01528)
>
> ##### [Yang Zhou](https://yang-zhou-me.github.io/)\* , [Hao Shao](http://hao-shao.com/)\* , [Letian Wang](http://letian-wang.github.io/) , [Zhuofan Zong](https://zongzhuofan.github.io/) , [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/) , [Steven L. Waslander](https://www.trailab.utias.utoronto.ca/) ("*" denotes equal contribution)

![pipeline](assets/pipeline_1229_crop.jpg)

## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Code Flow](#code-flow)
- [Setup Instructions](#setup-instructions)
- [Video Generation](#video-generation)
- [Evaluation](#evaluation)
- [Benchmarked Models](#benchmarked-models)
- [Citation](#citation)
- [License](#license)

## Updates <a name="updates"></a>

- [03/2026] Evaluation code released.
- [01/2026] DrivingGen is aceepted to ICLR 2026.
- [01/2026] We release our paper on [arXiv](https://arxiv.org/abs/2601.01528) and our dataset on [Hugging Face](https://huggingface.co/datasets/yangzhou99/DrivingGen).

## Overview <a name="overview"></a>

**DrivingGen** is the first comprehensive benchmark for generative driving world models. It combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources — spanning varied weather, time of day, geographic regions, and complex maneuvers. DrivingGen evaluates models from both a **visual perspective** (the realism and overall quality of generated videos) and a **robotics perspective** (the physical plausibility, consistency, and accuracy of generated trajectories).

## Code Flow <a name="code-flow"></a>

The DrivingGen evaluation pipeline proceeds in six sequential stages. The diagram below shows the data flow through the codebase.

```
Dataset Download
      │
      ▼
Video Generation (your model)
      │
      ├──► Ego Trajectory Extraction ──────────────────────────────┐
      │                                                             │
      └──► Agent Trajectory Extraction ─────────────────────────── ▼
                                                           Trajectory Metrics
                                                           (alignment, consistency,
                                                            quality, distribution)
      │
      ▼
Video Metric Evaluation
(FVD, P2020, CLIP-IQA+, DINOv3, Cosmos-Reason1)
```

---

### Stage 1 — Dataset Download

**Script:** `drivinggen/down_dataset.py`

Downloads the DrivingGen dataset from Hugging Face (`yangzhou99/DrivingGen`) into `./data/`. The dataset contains:
- `ego_condition.json` / `open_domain.json` — lists of scene IDs
- `data/{scene_id}/imgs/{scene_id}.jpg` — first-frame conditioning images
- `data/{scene_id}/caption/{scene_id}.txt` — text prompts for each scene
- `data/videos-fvd/` — ground-truth 100-frame clips for FVD computation
- Pre-computed ego and agent ground-truth trajectories in `.pkl` files

---

### Stage 2 — Video Generation

**Script:** `drivinggen/infer_example_wan.py` (example using Wan2.2-I2V-A14B)

```
JSON metadata (scene IDs)
        │
        ▼
Load conditioning image (first frame)  →  Resize to fit 576×1024 area
        │
        ▼
Load text caption from .txt file
        │
        ▼
Wan2.2 I2V Pipeline (HuggingFace Diffusers)
  - 101 frames, 576×1024 resolution
  - 40 DDIM inference steps, CFG scale 3.5
  - Fixed seed (generator=0) for reproducibility
        │
        ▼
Output: video.mp4  +  images/00000.png … 00100.png
Saved to: {out_dir}/{split}/{scene_id}/{model}/{exp_id}/
```

The pipeline patches `torch.linalg.solve` with a CPU fallback to handle cuSolver instability that can occur during diffusion sampling with bfloat16.

---

### Stage 3 — Ego Trajectory Extraction

**Script:** `drivinggen/func/extract_traj_ego_unidepth.py`

Estimates the ego-vehicle (camera) trajectory from the generated video frames using monocular depth + visual SLAM.

```
Generated video frames (PNG sequence)
        │
        ▼
YOLOv10x detection  ──►  Movable object mask (H×W)
        │                 (masks out cars/people so SLAM ignores dynamic regions)
        ▼
UniDepthV2-ViT-L14 inference per frame
  - Input:  RGB frame  (3, H, W)
  - Output: depth map  (1, H, W)   [metres]
            intrinsics (3, 3)
            point cloud (3, H, W)
        │
        ▼
Visual SLAM  [drivinggen/func/visual_slam/]
  dataset.py:
    DatasetHandler — wraps lists of (RGB, depth, intrinsics)
                     converts RGB→grayscale for feature detection
  vo.py:
    1. SIFT feature extraction per frame  (nfeatures=2500)
    2. FLANN kNN matching between consecutive frames (k=2 neighbours)
    3. Lowe's ratio test filtering  (threshold=0.7)
    4. PnP-RANSAC pose estimation:
         - Backproject 2D keypoints using depth: p_c = K⁻¹ · [u,v,1] · Z
         - solvePnPRansac(objectpoints, imagepoints, K) → rvec, tvec
         - Refine with solvePnP on inliers
    5. Accumulate: T_world = T_world · T_local⁻¹  (4×4 homogeneous)
    6. Fallback for failed frames: random yaw perturbation, same step length
        │
        ▼
Output saved to:  {outdir}/{scene}/{model}/unidepth-estimate_ego_traj.pkl
  - locs:       (N, 2)  XZ ground-plane coordinates [metres]
  - poses_3x3:  list of N × (R: 3×3, t: 3,) world-from-camera poses
```

**Task parallelism:** `set_task_list()` reads `RANK` / `WORLD_SIZE` env vars to shard scenes across multiple GPUs/nodes.

---

### Stage 4 — Agent Trajectory Extraction

**Script:** `drivinggen/func/extract_traj_agent_unidepth.py`

Extracts trajectories of other traffic agents (vehicles, cyclists, pedestrians) visible in the generated video.

```
Generated video frames (PNG sequence)
        │
        ▼
Frame 0 only:
  YOLOv10x detection  →  agent bounding boxes (x1,y1,x2,y2) + class labels
  Confidence threshold: ≥ 0.3 for vehicles/persons
        │
        ▼
UniDepthV2-ViT-L14 on every frame
  (same as Stage 3 — depth + intrinsics per frame)
        │
        ▼
SAMURAI tracker  [third_parties/samurai/]
  Input:  video frames + initial bounding box from frame 0
  Output: tracked (frame_id, bbox) sequence across all 101 frames
  (One SAMURAI run per detected agent)
        │
        ▼
Per-frame depth estimation for each tracked agent:
  estimate_depth_from_mask(depth_map, segmentation_mask)
    - Uses top-50% of the bounding box region (closer to camera = more reliable)
    - Returns median depth [metres]
        │
        ▼
3D reconstruction using ego camera poses:
  cam_coord  = K⁻¹ · [u, v, 1] · Z
  world_coord = R · cam_coord + t
  → Keep only (X, Z) for top-down 2D trajectory
        │
        ▼
Output saved to:
  *-estimate_agents_traj.pkl   — list of (frame_ids, [(x,z)…], class_label)
  *-estimate_agents_bbox.pkl   — tracked bounding boxes per agent
  *-estimate_agents_bbox_label.pkl — class labels per initial detection
```

---

### Stage 5 — Video Metric Evaluation

**Top-level script:** `drivinggen/z-sample_fvd.py`

This orchestration script loads the generated video frames and dispatches to individual metric modules, selected via `--metric {fvd|obj_q|sub_q|v_consist|a_consist|a_missing|all}`.

#### 5.1 Distribution — FVD

**Module:** `drivinggen/videos/video_distribution.py`

```
Generated videos (100 frames each, symlinked)     GT videos (100 frames each)
        │                                                   │
        └───────────────────────────────────────────────────┘
                              │
                              ▼
              StyleGAN-V  calc_metrics_('fvd2048_100f')
              (I3D network: (B,T,C,H,W) → feature vectors)
                              │
                              ▼
              Fréchet Video Distance (FVD) — scalar ↓ is better
```

#### 5.2 Objective Quality — P2020

**Module:** `drivinggen/videos/video_obj_q.py` → `drivinggen/videos/p2020_v2.py`

```
Video frames (list of image paths)
        │
        ├─ Per-frame metrics (p2020_v2.single_frame_metrics):
        │    • MTF50/MTF10 — spatial frequency response (sharpness)
        │    • Contrast Transfer Accuracy
        │    • Edge Rise Time
        │    • Total Distortion — radial distortion proxy
        │    • Flare Attenuation — center/periphery brightness ratio
        │    • Gradient Entropy — texture richness
        │    • Blur Extent — high/low frequency energy ratio
        │    • Chroma Aberration — R-B channel edge deviation
        │
        └─ Per-video metrics (p2020_v2.video_metrics):
             • MMP (Modulation Mitigation Probability) — sliding-window FFT
               measures temporal flicker/alias energy ratio ↑ is better

Parallel processing via ProcessPoolExecutor.
Final score: mean of mmp_alias across all videos.
```

#### 5.3 Subjective Quality — CLIP-IQA+

**Module:** `drivinggen/videos/video_sub_q.py`

```
Video frames (image paths)
        │
        ▼
Resize each frame to 512×512 → ToTensor → stack batch
        │
        ▼
CLIP-IQA+ model (pyiqa)
  — no-reference quality metric trained on human opinion scores (MOS)
  — CLIP visual encoder + lightweight IQA regression head
        │
        ▼
Mean score across all frames and videos  ↑ is better (0–1 range)
```

#### 5.4 Scene (Visual) Consistency — DINOv3

**Module:** `drivinggen/videos/video_v_consist.py`

```
Video frames (image paths)
        │
        ▼
SEA-RAFT optical flow  (RAFT-based, pretrained on TartanAir+KITTI)
  Input:  consecutive frame pair  (1, 3, H, W) each
  Output: flow  (1, 2, H, W)  → median magnitude per pair → (T-1,) array
        │
        ▼
Arc-length keyframe selection:
  m_bar = mean optical flow magnitude
  K = lerp(min_k=3, max_k=20, clamp((m_bar - v_low) / (v_high - v_low)))
  Select K keyframes equidistant in cumulative arc-length space
        │
        ▼
DINOv3-ViT-H16+ (pooler_output  →  1024-d embedding per frame)
        │
        ▼
Adjacent cosine similarity:  S = mean(cos(F_t, F_{t+1}))  ↑ is better
```

#### 5.5 Agent Appearance Consistency — DINOv3

**Module:** `drivinggen/videos/video_a_consist.py`

```
Per-agent tracked bounding boxes  [(frame_id, (x1,y1,x2,y2)), ...]
        │
        ▼
Crop agent region from each frame (min_size=32 px guard)
        │
        ▼
DINOv3-ViT-H16+  →  pooler_output  (1024-d per crop)
Cached to .pkl to avoid redundant forward passes
        │
        ├─ Reference consistency R = mean cosine sim to frame-0 embedding
        └─ Adjacent consistency  A = mean cosine sim between consecutive frames
        │
        ▼
Score = (R + A) / 2  per agent → mean over all agents and scenes  ↑ is better
```

#### 5.6 Agent Missing Detection — Cosmos-Reason1

**Module:** `drivinggen/videos/video_a_missing.py`

```
Per-agent tracked bounding boxes
        │
        ▼
For each agent that disappears before frame 100:
  Extract 6 annotated frames:
    - 2 frames from agent's first appearance (green-boxed)
    - 2 frames from agent's last appearance (green-boxed)
    - 2 frames immediately after disappearance
        │
        ▼
Cosmos-Reason1-7B  (NVIDIA, vllm inference)
  Prompt: classify disappearance as Natural (occlusion, exit frame)
          vs Unnatural (abrupt/non-physical)
  Output: "Natural" → not missing penalty
          "Unnatural" → counted as unnatural disappearance
        │
        ▼
Missing rate = fraction of agents with unnatural disappearance
Score = 1 - missing_rate  ↑ is better
```

---

### Stage 6 — Trajectory Metric Evaluation

Uses pre-extracted ego/agent trajectories from Stages 3–4.

#### 6.1 Alignment Metrics — `drivinggen/trajs/traj_alignment.py`

All metrics operate on `(N, T, 2)` arrays (N trajectories, T timesteps, XZ coords).

| Metric | Formula | Meaning |
|--------|---------|---------|
| ADE | mean‖pred_t − gt_t‖ over t | Average displacement at every step |
| FDE | ‖pred_T − gt_T‖ | Displacement only at final step |
| Success Rate | FDE < 3.0 m | Fraction of near-perfect predictions |
| Hausdorff | max(min_j d(p_i,g_j), min_i d(p_i,g_j)) | Worst-case coverage |
| DTW | dynamic programming alignment cost | Shape similarity ignoring temporal shift |

#### 6.2 Trajectory Consistency — `drivinggen/trajs/traj_consistency.py`

```
Trajectory (N, T, 2)
        │
        ▼
velocity v_t = ‖Δxy_t‖ / dt          (N, T-1)
acceleration a_t = Δv_t / dt          (N, T-2)
        │
        ▼
S = 0.5 · [exp(−σ_v / (μ_v + ε)) + exp(−σ_a / (μ_a + ε))]
                                       ↑ is better (0–1)
```

#### 6.3 Trajectory Quality — `drivinggen/trajs/traj_quality.py`

```
Trajectory (N, T, 2)
        │
        ├─ Comfort Score:
        │    jerk       = d³x/dt³   (3rd derivative of position)
        │    yaw-rate   = dθ/dt     (heading rate from velocity direction)
        │    S_j = 1 / (1 + jerk_per_meter)    (higher = smoother)
        │    S_comf = geometric_mean(S_j, S_a, S_y)
        │
        ├─ Curvature RMS:
        │    κ(t) = |ẋÿ − ẏẍ| / (ẋ² + ẏ²)^1.5  (Frenet curvature)
        │    S_curv = 1 / (1 + RMS(κ))
        │
        └─ Speed Score:
             S_speed = log(1 + v_mean) / log(1 + v_max)  ∈ (0, 1)
             (higher = closer to typical driving speed)
```

#### 6.4 Trajectory Distribution — FTD — `drivinggen/trajs/traj_distribution.py`

```
Predicted/GT trajectories
        │
        ▼
Sliding 11-frame windows (stride=10):
  Each window → MTR coordinate normalization (ego-centric frame)
              → Feature tensor:  (1, 1, 11, 29+T)
                (pos 6d + type 5d + time embed T+1 d + heading 2d + vel 2d + acc 2d)
        │
        ▼
MTR context encoder (agent_polyline_encoder):
  PointNet-style MLP on each 11-frame window
  Output: 256-d feature vector per window
  Mean pooled across windows → 1 feature per trajectory
        │
        ▼
Stack all features:  Fp (N_pred, 256)   Fg (N_gt, 256)
        │
        ▼
Fréchet Distance  =  ‖μ_p − μ_g‖² + Tr(Σ_p + Σ_g − 2√(Σ_p Σ_g))
FTD  ↓ is better
```

---

## Setup Instructions <a name="setup-instructions"></a>

#### 1. Clone the Repository

```shell
git clone https://github.com/youngzhou1999/DrivingGen.git
cd DrivingGen
```

#### 2. Environment Setup

```shell
conda create -n drivinggen python=3.10
conda activate drivinggen
```

#### 3. Install Dependencies

We recommend using the provided `environment.yml` for a full environment setup:
```shell
conda env create -f environment.yml
conda activate drivinggen
```

Alternatively, if you prefer installing into an existing environment:
```shell
pip install -r requirements.txt
```

Then install third-party packages:
```shell
cd third_parties/UniDepth && pip install -e . && cd ../..
cd third_parties/yolov10 && pip install -e . && cd ../..
cd third_parties/samurai && pip install -e . && cd ../..
```

#### 4. Download Dataset

Download the **DrivingGen** dataset from Hugging Face. First, update your Hugging Face token in `drivinggen/down_dataset.py`, then run:

```shell
bash scripts/0-down_data.sh
```

The dataset will be downloaded to `./data/`.

## Video Generation <a name="video-generation"></a>

This section guides you through generating videos for evaluation using your own world generation model. We provide an example using **Wan2.2-14B** (Image-to-Video).

#### 1. Run Inference

Configure and run video generation:

```shell
bash scripts/1-example_infer_model.sh
```

Key parameters in the script:

```shell
video_path=data/ego_condition.json   # Input metadata
out_dir=cache/infer_results          # Output directory
split=ego_condition                  # Data split (ego_condition / open_domain)
model=wan2.2-14b                     # Model name
exp_id=default_prompt                # Experiment ID
```

The generated videos (101 frames at 10 fps, 576x1024 resolution) will be saved as both MP4 videos and individual PNG frames.

#### 2. Extract Ego Trajectory

Extract ego vehicle trajectory from generated videos using UniDepthV2 and Visual SLAM:

```shell
bash scripts/2-get_ego_traj.sh
```

#### 3. Extract Agent Trajectories

Extract agent trajectories using YOLOv10 detection and depth estimation:

```shell
bash scripts/3-get_agent_traj.sh
```

## Evaluation <a name="evaluation"></a>

DrivingGen evaluates generated videos using comprehensive **video metrics** and **trajectory metrics**.

### Video Metrics

Evaluates visual quality and temporal coherence of generated videos:

| Category | Metrics |
| --- | --- |
| **Distribution** | FVD (Frechet Video Distance) |
| **Objective Quality** | IEEE P2020 automotive imaging metrics (sharpness, exposure, contrast, color, noise, artifacts, texture, temporal) |
| **Subjective Quality** | CLIP-IQA+ based assessment |
| **Scene Consistency** | DINOv3 feature-based consistency |
| **Agent Consistency** | Agent appearance consistency and missing detection |
| **Perceptual** | LPIPS, SSIM |

Run video evaluation:

```shell
bash scripts/4-get_video_metrics.sh
```

### Trajectory Metrics

Evaluates the physical plausibility and accuracy of generated trajectories:

| Category | Metrics |
| --- | --- |
| **Distribution** | FTD (Frechet Trajectory Distance) via Motion Transformer encoder |
| **Alignment** | ADE, FDE, Success Rate, Hausdorff Distance, DTW |
| **Quality** | Comfort Score (jerk, acceleration, yaw rate), Curvature RMS, Speed Score |
| **Consistency** | Velocity Consistency, Acceleration Consistency |

Run trajectory evaluation:

```shell
bash scripts/5-get_traj_metrics.sh
```

Results will be saved to `cache/eval_logs/`.

## Benchmarked Models <a name="benchmarked-models"></a>

DrivingGen benchmarks **14 state-of-the-art models** across three categories:

| Category | Models |
| --- | --- |
| **General Video World Models** | Gen-3, Kling, CogVideoX, Wan, HunyuanVideo, LTX-Video, SkyReels |
| **Physical World Models** | Cosmos-Predict1, Cosmos-Predict2 |
| **Driving-Specific World Models** | Vista, DrivingDojo, GEM, VaViM, UniFuture |

## Citation <a name="citation"></a>

If you find our research useful, please cite us as:

```bibtex
@misc{zhou2026drivinggencomprehensivebenchmarkgenerative,
      title={DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving},
      author={Yang Zhou and Hao Shao and Letian Wang and Zhuofan Zong and Hongsheng Li and Steven L. Waslander},
      year={2026},
      eprint={2601.01528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01528},
}
```

## License <a name="license"></a>

All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
