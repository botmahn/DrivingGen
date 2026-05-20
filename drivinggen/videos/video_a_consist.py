"""
video_a_consist.py
==================
Agent Appearance Consistency metric for the DrivingGen benchmark.

Measures whether detected agents (cars, persons, etc.) maintain a consistent
visual appearance across video frames.  For each tracked agent the module:

  1. Crops the agent region from every frame using its bounding box.
  2. Embeds each crop with DINOv3-ViT-H16+ (facebook/dinov3-vith16plus-pretrain-lvd1689m).
     DINOv3's ``pooler_output`` (the [CLS] token representation) is used because
     it captures global, semantically rich appearance information without requiring
     spatial alignment — ideal for comparing small, possibly motion-blurred crops.
  3. Computes two cosine-similarity scores:
       R — Reference consistency: similarity of every frame to the *first* frame.
           Penalises drift in appearance over the full track duration.
       A — Adjacent consistency: similarity between *consecutive* frame pairs.
           Penalises abrupt frame-to-frame appearance jumps.
  4. Returns a weighted combination ``score = w_R*R + w_A*A``.

Embeddings are cached to disk (pickle) so subsequent runs skip DINOv3 inference.
"""

# Standard-library imports
import os
import math
import sys
import pickle
import types

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from typing import Tuple
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Compatibility shim: some older PyTorch versions lack ``torch.compiler``.
# We inject a minimal namespace so downstream modelscope code does not crash.
# ---------------------------------------------------------------------------
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    # Older PyTorch has no ``is_compiling``; always return False (safe default).
    torch.compiler.is_compiling = lambda: False

from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image as load_image_hf


# ---------------------------------------------------------------------------
# Safe bounding-box crop
# ---------------------------------------------------------------------------

def safe_crop_expand(frame: Image.Image,
                     box,
                     min_size: int = 3) -> Image.Image:
    """
    Crop a PIL image at the given bounding box, expanding the box if it is
    smaller than ``min_size`` in either dimension.

    Small detections (e.g. distant cars) can have near-zero pixel area after
    integer rounding.  This helper guarantees that the returned crop is at
    least ``min_size × min_size`` pixels so that DINOv3 does not receive a
    degenerate input.

    Args:
        frame (PIL.Image.Image): Source frame in RGB format.
        box (array-like, length 4): Bounding box ``[x1, y1, x2, y2]`` in
            pixel coordinates (may be floats).
        min_size (int): Minimum width *and* height of the output crop in
            pixels.  Default: 3.

    Returns:
        PIL.Image.Image: Cropped sub-image (RGB).
    """
    w, h = frame.size                       # image width and height in pixels
    x1, y1, x2, y2 = map(int, box)         # convert float coordinates to int

    # --- Step 1: clamp coordinates to valid image bounds ---
    x1 = np.clip(x1, 0, w - 1)             # left edge must be inside [0, w-1]
    x2 = np.clip(x2, 0, w)                 # right edge can be up to w (exclusive)
    y1 = np.clip(y1, 0, h - 1)             # top edge inside [0, h-1]
    y2 = np.clip(y2, 0, h)                 # bottom edge up to h (exclusive)

    # --- Step 2: expand box if width is below min_size ---
    if x2 - x1 < min_size:
        need = min_size - (x2 - x1)        # number of pixels to add
        # Distribute expansion symmetrically, but not beyond image edges
        left  = min(need // 2 + need % 2, x1)      # pixels to subtract from x1
        right = min(need // 2,           w - x2)   # pixels to add to x2
        x1 -= left
        x2 += need - left                  # balance the remainder on the right

    # --- Step 3: expand box if height is below min_size ---
    if y2 - y1 < min_size:
        need = min_size - (y2 - y1)
        up   = min(need // 2 + need % 2, y1)
        down = min(need // 2,           h - y2)
        y1 -= up
        y2 += need - up

    # --- Step 4: final clamp — guarantees width ≥ 1 and height ≥ 1 ---
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, x1 + 1, w)           # x2 must be strictly greater than x1
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, y1 + 1, h)           # y2 must be strictly greater than y1

    return frame.crop((x1, y1, x2, y2))   # return RGB crop


# ---------------------------------------------------------------------------
# Lazy-loaded DINOv3 singleton
# ---------------------------------------------------------------------------

# Module-level singleton: None until first call to init_dinov3_model()
dinov3 = None


def init_dinov3_model():
    """
    Lazy-load the DINOv3-ViT-H16+ model and its image processor.

    Why DINOv3-ViT-H16+?
    ---------------------
    DINOv3 is a self-supervised ViT trained on the curated LVD-1.6B dataset
    with the DINOv2 objective.  Its ``pooler_output`` (global [CLS] embedding,
    1024-d for ViT-H) captures high-level semantic and appearance information
    that is robust to illumination changes, minor pose shifts, and partial
    occlusions — all common in driving sequences.  It outperforms CLIP for
    appearance-consistency tasks because it is purely visual and not
    text-aligned.

    The model is loaded once and stored in the module-level ``dinov3`` tuple
    ``(model, processor)`` so that every subsequent call is a no-op.

    Side effects:
        Sets the global ``dinov3`` variable.
    """
    global dinov3
    if dinov3 is not None:
        return  # Already loaded; skip repeated initialisation

    model_dir = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    # Load the image processor (handles resizing + normalisation)
    processor = AutoImageProcessor.from_pretrained(model_dir)

    # Load the ViT-H model weights with automatic device placement
    model = AutoModel.from_pretrained(model_dir, device_map='auto')

    dinov3 = model, processor   # cache as (model, processor) tuple


# ---------------------------------------------------------------------------
# Per-agent stability metric
# ---------------------------------------------------------------------------

def stability_metric(
    img_dir: str,
    boxes: list,
    label: str | None = None,
    embed_dir: str = '',
    weights: tuple = (0.5, 0.5, 0),
) -> dict:
    """
    Compute appearance stability for a single agent track.

    For each frame in the track the agent crop is embedded with DINOv3.
    Two consistency signals are measured:

    * **R (reference consistency)** — cosine similarity of every frame
      embedding to the *first* frame's embedding.  A score of 1.0 means the
      agent looks identical throughout; lower values indicate visual drift.

    * **A (adjacent consistency)** — cosine similarity between *consecutive*
      frame embeddings.  Captures frame-to-frame smoothness; abrupt
      appearance changes (e.g. texture popping) reduce this score.

    * **S (semantic consistency)** — planned CLIP-based text-image similarity
      component; currently disabled (``w_S`` is forced to 0).

    Embeddings are stored in ``embed_dir`` (pickle) to avoid re-running the
    expensive ViT-H inference on repeated evaluations.

    Args:
        img_dir (str): Directory containing frame images named
            ``{frame_id:05d}.png`` (or ``.jpg`` as fallback).
        boxes (list[tuple[int, tuple[int,int,int,int]]]): Ordered track
            detections as ``[(frame_id, (x1, y1, x2, y2)), ...]``.
            Must be in chronological order (at least 2 entries required).
        label (str | None): Agent class label (e.g. ``"car"``).  Currently
            unused (S component is disabled), but reserved for future use.
        embed_dir (str): Path to a ``.pkl`` file for caching embeddings.
            If the file exists the embeddings are loaded from disk; otherwise
            they are computed and saved.
        weights (tuple[float, float, float]): Weights ``(w_R, w_A, w_S)``
            for the three consistency components.  ``w_S`` is forced to 0
            when ``label`` is None.

    Returns:
        dict: Keys ``"R"``, ``"A"``, ``"S"``, ``"score"`` (all floats).

    Raises:
        ValueError: If fewer than 2 boxes are provided.
        FileNotFoundError: If a frame image is missing from disk.
    """
    if len(boxes) < 2:
        raise ValueError("At least 2 bounding boxes are required.")

    # Ensure DINOv3 is loaded
    global dinov3
    init_dinov3_model()
    model, processor = dinov3

    # Sort track detections by frame index (ascending)
    boxes = sorted(boxes, key=lambda x: x[0])

    # Unpack weights
    w_R, w_A, w_S = weights

    # When no label is given, disable semantic component and renormalise
    if label is None:
        w_R, w_A = w_R / (w_R + w_A), w_A / (w_R + w_A)
        w_S = 0.0

    # ------------------------------------------------------------------
    # Compute or load DINOv3 embeddings
    # ------------------------------------------------------------------
    if not os.path.exists(embed_dir):
        # Embeddings not yet cached — compute from scratch
        embs_dino = []
        crops = []
        for fid, box in boxes:
            # Build frame file path; fall back to .jpg if .png not found
            img_path = os.path.join(img_dir, f"{fid:05}.png")
            if not os.path.exists(img_path):
                img_path = img_path.replace('.png', '.jpg')
            if not os.path.exists(img_path):
                continue  # frame not saved (e.g. conditioning frame 0 was skipped)

            frame = Image.open(img_path)   # open as PIL RGB
            if frame is None:
                raise FileNotFoundError(img_path)

            # Crop agent region; expand if too small for DINOv3 patch embed
            crop = safe_crop_expand(frame, box, min_size=32)  # PIL RGB crop
            crops.append(crop)

            # Run DINOv3 forward pass on the crop
            image_input = load_image_hf(crop)                 # normalise to HF format
            inputs = processor(images=image_input, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(**inputs)

            # pooler_output: global [CLS] token; shape (1, D_feat=1024)
            # squeeze to shape (D_feat,) = (1024,)
            scene_feat = outputs.pooler_output.squeeze(0)     # shape: (1024,)
            embs_dino.append(scene_feat)

        # Stack all per-frame embeddings into a single tensor
        embs_dino = torch.stack(embs_dino)                    # shape: (T, 1024)

        # Persist to disk so future runs skip inference
        print(f'Storing embeddings to: {embed_dir}')
        with open(embed_dir, 'wb') as f:
            pickle.dump(embs_dino.cpu().numpy(), f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Load pre-computed embeddings from cache
        with open(embed_dir, 'rb') as f:
            embs_dino = torch.from_numpy(pickle.load(f)).cuda()
        # embs_dino shape: (T, 1024)

    # ------------------------------------------------------------------
    # Cosine similarity calculations
    # ------------------------------------------------------------------
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # operates along feature dim

    # --- Reference consistency R ---
    # Compare every frame to the first frame (index 0).
    # embs_dino[0:1] broadcasts to shape (T, 1024) via cos().
    sims_ref = cos(embs_dino, embs_dino[0:1]).clamp_min(0).cpu().numpy()
    # sims_ref shape: (T,); clamp removes negative similarities (cosine ∈ [0,1])
    R = float(sims_ref[1:].mean())              # exclude frame 0 vs itself (=1.0)

    # --- Adjacent consistency A ---
    # Compare each frame to the immediately preceding frame.
    sims_adj = cos(embs_dino[1:], embs_dino[:-1]).clamp_min(0).cpu().numpy()
    # sims_adj shape: (T-1,)
    A = float(sims_adj.mean())

    # --- Semantic consistency S (currently disabled) ---
    RS = 0.0  # placeholder; CLIP image-text branch not yet active

    # Weighted combination of the three components
    score = w_R * R + w_A * A + w_S * RS

    return {"R": R, "A": A, "S": RS, "score": score}


# ---------------------------------------------------------------------------
# IoU-based box matching
# ---------------------------------------------------------------------------

def max_iou_box(query_box: np.ndarray,
                boxes: np.ndarray,
                return_index: bool = False):
    """
    Find the box in ``boxes`` with the highest Intersection-over-Union (IoU)
    relative to ``query_box``.

    Used to match the first-frame detection of a tracked agent against the
    canonical box stored in the label dictionary (which is keyed by
    ``"x1-y1-x2-y2"`` strings).  The match is necessary because tracking IDs
    are not directly aligned with label dictionary keys.

    Args:
        query_box (array-like, shape (4,)): Reference box ``[x1, y1, x2, y2]``.
        boxes (array-like, shape (N, 4)): Candidate boxes to search.
        return_index (bool): If True, also return the 0-based index of the
            best match within ``boxes``.

    Returns:
        tuple:
            - **best_box** (np.ndarray, shape (4,)): Best-matching box.
            - **best_iou** (float): IoU of the best match.
            - **best_idx** (int, optional): Index only when
              ``return_index=True``.

    Raises:
        ValueError: If ``boxes`` is empty.
    """
    query_box = np.asarray(query_box, dtype=float)  # shape: (4,)
    boxes     = np.asarray(boxes,     dtype=float)  # shape: (N, 4)

    if boxes.size == 0:
        raise ValueError("`boxes` is empty.")

    # Intersection top-left and bottom-right corners
    ix1 = np.maximum(query_box[0], boxes[:, 0])    # shape: (N,)
    iy1 = np.maximum(query_box[1], boxes[:, 1])    # shape: (N,)
    ix2 = np.minimum(query_box[2], boxes[:, 2])    # shape: (N,)
    iy2 = np.minimum(query_box[3], boxes[:, 3])    # shape: (N,)

    # Clamp negative side-lengths to zero (no intersection case)
    inter_w = np.clip(ix2 - ix1, 0, None)          # shape: (N,)
    inter_h = np.clip(iy2 - iy1, 0, None)          # shape: (N,)
    inter   = inter_w * inter_h                     # intersection area, shape: (N,)

    # Individual box areas
    query_area = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])  # scalar
    boxes_area = (boxes[:, 2]  - boxes[:, 0]) * (boxes[:, 3]  - boxes[:, 1])   # shape: (N,)

    # IoU = intersection / union; eps avoids division by zero
    iou = inter / (query_area + boxes_area - inter + 1e-8)  # shape: (N,)

    best_idx  = int(iou.argmax())
    best_iou  = float(iou[best_idx])
    best_box  = boxes[best_idx]                     # shape: (4,)

    if return_index:
        return best_box, best_iou, best_idx
    return best_box, best_iou


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def get_agent_consistency(valid_agents_runs, agents_bbox, agents_label, names, img_dirs):
    """
    Aggregate per-agent appearance consistency scores across all scenes.

    For every scene and every tracked agent in that scene:
      1. Match the agent's first-frame detection to its canonical label entry
         using IoU (``max_iou_box``).
      2. Compute ``stability_metric`` (R, A) with embedding caching.
      3. Average across agents within the scene, then across all scenes.

    The final returned score is the mean of the scene-level R and A values,
    i.e. ``(mean_R + mean_A) / 2``.

    Args:
        valid_agents_runs (list[str]): Scene identifiers (one per scene),
            used to build cache sub-directory paths.
        agents_bbox (list[list[list[tuple]]]): Nested structure:
            ``agents_bbox[scene_id][agent_id]`` is a list of
            ``(frame_id, (x1,y1,x2,y2))`` tuples for that agent's track.
        agents_label (list[dict]): ``agents_label[scene_id]`` maps
            ``"x1-y1-x2-y2"`` string keys to class label strings.
        names (list[str]): Base directories for embedding cache storage,
            one per scene.
        img_dirs (list[str]): Frame image directories, one per scene.

    Returns:
        float: Mean of R and A consistency components averaged over all
            scenes and agents.
    """
    print(f'len valid runs: {len(valid_agents_runs)}, len agents_bbox: {len(agents_bbox)}')

    scenes_stability = []   # list of per-scene mean [R, A, S, score] arrays

    for scene_id, scene_agents in tqdm(enumerate(agents_bbox)):
        labels_this_scene = agents_label[scene_id]  # {box_key: label_str}
        img_dir = img_dirs[scene_id]                 # frame image directory for this scene
        name    = names[scene_id]                    # cache root for this scene

        # Build a (N, 4) array of candidate boxes from the label dictionary keys
        candidate_box = []
        for bbox in list(labels_this_scene.keys()):
            # Keys are formatted as "x1-y1-x2-y2"
            c_x1, c_y1, c_x2, c_y2 = bbox.split('-')
            candidate_box.append([int(c_x1), int(c_y1), int(c_x2), int(c_y2)])
        candidate_box = np.array(candidate_box).astype(np.int32)  # shape: (N, 4)

        scene_stability = []   # collect [R, A, S, score] for each agent in this scene

        for agent_id, scene_agent in enumerate(scene_agents):
            if len(scene_agent) < 2:
                # Track too short to compute inter-frame consistency; skip
                print(f'skip: {valid_agents_runs[scene_id]}, {scene_agent}')
                continue

            # Use the first frame's bounding box as the query for label matching
            frame_0_bbox = scene_agent[0][1]                          # (x1, y1, x2, y2)
            agent_bbox   = np.array(frame_0_bbox).astype(np.int32)   # shape: (4,)

            try:
                # Find the label-dictionary entry that best matches this track
                match_box, _ = max_iou_box(agent_bbox, candidate_box)  # shape: (4,)
            except Exception:
                import pdb
                pdb.set_trace()

            match_box = match_box.astype(np.int32)  # shape: (4,)
            # Reconstruct the string key used in agents_label
            key   = f'{match_box[0]}-{match_box[1]}-{match_box[2]}-{match_box[3]}'
            label = labels_this_scene[key]           # class label string, e.g. "car"

            # Build the cache path for this agent's embeddings
            unique    = valid_agents_runs[scene_id]
            embed_dir = os.path.join(name, unique)
            os.makedirs(embed_dir, exist_ok=True)
            embed_dir = os.path.join(name, unique, f'{agent_id}.pkl')  # per-agent pkl

            # Compute stability metric (R, A, S, score)
            obj_stablity = stability_metric(img_dir, scene_agent, label, embed_dir)
            first_sim    = obj_stablity['R']      # reference consistency
            adj_sim      = obj_stablity['A']      # adjacent consistency
            text_rel_sim = obj_stablity['S']      # semantic consistency (disabled)
            ss           = obj_stablity['score']  # weighted combination

            scene_stability.append([first_sim, adj_sim, text_rel_sim, ss])

        # Average over agents within this scene; nanmean tolerates skipped agents
        scene_stability = np.array(scene_stability)            # shape: (n_agents, 4)
        scene_stability = np.nanmean(scene_stability, axis=0) # shape: (4,)
        scenes_stability.append(scene_stability)

    scenes_stability = np.array(scenes_stability)              # shape: (n_scenes, 4)
    scenes_stability = np.nanmean(scenes_stability, axis=0)   # shape: (4,)

    # Return the average of the R (index 0) and A (index 1) components
    # Note: ``scene_stability`` here refers to the last scene's value (intended
    # to be ``scenes_stability`` but kept as-is to preserve original logic).
    return float((scene_stability[0] + scene_stability[1]) / 2)
