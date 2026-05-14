"""
video_a_missing.py
==================
Agent Missing Detection metric for the DrivingGen benchmark.

Determines whether agent disappearances in generated driving videos are
"natural" (occlusion, exit from camera frustum) or "unnatural" (abrupt,
non-physical, a generation artefact).

Pipeline for each tracked agent:
  1. Detect if the track ends before the last video frame.
  2. Assemble a set of annotated key frames: early track frames (green box
     around agent) and the two frames immediately *after* the track ends
     (no box, so the VLM can see the scene without the agent).
  3. Query Cosmos-Reason1-7B (a 7-billion-parameter Vision Language Model
     running under vLLM) with the frames and a natural-language question.
  4. Parse the model's structured ``<answer>`` tag to decide
     ``natural`` / ``unnatural``.
  5. Aggregate: the scene-level missing rate is the fraction of agents that
     disappeared unnaturally; the dataset score is ``1 - mean_missing_rate``
     (higher = fewer unnatural disappearances = better generation quality).

The VLM is loaded once and kept in the module-level singleton ``cosmos_r``.

Dependencies: vllm, cosmos_reason1_utils, qwen_vl_utils, transformers.
"""

# ---------------------------------------------------------------------------
# Multiprocessing start method must be set before any other imports that
# might fork subprocesses.  ``spawn`` is required to avoid CUDA context
# inheritance issues when vLLM workers start.
# ---------------------------------------------------------------------------
import multiprocessing as mp
if __name__ == '__main__' or True:   # always execute this block
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

import os
# Tell vLLM worker processes to also use spawn
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import types
import torch

# ---------------------------------------------------------------------------
# Compatibility shim for older PyTorch versions that lack torch.compiler.
# ---------------------------------------------------------------------------
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    # Older PyTorch does not expose is_compiling; returning False is safe.
    torch.compiler.is_compiling = lambda: False

# Cosmos-Reason1 helper utilities (bundled in third_parties/)
from cosmos_reason1_utils.script import init_script

# Run any one-time setup required by the cosmos utilities (e.g. sys.path fixes)
init_script()

import argparse
import collections
import pathlib
import textwrap
import re
import math
import random
import json
import glob
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from PIL import Image

import qwen_vl_utils
import transformers
import vllm
import yaml
from rich import print
from rich.pretty import pprint

from cosmos_reason1_utils.text import (
    PromptConfig,
    create_conversation,
    extract_tagged_text,
)
from cosmos_reason1_utils.vision import (
    VisionConfig,
    overlay_text_on_tensor,
    save_tensor,
)
from typing import List, Tuple, Dict, Sequence, Optional, TypeVar

import pickle
import torch
from numpy.typing import ArrayLike

# Paths to Cosmos-Reason1 config and prompt files (relative to repo root)
ROOT      = 'third_parties/cosmos-reason1'
SEPARATOR = "-" * 20  # visual divider for console output


def pprint_dict(d: dict, name: str):
    """
    Pretty-print a plain dictionary as a named tuple for readability.

    Args:
        d (dict): Dictionary to display.
        name (str): Label used as the named-tuple class name in output.
    """
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


# ---------------------------------------------------------------------------
# Cosmos-Reason1 VLM singleton
# ---------------------------------------------------------------------------

# Module-level singleton; None until first call to init_glm()
cosmos_r = None


def init_glm():
    """
    Lazy-load the Cosmos-Reason1-7B Vision Language Model (VLM) via vLLM.

    Why Cosmos-Reason1-7B?
    ----------------------
    Cosmos-Reason1 is a physics-aware VLM fine-tuned specifically on
    autonomous-driving data.  It understands scene geometry, object motion,
    and occlusion patterns — making it far more reliable for judging
    "natural vs. unnatural disappearance" than a generic VLM.

    The model is served by vLLM for efficient batched inference and uses the
    Qwen2.5-VL token format.  The prompt system instructs the model to
    produce chain-of-thought reasoning followed by a structured
    ``<answer>Natural|Unnatural</answer>`` tag.

    Side effects:
        Sets the global ``cosmos_r`` variable to a 5-tuple:
        ``(llm, processor, system_prompt, vision_kwargs, sampling_params)``.
    """
    global cosmos_r

    # Attempt to switch to spawn in case the method was reset elsewhere
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print('mp error: start method already set')

    # --- Load prompt and config files ---
    prompt          = "third_parties/cosmos-reason1/prompts/question.yaml"
    vision_config   = f"{ROOT}/configs/vision_config.yaml"
    sampling_params_path = f"{ROOT}/configs/sampling_params.yaml"
    verbose         = True

    # Parse YAML configs
    prompt_kwargs   = yaml.safe_load(open(prompt, "rb"))
    prompt_config   = PromptConfig.model_validate(prompt_kwargs)

    vision_kwargs   = yaml.safe_load(open(vision_config, "rb"))
    _vision_config  = VisionConfig.model_validate(vision_kwargs)

    sampling_kwargs = yaml.safe_load(open(sampling_params_path, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    if verbose:
        pprint_dict(vision_kwargs,   "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # --- Build system prompt ---
    # Cosmos-Reason1 uses a layered system prompt: English instruction +
    # task-specific prompt + chain-of-thought reasoning instruction.
    system_prompts = [open(f"{ROOT}/prompts/addons/english.txt").read()]
    if prompt_config.system_prompt:
        system_prompts.append(prompt_config.system_prompt)

    # Add reasoning addon if the prompt does not already include <think>
    if True and "<think>" not in prompt_config.system_prompt:
        if extract_tagged_text(prompt_config.system_prompt)[0]:
            raise ValueError(
                "Prompt already contains output format.  Cannot add reasoning."
            )
        system_prompts.append(open(f"{ROOT}/prompts/addons/reasoning.txt").read())

    # Concatenate all system-prompt sections
    system_prompt = "\n\n".join(map(str.rstrip, system_prompts))

    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(system_prompt.rstrip(), "  "))
    print(SEPARATOR)

    # --- Instantiate vLLM LLM engine ---
    llm = vllm.LLM(
        model="nvidia/Cosmos-Reason1-7B",
        # Cap multi-modal inputs: up to 120 images OR 1 video per prompt
        limit_mm_per_prompt={"image": 120, "video": 1},
        enforce_eager=True,         # disable CUDA graph tracing for stability
        gpu_memory_utilization=0.6  # reserve 40% VRAM for activations
    )

    # --- Load Qwen2.5-VL processor (tokenizer + image processor) ---
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
    )

    # Cache everything in the module-level singleton
    cosmos_r = llm, processor, system_prompt, vision_kwargs, sampling_params


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str):
    """
    Parse a Cosmos-Reason1 output string and extract the naturalness label.

    The VLM is prompted to emit its verdict inside an ``<answer>`` tag, e.g.::

        <answer>Natural</answer>

    This function finds that tag, strips any inner XML markup, and matches
    the keyword ``natural`` or ``unnatural`` case-insensitively.

    Args:
        text (str): Full raw output string from the VLM.

    Returns:
        str | None: ``"natural"`` or ``"unnatural"`` if found, else ``None``.
    """
    # Locate the <answer>...</answer> block (case-insensitive, dot-all mode)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if not m:
        return None  # No answer tag found

    # Strip any nested XML tags from the captured group
    inner = re.sub(r"<[^>]+>", "", m.group(1))

    # Extract the verdict keyword
    m2 = re.search(r"\b(natural|unnatural)\b", inner, flags=re.I)
    return m2.group(1).lower() if m2 else None


# ---------------------------------------------------------------------------
# VLM inference helpers
# ---------------------------------------------------------------------------

def glm_reasoning(gen_video_path, first_fids, first_box, last_fids, last_box,
                  next_fids, glm_dir):
    """
    Run Cosmos-Reason1-7B on 6 annotated frames to judge whether an agent's
    disappearance is natural or unnatural.

    Frame selection strategy:
      - ``first_fids[0]``, ``first_fids[1]`` — two early frames with a green
        bounding box showing the agent while it is still present.
      - ``last_fids[0]``, ``last_fids[-1]`` — two late frames (near the track
        end) with a green bounding box.
      - ``next_fids[0]``, ``next_fids[1]`` — the two frames *immediately
        after* the track ends; no bounding box, so the VLM can observe
        whether the scene transition looks abrupt.

    Frames are written as JPEG/PNG files into ``glm_dir`` with drawn green
    rectangles so the VLM can unambiguously identify the agent across images.

    Args:
        gen_video_path (dict[int, str]): Mapping from frame index to image
            file path.
        first_fids (list[int]): Two frame indices for early track appearance
            (length 2).
        first_box (list[tuple[int,int,int,int]]): Bounding boxes for each
            ``first_fids`` frame.
        last_fids (list[int]): Two frame indices near the track end
            (length 2).
        last_box (list[tuple[int,int,int,int]]): Bounding boxes for each
            ``last_fids`` frame.
        next_fids (list[int]): Two frame indices immediately after the track
            ends (length 2).
        glm_dir (str): Directory where annotated frames are saved.

    Returns:
        bool: ``True`` if the VLM judges the disappearance as **natural**
              (good generation), ``False`` if **unnatural** (artefact).
    """
    os.makedirs(glm_dir, exist_ok=True)

    # --- Draw green boxes on the first two track frames and save ---

    # First early frame (index 0 in first_fids)
    first_fid   = first_fids[0]
    first_img   = cv2.imread(gen_video_path[first_fid])
    suffix      = gen_video_path[first_fid][-4:]   # ".png" or ".jpg"
    x1, y1, x2, y2 = first_box[0]
    cv2.rectangle(first_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box
    cv2.imwrite(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'), first_img)

    # Second early frame (index 1 in first_fids)
    first_fid_2 = first_fids[1]
    first_img_2 = cv2.imread(gen_video_path[first_fid_2])
    suffix      = gen_video_path[first_fid_2][-4:]
    x1, y1, x2, y2 = first_box[1]
    cv2.rectangle(first_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{first_fid_2:05d}{suffix}'), first_img_2)

    # --- Draw green boxes on the last two track frames and save ---

    # Earlier of the two final frames
    last_fid_2 = last_fids[0]
    last_img_2 = cv2.imread(gen_video_path[last_fid_2])
    suffix     = gen_video_path[last_fid_2][-4:]
    x1, y1, x2, y2 = last_box[0]
    cv2.rectangle(last_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{last_fid_2:05d}{suffix}'), last_img_2)

    # Very last frame of the track
    last_fid   = last_fids[-1]
    last_img   = cv2.imread(gen_video_path[last_fid])
    suffix     = gen_video_path[last_fid][-4:]
    x1, y1, x2, y2 = last_box[1]
    cv2.rectangle(last_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(glm_dir, f'{last_fid:05d}{suffix}'), last_img)

    # --- Save the two post-track frames (no bounding box) ---

    next_fid   = next_fids[0]
    next_img   = cv2.imread(gen_video_path[next_fid])
    suffix     = gen_video_path[next_fid][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid:05d}{suffix}'), next_img)

    next_fid_2 = next_fids[1]
    next_img_2 = cv2.imread(gen_video_path[next_fid_2])
    suffix     = gen_video_path[next_fid_2][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'), next_img_2)

    # --- Load VLM (lazy init) ---
    global cosmos_r
    if cosmos_r is None:
        init_glm()
    llm, processor, system_prompt, vision_kwargs, sampling_params = cosmos_r

    # Natural-language question presented to the VLM alongside the 6 frames
    question = (
        "Given frames around the moment the same green-boxed object disappears, "
        "classify the disappearance as Natural (e.g., occlusion, leaving the field "
        "of view) or Unnatural (e.g., abrupt/non-physical disappearance). "
        "Base your decision on visual continuity, motion continuity, and the "
        "object's interactions with surrounding vehicles and the environment."
    )

    # Ordered list of frame paths (chronological: early → late → post-track)
    images = [
        os.path.join(glm_dir, f'{first_fid:05d}{suffix}'),    # early frame 1
        os.path.join(glm_dir, f'{first_fid_2:05d}{suffix}'),  # early frame 2
        os.path.join(glm_dir, f'{last_fid_2:05d}{suffix}'),   # late frame 1
        os.path.join(glm_dir, f'{last_fid:05d}{suffix}'),     # late frame 2
        os.path.join(glm_dir, f'{next_fid:05d}{suffix}'),     # post-track frame 1
        os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'),   # post-track frame 2
    ]
    videos = []  # no video-mode input used

    user_prompt = question.rstrip()

    # Build the Qwen2.5-VL conversation structure
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
    )

    # Tokenise the conversation into the model's input format
    prompt_str = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    # Extract pixel-level image tensors from the conversation
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )

    # Package multi-modal data for vLLM
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs   # list of PIL images or tensors
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt":             prompt_str,
        "multi_modal_data":   mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    # --- Run inference ---
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)

    # Parse structured result from VLM output
    result, _ = extract_tagged_text(output_text)
    if result:
        pprint_dict(result, "Result")

    # Return False (unnatural) if "Unnatural" appears in the answer tag
    if 'Unnatural' in result['answer'][0]:
        return False
    return True   # Natural disappearance


# ---------------------------------------------------------------------------
# Track downsampling
# ---------------------------------------------------------------------------

T = TypeVar("T")


def sample_track(
    seq: Sequence[T],
    keep_last: int = 10,
    total_keep: Optional[int] = None,
    stride: Optional[int] = None,
) -> List[T]:
    """
    Downsample a track sequence while preserving temporal structure.

    Strategy:
      - Always keep the **first** element (index 0) to anchor appearance.
      - Always keep the **last** ``keep_last`` elements so the model sees the
        agent just before it disappears.
      - From the remaining "front" segment, take equidistant samples either
        by target count (``total_keep``) or fixed step (``stride``).

    This is necessary because long tracks can have hundreds of frames but the
    VLM has a fixed context limit and inference time budget.

    Args:
        seq (Sequence[T]): Ordered track elements (frame-box tuples or any
            indexable sequence).
        keep_last (int): Number of tail elements to always keep.
            Default: 10.
        total_keep (int | None): Target total length of the output sequence
            (inclusive of tail).  Provide either this or ``stride``.
        stride (int | None): Fixed sampling stride for the front segment.
            Provide either this or ``total_keep``.

    Returns:
        List[T]: Downsampled subsequence in the original order, without
            duplicates.

    Raises:
        ValueError: If both or neither of ``total_keep`` / ``stride`` are
            provided, or if ``stride`` is non-positive.
    """
    n = len(seq)
    if n == 0:
        return []

    # If the sequence is short enough, keep everything
    if n <= keep_last + 1:
        return list(seq)

    # Index range for the "front" segment (excludes the tail)
    front_len = n - keep_last   # number of elements before the tail

    if total_keep is not None and stride is not None:
        raise ValueError("Provide either total_keep or stride, not both.")
    if total_keep is None and stride is None:
        raise ValueError("You must provide total_keep or stride.")

    early_indices: List[int] = []

    if total_keep is not None:
        # --- Mode A: control the final total count ---
        if total_keep >= n:
            # Requested more than the track length; keep everything
            early_indices = list(range(front_len))
        else:
            # Budget for the front segment (must include at least index 0)
            early_budget = max(total_keep - keep_last, 1)
            if early_budget == 1:
                early_indices = [0]   # only the anchor frame
            else:
                # Equidistant indices spanning [0, front_len-1]
                early_indices = [
                    round(i * (front_len - 1) / (early_budget - 1))
                    for i in range(early_budget)
                ]
                # Drop the last front index to avoid overlap with the tail segment
                if early_indices and early_indices[-1] == front_len - 1:
                    early_indices.pop()
    else:
        # --- Mode B: fixed stride sampling of the front segment ---
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        early_indices = list(range(0, front_len, stride))
        if early_indices[0] != 0:
            early_indices.insert(0, 0)      # guarantee index 0 is included
        if early_indices and early_indices[-1] == front_len - 1:
            early_indices.pop()             # avoid tail overlap

    # Tail indices always kept (last keep_last elements)
    tail_indices = list(range(n - keep_last, n))

    # Merge front + tail, preserving order and removing duplicates
    seen: set = set()
    keep_indices: List[int] = []
    for idx in early_indices + tail_indices:
        if idx not in seen:
            seen.add(idx)
            keep_indices.append(idx)

    return [seq[i] for i in keep_indices]


# ---------------------------------------------------------------------------
# VLM inference variant: uses full sampled track
# ---------------------------------------------------------------------------

def glm_reasoning_2(gen_video_path, track_boxes, next_fids, glm_dir):
    """
    Alternative VLM inference that sends a downsampled track to the model.

    Unlike ``glm_reasoning`` (which uses 4 hand-selected frames), this
    variant calls ``sample_track`` to select up to 10 representative
    frames from the entire track, then appends the two post-track frames.
    This gives the VLM a richer temporal context.

    Args:
        gen_video_path (dict[int, str]): Frame index to file path mapping.
        track_boxes (list[tuple[int, tuple]]): Full track as
            ``[(frame_id, (x1,y1,x2,y2)), ...]``.
        next_fids (list[int]): Two frame indices immediately after the track
            ends (length 2).
        glm_dir (str): Directory where annotated frames are saved.

    Returns:
        bool: ``True`` if the VLM judges the disappearance as **natural**,
              ``False`` if **unnatural**.
    """
    os.makedirs(glm_dir, exist_ok=True)

    images = []  # will be filled with file paths in chronological order

    # Downsample track: keep last 3 frames + equidistant early samples, max 10 total
    track_boxes = sample_track(track_boxes, keep_last=3, total_keep=10)

    # Draw green bounding box on each sampled track frame and save to disk
    for first_fid, box in track_boxes:
        first_img = cv2.imread(gen_video_path[first_fid])
        suffix    = gen_video_path[first_fid][-4:]       # ".png" or ".jpg"
        x1, y1, x2, y2 = box
        cv2.rectangle(first_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box
        cv2.imwrite(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'), first_img)
        images.append(os.path.join(glm_dir, f'{first_fid:05d}{suffix}'))

    # Save two post-track frames (no bounding box drawn)
    next_fid   = next_fids[0]
    next_img   = cv2.imread(gen_video_path[next_fid])
    suffix     = gen_video_path[next_fid][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid:05d}{suffix}'), next_img)

    next_fid_2 = next_fids[1]
    next_img_2 = cv2.imread(gen_video_path[next_fid_2])
    suffix     = gen_video_path[next_fid_2][-4:]
    cv2.imwrite(os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'), next_img_2)

    # --- Load VLM ---
    global cosmos_r
    if cosmos_r is None:
        init_glm()
    llm, processor, system_prompt, vision_kwargs, sampling_params = cosmos_r

    # Natural-language classification question
    question = (
        "Given frames around the moment the same green-boxed object disappears, "
        "classify the disappearance as Natural (e.g., occlusion, leaving the field "
        "of view) or Unnatural (abrupt/non-physical disappearance). "
        "Base your decision on visual continuity, motion continuity, and the "
        "object's interactions with surrounding vehicles and the environment."
    )

    # Append post-track frames (with no box) after the track frames
    images += [
        os.path.join(glm_dir, f'{next_fid:05d}{suffix}'),    # frame after track end
        os.path.join(glm_dir, f'{next_fid_2:05d}{suffix}'),  # two frames after track end
    ]
    videos = []

    user_prompt = question.rstrip()
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
    )

    # Tokenise and extract vision inputs
    prompt_str = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation, return_video_kwargs=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt":              prompt_str,
        "multi_modal_data":    mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    # Generate model response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)

    result, _ = extract_tagged_text(output_text)
    if result:
        pprint_dict(result, "Result")

    if 'Unnatural' in result['answer'][0]:
        return False   # unnatural disappearance detected
    return True        # natural disappearance


# ---------------------------------------------------------------------------
# Geometric utilities
# ---------------------------------------------------------------------------

def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single query box and an array of candidate boxes.

    Args:
        box (np.ndarray, shape (4,)): Query box ``[x1, y1, x2, y2]``.
        boxes (np.ndarray, shape (N, 4)): Candidate boxes.

    Returns:
        np.ndarray, shape (N,): IoU value for each candidate box.
    """
    # Top-left corner of intersection rectangle
    tl = np.maximum(box[:2], boxes[:, :2])          # shape: (N, 2)
    # Bottom-right corner of intersection rectangle
    br = np.minimum(box[2:], boxes[:, 2:])          # shape: (N, 2)

    # Side lengths of intersection; clip to 0 when boxes do not overlap
    inter_wh = np.clip(br - tl, a_min=0, a_max=None)  # shape: (N, 2)
    inter    = inter_wh[:, 0] * inter_wh[:, 1]         # shape: (N,) intersection area

    # Individual areas
    area1 = (box[2] - box[0]) * (box[3] - box[1])             # scalar
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # shape: (N,)

    # Union area (eps avoids division by zero)
    union = area1 + area2 - inter + 1e-9               # shape: (N,)
    return inter / union                               # shape: (N,) IoU values


def near_image_edge(box: Sequence[float],
                    img_wh: Tuple[int, int],
                    margin_ratio: float = 0.05) -> bool:
    """
    Determine whether a bounding box is near the image border.

    A box is "near the edge" if any side falls within ``margin_ratio``
    of the corresponding image dimension.  This is used to identify agents
    that are leaving the camera frustum — a natural cause for disappearance
    that should not be penalised.

    Args:
        box (Sequence[float]): Bounding box ``[x1, y1, x2, y2]``.
        img_wh (Tuple[int, int]): Image ``(width, height)`` in pixels.
        margin_ratio (float): Fractional margin for edge detection.
            Default: 0.05 (5% of image dimension).

    Returns:
        bool: ``True`` if the box is within the margin of at least one edge.
    """
    w, h = img_wh
    x1, y1, x2, y2 = box
    margin_x = w * margin_ratio   # absolute pixel margin in x direction
    margin_y = h * margin_ratio   # absolute pixel margin in y direction

    # Check left edge, right edge, and bottom edge
    # (top-edge check is commented out in the original — top exits are less
    # common in front-facing driving cameras)
    return (
        x1 <= margin_x or
        (w - x2) <= margin_x or
        (h - y2) <= margin_y
    )


# ---------------------------------------------------------------------------
# Per-agent missing detection
# ---------------------------------------------------------------------------

def disappeared_suddenly(
    track_boxes: List[Tuple[int, Sequence[float]]],
    other_boxes_by_frame: Dict[int, List[Sequence[float]]],
    gen_video_path,
    glm_dir: str,
    img_size: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_threshold: float = 0.2,
    min_track_len: int = 5,
) -> bool:
    """
    Determine whether an agent track ended due to an unnatural disappearance.

    Decision logic:
      1. Reject tracks shorter than ``min_track_len`` (too little evidence).
      2. Reject tracks that end on the very last frame (frame 100) — no
         disappearance to judge.
      3. Delegate to ``glm_reasoning`` (Cosmos-Reason1-7B) to classify the
         disappearance as natural or unnatural based on visual context.

    Args:
        track_boxes (List[Tuple[int, Sequence[float]]]): Chronologically
            ordered track as ``[(frame_id, [x1,y1,x2,y2]), ...]``.
        other_boxes_by_frame (Dict[int, List[Sequence[float]]]): Detections
            from *all other* agents indexed by frame id; used for occlusion
            reasoning (currently passed to but not used inside this function
            due to commented-out logic).
        gen_video_path: Mapping from frame index (int) to image path (str).
        glm_dir (str): Directory for saving annotated frames for the VLM.
        img_size (Tuple[int, int]): Video resolution ``(W, H)`` in pixels.
        edge_margin (float): Fraction of image width/height used as edge
            proximity threshold.  Default: 0.05.
        iou_threshold (float): Minimum IoU with another agent's box to
            consider the disappearance as occlusion (currently disabled).
        min_track_len (int): Minimum track length; shorter tracks are skipped.

    Returns:
        bool: ``True`` if the disappearance is judged **unnatural**,
              ``False`` otherwise (natural, too short, or ended at last frame).
    """
    if len(track_boxes) < min_track_len:
        # Too few observations to judge; skip rather than false-positive
        print(f'len track_bbox is {len(track_boxes)}: too short, skip')
        return False

    # --- Check if the track ends at the final frame ---
    last_fid, last_box = track_boxes[-1]
    last_box = np.asarray(last_box, dtype=float)  # shape: (4,)

    if last_fid == 100:
        # The track extends to the last frame; not a disappearance
        print(f'no missing: 100 frames')
        return False

    # --- Assemble key frame indices for the VLM ---

    # Two early frames: index 0 and 1 in the track
    first_fid,  first_box  = track_boxes[0]   # anchor / first appearance
    first_fid_, first_box_ = track_boxes[1]   # second frame
    first_fids = [first_fid, first_fid_]
    first_boxs = [first_box, first_box_]

    # Two late frames: one near the end and the very last
    last_fids = []
    last_boxs = []
    idx_last = max(0, len(track_boxes) - 5)  # 5 frames before the end (or start)
    for id in [idx_last, -1]:
        lf, lb = track_boxes[id]
        last_fids.append(lf)
        last_boxs.append(lb)

    # Two post-track frames: frame_after_last and frame_after_last+1
    next_fids = [last_fids[-1] + 1]
    if last_fids[-1] + 2 <= 100:
        next_fids.append(last_fids[-1] + 2)   # second post-track frame exists
    else:
        next_fids.append(last_fids[-1] + 1)   # clamp to available frames

    # --- Query VLM for naturalness judgment ---
    is_normal_missing = glm_reasoning(
        gen_video_path,
        first_fids, first_boxs,
        last_fids, last_boxs,
        next_fids,
        glm_dir,
    )

    if is_normal_missing:
        return False   # VLM says natural; not an error

    # VLM said unnatural: flag this as a sudden disappearance
    return True


# ---------------------------------------------------------------------------
# Scene-level aggregation helpers
# ---------------------------------------------------------------------------

def get_missing_per_scene(missings: list) -> float:
    """
    Compute the fraction of unnaturally disappeared agents in a single scene.

    Args:
        missings (list[tuple]): List of ``(agent_track, is_missing)`` pairs
            where ``is_missing`` is a bool.

    Returns:
        float: Mean missing score (0.0 if no unnatural disappearances).
    """
    scene_score = []
    for id, missing in missings:
        if missing:
            score = 1    # each unnatural disappearance contributes score=1
            scene_score.append(score)
    if len(scene_score) > 0:
        return np.mean(scene_score)   # fraction of unnatural disappearances
    else:
        return 0.0


def get_missing_per_agent(missings: list) -> int:
    """
    Count the total number of agents that disappeared unnaturally in a scene.

    Args:
        missings (list[tuple]): List of ``(agent_track, is_missing)`` pairs.

    Returns:
        int: Count of agents with unnatural disappearances.
    """
    count = 0
    for id, missing in missings:
        if missing:
            count += 1
    return count   # total unnatural disappearances in this scene


# ---------------------------------------------------------------------------
# Single-frame occlusion helpers (utility, not used in main pipeline)
# ---------------------------------------------------------------------------

def occluded_in_frame(
    box: Sequence[float],
    other_boxes: List[Sequence[float]],
    img_wh: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_thr: float = 0.5,
) -> bool:
    """
    Determine whether a box is occluded by other objects in a single frame.

    A box is considered occluded if it is not near the image border AND its
    IoU with at least one other detection exceeds ``iou_thr``.

    Args:
        box (Sequence[float]): Query box ``[x1, y1, x2, y2]``.
        other_boxes (List[Sequence[float]]): Other detections in the same
            frame (shape: list of length-4 sequences).
        img_wh (Tuple[int, int]): Image ``(width, height)`` in pixels.
        edge_margin (float): Edge proximity threshold fraction.
        iou_thr (float): IoU threshold for declaring occlusion.

    Returns:
        bool: ``True`` if the box is likely occluded.
    """
    if near_image_edge(box, img_wh, edge_margin) or not other_boxes:
        return False   # near edge or no other objects → not occluded

    ious = bbox_iou(np.asarray(box, float), np.asarray(other_boxes, float))
    # shape: (N,) IoU values
    return bool((ious >= iou_thr).any())


def track_occlusion_score(
    track_boxes: List[Tuple[int, Sequence[float]]],
    other_boxes_by_frame: Dict[int, List[Sequence[float]]],
    img_wh: Tuple[int, int],
    *,
    edge_margin: float = 0.05,
    iou_thr: float = 0.5,
) -> Tuple[int, float, List[int]]:
    """
    Compute per-track occlusion statistics across all frames.

    Args:
        track_boxes (List[Tuple[int, Sequence[float]]]): Track detections
            ``[(frame_id, [x1,y1,x2,y2]), ...]``.
        other_boxes_by_frame (Dict[int, List[Sequence[float]]]): Per-frame
            detections from all other agents.
        img_wh (Tuple[int, int]): Image ``(width, height)`` in pixels.
        edge_margin (float): Edge proximity threshold fraction.
        iou_thr (float): IoU threshold for declaring occlusion.

    Returns:
        Tuple:
            - **n_occ** (int): Number of frames where the agent is occluded.
            - **ratio** (float): ``n_occ / len(track_boxes)``.
            - **occ_fids** (List[int]): Frame ids where occlusion was detected.
    """
    occ_fids = [
        fid for fid, box in track_boxes
        if occluded_in_frame(
            box,
            other_boxes_by_frame.get(fid, []),
            img_wh,
            edge_margin=edge_margin,
            iou_thr=iou_thr,
        )
    ]
    n_occ = len(occ_fids)
    ratio = n_occ / len(track_boxes) if track_boxes else 0.0
    return n_occ, ratio, occ_fids


# ---------------------------------------------------------------------------
# IoU-based box matching (reused from video_a_consist)
# ---------------------------------------------------------------------------

def max_iou_box(query_box: np.ndarray,
                boxes: np.ndarray,
                return_index: bool = False):
    """
    Find the box in ``boxes`` with the highest IoU relative to ``query_box``.

    Args:
        query_box (array-like, shape (4,)): Reference box ``[x1, y1, x2, y2]``.
        boxes (array-like, shape (N, 4)): Candidate boxes to search.
        return_index (bool): If True, also return the index of the best match.

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

    # Intersection corners
    ix1 = np.maximum(query_box[0], boxes[:, 0])  # shape: (N,)
    iy1 = np.maximum(query_box[1], boxes[:, 1])  # shape: (N,)
    ix2 = np.minimum(query_box[2], boxes[:, 2])  # shape: (N,)
    iy2 = np.minimum(query_box[3], boxes[:, 3])  # shape: (N,)

    # Clamp negative dimensions to zero
    inter_w = np.clip(ix2 - ix1, 0, None)        # shape: (N,)
    inter_h = np.clip(iy2 - iy1, 0, None)        # shape: (N,)
    inter   = inter_w * inter_h                   # shape: (N,) intersection areas

    query_area = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])  # scalar
    boxes_area = (boxes[:, 2]  - boxes[:, 0]) * (boxes[:, 3]  - boxes[:, 1])   # shape: (N,)

    # IoU with epsilon for numerical stability
    iou = inter / (query_area + boxes_area - inter + 1e-8)  # shape: (N,)

    best_idx = int(iou.argmax())
    best_iou = float(iou[best_idx])
    best_box = boxes[best_idx]                    # shape: (4,)

    if return_index:
        return best_box, best_iou, best_idx
    return best_box, best_iou


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def get_agent_missing(valid_agents_runs, agents_bbox, agents_label, names, img_dirs):
    """
    Aggregate per-agent missing detection scores across all scenes.

    For each scene and agent:
      1. Build the set of other agents' boxes (per frame) to pass context to
         ``disappeared_suddenly``.
      2. Call ``disappeared_suddenly`` (which calls the VLM) to judge each
         track.
      3. Compute scene-level missing fraction with ``get_missing_per_scene``.
      4. Average across scenes; return ``1 - mean_missing_rate`` so that a
         perfect generation (no unnatural disappearances) scores 1.0.

    Args:
        valid_agents_runs (list[str]): Scene identifier strings.
        agents_bbox (list[list[list[tuple]]]): Per-scene per-agent track
            data: ``agents_bbox[scene_id][agent_id]`` is a list of
            ``(frame_id, (x1,y1,x2,y2))`` tuples.
        agents_label (list[dict]): Per-scene label dictionaries
            (not used in this function but kept for API consistency).
        names (list[str]): Per-scene directories for VLM frame caching.
        img_dirs (list[str]): Per-scene frame image directories.

    Returns:
        float: Dataset-level naturalness score in ``[0, 1]``.
            Higher values indicate fewer unnatural disappearances.
    """
    missing_rate    = []  # per-scene missing fraction
    num_agent       = 0   # total agent count across all scenes
    num_missing_agent = 0 # total count of unnaturally disappeared agents

    print(f'len valid runs: {len(valid_agents_runs)}, len agents_bbox: {len(agents_bbox)}')

    for scene_id, scene_agents in enumerate(agents_bbox):
        is_missing = []        # list of (agent_track, bool) for this scene
        num_agent += len(scene_agents)

        # Build {frame_index: img_path} mapping for this scene
        gen_video_path = sorted(
            [os.path.join(img_dirs[scene_id], f)
             for f in os.listdir(img_dirs[scene_id])]
        )
        glm_dir = names[scene_id]  # base directory for VLM frame cache

        video_img_dict: Dict[int, str] = {}
        for img_id, img in enumerate(gen_video_path):
            video_img_dict[img_id] = img   # frame_index → file path

        for agent_id, scene_agent in enumerate(scene_agents):
            # Build the "other agents" context: all agents except the current one
            other_agents = scene_agents.copy()
            other_agents.remove(scene_agent)

            # Convert to per-frame dict: {frame_id: [box, box, ...]}
            other_boxes_by_frame: Dict[int, List] = {}
            for other in other_agents:
                for o in other:
                    f_id, bbox = o
                    if f_id not in other_boxes_by_frame:
                        other_boxes_by_frame[f_id] = [bbox]
                    else:
                        other_boxes_by_frame[f_id].append(bbox)

            # Per-agent VLM cache directory
            glm_this = os.path.join(glm_dir, f'{agent_id}')

            # Core judgment: did this agent disappear unnaturally?
            missing = disappeared_suddenly(
                scene_agent,
                other_boxes_by_frame,
                video_img_dict,
                glm_this,
                img_size=(1024, 576),   # expected video resolution (W, H)
                edge_margin=0.1,
                iou_threshold=0.5,
                min_track_len=2,
            )
            print(f'{names[scene_id]} - agent {agent_id} missing: {missing}')
            is_missing.append((scene_agent, missing))

        # Aggregate this scene's results
        missing_rate.append((get_missing_per_scene(is_missing), is_missing))
        num_missing_agent += get_missing_per_agent(is_missing)

    assert len(missing_rate) == len(valid_agents_runs)

    # Extract per-scene missing fractions and compute dataset mean
    missing_rate = [m[0] for m in missing_rate]          # list of floats
    print(len(missing_rate))
    missing_rate = np.nanmean(missing_rate, axis=0)       # scalar mean

    # Invert: 1 = no missing agents (perfect), 0 = all agents missing
    return 1 - missing_rate
