#!/usr/bin/env python3
"""
caption_usage_eval_v3.py

Evaluate how much of a caption was followed by a generated ego-vehicle-view video.

Supports:
    1. Single video-caption pair
    2. Folder of videos + folder of matching caption .txt files
    3. Folder of videos + JSON mapping of captions
    4. Uniform frame sampling, e.g. 8 or 16 frames
    5. Dynamic aspect-ratio-preserving resize before Qwen3-VL
    6. Correct Qwen3-VL integrated video tokenization path
    7. Safe OpenCV sampled-frame fallback if direct video processing fails
    8. Qwen3-VL text-only claim extraction
    9. Qwen3-VL video claim verification
    10. YOLO tracking with Ultralytics
    11. CUDA cleanup after every video

Install:
    pip install -U torch torchvision transformers accelerate "qwen-vl-utils[decord]>=0.0.14" \
        decord opencv-python pillow ultralytics numpy tqdm

Recommended batch command:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python caption_usage_eval_v3.py \
        --video_dir ./datasets/output_videos/ltx-video-13b-drivinggen-samples-outputs/videos/ \
        --caption_dir ./datasets/input_captions/ \
        --outdir ./datasets/output_cue/ltx_13b \
        --num_frames 8 \
        --resize_long_side 448 \
        --yolo_device cpu \
        --torch_dtype float16
"""

import argparse
import collections
import gc
import html
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO


# -----------------------------
# Constants
# -----------------------------

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

VALID_CATEGORIES = {
    "ego_vehicle",
    "traffic_agents",
    "actions",
    "environment",
    "global_subjective",
}

VEHICLE_CLASS_NAMES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bicycle",
    "van",
}

PERSON_CLASS_NAMES = {
    "person",
    "pedestrian",
}

TRAFFIC_LIGHT_CLASS_NAMES = {
    "traffic light",
    "traffic_light",
}


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class VideoCaptionPair:
    video_path: str
    caption: str
    pair_id: str


@dataclass
class TrackRecord:
    frame_idx: int
    track_id: int
    class_id: int
    class_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    center: Tuple[float, float]


# -----------------------------
# Utility functions
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ensure_dir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def read_caption(args: argparse.Namespace) -> str:
    if args.caption_file:
        return Path(args.caption_file).read_text(encoding="utf-8").strip()
    if args.caption:
        return args.caption.strip()
    raise ValueError("Provide either --caption or --caption_file.")


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else None

    cap.release()

    return {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def get_video_hw(video_path: str) -> Tuple[int, int]:
    info = get_video_info(video_path)
    return int(info["height"]), int(info["width"])


def parse_torch_dtype(dtype_str: str):
    dtype_str = str(dtype_str).lower()

    if dtype_str == "auto":
        return "auto"
    if dtype_str in {"none", "null"}:
        return None
    if dtype_str in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_str in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_str in {"float32", "fp32"}:
        return torch.float32

    raise ValueError(
        f"Unsupported --torch_dtype {dtype_str}. "
        "Use auto, float16, bfloat16, float32, or none."
    )


def round_to_divisor(x: int, divisor: int = 28) -> int:
    return max(divisor, int(round(x / divisor) * divisor))


def compute_resized_hw(
    height: int,
    width: int,
    long_side: int,
    divisor: int = 28,
) -> Tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid video dimensions: height={height}, width={width}")

    scale = float(long_side) / float(max(height, width))
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))

    new_h = round_to_divisor(new_h, divisor)
    new_w = round_to_divisor(new_w, divisor)

    return new_h, new_w


def safe_json_loads(text: str) -> Any:
    text = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = text[obj_start:obj_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidate = text[arr_start:arr_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text[:2000]}")


def normalize_score(x: Any) -> float:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes", "true", "followed", "clearly followed", "1", "1.0"}:
            return 1.0
        if s in {"partial", "partially", "ambiguous", "unclear", "0.5"}:
            return 0.5
        if s in {"no", "false", "not followed", "ignored", "missing", "0", "0.0"}:
            return 0.0
        try:
            x = float(s)
        except Exception:
            return 0.5

    try:
        val = float(x)
    except Exception:
        return 0.5

    val = max(0.0, min(1.0, val))

    if val >= 0.75:
        return 1.0
    if val <= 0.25:
        return 0.0
    return 0.5


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def score_to_label(score: float) -> str:
    if score >= 0.75:
        return "followed"
    if score <= 0.25:
        return "ignored"
    return "partially_followed"


def score_to_color(score: float) -> str:
    if score >= 0.75:
        return "#c8f7c5"
    if score <= 0.25:
        return "#ffc7c7"
    return "#fff3b0"


def short_direction(dx: float, dy: float, stationary_thresh: float = 8.0) -> str:
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < stationary_thresh:
        return "stationary"

    if abs(dx) >= abs(dy):
        return "rightward" if dx > 0 else "leftward"
    return "downward" if dy > 0 else "upward"


def parse_video_exts(exts: str) -> List[str]:
    parsed = []
    for e in exts.split(","):
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        parsed.append(e)
    return parsed


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def make_pair_id(video_path: Path, video_root: Optional[Path] = None) -> str:
    if video_root is not None:
        try:
            rel = video_path.relative_to(video_root)
            no_ext = rel.with_suffix("")
            return sanitize_name("__".join(no_ext.parts))
        except ValueError:
            pass

    return sanitize_name(video_path.stem)


def load_caption_json(caption_json: str) -> Dict[str, str]:
    path = Path(caption_json)
    data = json.loads(path.read_text(encoding="utf-8"))

    mapping: Dict[str, str] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                mapping[str(key)] = value.strip()
            elif isinstance(value, dict):
                if "caption" in value:
                    mapping[str(key)] = str(value["caption"]).strip()
                elif "caption_file" in value:
                    mapping[str(key)] = Path(value["caption_file"]).read_text(
                        encoding="utf-8"
                    ).strip()
        return mapping

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            video_key = (
                item.get("video")
                or item.get("video_path")
                or item.get("filename")
                or item.get("file")
            )

            if not video_key:
                continue

            if "caption" in item:
                caption = str(item["caption"]).strip()
            elif "caption_file" in item:
                caption = Path(item["caption_file"]).read_text(
                    encoding="utf-8"
                ).strip()
            else:
                continue

            mapping[str(video_key)] = caption

        return mapping

    raise ValueError(f"Unsupported caption JSON format: {caption_json}")


def find_caption_for_video_from_mapping(
    video_path: Path,
    mapping: Dict[str, str],
) -> Optional[str]:
    candidates = [
        str(video_path),
        str(video_path.resolve()),
        video_path.name,
        video_path.stem,
    ]

    for key in candidates:
        if key in mapping:
            return mapping[key]

    return None


def find_video_caption_pairs(args: argparse.Namespace) -> List[VideoCaptionPair]:
    pairs: List[VideoCaptionPair] = []

    if args.video:
        caption = read_caption(args)
        video_path = Path(args.video).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        pairs.append(
            VideoCaptionPair(
                video_path=str(video_path),
                caption=caption,
                pair_id=make_pair_id(video_path),
            )
        )

        return pairs

    if not args.video_dir:
        raise ValueError("Provide either --video or --video_dir.")

    video_root = Path(args.video_dir).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"Video directory not found: {video_root}")

    if not args.caption_dir and not args.caption_json:
        raise ValueError(
            "For --video_dir mode, provide either --caption_dir or --caption_json."
        )

    video_exts = parse_video_exts(args.video_exts)

    if args.recursive:
        video_paths = [
            p for p in video_root.rglob("*")
            if p.is_file() and p.suffix.lower() in video_exts
        ]
    else:
        video_paths = [
            p for p in video_root.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        ]

    video_paths = sorted(video_paths)

    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {video_root} with extensions {video_exts}"
        )

    caption_mapping = None
    if args.caption_json:
        caption_mapping = load_caption_json(args.caption_json)

    caption_root = Path(args.caption_dir).resolve() if args.caption_dir else None

    missing_captions = []

    for video_path in video_paths:
        caption: Optional[str] = None

        if caption_mapping is not None:
            caption = find_caption_for_video_from_mapping(video_path, caption_mapping)

        if caption is None and caption_root is not None:
            caption_ext = args.caption_ext
            if not caption_ext.startswith("."):
                caption_ext = "." + caption_ext

            if args.recursive:
                rel = video_path.relative_to(video_root)
                caption_path = caption_root / rel.with_suffix(caption_ext)
            else:
                caption_path = caption_root / f"{video_path.stem}{caption_ext}"

            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8").strip()

        if caption is None or not caption.strip():
            missing_captions.append(str(video_path))
            continue

        pairs.append(
            VideoCaptionPair(
                video_path=str(video_path),
                caption=caption.strip(),
                pair_id=make_pair_id(video_path, video_root),
            )
        )

    if missing_captions:
        logging.warning(
            f"Skipped {len(missing_captions)} videos because captions were missing."
        )

        missing_path = Path(args.outdir) / "missing_captions.json"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(
            json.dumps(missing_captions, indent=2),
            encoding="utf-8",
        )
        logging.warning(f"Saved missing caption list to: {missing_path}")

    if not pairs:
        raise RuntimeError("No valid video-caption pairs found.")

    return pairs


def print_extracted_claims(claims: List[Dict[str, Any]]) -> None:
    print("\n==============================")
    print("Extracted Atomic Caption Claims")
    print("==============================")

    for c in claims:
        print(
            f"[{c.get('claim_id', '')}] "
            f"({c.get('category', '')}, verifier={c.get('preferred_verifier', '')}, "
            f"importance={c.get('importance', '')})"
        )
        print(f"Claim: {c.get('claim', '')}")
        print(f"Question: {c.get('question', '')}")
        print("-" * 80)

    print("")


def manual_sample_video_frames(
    video_path: str,
    num_frames: Optional[int],
    sample_fps: Optional[float],
    resize_long_side: Optional[int],
    resize_divisor: int,
    max_pixels: int,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Safe fallback when qwen-vl-utils returns empty video inputs.

    Returns:
        frames: list[PIL.Image]
        metadata:
            {
              "fps": original_video_fps,
              "frame_indices": sampled frame indices,
              "total_num_frames": total frames,
            }
    """
    info = get_video_info(video_path)
    video_fps = float(info["fps"]) if info["fps"] and info["fps"] > 0 else 24.0
    total_frames = int(info["frame_count"])
    width = int(info["width"])
    height = int(info["height"])

    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    if num_frames is not None:
        n = max(1, int(num_frames))
        frame_indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()
    elif sample_fps is not None and sample_fps > 0:
        duration = total_frames / video_fps
        n = max(1, int(round(duration * float(sample_fps))))
        frame_indices = np.linspace(0, total_frames - 1, n, dtype=int).tolist()
    else:
        frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int).tolist()

    if resize_long_side is not None:
        resized_h, resized_w = compute_resized_hw(
            height=height,
            width=width,
            long_side=int(resize_long_side),
            divisor=int(resize_divisor),
        )
    else:
        # If no explicit resize_long_side, derive a safe size from max_pixels.
        scale = math.sqrt(float(max_pixels) / max(1.0, float(height * width)))
        if scale < 1.0:
            resized_h = round_to_divisor(int(round(height * scale)), resize_divisor)
            resized_w = round_to_divisor(int(round(width * scale)), resize_divisor)
        else:
            resized_h, resized_w = height, width

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for manual sampling: {video_path}")

    frames: List[Image.Image] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if resized_h != height or resized_w != width:
            frame = cv2.resize(frame, (int(resized_w), int(resized_h)), interpolation=cv2.INTER_AREA)

        frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Manual fallback sampled zero frames from: {video_path}")

    metadata = {
        "fps": video_fps,
        "frame_indices": [int(x) for x in frame_indices[:len(frames)]],
        "total_num_frames": total_frames,
    }

    return frames, metadata


# -----------------------------
# Qwen runner
# -----------------------------

class QwenRunner:
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str = "none",
    ) -> None:
        logging.info(f"Loading Qwen model: {model_id}")

        dtype = parse_torch_dtype(torch_dtype)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if dtype is not None:
            # Transformers >= 4.56 prefers dtype, but older versions may still prefer torch_dtype.
            model_kwargs["dtype"] = dtype

        if attn_implementation and attn_implementation != "none":
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs,
            )
        except TypeError:
            if "dtype" in model_kwargs:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs,
            )

        self.model.eval()

    def _input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self._input_device()
        moved = {}

        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = v.to(device)
            else:
                moved[k] = v

        return moved

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        inputs = None
        output_ids = None
        generated = None

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = self._move_inputs_to_device(inputs)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            input_len = inputs["input_ids"].shape[-1]
            generated = output_ids[:, input_len:]

            text = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return text.strip()

        finally:
            del inputs
            del output_ids
            del generated
            cleanup_memory()

    @torch.inference_mode()
    def generate_video(
        self,
        video_path: str,
        prompt: str,
        sample_fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        resize_long_side: Optional[int] = None,
        resize_divisor: int = 28,
        max_pixels: int = 360 * 640,
        max_new_tokens: int = 4096,
    ) -> str:
        """
        Qwen3-VL video inference using the Qwen3-VL processor's integrated
        chat-template tokenization path.

        Why this is different from the older Qwen2-VL style path:
            The older pattern builds the chat-template text separately, calls
            qwen_vl_utils.process_vision_info(...), and then calls the processor
            with text + videos. With newer Qwen3-VL/Transformers combinations,
            this can create video features without inserting the corresponding
            video placeholder tokens into input_ids, producing:

                ValueError: Video features and video tokens do not match,
                tokens: 0, features: N

        This function first uses:
            processor.apply_chat_template(..., tokenize=True, return_dict=True)
        directly on the multimodal messages. That lets the Qwen3-VL processor
        create both the visual features and matching placeholder tokens.

        If direct video decoding/tokenization still fails because of a local
        version mismatch, it falls back to OpenCV uniform frame sampling and
        sends the sampled frames as an ordered image sequence.
        """
        if sample_fps is not None and num_frames is not None:
            raise ValueError("Use either --sample_fps or --num_frames, not both.")

        def build_video_item() -> Dict[str, Any]:
            item: Dict[str, Any] = {
                "type": "video",
                "video": str(video_path),
            }

            if num_frames is not None:
                item["nframes"] = int(num_frames)
            elif sample_fps is not None:
                item["fps"] = float(sample_fps)
            else:
                item["nframes"] = 8

            if resize_long_side is not None:
                h, w = get_video_hw(video_path)
                resized_h, resized_w = compute_resized_hw(
                    height=h,
                    width=w,
                    long_side=int(resize_long_side),
                    divisor=int(resize_divisor),
                )
                item["resized_height"] = int(resized_h)
                item["resized_width"] = int(resized_w)
            else:
                item["max_pixels"] = int(max_pixels)

            return item

        def run_messages(messages: List[Dict[str, Any]]) -> str:
            inputs = None
            output_ids = None
            generated = None

            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = self._move_inputs_to_device(inputs)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

                input_len = inputs["input_ids"].shape[-1]
                generated = output_ids[:, input_len:]

                out_text = self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                return out_text.strip()

            finally:
                del inputs
                del output_ids
                del generated
                cleanup_memory()

        video_messages = [
            {
                "role": "user",
                "content": [
                    build_video_item(),
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            return run_messages(video_messages)

        except Exception as direct_error:
            # The specific error in the provided log is:
            #   Video features and video tokens do not match, tokens: 0, features: 1792
            # This fallback also protects against minor Transformers/Qwen3-VL
            # processor version mismatches around video decoding.
            err = str(direct_error)
            logging.warning(
                "Qwen3-VL direct video processor path failed. "
                "Falling back to sampled-frame image-sequence mode. "
                f"Error: {err[:500]}"
            )
            cleanup_memory()

            fallback_num_frames = num_frames
            if "out of memory" in err.lower() and num_frames is not None:
                # Give the fallback a chance to complete on constrained GPUs.
                fallback_num_frames = max(4, min(16, int(num_frames)))
                if fallback_num_frames != num_frames:
                    logging.warning(
                        f"CUDA OOM detected. Reducing fallback frames from "
                        f"{num_frames} to {fallback_num_frames}."
                    )

            frames, metadata = manual_sample_video_frames(
                video_path=video_path,
                num_frames=fallback_num_frames,
                sample_fps=sample_fps if fallback_num_frames is None else None,
                resize_long_side=resize_long_side,
                resize_divisor=resize_divisor,
                max_pixels=max_pixels,
            )

            frame_note = (
                "The following images are uniformly sampled frames from one "
                "silent driving video, shown in chronological order. Treat them "
                "as a short video sequence, not as independent images.\n"
                f"Original video FPS: {metadata.get('fps')}.\n"
                f"Original total frames: {metadata.get('total_num_frames')}.\n"
                f"Sampled frame indices: {metadata.get('frame_indices')}.\n\n"
            )

            image_content: List[Dict[str, Any]] = [
                {"type": "image", "image": frame}
                for frame in frames
            ]

            image_messages = [
                {
                    "role": "user",
                    "content": image_content + [
                        {"type": "text", "text": frame_note + prompt}
                    ],
                }
            ]

            return run_messages(image_messages)


# -----------------------------
# Caption claim extraction
# -----------------------------

def build_claim_extraction_prompt(caption: str) -> str:
    return f"""
You are decomposing a driving-video generation caption into atomic, visually verifiable claims.

Caption:
{caption}

Task:
Split the caption into small claims that can be checked in the generated video.

Use only these categories:
- ego_vehicle
- traffic_agents
- actions
- environment
- global_subjective

Use only these preferred_verifier values:
- qwen_vl
- yolo
- both

Rules:
1. Each claim must be atomic.
2. Split temporal and causal events into separate claims when useful.
3. Prefer concrete visual claims over subjective claims.
4. Keep subjective claims only if they are visually inferable.
5. Do not include claims that cannot be checked visually from a silent video.
6. Make the claims useful for an autonomous-driving / ego-vehicle-view video.
7. Give higher importance to safety-critical and concrete claims.
8. Keep each claim short.
9. Do not assign yolo as verifier for wet road, overcast sky, windshield wipers, trees, signs, or weather. Use qwen_vl for those.
10. Use yolo mainly for cars, trucks, buses, people, traffic lights, and object motion.

Return valid JSON only with this schema:

{{
  "claims": [
    {{
      "claim_id": "C1",
      "claim": "The camera appears to be mounted on a car or in an ego-vehicle viewpoint.",
      "category": "ego_vehicle",
      "claim_type": "viewpoint",
      "preferred_verifier": "qwen_vl",
      "importance": 5,
      "question": "Does the video appear to be captured from a camera mounted on a car or from an ego-vehicle viewpoint?"
    }}
  ]
}}
""".strip()


def fallback_claims_from_caption(caption: str) -> List[Dict[str, Any]]:
    parts = re.split(r"(?<=[.!?])\s+", caption.strip())
    claims: List[Dict[str, Any]] = []

    for i, sent in enumerate(parts, start=1):
        sent = sent.strip()
        if not sent:
            continue

        lower = sent.lower()

        if any(x in lower for x in ["camera", "ego", "mounted", "forward", "stationary"]):
            category = "ego_vehicle"
        elif any(x in lower for x in ["vehicle", "vehicles", "cars", "truck", "bus", "pedestrian", "traffic light"]):
            category = "traffic_agents"
        elif any(x in lower for x in ["change", "move", "pass", "cross", "turn", "stop", "follows"]):
            category = "actions"
        elif any(x in lower for x in ["night", "urban", "street", "intersection", "wet", "building", "crosswalk", "rain", "overcast", "tree", "barrier", "sign"]):
            category = "environment"
        else:
            category = "global_subjective"

        preferred = "both" if category in {"traffic_agents", "actions"} else "qwen_vl"

        claims.append(
            {
                "claim_id": f"C{i}",
                "claim": sent,
                "category": category,
                "claim_type": "unknown",
                "preferred_verifier": preferred,
                "importance": 3,
                "question": f"Is the following claim visible in the video: {sent}",
            }
        )

    return claims


def extract_claims(
    qwen: QwenRunner,
    caption: str,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    prompt = build_claim_extraction_prompt(caption)
    raw = qwen.generate_text(prompt, max_new_tokens=max_new_tokens)

    try:
        parsed = safe_json_loads(raw)
        claims = parsed.get("claims", parsed if isinstance(parsed, list) else [])
        if not isinstance(claims, list) or not claims:
            raise ValueError("No claims found.")
    except Exception as e:
        logging.warning(
            "Claim extraction JSON parse failed. "
            f"Falling back to sentence split. Error: {e}"
        )
        claims = fallback_claims_from_caption(caption)

    cleaned: List[Dict[str, Any]] = []

    for idx, c in enumerate(claims, start=1):
        claim_id = str(c.get("claim_id", f"C{idx}")).strip()
        claim = str(c.get("claim", "")).strip()

        if not claim:
            continue

        category = str(c.get("category", "environment")).strip()
        if category not in VALID_CATEGORIES:
            category = "environment"

        preferred = str(c.get("preferred_verifier", "qwen_vl")).strip()
        if preferred not in {"qwen_vl", "yolo", "both"}:
            preferred = "qwen_vl"

        # Force obviously non-YOLO concepts back to qwen_vl.
        claim_lower = claim.lower()
        non_yolo_terms = [
            "wet", "overcast", "rain", "wiper", "windshield", "tree",
            "barrier", "sign", "speed limit", "sky", "weather", "road surface",
            "calm", "quiet", "atmosphere", "headlights", "reflections"
        ]
        if any(t in claim_lower for t in non_yolo_terms):
            preferred = "qwen_vl"

        try:
            importance = int(c.get("importance", 3))
        except Exception:
            importance = 3

        importance = max(1, min(5, importance))

        question = str(
            c.get("question", f"Is this claim visible in the video: {claim}")
        ).strip()

        cleaned.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "category": category,
                "claim_type": str(c.get("claim_type", "unknown")).strip(),
                "preferred_verifier": preferred,
                "importance": importance,
                "question": question,
            }
        )

    return cleaned


def print_extracted_claims(claims: List[Dict[str, Any]]) -> None:
    print("\n==============================")
    print("Extracted Atomic Caption Claims")
    print("==============================")

    for c in claims:
        print(
            f"[{c.get('claim_id', '')}] "
            f"({c.get('category', '')}, verifier={c.get('preferred_verifier', '')}, "
            f"importance={c.get('importance', '')})"
        )
        print(f"Claim: {c.get('claim', '')}")
        print(f"Question: {c.get('question', '')}")
        print("-" * 80)

    print("")


# -----------------------------
# YOLO tracking
# -----------------------------

def run_yolo_tracking(
    video_path: str,
    yolo_model_path: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    yolo_device: str = "cpu",
    max_frames: Optional[int] = None,
) -> Tuple[List[TrackRecord], Dict[str, Any]]:
    logging.info(
        f"Running YOLO tracking: model={yolo_model_path}, tracker={tracker}, device={yolo_device}"
    )

    yolo = YOLO(yolo_model_path)
    names = yolo.names

    records: List[TrackRecord] = []
    results = None

    try:
        results = yolo.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=yolo_device,
            verbose=False,
        )

        for frame_idx, result in enumerate(tqdm(results, desc="YOLO tracking")):
            if max_frames is not None and frame_idx >= max_frames:
                break

            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            track_ids = boxes.id.detach().cpu().numpy().astype(int)

            for box, cls_id, score, tid in zip(xyxy, cls, confs, track_ids):
                class_name = names.get(int(cls_id), str(cls_id))
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                records.append(
                    TrackRecord(
                        frame_idx=int(frame_idx),
                        track_id=int(tid),
                        class_id=int(cls_id),
                        class_name=str(class_name),
                        conf=float(score),
                        xyxy=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                )

        summary = summarize_tracks(records, get_video_info(video_path))
        return records, summary

    finally:
        del yolo
        del results
        cleanup_memory()


def summarize_tracks(
    records: List[TrackRecord],
    video_info: Dict[str, Any],
) -> Dict[str, Any]:
    width = int(video_info["width"])
    height = int(video_info["height"])
    diag = math.sqrt(width * width + height * height)
    motion_thresh = 0.04 * diag

    class_det_counts = collections.Counter()
    tracks_by_id: Dict[int, List[TrackRecord]] = collections.defaultdict(list)

    for r in records:
        class_det_counts[r.class_name] += 1
        tracks_by_id[r.track_id].append(r)

    track_summaries = []
    unique_tracks_by_class = collections.Counter()

    for tid, items in tracks_by_id.items():
        items = sorted(items, key=lambda x: x.frame_idx)

        class_name = collections.Counter(
            [x.class_name for x in items]
        ).most_common(1)[0][0]

        unique_tracks_by_class[class_name] += 1

        first = items[0]
        last = items[-1]

        dx = last.center[0] - first.center[0]
        dy = last.center[1] - first.center[1]
        disp = math.sqrt(dx * dx + dy * dy)

        direction = short_direction(dx, dy, stationary_thresh=motion_thresh)
        avg_conf = float(np.mean([x.conf for x in items]))

        track_summaries.append(
            {
                "track_id": int(tid),
                "class_name": class_name,
                "frames_seen": int(len(items)),
                "first_frame": int(first.frame_idx),
                "last_frame": int(last.frame_idx),
                "avg_conf": round(avg_conf, 3),
                "start_center": [
                    round(first.center[0], 1),
                    round(first.center[1], 1),
                ],
                "end_center": [
                    round(last.center[0], 1),
                    round(last.center[1], 1),
                ],
                "displacement_px": round(disp, 2),
                "displacement_norm": round(disp / diag, 4) if diag > 0 else 0.0,
                "direction": direction,
            }
        )

    vehicle_tracks = [
        t for t in track_summaries
        if t["class_name"].lower() in VEHICLE_CLASS_NAMES
    ]

    moving_vehicle_tracks = [
        t for t in vehicle_tracks
        if t["direction"] != "stationary"
    ]

    person_tracks = [
        t for t in track_summaries
        if t["class_name"].lower() in PERSON_CLASS_NAMES
    ]

    traffic_light_tracks = [
        t for t in track_summaries
        if t["class_name"].lower() in TRAFFIC_LIGHT_CLASS_NAMES
    ]

    top_tracks = sorted(
        track_summaries,
        key=lambda x: (x["frames_seen"], x["displacement_px"]),
        reverse=True,
    )[:50]

    return {
        "video_info": video_info,
        "num_detections": int(len(records)),
        "class_detection_counts": dict(class_det_counts),
        "unique_tracks_by_class": dict(unique_tracks_by_class),
        "num_unique_tracks": int(len(tracks_by_id)),
        "num_vehicle_tracks": int(len(vehicle_tracks)),
        "num_moving_vehicle_tracks": int(len(moving_vehicle_tracks)),
        "num_person_tracks": int(len(person_tracks)),
        "num_traffic_light_tracks": int(len(traffic_light_tracks)),
        "top_tracks": top_tracks,
    }


# -----------------------------
# Simple CV checks
# -----------------------------

def estimate_camera_stability(
    video_path: str,
    max_pairs: int = 30,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"available": False, "reason": "could_not_open_video"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = math.sqrt(width * width + height * height)

    if frame_count < 3:
        cap.release()
        return {"available": False, "reason": "too_few_frames"}

    indices = np.linspace(
        0,
        frame_count - 1,
        num=min(max_pairs + 1, frame_count),
        dtype=int,
    )

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if width > 640:
            new_w = 640
            new_h = int(height * new_w / width)
            gray = cv2.resize(gray, (new_w, new_h))

        frames.append(gray)

    cap.release()

    if len(frames) < 2:
        return {"available": False, "reason": "too_few_decoded_frames"}

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    translations = []
    rotations_deg = []
    valid_pairs = 0

    for a, b in zip(frames[:-1], frames[1:]):
        kp1, des1 = orb.detectAndCompute(a, None)
        kp2, des2 = orb.detectAndCompute(b, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue

        matches = bf.match(des1, des2)
        if len(matches) < 10:
            continue

        matches = sorted(matches, key=lambda m: m.distance)[:100]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        M, _ = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if M is None:
            continue

        dx = float(M[0, 2])
        dy = float(M[1, 2])
        trans = math.sqrt(dx * dx + dy * dy)
        angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))

        translations.append(trans)
        rotations_deg.append(abs(angle))
        valid_pairs += 1

    if not translations:
        return {"available": False, "reason": "no_valid_feature_pairs"}

    avg_translation = float(np.mean(translations))
    avg_rotation = float(np.mean(rotations_deg))
    norm_translation = avg_translation / max(diag, 1.0)

    if norm_translation < 0.005 and avg_rotation < 0.5:
        stability_score = 1.0
    elif norm_translation < 0.02 and avg_rotation < 2.0:
        stability_score = 0.5
    else:
        stability_score = 0.0

    return {
        "available": True,
        "valid_pairs": int(valid_pairs),
        "avg_translation_px": round(avg_translation, 3),
        "avg_translation_norm": round(norm_translation, 5),
        "avg_rotation_deg": round(avg_rotation, 3),
        "camera_stability_score": stability_score,
    }


def analyze_traffic_light_colors(
    video_path: str,
    records: List[TrackRecord],
    max_samples: int = 200,
) -> Dict[str, Any]:
    tl_records = [
        r for r in records
        if r.class_name.lower() in TRAFFIC_LIGHT_CLASS_NAMES
    ]

    if not tl_records:
        return {
            "available": False,
            "reason": "no_traffic_light_detections",
        }

    if len(tl_records) > max_samples:
        idxs = np.linspace(0, len(tl_records) - 1, max_samples, dtype=int)
        tl_records = [tl_records[i] for i in idxs]

    by_frame: Dict[int, List[TrackRecord]] = collections.defaultdict(list)
    for r in tl_records:
        by_frame[r.frame_idx].append(r)

    needed_frames = set(by_frame.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "available": False,
            "reason": "could_not_open_video",
        }

    color_events = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in needed_frames:
            for r in by_frame[frame_idx]:
                x1, y1, x2, y2 = [int(round(v)) for v in r.xyxy]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                red1 = cv2.inRange(
                    hsv,
                    np.array([0, 60, 60]),
                    np.array([12, 255, 255]),
                )
                red2 = cv2.inRange(
                    hsv,
                    np.array([165, 60, 60]),
                    np.array([180, 255, 255]),
                )
                red_mask = cv2.bitwise_or(red1, red2)

                green_mask = cv2.inRange(
                    hsv,
                    np.array([35, 50, 50]),
                    np.array([95, 255, 255]),
                )

                area = max(1, crop.shape[0] * crop.shape[1])
                red_ratio = float(np.count_nonzero(red_mask) / area)
                green_ratio = float(np.count_nonzero(green_mask) / area)

                color = "unknown"

                if red_ratio > 0.02 and red_ratio > green_ratio * 1.5:
                    color = "red"
                elif green_ratio > 0.02 and green_ratio > red_ratio * 1.5:
                    color = "green"

                if color != "unknown":
                    color_events.append(
                        {
                            "frame_idx": int(frame_idx),
                            "track_id": int(r.track_id),
                            "color": color,
                            "red_ratio": round(red_ratio, 4),
                            "green_ratio": round(green_ratio, 4),
                        }
                    )

        frame_idx += 1

    cap.release()

    red_frames = [e["frame_idx"] for e in color_events if e["color"] == "red"]
    green_frames = [e["frame_idx"] for e in color_events if e["color"] == "green"]

    red_to_green = False
    if red_frames and green_frames:
        red_to_green = min(red_frames) < max(green_frames)

    return {
        "available": True,
        "num_color_events": int(len(color_events)),
        "observed_red": bool(red_frames),
        "observed_green": bool(green_frames),
        "red_to_green_possible": bool(red_to_green),
        "first_red_frame": min(red_frames) if red_frames else None,
        "first_green_frame": min(green_frames) if green_frames else None,
        "sample_events": color_events[:30],
    }


# -----------------------------
# Qwen video evaluation
# -----------------------------

def compact_tracking_context(
    tracking_summary: Dict[str, Any],
    camera_motion: Dict[str, Any],
    traffic_light_colors: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "video_info": tracking_summary.get("video_info", {}),
        "class_detection_counts": tracking_summary.get("class_detection_counts", {}),
        "unique_tracks_by_class": tracking_summary.get("unique_tracks_by_class", {}),
        "num_vehicle_tracks": tracking_summary.get("num_vehicle_tracks", 0),
        "num_moving_vehicle_tracks": tracking_summary.get("num_moving_vehicle_tracks", 0),
        "num_person_tracks": tracking_summary.get("num_person_tracks", 0),
        "num_traffic_light_tracks": tracking_summary.get("num_traffic_light_tracks", 0),
        "top_tracks": tracking_summary.get("top_tracks", [])[:30],
        "camera_motion_estimate": camera_motion,
        "traffic_light_color_estimate": traffic_light_colors,
    }


def build_video_eval_prompt(
    caption: str,
    claims: List[Dict[str, Any]],
    tracking_context: Dict[str, Any],
) -> str:
    return f"""
You are evaluating whether a generated ego-vehicle-view driving video followed its input caption.

Original caption:
{caption}

Atomic claims to evaluate:
{json.dumps(claims, indent=2)}

Additional YOLO/tracking/CV summary:
{json.dumps(tracking_context, indent=2)}

Instructions:
1. Watch the video carefully.
2. For every claim, assign:
   - qwen_score: 1.0 if clearly followed
   - qwen_score: 0.5 if partially followed or ambiguous
   - qwen_score: 0.0 if not followed
3. Provide short visual evidence.
4. Provide a short failure_reason if qwen_score < 1.0.
5. Use the YOLO/tracking summary only as supporting evidence.
6. Trust the video if it conflicts with YOLO.
7. Be strict for temporal/action claims:
   - Object presence is not enough for an action claim.
   - Example: "traffic light changes from red to green" requires a visible temporal change.
8. Be careful with subjective claims such as "calm and quiet"; score them only if visually inferable.

Return valid JSON only with this schema:

{{
  "claim_results": [
    {{
      "claim_id": "C1",
      "qwen_score": 1.0,
      "evidence": "The video shows a forward-facing road view from a vehicle-like perspective.",
      "failure_reason": ""
    }}
  ],
  "overall_observation": "Short summary of which caption parts are most and least followed."
}}
""".strip()


def evaluate_claims_with_qwen_video(
    qwen: QwenRunner,
    video_path: str,
    caption: str,
    claims: List[Dict[str, Any]],
    tracking_context: Dict[str, Any],
    sample_fps: Optional[float],
    num_frames: Optional[int],
    resize_long_side: Optional[int],
    resize_divisor: int,
    max_pixels: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt = build_video_eval_prompt(
        caption=caption,
        claims=claims,
        tracking_context=tracking_context,
    )

    raw = qwen.generate_video(
        video_path=video_path,
        prompt=prompt,
        sample_fps=sample_fps,
        num_frames=num_frames,
        resize_long_side=resize_long_side,
        resize_divisor=resize_divisor,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
    )

    try:
        parsed = safe_json_loads(raw)
    except Exception as e:
        logging.warning(f"Qwen video eval JSON parse failed. Error: {e}")
        parsed = {
            "claim_results": [],
            "overall_observation": raw[:2000],
        }

    if "claim_results" not in parsed or not isinstance(parsed["claim_results"], list):
        parsed["claim_results"] = []

    return parsed


# -----------------------------
# Scoring
# -----------------------------

def yolo_score_for_claim(
    claim: Dict[str, Any],
    tracking_summary: Dict[str, Any],
    camera_motion: Dict[str, Any],
    traffic_light_colors: Dict[str, Any],
) -> Optional[float]:
    text = (claim.get("claim", "") + " " + claim.get("question", "")).lower()

    num_vehicle_tracks = int(tracking_summary.get("num_vehicle_tracks", 0))
    num_moving_vehicle_tracks = int(tracking_summary.get("num_moving_vehicle_tracks", 0))
    num_person_tracks = int(tracking_summary.get("num_person_tracks", 0))
    num_traffic_light_tracks = int(tracking_summary.get("num_traffic_light_tracks", 0))

    if any(w in text for w in ["car", "cars", "vehicle", "vehicles", "truck", "bus"]):
        action_words = [
            "pass",
            "passes",
            "move",
            "moves",
            "moving",
            "drives",
            "drive",
            "through",
            "cross",
            "travel",
            "traveling",
        ]

        if any(w in text for w in action_words):
            if num_moving_vehicle_tracks >= 2:
                return 1.0
            if num_moving_vehicle_tracks == 1:
                return 0.5
            return 0.0

        if num_vehicle_tracks >= 1:
            return 1.0
        return 0.0

    if any(w in text for w in ["pedestrian", "person", "people"]):
        if num_person_tracks >= 1:
            return 1.0
        return 0.0

    if "traffic light" in text or "traffic lights" in text:
        if any(w in text for w in ["red to green", "change", "changes", "turns green"]):
            if traffic_light_colors.get("available") and traffic_light_colors.get("red_to_green_possible"):
                return 1.0
            if num_traffic_light_tracks >= 1:
                return 0.5
            return 0.0

        if num_traffic_light_tracks >= 1:
            return 1.0
        return 0.0

    if any(w in text for w in ["stationary", "stable", "consistent view", "camera remains"]):
        if camera_motion.get("available"):
            return normalize_score(camera_motion.get("camera_stability_score", 0.5))
        return None

    return None


def fuse_scores(
    qwen_score: float,
    yolo_score: Optional[float],
    preferred_verifier: str,
    fuse_yolo: bool = True,
) -> float:
    if not fuse_yolo or yolo_score is None:
        return qwen_score

    if preferred_verifier == "yolo":
        return round(0.7 * yolo_score + 0.3 * qwen_score, 3)

    if preferred_verifier == "both":
        return round(0.5 * yolo_score + 0.5 * qwen_score, 3)

    return round(0.8 * qwen_score + 0.2 * yolo_score, 3)


def combine_claim_results(
    claims: List[Dict[str, Any]],
    qwen_eval: Dict[str, Any],
    tracking_summary: Dict[str, Any],
    camera_motion: Dict[str, Any],
    traffic_light_colors: Dict[str, Any],
    fuse_yolo: bool,
) -> List[Dict[str, Any]]:
    qwen_by_id = {}

    for r in qwen_eval.get("claim_results", []):
        cid = str(r.get("claim_id", "")).strip()
        if cid:
            qwen_by_id[cid] = r

    final_results = []

    for c in claims:
        cid = c["claim_id"]
        q = qwen_by_id.get(cid, {})

        qwen_score = normalize_score(q.get("qwen_score", 0.5))

        yolo_score = yolo_score_for_claim(
            claim=c,
            tracking_summary=tracking_summary,
            camera_motion=camera_motion,
            traffic_light_colors=traffic_light_colors,
        )

        final_score = fuse_scores(
            qwen_score=qwen_score,
            yolo_score=yolo_score,
            preferred_verifier=c.get("preferred_verifier", "qwen_vl"),
            fuse_yolo=fuse_yolo,
        )

        final_results.append(
            {
                **c,
                "qwen_score": qwen_score,
                "yolo_or_cv_score": yolo_score,
                "final_score": final_score,
                "usage_label": score_to_label(final_score),
                "evidence": str(q.get("evidence", "")).strip(),
                "failure_reason": str(q.get("failure_reason", "")).strip(),
            }
        )

    return final_results


# -----------------------------
# Reports
# -----------------------------

def compute_report(
    caption: str,
    video_info: Dict[str, Any],
    claim_results: List[Dict[str, Any]],
    tracking_summary: Dict[str, Any],
    camera_motion: Dict[str, Any],
    traffic_light_colors: Dict[str, Any],
    qwen_overall_observation: str,
    sampling_config: Dict[str, Any],
) -> Dict[str, Any]:
    scores = [float(r["final_score"]) for r in claim_results]

    weighted_scores = [
        float(r["final_score"]) * float(r.get("importance", 3))
        for r in claim_results
    ]

    weights = [float(r.get("importance", 3)) for r in claim_results]

    category_scores = {}

    for cat in sorted(VALID_CATEGORIES):
        vals = [
            float(r["final_score"])
            for r in claim_results
            if r["category"] == cat
        ]
        category_scores[cat] = mean_or_none(vals)

    most_followed = sorted(
        [r for r in claim_results if r["final_score"] >= 0.75],
        key=lambda x: x["final_score"],
        reverse=True,
    )

    partially_followed = sorted(
        [r for r in claim_results if 0.25 < r["final_score"] < 0.75],
        key=lambda x: x["final_score"],
        reverse=True,
    )

    least_followed = sorted(
        [r for r in claim_results if r["final_score"] <= 0.25],
        key=lambda x: x["final_score"],
    )

    return {
        "caption": caption,
        "video_info": video_info,
        "sampling_config": sampling_config,
        "caption_usage_score": round(100.0 * mean_or_none(scores), 2) if scores else None,
        "weighted_caption_usage_score": round(
            100.0 * sum(weighted_scores) / max(1.0, sum(weights)),
            2,
        ) if scores else None,
        "category_scores": {
            k: round(100.0 * v, 2) if v is not None else None
            for k, v in category_scores.items()
        },
        "num_claims": int(len(claim_results)),
        "claim_results": claim_results,
        "most_followed_claims": most_followed,
        "partially_followed_claims": partially_followed,
        "least_followed_claims": least_followed,
        "tracking_summary": tracking_summary,
        "camera_motion": camera_motion,
        "traffic_light_colors": traffic_light_colors,
        "qwen_overall_observation": qwen_overall_observation,
    }


def save_json(report: Dict[str, Any], outpath: Path) -> None:
    outpath.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_markdown(report: Dict[str, Any], outpath: Path) -> None:
    lines = []

    lines.append("# Caption Usage Evaluation Report\n")

    lines.append("## Overall Scores\n")
    lines.append(f"- Caption Usage Score: **{report['caption_usage_score']} / 100**")
    lines.append(f"- Weighted Caption Usage Score: **{report['weighted_caption_usage_score']} / 100**")
    lines.append(f"- Number of claims: **{report['num_claims']}**\n")

    lines.append("## Sampling Config\n")
    for k, v in report["sampling_config"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Category Scores\n")
    for cat, score in report["category_scores"].items():
        if score is not None:
            lines.append(f"- {cat}: **{score} / 100**")
    lines.append("")

    lines.append("## Qwen Overall Observation\n")
    lines.append(str(report.get("qwen_overall_observation", "")).strip() + "\n")

    lines.append("## Claim-Level Results\n")
    lines.append(
        "| ID | Category | Claim | Qwen | YOLO/CV | Final | Label | Evidence | Failure Reason |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---|---|")

    for r in report["claim_results"]:
        yolo_score = r["yolo_or_cv_score"]
        yolo_str = "" if yolo_score is None else str(yolo_score)

        lines.append(
            "| {cid} | {cat} | {claim} | {qwen} | {yolo} | {final} | {label} | {evidence} | {failure} |".format(
                cid=str(r["claim_id"]).replace("|", "\\|"),
                cat=str(r["category"]).replace("|", "\\|"),
                claim=str(r["claim"]).replace("|", "\\|"),
                qwen=r["qwen_score"],
                yolo=yolo_str,
                final=r["final_score"],
                label=str(r["usage_label"]).replace("|", "\\|"),
                evidence=str(r.get("evidence", "")).replace("|", "\\|"),
                failure=str(r.get("failure_reason", "")).replace("|", "\\|"),
            )
        )

    lines.append("\n## Most Followed Claims\n")
    if report["most_followed_claims"]:
        for r in report["most_followed_claims"]:
            lines.append(f"- **{r['claim_id']}**: {r['claim']}")
    else:
        lines.append("- None")

    lines.append("\n## Partially Followed Claims\n")
    if report["partially_followed_claims"]:
        for r in report["partially_followed_claims"]:
            lines.append(f"- **{r['claim_id']}**: {r['claim']}")
    else:
        lines.append("- None")

    lines.append("\n## Least Followed Claims\n")
    if report["least_followed_claims"]:
        for r in report["least_followed_claims"]:
            lines.append(f"- **{r['claim_id']}**: {r['claim']}")
    else:
        lines.append("- None")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def save_html_heatmap(report: Dict[str, Any], outpath: Path) -> None:
    rows = []

    for r in report["claim_results"]:
        score = float(r["final_score"])
        color = score_to_color(score)

        rows.append(
            f"""
            <tr style="background-color:{color};">
                <td>{html.escape(str(r["claim_id"]))}</td>
                <td>{html.escape(str(r["category"]))}</td>
                <td>{html.escape(str(r["claim"]))}</td>
                <td>{score:.2f}</td>
                <td>{html.escape(str(r["usage_label"]))}</td>
                <td>{html.escape(str(r.get("evidence", "")))}</td>
                <td>{html.escape(str(r.get("failure_reason", "")))}</td>
            </tr>
            """
        )

    category_items = "".join(
        f"<li>{html.escape(str(k))}: {html.escape(str(v))} / 100</li>"
        for k, v in report["category_scores"].items()
        if v is not None
    )

    sampling_items = "".join(
        f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>"
        for k, v in report["sampling_config"].items()
    )

    doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Caption Usage Heatmap</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 24px;
    line-height: 1.4;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
th, td {{
    border: 1px solid #999;
    padding: 8px;
    vertical-align: top;
}}
th {{
    background: #eee;
}}
.score {{
    font-size: 24px;
    font-weight: bold;
}}
.caption {{
    background: #f7f7f7;
    padding: 12px;
    border-left: 4px solid #444;
}}
.small {{
    color: #555;
}}
</style>
</head>
<body>

<h1>Caption Usage Heatmap</h1>

<p class="score">Caption Usage Score: {report["caption_usage_score"]} / 100</p>
<p class="score">Weighted Caption Usage Score: {report["weighted_caption_usage_score"]} / 100</p>

<h2>Original Caption</h2>
<div class="caption">{html.escape(report["caption"])}</div>

<h2>Sampling Config</h2>
<ul>
{sampling_items}
</ul>

<h2>Category Scores</h2>
<ul>
{category_items}
</ul>

<h2>Qwen Overall Observation</h2>
<p>{html.escape(str(report.get("qwen_overall_observation", "")))}</p>

<h2>Claim Heatmap</h2>
<p class="small">Green = followed, Yellow = partially followed, Red = ignored.</p>

<table>
<tr>
    <th>ID</th>
    <th>Category</th>
    <th>Claim</th>
    <th>Final Score</th>
    <th>Label</th>
    <th>Evidence</th>
    <th>Failure Reason</th>
</tr>
{''.join(rows)}
</table>

</body>
</html>
"""
    outpath.write_text(doc, encoding="utf-8")


# -----------------------------
# Per-pair processing
# -----------------------------

def process_one_pair(
    pair: VideoCaptionPair,
    qwen: QwenRunner,
    args: argparse.Namespace,
    root_outdir: Path,
    resize_long_side: Optional[int],
    sampling_config: Dict[str, Any],
) -> Dict[str, Any]:
    pair_outdir = root_outdir / pair.pair_id
    pair_outdir.mkdir(parents=True, exist_ok=True)

    video_path = pair.video_path
    caption = pair.caption

    logging.info("=" * 80)
    logging.info(f"Processing pair: {pair.pair_id}")
    logging.info(f"Video: {video_path}")
    logging.info(f"Output: {pair_outdir}")
    logging.info("=" * 80)
    logging.info(f"Caption: {caption}")

    video_info = get_video_info(video_path)

    (pair_outdir / "caption.txt").write_text(caption, encoding="utf-8")

    if args.claims_json:
        logging.info(f"Loading claims from {args.claims_json}")
        claims_data = json.loads(Path(args.claims_json).read_text(encoding="utf-8"))
        claims = claims_data.get(
            "claims",
            claims_data if isinstance(claims_data, list) else [],
        )
    else:
        logging.info("Extracting atomic caption claims with Qwen3-VL text-only mode.")

        claims = extract_claims(
            qwen=qwen,
            caption=caption,
            max_new_tokens=args.claim_max_new_tokens,
        )

        print_extracted_claims(claims)

    claims_out = pair_outdir / "atomic_claims.json"
    claims_out.write_text(
        json.dumps({"claims": claims}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cleanup_memory()

    records, tracking_summary = run_yolo_tracking(
        video_path=video_path,
        yolo_model_path=args.yolo_model,
        tracker=args.tracker,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        yolo_device=args.yolo_device,
        max_frames=args.max_yolo_frames,
    )

    tracking_out = pair_outdir / "yolo_tracking_summary.json"
    tracking_out.write_text(
        json.dumps(tracking_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cleanup_memory()

    logging.info("Estimating camera stability.")
    camera_motion = estimate_camera_stability(video_path)

    logging.info("Estimating traffic-light color changes from YOLO traffic-light crops.")
    traffic_light_colors = analyze_traffic_light_colors(video_path, records)

    cv_out = pair_outdir / "simple_cv_checks.json"
    cv_out.write_text(
        json.dumps(
            {
                "camera_motion": camera_motion,
                "traffic_light_colors": traffic_light_colors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cleanup_memory()

    tracking_context = compact_tracking_context(
        tracking_summary=tracking_summary,
        camera_motion=camera_motion,
        traffic_light_colors=traffic_light_colors,
    )

    logging.info("Evaluating claims with Qwen3-VL over sampled/resized video frames.")

    qwen_eval = evaluate_claims_with_qwen_video(
        qwen=qwen,
        video_path=video_path,
        caption=caption,
        claims=claims,
        tracking_context=tracking_context,
        sample_fps=args.sample_fps,
        num_frames=args.effective_num_frames,
        resize_long_side=resize_long_side,
        resize_divisor=args.resize_divisor,
        max_pixels=args.max_pixels,
        max_new_tokens=args.eval_max_new_tokens,
    )

    qwen_eval_out = pair_outdir / "qwen_video_claim_eval_raw.json"
    qwen_eval_out.write_text(
        json.dumps(qwen_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cleanup_memory()

    claim_results = combine_claim_results(
        claims=claims,
        qwen_eval=qwen_eval,
        tracking_summary=tracking_summary,
        camera_motion=camera_motion,
        traffic_light_colors=traffic_light_colors,
        fuse_yolo=not args.no_fuse_yolo,
    )

    report = compute_report(
        caption=caption,
        video_info=video_info,
        claim_results=claim_results,
        tracking_summary=tracking_summary,
        camera_motion=camera_motion,
        traffic_light_colors=traffic_light_colors,
        qwen_overall_observation=str(qwen_eval.get("overall_observation", "")),
        sampling_config=sampling_config,
    )

    report["pair_id"] = pair.pair_id

    json_report_path = pair_outdir / "caption_usage_report.json"
    md_report_path = pair_outdir / "caption_usage_report.md"
    html_report_path = pair_outdir / "caption_usage_heatmap.html"

    save_json(report, json_report_path)
    save_markdown(report, md_report_path)
    save_html_heatmap(report, html_report_path)

    logging.info(f"Saved JSON report: {json_report_path}")
    logging.info(f"Saved Markdown report: {md_report_path}")
    logging.info(f"Saved HTML heatmap: {html_report_path}")

    cleanup_memory()

    return {
        "pair_id": pair.pair_id,
        "video_path": video_path,
        "output_dir": str(pair_outdir),
        "caption_usage_score": report["caption_usage_score"],
        "weighted_caption_usage_score": report["weighted_caption_usage_score"],
        "category_scores": report["category_scores"],
        "num_claims": report["num_claims"],
        "status": "success",
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate which parts of a driving-video caption were followed "
            "by one or more generated videos."
        )
    )

    parser.add_argument("--video", default=None, help="Path to a single generated video.")
    parser.add_argument("--caption", default=None, help="Caption string for single-video mode.")
    parser.add_argument("--caption_file", default=None, help="Caption .txt file for single-video mode.")

    parser.add_argument("--video_dir", default=None, help="Folder containing generated videos.")
    parser.add_argument("--caption_dir", default=None, help="Folder containing caption .txt files.")
    parser.add_argument("--caption_json", default=None, help="JSON mapping videos to captions.")
    parser.add_argument(
        "--caption_ext",
        default=".txt",
        help="Caption file extension for --caption_dir mode.",
    )
    parser.add_argument(
        "--video_exts",
        default=".mp4,.avi,.mov,.mkv,.webm",
        help="Comma-separated video extensions for --video_dir mode.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search --video_dir and mirror relative caption paths under --caption_dir.",
    )

    parser.add_argument("--outdir", default="caption_usage_outputs", help="Output directory.")

    parser.add_argument("--model_id", default=DEFAULT_QWEN_MODEL_ID, help="Qwen model ID.")
    parser.add_argument("--device_map", default="auto", help="Transformers device_map.")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        help="auto, float16, bfloat16, float32, or none.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="none",
        help="Optional: flash_attention_2, sdpa, eager, or none.",
    )

    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Uniformly sample this many frames for Qwen3-VL. Recommended: 8 or 16.",
    )
    sampling.add_argument(
        "--sample_fps",
        type=float,
        default=None,
        help="FPS-based sampling for Qwen3-VL. Mutually exclusive with --num_frames.",
    )

    parser.add_argument(
        "--resize_long_side",
        type=int,
        default=448,
        help=(
            "Resize video frames before Qwen3-VL so the longer side is approximately this value. "
            "Aspect ratio is preserved. Set -1 to disable and use --max_pixels."
        ),
    )
    parser.add_argument(
        "--resize_divisor",
        type=int,
        default=28,
        help="Round resized height/width to this divisor.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=360 * 640,
        help="Used only when --resize_long_side is disabled.",
    )

    parser.add_argument("--claim_max_new_tokens", type=int, default=2048)
    parser.add_argument("--eval_max_new_tokens", type=int, default=4096)

    parser.add_argument("--yolo_model", default="yolov10n.pt", help="YOLO model path/name.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML.")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.7)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--max_yolo_frames", type=int, default=None)
    parser.add_argument(
        "--yolo_device",
        default="cpu",
        help=(
            "Device for YOLO tracking. Use 'cpu' to avoid competing with Qwen GPU memory. "
            "Use '0' or 'cuda:0' only if you have enough VRAM."
        ),
    )

    parser.add_argument(
        "--no_fuse_yolo",
        action="store_true",
        help="If set, final score uses only Qwen-VL claim scores.",
    )

    parser.add_argument(
        "--claims_json",
        default=None,
        help=(
            "Optional path to precomputed claims JSON. "
            "Recommended only for single-video mode."
        ),
    )

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop batch processing if one video fails.",
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    root_outdir = ensure_dir(args.outdir)

    if args.sample_fps is None and args.num_frames is None:
        args.effective_num_frames = 8
    else:
        args.effective_num_frames = args.num_frames

    if args.video_dir and args.claims_json:
        logging.warning(
            "--claims_json was provided in folder mode. "
            "The same claims will be reused for every video. "
            "Usually this is not recommended unless all captions share the same claim set."
        )

    resize_long_side = (
        None
        if args.resize_long_side is not None and args.resize_long_side < 0
        else args.resize_long_side
    )

    sampling_config = {
        "num_frames": args.effective_num_frames,
        "sample_fps": args.sample_fps,
        "resize_long_side": resize_long_side,
        "resize_divisor": args.resize_divisor,
        "max_pixels": args.max_pixels,
    }

    logging.info("Finding video-caption pairs.")
    pairs = find_video_caption_pairs(args)

    dataset_index = [
        {
            "pair_id": p.pair_id,
            "video_path": p.video_path,
            "caption_preview": p.caption[:200],
        }
        for p in pairs
    ]

    dataset_index_path = root_outdir / "dataset_index.json"
    dataset_index_path.write_text(
        json.dumps(dataset_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info(f"Found {len(pairs)} valid video-caption pairs.")
    logging.info(f"Saved dataset index: {dataset_index_path}")

    logging.info("Sampling config:")
    logging.info(json.dumps(sampling_config, indent=2))

    logging.info("Loading Qwen model once for all pairs.")
    qwen = QwenRunner(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    batch_summary = []
    failed = []

    for pair in tqdm(pairs, desc="Processing video-caption pairs"):
        try:
            summary = process_one_pair(
                pair=pair,
                qwen=qwen,
                args=args,
                root_outdir=root_outdir,
                resize_long_side=resize_long_side,
                sampling_config=sampling_config,
            )

            batch_summary.append(summary)

            print("\n==============================")
            print(f"Done: {pair.pair_id}")
            print("==============================")
            print(f"CUS: {summary['caption_usage_score']} / 100")
            print(f"Weighted CUS: {summary['weighted_caption_usage_score']} / 100")
            print(f"Output: {summary['output_dir']}")

        except Exception as e:
            logging.exception(f"Failed on pair: {pair.pair_id}")

            failure_record = {
                "pair_id": pair.pair_id,
                "video_path": pair.video_path,
                "error": str(e),
                "status": "failed",
            }

            failed.append(failure_record)
            batch_summary.append(failure_record)

            if args.stop_on_error:
                raise

        finally:
            cleanup_memory()

    batch_summary_path = root_outdir / "batch_summary.json"
    batch_summary_path.write_text(
        json.dumps(batch_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    batch_md_path = root_outdir / "batch_summary.md"

    lines = []
    lines.append("# Batch Caption Usage Summary\n")
    lines.append("| Pair ID | Status | CUS | Weighted CUS | Output |")
    lines.append("|---|---|---:|---:|---|")

    for item in batch_summary:
        lines.append(
            "| {pair_id} | {status} | {cus} | {wcus} | {out} |".format(
                pair_id=item.get("pair_id", ""),
                status=item.get("status", ""),
                cus=item.get("caption_usage_score", ""),
                wcus=item.get("weighted_caption_usage_score", ""),
                out=item.get("output_dir", ""),
            )
        )

    batch_md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n==============================")
    print("Batch Evaluation Complete")
    print("==============================")
    print(f"Total pairs: {len(pairs)}")
    print(f"Successful: {sum(1 for x in batch_summary if x.get('status') == 'success')}")
    print(f"Failed: {len(failed)}")
    print(f"Batch summary JSON: {batch_summary_path}")
    print(f"Batch summary Markdown: {batch_md_path}")
    print(f"All outputs saved under: {root_outdir}")


if __name__ == "__main__":
    main()