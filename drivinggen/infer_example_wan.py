"""
infer_example_wan.py
====================
Video generation inference script for the DrivingGen benchmark using the
Wan2.2-I2V-A14B model (image-to-video diffusion pipeline).

Given a JSON metadata file that lists scene IDs with associated conditioning
images and text captions, this script:
    1. Loads each scene's first frame and caption.
    2. Runs the Wan 2.2 image-to-video pipeline at 576 x 1024 resolution for
       101 frames with 40 denoising steps.
    3. Saves the generated video as MP4 and individual PNG frames.

A cuSolver fallback patch is applied so that torch.linalg.solve automatically
retries on CPU when the GPU solver encounters internal errors.
"""

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import json
import argparse
import os

import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from PIL import Image
from einops import rearrange, repeat
import numpy as np
import imageio


# ---------------------------------------------------------------------------
# Video / image I/O utilities
# ---------------------------------------------------------------------------

def save_img_seq_to_video(out_path: str, img_seq: np.ndarray, fps: int):
    """Write a sequence of RGB images to an MP4 video file using imageio.

    Args:
        out_path (str): Absolute path to the output MP4 file.
        img_seq (np.ndarray): Sequence of frames to write.
            shape: (T, H, W, 3) — uint8, values in [0, 255].
        fps (int): Frames per second for the output video.
    """
    # Open an imageio writer in MP4 format with the requested frame rate.
    writer = imageio.get_writer(out_path, fps=fps)

    # Append each frame sequentially; imageio handles codec encoding.
    for img in img_seq:
        # img shape: (H, W, 3) — uint8
        writer.append_data(img)

    # Finalise and flush the video container to disk.
    writer.close()


def perform_save_locally(save_path: str, samples: np.ndarray, mode: str):
    """Save generated frames to disk as individual PNG images or as an MP4 video.

    Args:
        save_path (str): Root directory where outputs are written.  Sub-folders
            ``images/`` or ``videos/`` are created automatically as needed.
        samples (np.ndarray): Generated frame sequence in floating-point format.
            shape: (T, H, W, 3) — float, values in [0.0, 1.0].
        mode (str): Output format — one of ``"images"`` or ``"videos"``.
            ``"grids"`` is declared but not implemented.

    Raises:
        NotImplementedError: If ``mode`` is ``"grids"`` (not yet supported).
    """
    assert mode in ["images", "grids", "videos"], \
        f"mode must be 'images', 'grids', or 'videos', got '{mode}'"

    # Create the per-mode sub-directory for images and grids.
    # Videos are saved directly inside save_path without a sub-folder.
    if mode != 'videos':
        merged_path = os.path.join(save_path, mode)
        os.makedirs(merged_path, exist_ok=True)

    if mode == "images":
        # --- Save each frame as a numbered PNG file. ---
        frame_count = 0
        for sample in samples:
            # sample shape: (H, W, 3) — float in [0, 1]

            # Scale from [0, 1] float to [0, 255] uint8 for image saving.
            sample = sample * 255.0
            # sample shape: (H, W, 3) — float in [0, 255]

            image_save_path = os.path.join(merged_path, f"{frame_count:05}.png")

            # Convert to uint8 and save via PIL.
            Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
            frame_count += 1

    elif mode == "videos":
        # --- Save the entire sequence as a single MP4 video. ---
        # Scale from [0, 1] float to [0, 255] uint8.
        img_seq = samples * 255.0
        # img_seq shape: (T, H, W, 3) — float in [0, 255]

        video_save_path = os.path.join(save_path, "video.mp4")

        # Write at 10 fps to match the benchmark's standard frame rate.
        save_img_seq_to_video(video_save_path, img_seq.astype(np.uint8), fps=10)

    else:
        # "grids" mode is declared but not implemented.
        raise NotImplementedError(f"mode '{mode}' is not implemented.")


def deal_img(img_path: str) -> Image.Image:
    """Load and resize a conditioning image to fit within the 576 x 1024 target area.

    The image is resized so that:
        1. Its total pixel area does not exceed ``max_area = 576 * 1024``.
        2. Its aspect ratio is preserved.
        3. Both dimensions are integer multiples of
           ``pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]``
           (the model's spatial quantisation factor), so that the VAE encoder
           and transformer patch embedding can tile the image without padding.

    Resize formula:
        height = round(sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width  = round(sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    where aspect_ratio = image.height / image.width.

    Args:
        img_path (str): Path to the input image file.

    Returns:
        image (PIL.Image.Image): Resized conditioning image.
            Dimensions are multiples of mod_value and fit within 576 x 1024.

    Notes:
        This function references the global variable ``pipe`` (the loaded
        WanImageToVideoPipeline instance) to determine ``mod_value``.
    """
    # Load the image from disk.  load_image handles both local paths and URLs.
    image = load_image(img_path)

    # Maximum allowed pixel area (576 rows * 1024 columns).
    max_area = 576 * 1024

    # Preserve the original aspect ratio (height / width).
    aspect_ratio = image.height / image.width

    # Compute the spatial quantisation modulus from the pipeline's VAE and
    # transformer configuration.  Both height and width must be divisible by
    # this value for the model to run without errors.
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    # Compute target height: solve for h in h * w = max_area with h/w = aspect_ratio,
    # giving h = sqrt(max_area * aspect_ratio), then floor to mod_value multiple.
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value

    # Compute target width: w = sqrt(max_area / aspect_ratio), floor to mod_value.
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    # Resize the image using PIL's default resampling filter.
    image = image.resize((width, height))

    return image


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # cuSolver fallback patch
    # -----------------------------------------------------------------------
    # On some GPU/driver combinations torch.linalg.solve raises a
    # CUSOLVER_STATUS_INTERNAL_ERROR for small matrices.  This monkey-patch
    # retries on the CPU and moves the result back to GPU/original dtype,
    # which is sufficient for the small linear systems encountered in the
    # attention layers of the transformer.

    # Save a reference to the original GPU solve function.
    _orig_solve = torch.linalg.solve

    def solve_with_cpu_fallback(A, B):
        """Wrapper around torch.linalg.solve that falls back to CPU on cuSolver errors.

        Args:
            A (torch.Tensor): Coefficient matrix.
                shape: (..., N, N).
            B (torch.Tensor): Right-hand side matrix or vector.
                shape: (..., N, K) or (..., N).

        Returns:
            X (torch.Tensor): Solution tensor on the same device and dtype as A.
                shape: (..., N, K) or (..., N).
        """
        try:
            # Attempt GPU solve first (fast path).
            return _orig_solve(A, B)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusolver" in msg or "cusolver_status_internal_error" in msg:
                # cuSolver internal error encountered; fall back to CPU.
                # Cast to float32 (CPU solver requires at least float32) and
                # detach from the computation graph before moving to CPU.
                X = _orig_solve(
                    A.detach().float().cpu(),
                    B.detach().float().cpu()
                )
                # Restore to original device and dtype so downstream ops work normally.
                return X.to(A.device, dtype=A.dtype)
            # Re-raise any other RuntimeError unchanged.
            raise

    # Install the patched solver globally before loading the pipeline.
    torch.linalg.solve = solve_with_cpu_fallback

    # -----------------------------------------------------------------------
    # Argument parsing
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Generate a video from a conditioning image and text prompt using Wan2.2."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a JSON metadata file (with .json extension) or a single image file "
             "to use as the conditioning frame for image-to-video generation."
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        default=None,
        help="Root directory where generated videos and frames will be saved."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wan2.2-14b",
        help="Model identifier used as a sub-directory name in the output path. "
             "Default: 'wan2.2-14b'."
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="default_prompt",
        help="Experiment identifier used as a sub-directory name in the output path. "
             "Default: 'default_prompt'."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="ego_condition",
        help="Dataset split name used as a sub-directory in the output path. "
             "One of 'ego_condition' or 'open_domain'. Default: 'ego_condition'."
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    # Hugging Face model identifier for the Wan2.2 image-to-video pipeline.
    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    # Use bfloat16 for lower memory footprint while preserving dynamic range.
    dtype = torch.bfloat16

    # Target compute device.
    device = "cuda"

    # Load the full diffusion pipeline from the Hugging Face Hub in bfloat16.
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)  # move all model weights to GPU

    # -----------------------------------------------------------------------
    # Scene iteration and video generation
    # -----------------------------------------------------------------------

    # Derive the base directory from the JSON path (strip the '.json' suffix).
    base_path = args.image[:-5]  # remove ".json" — 5 characters

    if '.json' in args.image:
        # Load the JSON metadata file which is a list of scene ID strings.
        with open(args.image, 'r') as f:
            data = json.load(f)

        prompts = []  # list of dicts with 'prompt' and 'visual_input' keys
        runs    = []  # corresponding scene ID strings

        # Iterate over scenes ([:2] slice is a quick debug limiter — remove for full runs).
        for d in data[:2]:
            # Construct the path to the conditioning first frame (JPEG preferred).
            first_img = os.path.join(base_path, 'imgs', d + '.jpg')

            if not os.path.exists(first_img):
                # Attempt PNG fallback if JPEG is not found.
                first_img[-4:] = ".png"
                if not os.path.exists(first_img):
                    print(f'first image not found. {first_img}')

            # Load the text caption for this scene from a corresponding .txt file.
            prompt_path = os.path.join(base_path, 'caption', d + '.txt')

            with open(prompt_path) as f:
                prompt = f.read()  # plain text caption string

            prompts.append({
                'prompt':       prompt,
                'visual_input': first_img
            })
            runs.append(d)  # retain the scene ID for output path construction

        # -----------------------------------------------------------------------
        # Generate one video per scene.
        # -----------------------------------------------------------------------
        for idx, prompt in enumerate(prompts):
            image_path = prompt['visual_input']   # path to conditioning frame
            prompt     = prompt['prompt']          # text prompt string

            # Resize the conditioning image to the model's expected resolution.
            image = deal_img(image_path)

            # Negative prompt (translated from Chinese):
            # Discourages: vivid colors, overexposure, static scenes, blurry details,
            # subtitles, artistic styles, still frames, gray tone, low quality,
            # JPEG artifacts, ugly/deformed limbs, fused fingers, messy backgrounds,
            # three legs, crowds, walking backwards.
            negative_prompt = (
                "vivid colors, overexposed, static, blurry details, subtitles, "
                "stylized, artwork, painting, still, overall gray, worst quality, "
                "low quality, JPEG compression artifacts, ugly, incomplete, "
                "extra fingers, poorly drawn hands, poorly drawn face, deformed, "
                "disfigured, mutilated limbs, fused fingers, frozen frame, "
                "cluttered background, three legs, crowded background, walking backwards"
            )

            # Fix the random seed for reproducible generation.
            generator = torch.Generator(device=device).manual_seed(0)

            # Run the image-to-video denoising pipeline.
            output = pipe(
                image=image,                  # conditioning first frame
                prompt=prompt,                # text guidance
                negative_prompt=negative_prompt,
                height=576,                   # output video height in pixels
                width=1024,                   # output video width in pixels
                num_frames=101,               # 1 conditioning frame + 100 generated frames
                guidance_scale=3.5,           # classifier-free guidance scale
                num_inference_steps=40,       # number of DDIM/flow-matching denoising steps
                generator=generator,
            ).frames[0]
            # output shape: list of PIL images (101 frames) or numpy array

            # Retrieve the scene ID for constructing the output directory path.
            run = runs[idx]

            # Build the hierarchical output directory:
            # <video_save_folder>/<split>/<scene_id>/<model>/<exp_id>/
            virtual_path = os.path.join(
                args.video_save_folder,
                args.split,   # e.g. "ego_condition"
                run,          # scene ID
                args.model,   # e.g. "wan2.2-14b"
                args.exp_id   # e.g. "default_prompt"
            )
            os.makedirs(virtual_path, exist_ok=True)

            # Trim to exactly 101 frames (1 conditioning + 100 generated).
            samples = output[:101]
            # samples shape: (101, H, W, 3) — float in [0, 1]

            # Save as both a video file and individual PNG frames.
            perform_save_locally(virtual_path, samples, "videos")  # writes video.mp4
            perform_save_locally(virtual_path, samples, "images")  # writes 00000.png … 00100.png
