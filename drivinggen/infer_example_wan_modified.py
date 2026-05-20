"""
infer_example_wan.py
====================
Video generation inference script for the DrivingGen benchmark using the
Wan2.2-I2V-A14B model (image-to-video diffusion pipeline).

Pipeline overview
------------------
1. **Input**: JSON metadata file listing scene IDs with associated conditioning
   images and text captions (or a single image file).
2. **Processing**: For each scene:
   - Load the first frame (JPEG/PNG) and resize to fit the model.
   - Load the text caption.
   - Run Wan 2.2 image-to-video diffusion at 576 × 1024 for 101 frames.
3. **Output**: Generated video as MP4 + individual PNG frames, saved to a
   hierarchical output directory:
   ``<output_dir>/<split>/<scene_id>/<model>/<exp_id>/``

Quirks & fixes
--------------
A cuSolver fallback patch is applied so that torch.linalg.solve automatically
retries on CPU when the GPU solver encounters cuSolver internal errors.

Usage examples
--------------
Single image::

    python infer_example_wan.py \\
      --image /path/to/image.jpg \\
      --video_save_folder /output

JSON batch (first N scenes)::

    python infer_example_wan.py \\
      --image /path/to/scenes.json \\
      --video_save_folder /output \\
      --num_scenes 10 \\
      --split ego_condition \\
      --model wan2.2-14b \\
      --exp_id default_prompt

Full dataset (all scenes)::

    python infer_example_wan.py \\
      --image /path/to/scenes.json \\
      --video_save_folder /output \\
      --num_scenes -1

Custom generation parameters::

    python infer_example_wan.py \\
      --image /path/to/image.jpg \\
      --video_save_folder /output \\
      --height 576 --width 1024 --num_frames 101 \\
      --guidance_scale 3.5 --num_steps 40 --seed 42 \\
      --fps 10
"""

import torch
import argparse
import json
import os

import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image
from PIL import Image
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

# ---------------------------------------------------------------------------
# cuSolver fallback patch
# ---------------------------------------------------------------------------

def install_cusolver_fallback():
    """Install a cuSolver error recovery patch for torch.linalg.solve.

    On some GPU/driver combinations, torch.linalg.solve raises
    CUSOLVER_STATUS_INTERNAL_ERROR for small matrices (e.g., in attention
    layers).  This patch monkeymonkeys the solver to retry on CPU and move
    the result back to GPU/original dtype.

    Should be called *before* loading any models that use the solver.
    """
    _orig_solve = torch.linalg.solve   # reference to original GPU implementation

    def solve_with_cpu_fallback(A, B):
        """Wrapper around torch.linalg.solve with CPU fallback for cuSolver errors.

        Args:
            A (torch.Tensor): Coefficient matrix, shape ``(..., N, N)``.
            B (torch.Tensor): Right-hand side matrix/vector, shape ``(..., N, K)`` or
                ``(..., N)``.

        Returns:
            torch.Tensor: Solution tensor on the same device and dtype as ``A``,
                shape ``(..., N, K)`` or ``(..., N)``.
        """
        try:
            return _orig_solve(A, B)  # fast path: GPU solve
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusolver" in msg or "cusolver_status_internal_error" in msg:
                # cuSolver failed; retry on CPU
                X = _orig_solve(
                    A.detach().float().cpu(),  # cast to float32; move to CPU
                    B.detach().float().cpu()
                )
                return X.to(A.device, dtype=A.dtype)  # restore to original device/dtype
            raise  # re-raise non-cuSolver errors

    torch.linalg.solve = solve_with_cpu_fallback   # install patch globally


# ---------------------------------------------------------------------------
# Model and inference configuration
# ---------------------------------------------------------------------------


def load_model(device: str = "cuda", dtype=torch.bfloat16):
    """Load the Wan2.2-I2V-A14B diffusion pipeline.

    Args:
        device (str): Compute device ('cuda' or 'cpu'). Default: 'cuda'.
        dtype (torch.dtype): Precision for model weights. Default: torch.bfloat16.

    Returns:
        WanImageToVideoPipeline: Loaded and device-placed pipeline.
    """
    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    print(f"Loading model {model_id} on {device} with dtype {dtype}...")
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    print("Model loaded successfully.")
    return pipe


# ---------------------------------------------------------------------------
# Scene loading from JSON
# ---------------------------------------------------------------------------

def load_scenes_from_json(json_path: str, base_path: str, num_scenes: int = -1):
    """Load scene metadata (images and captions) from a JSON file.

    The JSON should be a list of scene ID strings (without extensions).  For
    each ID, the function looks for:
      - Conditioning image: ``<base_path>/imgs/<id>.jpg`` or ``.png``
      - Caption: ``<base_path>/caption/<id>.txt``

    Args:
        json_path (str): Path to the JSON metadata file.
        base_path (str): Base directory where image and caption subdirectories
            reside (derived from JSON path by stripping ``.json``).
        num_scenes (int): Max number of scenes to load. Use -1 to load all.
            Default: -1 (all).

    Returns:
        tuple:
            - **scenes** (list[dict]): List of dicts with keys 'scene_id',
              'image_path', 'prompt'.
            - **num_loaded** (int): Number of scenes actually loaded.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data: list of scene ID strings

    # Limit to num_scenes if positive
    if num_scenes > 0:
        data = data[:num_scenes]

    scenes = []
    for scene_id in data:
        # Try JPEG first, then PNG as fallback
        image_path = os.path.join(base_path, 'imgs', scene_id + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(base_path, 'imgs', scene_id + '.png')
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found for scene {scene_id}: {image_path}")
                continue

        # Load caption from text file
        caption_path = os.path.join(base_path, 'caption', scene_id + '.txt')
        if not os.path.exists(caption_path):
            print(f"⚠️  Caption not found for scene {scene_id}: {caption_path}")
            continue

        with open(caption_path, 'r') as f:
            prompt = f.read().strip()

        scenes.append({
            'scene_id': scene_id,
            'image_path': image_path,
            'prompt': prompt
        })

    return scenes, len(scenes)


# ---------------------------------------------------------------------------
# Video generation and saving
# ---------------------------------------------------------------------------

def generate_single_video(
    pipe,
    image_path: str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    device: str = "cuda",
):
    """Generate a single video using the Wan2.2 pipeline.

    Args:
        pipe (WanImageToVideoPipeline): Loaded diffusion pipeline.
        image_path (str): Path to the conditioning first frame.
        prompt (str): Text prompt for generation.
        height (int): Output frame height.
        width (int): Output frame width.
        num_frames (int): Number of output frames.
        guidance_scale (float): Classifier-free guidance scale.
        num_steps (int): Number of denoising steps.
        seed (int): Random seed for reproducibility.
        device (str): Compute device. Default: 'cuda'.

    Returns:
        np.ndarray: Generated video frames, shape ``(T, H, W, 3)``, float in
            ``[0, 1]``.
    """
    # Resize and prepare the conditioning image
    image = deal_img(image_path)

    # Standard negative prompt (guidance for what to avoid)
    negative_prompt = (
        "vivid colors, overexposed, static, blurry details, subtitles, "
        "stylized, artwork, painting, still, overall gray, worst quality, "
        "low quality, JPEG compression artifacts, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn face, deformed, "
        "disfigured, mutilated limbs, fused fingers, frozen frame, "
        "cluttered background, three legs, crowded background, walking backwards"
    )

    # Set up the random generator with the requested seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # Run the diffusion pipeline
    print(f"  Generating {num_frames} frames at {height}×{width}...")
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
    ).frames[0]
    # output: list of PIL images or numpy array

    # Convert to numpy and trim to exact frame count
    samples = np.array(output[:num_frames]) / 255.0 if isinstance(output[0], Image.Image) else output[:num_frames]
    # samples: shape (T, H, W, 3), float in [0, 1]

    return samples


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate videos from conditioning images using Wan2.2-I2V",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:

        Single image:
            python infer_example_wan.py --image img.jpg --video_save_folder /out

        Batch from JSON (first 10 scenes):
            python infer_example_wan.py --image scenes.json --video_save_folder /out --num_scenes 10

        Full dataset:
            python infer_example_wan.py --image scenes.json --video_save_folder /out --num_scenes -1

        Custom generation parameters:
            python infer_example_wan.py --image scenes.json --video_save_folder /out \\
            --height 576 --width 1024 --num_frames 101 --guidance_scale 3.5 \\
            --num_steps 40 --seed 42 --fps 10
                """
    )

    # Input/output paths
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to JSON metadata file (scenes.json) or a single image (jpg/png)."
    )
    parser.add_argument(
        "--video_save_folder",
        type=str,
        required=True,
        help="Root output directory. Generated videos saved to "
             "<folder>/<split>/<scene>/<model>/<exp_id>/"
    )

    # Directory structure
    parser.add_argument(
        "--split",
        type=str,
        default="ego_condition",
        help="Dataset split name (subdirectory). Default: 'ego_condition'."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wan2.2-14b",
        help="Model name (subdirectory). Default: 'wan2.2-14b'."
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="default_prompt",
        help="Experiment ID (subdirectory). Default: 'default_prompt'."
    )

    # Generation control
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=2,
        help="Max number of scenes to generate from JSON. Use -1 for all. Default: 2."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation. Default: 0."
    )

    # Video dimensions
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Video frame height in pixels. Default: 576."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Video frame width in pixels. Default: 1024."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=101,
        help="Number of output frames (1 conditioning + 100 generated). Default: 101."
    )

    # Diffusion parameters
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale. Default: 3.5."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=40,
        help="Number of denoising steps. Default: 40."
    )

    # Output video settings
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video frame rate. Default: 10 fps."
    )

    args = parser.parse_args()

    # Install the cuSolver fallback patch before loading models
    install_cusolver_fallback()

    # Load the diffusion pipeline
    pipe = load_model(device="cuda", dtype=torch.bfloat16)

    # Determine if input is JSON (batch) or single image
    if args.image.endswith('.json'):
        # Batch mode: load metadata from JSON
        base_path = args.image[:-5]  # strip ".json" suffix
        scenes, num_loaded = load_scenes_from_json(
            args.image, base_path, num_scenes=args.num_scenes
        )
        print(f"Loaded {num_loaded} scene(s).")

        if not scenes:
            print("No valid scenes found. Exiting.")
            exit(1)

        # Generate video for each scene
        for scene_num, scene in enumerate(scenes, start=1):
            print(f"\n[{scene_num}/{num_loaded}] Processing scene: {scene['scene_id']}")

            # Generate video
            samples = generate_single_video(
                pipe,
                image_path=scene['image_path'],
                prompt=scene['prompt'],
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_steps=args.num_steps,
                seed=args.seed,
                device="cuda",
            )
            # samples: shape (T, H, W, 3), float in [0, 1]

            # Construct output directory
            output_dir = os.path.join(
                args.video_save_folder,
                args.split,
                scene['scene_id'],
                args.model,
                args.exp_id,
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save video and frames
            print(f"  Saving to {output_dir}")
            perform_save_locally(output_dir, samples, "videos")
            perform_save_locally(output_dir, samples, "images")

    else:
        # Single image mode
        print(f"Processing single image: {args.image}")

        # Generate video
        samples = generate_single_video(
            pipe,
            image_path=args.image,
            prompt="A car driving on a road.",  # default prompt
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            device="cuda",
        )
        # samples: shape (T, H, W, 3), float in [0, 1]

        # Construct output directory
        output_dir = os.path.join(
            args.video_save_folder,
            "single_image",
            args.model,
            args.exp_id,
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save video and frames
        print(f"Saving to {output_dir}")
        perform_save_locally(output_dir, samples, "videos")
        perform_save_locally(output_dir, samples, "images")
