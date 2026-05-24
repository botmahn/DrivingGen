#!/usr/bin/env python3
"""
Generate a DOCX report for the current DrivingGen benchmark results.

This script intentionally avoids python-docx, pandas, and matplotlib because the
local environment may not have compatible installs. It renders charts with PIL
and writes a minimal OOXML .docx package directly.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rank_ego_trajectory_displacement import (  # noqa: E402
    TrajectoryJob,
    add_ranks,
    aggregate_model_metrics,
    build_jobs,
    compute_scene_metrics,
    discover_models,
    discover_scenes,
    find_pred_traj,
    prepare_trajectories,
)


REPORT_DIR = REPO_ROOT / "benchmark_results"
ASSET_DIR = REPORT_DIR / "drivinggen_report_assets"
REPORT_PATH = REPORT_DIR / "drivinggen_benchmark_report.docx"
CAPTION_ROOT = REPORT_DIR / "caption_following" / "output_cue"
INPUT_ROOT = REPO_ROOT / "drivinggen_input_dataset"
BENCHMARK_ROOT = REPORT_DIR

MODEL_DISPLAY = {
    "ltx_2b": "LTX-2B",
    "ltx_13b": "LTX-13B",
    "nvidia-cosmos-predict-2.5-2b": "NVIDIA Cosmos 2.5-2B",
    "vista-2b": "Vista-2B",
    "wan2.2-5b": "WAN2.2-5B",
    "ltx-video-2b": "LTX-2B",
    "ltx-video-13b": "LTX-13B",
    "nvidia-cosmos-predict2-2b": "NVIDIA Cosmos 2.5-2B",
    "vista": "Vista-2B",
    "wan2.2-5b": "WAN2.2-5B",
    "ltx-video-2bx": "LTX-2B",
    "ltx-video-13bx": "LTX-13B",
    "nvidia-cosmos-predict2.5-2b": "NVIDIA Cosmos 2.5-2B",
    "vista2b": "Vista-2B",
}

MODEL_COLORS = {
    "LTX-2B": "#2563eb",
    "LTX-13B": "#dc2626",
    "NVIDIA Cosmos 2.5-2B": "#16a34a",
    "Vista-2B": "#f59e0b",
    "WAN2.2-5B": "#7c3aed",
}

MODEL_DIRS = {
    "LTX-2B": REPORT_DIR / "ltx-video-2b-drivinggen-samples-outputs",
    "LTX-13B": REPORT_DIR / "ltx-video-13b-drivinggen-samples-outputs",
    "NVIDIA Cosmos 2.5-2B": REPORT_DIR / "nvidia-cosmos-predict2-2b-drivinggen-samples-outputs",
    "Vista-2B": REPORT_DIR / "vista-drivinggen-samples-outputs",
    "WAN2.2-5B": REPORT_DIR / "wan2.2-5b-drivinggen-samples-outputs",
}

PALETTE = {
    "ink": "#172033",
    "muted": "#5b6475",
    "light": "#eef3f8",
    "panel": "#f8fafc",
    "grid": "#d9e1ea",
    "good": "#16a34a",
    "warn": "#f59e0b",
    "bad": "#dc2626",
    "blue": "#2563eb",
    "teal": "#0891b2",
}


def display_model(name: str) -> str:
    return MODEL_DISPLAY.get(name, name)


def pct(value: float) -> str:
    return f"{value:.2f}"


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(values) / len(values)) if values else float("nan")


def stdev(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def shorten_scene(scene: str, limit: int = 46) -> str:
    text = scene.replace("+", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def scene_id_to_file_stem(pair_id: str) -> str:
    return pair_id.replace("_", "+", 1) if re.match(r"^[A-Z]{2}_\\d", pair_id) else pair_id.replace("_", "+")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = {
    "xs": load_font(16),
    "sm": load_font(20),
    "body": load_font(24),
    "body_b": load_font(24, bold=True),
    "h3": load_font(30, bold=True),
    "h2": load_font(38, bold=True),
    "h1": load_font(52, bold=True),
}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 7,
) -> int:
    x, y = xy
    for line in wrap_by_width(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def draw_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def save_bar_chart(
    path: Path,
    title: str,
    items: list[tuple[str, float]],
    subtitle: str = "",
    lower_better: bool = False,
    suffix: str = "",
    width: int = 1500,
    height: int | None = None,
) -> None:
    if height is None:
        height = max(620, 230 + 72 * len(items))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin_l, margin_r = 360, 90
    top = 126
    row_h = max(56, int((height - top - 80) / max(1, len(items))))
    draw.text((52, 36), title, font=FONT["h2"], fill=PALETTE["ink"])
    if subtitle:
        draw.text((52, 84), subtitle, font=FONT["sm"], fill=PALETTE["muted"])
    values = [v for _, v in items]
    max_value = max(values) if values else 1.0
    min_value = min(values) if values else 0.0
    scale_max = max_value * 1.08 if max_value > 0 else 1.0
    bar_w = width - margin_l - margin_r
    for idx, (label, value) in enumerate(items):
        y = top + idx * row_h
        color = MODEL_COLORS.get(label, PALETTE["blue"])
        draw.text((52, y + 12), label, font=FONT["body_b"], fill=PALETTE["ink"])
        x0 = margin_l
        x1 = margin_l + int((value / scale_max) * bar_w)
        draw_round_rect(draw, (x0, y + 10, margin_l + bar_w, y + 42), 10, PALETTE["light"])
        draw_round_rect(draw, (x0, y + 10, max(x0 + 2, x1), y + 42), 10, color)
        label_text = f"{value:.2f}{suffix}"
        if lower_better and value == min_value:
            label_text += "  best"
        elif not lower_better and value == max_value:
            label_text += "  best"
        draw.text((margin_l + bar_w + 18, y + 12), label_text, font=FONT["sm"], fill=PALETTE["ink"])
    img.save(path)


def color_ramp(value: float, vmin: float, vmax: float, high_good: bool = True) -> tuple[int, int, int]:
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if not high_good:
        t = 1.0 - t
    low = np.array(hex_to_rgb("#fee2e2"), dtype=np.float64)
    mid = np.array(hex_to_rgb("#fef3c7"), dtype=np.float64)
    high = np.array(hex_to_rgb("#dcfce7"), dtype=np.float64)
    if t < 0.5:
        rgb = low * (1 - 2 * t) + mid * (2 * t)
    else:
        rgb = mid * (2 - 2 * t) + high * (2 * t - 1)
    return tuple(int(x) for x in rgb)


def save_heatmap(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float | None]],
    vmin: float,
    vmax: float,
    high_good: bool = True,
    width: int = 1700,
    cell_h: int = 72,
) -> None:
    left, top = 380, 165
    cell_w = max(110, int((width - left - 60) / max(1, len(col_labels))))
    height = top + cell_h * len(row_labels) + 80
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 36), title, font=FONT["h2"], fill=PALETTE["ink"])
    draw.text((52, 88), "Darker green is better; red marks lower scores or higher errors.", font=FONT["sm"], fill=PALETTE["muted"])
    for j, label in enumerate(col_labels):
        x = left + j * cell_w
        short = shorten_scene(label, 18)
        lines = textwrap.wrap(short, width=15)[:2]
        for k, line in enumerate(lines):
            draw.text((x + 8, 118 + k * 20), line, font=FONT["xs"], fill=PALETTE["muted"])
    for i, row_label in enumerate(row_labels):
        y = top + i * cell_h
        draw.text((52, y + 20), row_label, font=FONT["body_b"], fill=PALETTE["ink"])
        for j, value in enumerate(values[i]):
            x = left + j * cell_w
            if value is None or math.isnan(float(value)):
                fill = "#f1f5f9"
                text = "NA"
            else:
                fill = "#{:02x}{:02x}{:02x}".format(*color_ramp(float(value), vmin, vmax, high_good))
                text = f"{float(value):.1f}"
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 5), fill=fill, outline="#ffffff")
            tw, th = text_size(draw, text, FONT["sm"])
            draw.text((x + (cell_w - tw) / 2 - 2, y + (cell_h - th) / 2 - 4), text, font=FONT["sm"], fill=PALETTE["ink"])
    img.save(path)


def save_grouped_metric_chart(
    path: Path,
    title: str,
    rows: list[dict[str, float | str]],
    metrics: list[tuple[str, str]],
    width: int = 1600,
    height: int = 760,
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 34), title, font=FONT["h2"], fill=PALETTE["ink"])
    draw.text((52, 85), "Metrics are shown on their native scale. Higher is better except FVD.", font=FONT["sm"], fill=PALETTE["muted"])
    plot_l, plot_t, plot_r, plot_b = 120, 170, width - 70, height - 130
    draw.line((plot_l, plot_b, plot_r, plot_b), fill=PALETTE["grid"], width=2)
    group_w = (plot_r - plot_l) / max(1, len(rows))
    bar_w = min(42, group_w / (len(metrics) + 1))
    for m_idx, (_, label) in enumerate(metrics):
        x = plot_l + m_idx * 250
        color = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"][m_idx % 5]
        draw.rectangle((x, height - 78, x + 24, height - 54), fill=color)
        draw.text((x + 34, height - 79), label, font=FONT["xs"], fill=PALETTE["ink"])
    for i, row in enumerate(rows):
        gx = plot_l + i * group_w + group_w / 2
        model = str(row["model"])
        for m_idx, (key, _) in enumerate(metrics):
            value = float(row[key])
            if key == "fvd":
                normalized = value / max(float(r[key]) for r in rows)
            else:
                normalized = value
            h = normalized * (plot_b - plot_t)
            x0 = gx - (len(metrics) * bar_w) / 2 + m_idx * bar_w
            color = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"][m_idx % 5]
            draw.rectangle((x0, plot_b - h, x0 + bar_w * 0.72, plot_b), fill=color)
        for k, line in enumerate(textwrap.wrap(model, width=14)[:2]):
            tw, _ = text_size(draw, line, FONT["xs"])
            draw.text((gx - tw / 2, plot_b + 16 + k * 19), line, font=FONT["xs"], fill=PALETTE["muted"])
    img.save(path)


def load_caption_data() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []
    for summary_path in sorted(CAPTION_ROOT.glob("*/batch_summary.json")):
        model_key = summary_path.parent.name
        model = display_model(model_key)
        rows = json.loads(summary_path.read_text())
        good_rows = [r for r in rows if r.get("status") == "success"]
        model_rows.append(
            {
                "model": model,
                "model_key": model_key,
                "num_scenes": len(good_rows),
                "mean_caption_usage_score": mean([float(r["caption_usage_score"]) for r in good_rows]),
                "std_caption_usage_score": stdev([float(r["caption_usage_score"]) for r in good_rows]),
                "mean_weighted_caption_usage_score": mean(
                    [float(r["weighted_caption_usage_score"]) for r in good_rows]
                ),
            }
        )
        for row in good_rows:
            scene_rows.append(
                {
                    "model": model,
                    "model_key": model_key,
                    "scene": str(row["pair_id"]),
                    "caption_usage_score": float(row["caption_usage_score"]),
                    "weighted_caption_usage_score": float(row["weighted_caption_usage_score"]),
                    "num_claims": int(row.get("num_claims", 0)),
                    "category_scores": row.get("category_scores", {}),
                }
            )
            report_path = summary_path.parent / str(row["pair_id"]) / "caption_usage_report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                for claim in report.get("claim_results", []):
                    claim_rows.append(
                        {
                            "model": model,
                            "scene": str(row["pair_id"]),
                            "category": claim.get("category", "unknown"),
                            "claim_type": claim.get("claim_type", "unknown"),
                            "importance": float(claim.get("importance", 0) or 0),
                            "final_score": float(claim.get("final_score", 0) or 0),
                            "usage_label": claim.get("usage_label", ""),
                            "claim": claim.get("claim", ""),
                        }
                    )
    return model_rows, scene_rows, claim_rows


def caption_category_table(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    categories = sorted(
        {
            cat
            for row in scene_rows
            for cat, value in dict(row["category_scores"]).items()
            if value is not None and cat != "global_subjective"
        }
    )
    models = sorted({str(row["model"]) for row in scene_rows})
    out = []
    for model in models:
        entry: dict[str, object] = {"model": model}
        for cat in categories:
            vals = [
                float(dict(row["category_scores"])[cat])
                for row in scene_rows
                if row["model"] == model and dict(row["category_scores"]).get(cat) is not None
            ]
            entry[cat] = mean(vals)
        out.append(entry)
    return out


def caption_common_summary(scene_rows: list[dict[str, object]]) -> tuple[list[str], list[dict[str, object]]]:
    by_model: dict[str, set[str]] = {}
    for row in scene_rows:
        by_model.setdefault(str(row["model"]), set()).add(str(row["scene"]))
    common = sorted(set.intersection(*by_model.values())) if by_model else []
    out = []
    for model in sorted(by_model):
        vals = [
            float(row["weighted_caption_usage_score"])
            for row in scene_rows
            if row["model"] == model and row["scene"] in common
        ]
        out.append({"model": model, "num_common_scenes": len(vals), "mean_common_weighted_score": mean(vals)})
    return common, out


def caption_scene_difficulty(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_scene: dict[str, list[dict[str, object]]] = {}
    for row in scene_rows:
        by_scene.setdefault(str(row["scene"]), []).append(row)
    out = []
    for scene, rows in by_scene.items():
        vals = [float(r["weighted_caption_usage_score"]) for r in rows]
        best = max(rows, key=lambda r: float(r["weighted_caption_usage_score"]))
        out.append(
            {
                "scene": scene,
                "mean_weighted_score": mean(vals),
                "num_models": len(rows),
                "best_model": best["model"],
                "best_score": float(best["weighted_caption_usage_score"]),
            }
        )
    return sorted(out, key=lambda r: float(r["mean_weighted_score"]))


def load_fvd_results() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for path in sorted(REPORT_DIR.glob("*/benchmarks/FVD/*/results.txt")):
        text = path.read_text().strip()
        if not text or ":" not in text:
            continue
        raw_model, payload = text.split(":", 1)
        data = ast.literal_eval(payload.strip())
        row = {"model": display_model(raw_model.strip()), "raw_model": raw_model.strip(), "path": str(path)}
        row.update(data)
        rows.append(row)

    metrics = [
        ("fvd", "lower"),
        ("objective_quality", "higher"),
        ("subjective_quality", "higher"),
        ("scene_consistency", "higher"),
        ("agent_consistency", "higher"),
    ]
    ranks: dict[str, list[int]] = {str(r["model"]): [] for r in rows}
    for key, direction in metrics:
        ordered = sorted(rows, key=lambda r: float(r[key]), reverse=(direction == "higher"))
        for idx, row in enumerate(ordered, 1):
            ranks[str(row["model"])].append(idx)
    rank_rows = []
    for row in rows:
        model = str(row["model"])
        rank_rows.append(
            {
                "model": model,
                "avg_rank": mean(ranks[model]),
                "raw_fvd": float(row["fvd"]),
                "objective_quality": float(row["objective_quality"]),
                "subjective_quality": float(row["subjective_quality"]),
                "scene_consistency": float(row["scene_consistency"]),
                "agent_consistency": float(row["agent_consistency"]),
            }
        )
    rank_rows.sort(key=lambda r: (float(r["avg_rank"]), str(r["model"])))
    for idx, row in enumerate(rank_rows, 1):
        row["rank"] = idx
    return rows, rank_rows


def compute_trajectory_rankings(common_scenes: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_dirs = discover_models(REPORT_DIR, None)
    scene_ids = discover_scenes(model_dirs, None)
    jobs, warnings = build_jobs(model_dirs, scene_ids, INPUT_ROOT, common_scenes=common_scenes, strict=False)
    scene_rows = []
    for job in jobs:
        row = compute_scene_metrics(job, "none", 1.0)
        row["model"] = display_model(str(row["model"]))
        scene_rows.append(row)
    aggregate_rows = aggregate_model_metrics(scene_rows)
    for row in aggregate_rows:
        row["model"] = display_model(str(row["model"]))
    ranked = add_ranks(aggregate_rows, "mean_combined_endpoint_distance_rel_error")
    return ranked, scene_rows


def save_cover(path: Path, caption_rank: list[dict[str, object]], traj_rank: list[dict[str, object]], fvd_rank: list[dict[str, object]]) -> None:
    img = Image.new("RGB", (1800, 1100), "#0f172a")
    draw = ImageDraw.Draw(img)
    for i in range(0, 1800, 60):
        shade = int(25 + 28 * (i / 1800))
        draw.line((i, 0, i + 520, 1100), fill=(shade, 40, 68), width=3)
    draw_round_rect(draw, (70, 70, 1730, 1030), 34, "#111827", "#334155", 3)
    draw.text((125, 140), "DrivingGen Benchmark Report", font=FONT["h1"], fill="#f8fafc")
    draw.text((128, 212), "Caption following, video quality, and ego-trajectory adherence", font=FONT["h3"], fill="#cbd5e1")
    cards = [
        ("Overall signal", "LTX-2B", "Best balanced model in this 10-scene slice"),
        ("Caption common scenes", str(caption_rank[0]["model"]), f"{float(caption_rank[0]['mean_common_weighted_score']):.2f} weighted CUS"),
        ("Ego trajectory", str(traj_rank[0]["model"]), f"{float(traj_rank[0]['mean_combined_endpoint_distance_rel_error']):.3f} rel. error"),
        ("FVD aggregate", str(fvd_rank[0]["model"]), f"avg metric rank {float(fvd_rank[0]['avg_rank']):.2f}"),
    ]
    x0, y0 = 125, 365
    for i, (label, value, note) in enumerate(cards):
        x = x0 + (i % 2) * 800
        y = y0 + (i // 2) * 245
        draw_round_rect(draw, (x, y, x + 720, y + 185), 26, "#1e293b", "#475569", 2)
        draw.text((x + 38, y + 30), label, font=FONT["sm"], fill="#94a3b8")
        draw.text((x + 38, y + 72), value, font=FONT["h2"], fill="#f8fafc")
        draw_wrapped(draw, (x + 38, y + 126), note, FONT["sm"], "#cbd5e1", 620, line_gap=4)
    draw.text((128, 920), f"Generated {datetime.now().strftime('%Y-%m-%d')} from benchmark_results/", font=FONT["body"], fill="#cbd5e1")
    img.save(path)


def save_caption_scene_heatmap(path: Path, common_scenes: list[str], scene_rows: list[dict[str, object]]) -> None:
    models = ["LTX-2B", "LTX-13B", "NVIDIA Cosmos 2.5-2B", "Vista-2B", "WAN2.2-5B"]
    values = []
    for model in models:
        row_vals = []
        for scene in common_scenes:
            matches = [r for r in scene_rows if r["model"] == model and r["scene"] == scene]
            row_vals.append(float(matches[0]["weighted_caption_usage_score"]) if matches else None)
        values.append(row_vals)
    save_heatmap(path, "Caption Following by Scene", models, common_scenes, values, 65, 100, True)


def save_category_heatmap(path: Path, category_rows: list[dict[str, object]]) -> None:
    categories = [c for c in ["actions", "ego_vehicle", "environment", "traffic_agents"] if any(c in r for r in category_rows)]
    models = ["LTX-2B", "LTX-13B", "NVIDIA Cosmos 2.5-2B", "Vista-2B", "WAN2.2-5B"]
    lookup = {str(r["model"]): r for r in category_rows}
    values = [[float(lookup.get(model, {}).get(cat, float("nan"))) for cat in categories] for model in models]
    save_heatmap(path, "Caption Category Scores", models, categories, values, 55, 100, True, width=1300)


def save_ltx_trajectory_comparison(path: Path, scene_rows: list[dict[str, object]]) -> None:
    scenes = []
    for row in scene_rows:
        if row["model"] == "LTX-2B":
            scenes.append(str(row["scene"]))
    rows = []
    for scene in scenes:
        a = next((r for r in scene_rows if r["model"] == "LTX-2B" and r["scene"] == scene), None)
        b = next((r for r in scene_rows if r["model"] == "LTX-13B" and r["scene"] == scene), None)
        if a and b:
            rows.append(
                (
                    shorten_scene(scene, 32),
                    float(a["combined_endpoint_distance_rel_error"]),
                    float(b["combined_endpoint_distance_rel_error"]),
                )
            )
    width, height = 1600, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 34), "LTX-2B vs LTX-13B Ego-Trajectory Error", font=FONT["h2"], fill=PALETTE["ink"])
    draw.text((52, 84), "Lower is better. Near-static scenes dominate the 13B failure pattern.", font=FONT["sm"], fill=PALETTE["muted"])
    plot_l, plot_t, plot_r, plot_b = 430, 150, width - 95, height - 80
    max_v = max(max(a, b) for _, a, b in rows) * 1.08
    row_h = (plot_b - plot_t) / max(1, len(rows))
    for i, (scene, v2, v13) in enumerate(rows):
        y = plot_t + i * row_h + 10
        draw_wrapped(draw, (52, int(y)), scene, FONT["xs"], PALETTE["ink"], 335, 2)
        for j, (value, model) in enumerate([(v2, "LTX-2B"), (v13, "LTX-13B")]):
            yy = y + j * 24
            x1 = plot_l + int((value / max_v) * (plot_r - plot_l))
            draw_round_rect(draw, (plot_l, int(yy), plot_r, int(yy + 16)), 8, PALETTE["light"])
            draw_round_rect(draw, (plot_l, int(yy), max(plot_l + 2, x1), int(yy + 16)), 8, MODEL_COLORS[model])
            draw.text((x1 + 8, int(yy - 4)), f"{value:.2f}", font=FONT["xs"], fill=PALETTE["ink"])
    draw.rectangle((52, height - 52, 76, height - 28), fill=MODEL_COLORS["LTX-2B"])
    draw.text((86, height - 55), "LTX-2B", font=FONT["sm"], fill=PALETTE["ink"])
    draw.rectangle((210, height - 52, 234, height - 28), fill=MODEL_COLORS["LTX-13B"])
    draw.text((244, height - 55), "LTX-13B", font=FONT["sm"], fill=PALETTE["ink"])
    img.save(path)


def read_video_frame(path: Path, frame_fraction: float = 0.5) -> Image.Image | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frame_fraction))))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def read_gt_image(scene: str) -> Image.Image | None:
    image_path = INPUT_ROOT / "ego_condition" / "imgs" / f"{scene}.jpg"
    if image_path.is_file():
        return Image.open(image_path).convert("RGB")
    frame_dir = INPUT_ROOT / "videos-fvd" / scene
    frames = sorted(frame_dir.glob("*.jpg")) if frame_dir.is_dir() else []
    if frames:
        return Image.open(frames[len(frames) // 2]).convert("RGB")
    return None


def make_thumbnail(image: Image.Image | None, size: tuple[int, int]) -> Image.Image:
    thumb = Image.new("RGB", size, "#e5e7eb")
    if image is None:
        draw = ImageDraw.Draw(thumb)
        draw.text((30, size[1] // 2 - 12), "missing", font=FONT["body"], fill=PALETTE["muted"])
        return thumb
    img = image.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    thumb.paste(img, (x, y))
    return thumb


def save_contact_sheet(path: Path, scene: str, title: str) -> None:
    columns = [
        ("GT", None),
        ("LTX-2B", MODEL_DIRS["LTX-2B"]),
        ("LTX-13B", MODEL_DIRS["LTX-13B"]),
        ("NVIDIA", MODEL_DIRS["NVIDIA Cosmos 2.5-2B"]),
        ("Vista", MODEL_DIRS["Vista-2B"]),
        ("WAN", MODEL_DIRS["WAN2.2-5B"]),
    ]
    tile_w, tile_h = 300, 176
    width, height = 1960, 410
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 34), title, font=FONT["h2"], fill=PALETTE["ink"])
    draw_wrapped(draw, (52, 86), scene, FONT["sm"], PALETTE["muted"], width - 120, 4)
    x, y = 52, 160
    for label, model_dir in columns:
        if model_dir is None:
            frame = read_gt_image(scene)
        else:
            video = model_dir / "generated_videos" / f"{scene}.mp4"
            frame = read_video_frame(video)
        draw_round_rect(draw, (x - 5, y - 5, x + tile_w + 5, y + tile_h + 5), 12, "#f8fafc", "#d8e0ea")
        img.paste(make_thumbnail(frame, (tile_w, tile_h)), (x, y))
        draw.text((x, y + tile_h + 16), label, font=FONT["body_b"], fill=MODEL_COLORS.get(label, PALETTE["ink"]))
        x += tile_w + 26
    img.save(path)


def trajectory_plot_points(points: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = box
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        finite = np.zeros((1, 2), dtype=np.float64)
    fmin, lmin = finite.min(axis=0)
    fmax, lmax = finite.max(axis=0)
    fspan = max(float(fmax - fmin), 1.0)
    lspan = max(float(lmax - lmin), 1.0)
    fmid = float((fmin + fmax) * 0.5)
    lmid = float((lmin + lmax) * 0.5)
    scale = min((x1 - x0) / lspan, (y1 - y0) / fspan) * 0.86
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    out = []
    for forward, lateral_left in points:
        px = cx - (float(lateral_left) - lmid) * scale
        py = cy - (float(forward) - fmid) * scale
        out.append((int(round(px)), int(round(py))))
    return out


def load_model_traj(model: str, scene: str) -> tuple[np.ndarray, np.ndarray] | None:
    model_dir = MODEL_DIRS[model]
    pred = find_pred_traj(model_dir, scene)
    gt = INPUT_ROOT / "ego_condition" / "ego_motion" / f"{scene}.npy"
    if pred is None or not gt.is_file():
        return None
    job = TrajectoryJob(
        model_dir=model_dir,
        model_name=model,
        scene=scene,
        pred_traj_path=pred,
        gt_traj_path=gt,
    )
    gt_traj, pred_traj, _ = prepare_trajectories(job, "none")
    return gt_traj, pred_traj


def save_trajectory_path_panel(path: Path, scene: str) -> None:
    models = ["LTX-2B", "LTX-13B"]
    loaded = {m: load_model_traj(m, scene) for m in models}
    width, height = 1500, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 34), "Trajectory Shape: Near-Static Failure Case", font=FONT["h2"], fill=PALETTE["ink"])
    draw_wrapped(draw, (52, 86), scene, FONT["sm"], PALETTE["muted"], width - 120, 4)
    for idx, model in enumerate(models):
        x0 = 80 + idx * 710
        box = (x0 + 30, 170, x0 + 620, 640)
        draw_round_rect(draw, (x0, 140, x0 + 650, 670), 18, "#f8fafc", "#d8e0ea")
        draw.text((x0 + 28, 154), model, font=FONT["h3"], fill=MODEL_COLORS[model])
        data = loaded[model]
        if data is None:
            draw.text((x0 + 80, 360), "trajectory missing", font=FONT["body"], fill=PALETTE["muted"])
            continue
        gt, pred = data
        all_points = np.vstack([gt, pred])
        gt_pts = trajectory_plot_points(gt, box)
        pred_pts = trajectory_plot_points(pred, box)
        for gx in range(box[0], box[2] + 1, 80):
            draw.line((gx, box[1], gx, box[3]), fill="#e5e7eb", width=1)
        for gy in range(box[1], box[3] + 1, 80):
            draw.line((box[0], gy, box[2], gy), fill="#e5e7eb", width=1)
        if len(gt_pts) > 1:
            draw.line(gt_pts, fill="#16a34a", width=5, joint="curve")
        if len(pred_pts) > 1:
            draw.line(pred_pts, fill=MODEL_COLORS[model], width=5, joint="curve")
        draw.ellipse((gt_pts[-1][0] - 7, gt_pts[-1][1] - 7, gt_pts[-1][0] + 7, gt_pts[-1][1] + 7), fill="#16a34a")
        draw.ellipse(
            (pred_pts[-1][0] - 7, pred_pts[-1][1] - 7, pred_pts[-1][0] + 7, pred_pts[-1][1] + 7),
            fill=MODEL_COLORS[model],
        )
    draw.rectangle((92, 672, 116, 696), fill="#16a34a")
    draw.text((126, 669), "GT", font=FONT["sm"], fill=PALETTE["ink"])
    draw.rectangle((205, 672, 229, 696), fill="#2563eb")
    draw.text((239, 669), "Predicted", font=FONT["sm"], fill=PALETTE["ink"])
    img.save(path)


def table_rows(headers: list[str], rows: list[list[object]], max_rows: int | None = None) -> list[list[str]]:
    rows = rows[:max_rows] if max_rows else rows
    return [headers] + [[str(cell) for cell in row] for row in rows]


@dataclass
class DocxImage:
    rid: str
    target: str
    path: Path


class DocxWriter:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.images: list[DocxImage] = []

    def paragraph(self, text: str = "", style: str | None = None, align: str | None = None) -> None:
        ppr = ""
        if style or align:
            parts = []
            if style:
                parts.append(f'<w:pStyle w:val="{style}"/>')
            if align:
                parts.append(f'<w:jc w:val="{align}"/>')
            ppr = "<w:pPr>" + "".join(parts) + "</w:pPr>"
        runs = []
        pieces = str(text).split("\n")
        for idx, piece in enumerate(pieces):
            if idx:
                runs.append("<w:r><w:br/></w:r>")
            runs.append(f"<w:r><w:t>{escape(piece)}</w:t></w:r>")
        self.body.append("<w:p>" + ppr + "".join(runs) + "</w:p>")

    def bullet(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Bullet"/></w:pPr>'
            f"<w:r><w:t>- {escape(text)}</w:t></w:r></w:p>"
        )

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows: list[list[object]], widths: list[int] | None = None) -> None:
        if not rows:
            return
        xml = [
            '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
            '<w:tblLook w:firstRow="1" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        ]
        for r_idx, row in enumerate(rows):
            xml.append("<w:tr>")
            for c_idx, cell in enumerate(row):
                shade = "D9EAF7" if r_idx == 0 else ("F8FAFC" if r_idx % 2 == 0 else "FFFFFF")
                width_xml = ""
                if widths and c_idx < len(widths):
                    width_xml = f'<w:tcW w:w="{widths[c_idx]}" w:type="dxa"/>'
                xml.append(f'<w:tc><w:tcPr>{width_xml}<w:shd w:fill="{shade}"/></w:tcPr>')
                text = str(cell)
                xml.append(f"<w:p><w:r>{'<w:rPr><w:b/></w:rPr>' if r_idx == 0 else ''}<w:t>{escape(text)}</w:t></w:r></w:p>")
                xml.append("</w:tc>")
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))

    def image(self, path: Path, width_in: float = 6.6, caption: str | None = None) -> None:
        image = Image.open(path)
        aspect = image.height / image.width
        height_in = width_in * aspect
        rid = f"rId{len(self.images) + 1}"
        target = f"media/image{len(self.images) + 1}{path.suffix.lower()}"
        self.images.append(DocxImage(rid, target, path))
        cx = int(width_in * 914400)
        cy = int(height_in * 914400)
        docpr_id = len(self.images)
        xml = f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{docpr_id}" name="Picture {docpr_id}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="{docpr_id}" name="{escape(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""
        self.body.append(xml)
        if caption:
            self.paragraph(caption, style="Caption", align="center")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
 <w:body>
  {''.join(self.body)}
  <w:sectPr>
   <w:pgSz w:w="12240" w:h="15840"/>
   <w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>
  </w:sectPr>
 </w:body>
</w:document>"""

        rels = [
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        ]
        content_overrides = [
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        ]
        for img in self.images:
            rels.append(
                f'<Relationship Id="{img.rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{img.target}"/>'
            )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="jpg" ContentType="image/jpeg"/>
 <Default Extension="jpeg" ContentType="image/jpeg"/>
 """
                + "\n".join(content_overrides)
                + "\n</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
            )
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(rels)
                + "</Relationships>",
            )
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/styles.xml", styles_xml())
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            zf.writestr(
                "docProps/core.xml",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>DrivingGen Benchmark Report</dc:title>
 <dc:creator>Codex</dc:creator>
 <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
            )
            zf.writestr(
                "docProps/app.xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Codex OOXML Writer</Application>
</Properties>""",
            )
            for img in self.images:
                zf.write(img.path, f"word/{img.target}")


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
  <w:name w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="21"/><w:color w:val="172033"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Title">
  <w:name w:val="Title"/><w:qFormat/>
  <w:pPr><w:spacing w:after="260"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:sz w:val="44"/><w:color w:val="172033"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Heading1">
  <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:before="360" w:after="180"/><w:keepNext/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:sz w:val="34"/><w:color w:val="0F172A"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Heading2">
  <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:before="240" w:after="140"/><w:keepNext/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:sz w:val="27"/><w:color w:val="1E3A8A"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Caption">
  <w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:spacing w:after="160"/></w:pPr>
  <w:rPr><w:i/><w:sz w:val="18"/><w:color w:val="5B6475"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Bullet">
  <w:name w:val="Bullet"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:ind w:left="360" w:hanging="180"/><w:spacing w:after="80"/></w:pPr>
  <w:rPr><w:sz w:val="21"/></w:rPr>
 </w:style>
 <w:style w:type="table" w:styleId="TableGrid">
  <w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:qFormat/>
  <w:tblPr><w:tblBorders>
   <w:top w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
   <w:left w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
   <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
   <w:right w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
   <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
   <w:insideV w:val="single" w:sz="4" w:space="0" w:color="D8E0EA"/>
  </w:tblBorders></w:tblPr>
 </w:style>
</w:styles>"""


def build_report() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    caption_models, caption_scenes, claim_rows = load_caption_data()
    common_scenes, caption_common = caption_common_summary(caption_scenes)
    caption_categories = caption_category_table(caption_scenes)
    scene_difficulty = caption_scene_difficulty(caption_scenes)
    fvd_rows, fvd_rank = load_fvd_results()
    traj_common, traj_common_scenes = compute_trajectory_rankings(common_scenes=True)
    traj_all, _ = compute_trajectory_rankings(common_scenes=False)

    caption_common_sorted = sorted(caption_common, key=lambda r: float(r["mean_common_weighted_score"]), reverse=True)
    caption_models_sorted = sorted(caption_models, key=lambda r: float(r["mean_weighted_caption_usage_score"]), reverse=True)

    cover = ASSET_DIR / "cover.png"
    caption_bar = ASSET_DIR / "caption_weighted_score.png"
    caption_heatmap = ASSET_DIR / "caption_scene_heatmap.png"
    category_heatmap = ASSET_DIR / "caption_category_heatmap.png"
    fvd_bar = ASSET_DIR / "fvd_bar.png"
    fvd_grouped = ASSET_DIR / "fvd_quality_grouped.png"
    traj_bar = ASSET_DIR / "trajectory_ranking.png"
    traj_ltx = ASSET_DIR / "ltx_trajectory_comparison.png"
    contact_static = ASSET_DIR / "contact_static_cloudy.png"
    contact_hard = ASSET_DIR / "contact_se_000237.png"
    traj_paths = ASSET_DIR / "trajectory_paths_static_cloudy.png"

    save_cover(cover, caption_common_sorted, traj_common, fvd_rank)
    save_bar_chart(
        caption_bar,
        "Caption Following Leaderboard",
        [(str(r["model"]), float(r["mean_weighted_caption_usage_score"])) for r in caption_models_sorted],
        "Mean weighted caption usage score across available evaluated scenes.",
        lower_better=False,
        suffix="",
    )
    save_caption_scene_heatmap(caption_heatmap, common_scenes, caption_scenes)
    save_category_heatmap(category_heatmap, caption_categories)
    save_bar_chart(
        fvd_bar,
        "Raw FVD",
        [(str(r["model"]), float(r["fvd"])) for r in sorted(fvd_rows, key=lambda r: float(r["fvd"]))],
        "Lower is better. WAN2.2-5B has no results.txt in the current benchmark tree.",
        lower_better=True,
    )
    save_grouped_metric_chart(
        fvd_grouped,
        "Quality and Consistency Metrics",
        sorted(fvd_rows, key=lambda r: display_model(str(r["model"]))),
        [
            ("objective_quality", "Objective quality"),
            ("subjective_quality", "Subjective quality"),
            ("scene_consistency", "Scene consistency"),
            ("agent_consistency", "Agent consistency"),
        ],
    )
    save_bar_chart(
        traj_bar,
        "Ego-Trajectory Ranking",
        [(str(r["model"]), float(r["mean_combined_endpoint_distance_rel_error"])) for r in traj_common],
        "Common-scene ranking. Lower combines path distance error and endpoint-vector error.",
        lower_better=True,
        width=1550,
    )
    save_ltx_trajectory_comparison(traj_ltx, traj_common_scenes)
    static_scene = "wide_road+normal+cloudy+Night+cdc81ce53be615a7"
    hard_scene = "SE+000237"
    save_contact_sheet(contact_static, static_scene, "Visual Sample: Near-Static Night Scene")
    save_contact_sheet(contact_hard, hard_scene, "Visual Sample: Hard Caption Scene")
    save_trajectory_path_panel(traj_paths, static_scene)

    doc = DocxWriter()
    doc.image(cover, width_in=7.0)
    doc.paragraph("DrivingGen Benchmark Report", style="Title", align="center")
    doc.paragraph(
        "A consolidated analysis of caption following, video quality/FVD-style results, and ego-trajectory adherence for the current 10-sample benchmark subset.",
        align="center",
    )
    doc.page_break()

    doc.paragraph("Executive Summary", style="Heading1")
    doc.bullet("LTX-2B is the strongest balanced model in this benchmark slice. It leads the ego-trajectory ranking, is the best complete-coverage caption model, and wins the aggregate ranking over the available results.txt metrics.")
    doc.bullet("NVIDIA Cosmos 2.5-2B is highly competitive on caption following and leads the common-scene caption mean, but it is missing one caption scene and some trajectory coverage.")
    doc.bullet("LTX-13B has the best raw FVD, but it loses the aggregate quality/consistency ranking and has severe near-static ego-trajectory drift on two night scenes.")
    doc.bullet("Caption following is easiest for ego-view and environment claims; action and traffic-agent claims remain the most fragile categories.")
    doc.bullet("The current results are from 10 scenes, so the model ordering should be treated as a high-signal early benchmark rather than a final 400-sample conclusion.")

    doc.paragraph("At-a-Glance Leaderboard", style="Heading2")
    doc.table(
        table_rows(
            ["Dimension", "Best model", "Notes"],
            [
                ["Caption following, complete coverage", caption_models_sorted[0]["model"], f"{float(caption_models_sorted[0]['mean_weighted_caption_usage_score']):.2f} mean weighted score over {caption_models_sorted[0]['num_scenes']} scenes"],
                ["Caption following, common scenes", caption_common_sorted[0]["model"], f"{float(caption_common_sorted[0]['mean_common_weighted_score']):.2f} mean weighted score"],
                ["Ego trajectory", traj_common[0]["model"], f"{float(traj_common[0]['mean_combined_endpoint_distance_rel_error']):.3f} mean relative endpoint-distance error"],
                ["FVD/quality aggregate", fvd_rank[0]["model"], f"{float(fvd_rank[0]['avg_rank']):.2f} average rank across usable results.txt metrics"],
                ["Raw FVD only", min(fvd_rows, key=lambda r: float(r["fvd"]))["model"], f"{min(float(r['fvd']) for r in fvd_rows):.2f} FVD"],
            ],
        ),
        widths=[2900, 3000, 5200],
    )

    doc.paragraph("Caption Following", style="Heading1")
    doc.paragraph(
        "Caption usage score measures whether visible video content follows decomposed caption claims. The weighted score emphasizes more important claims. NVIDIA and LTX-2B are essentially tied on shared coverage, while LTX-2B is the strongest model with all 10 scenes present."
    )
    doc.image(caption_bar, width_in=7.0, caption="Mean weighted caption usage score by model.")
    doc.table(
        table_rows(
            ["Model", "Scenes", "Mean CUS", "Std CUS", "Mean weighted CUS"],
            [
                [
                    r["model"],
                    r["num_scenes"],
                    pct(float(r["mean_caption_usage_score"])),
                    pct(float(r["std_caption_usage_score"])),
                    pct(float(r["mean_weighted_caption_usage_score"])),
                ]
                for r in caption_models_sorted
            ],
        ),
        widths=[2900, 1200, 1600, 1500, 2200],
    )
    doc.image(caption_heatmap, width_in=7.1, caption="Per-scene weighted caption usage scores on the common scene subset.")
    doc.image(category_heatmap, width_in=6.4, caption="Caption category means. Environment and ego-view claims are consistently easier than action/agent claims.")

    category_overall = []
    for cat in ["actions", "ego_vehicle", "environment", "traffic_agents"]:
        vals = []
        for row in caption_scenes:
            value = dict(row["category_scores"]).get(cat)
            if value is not None:
                vals.append(float(value))
        category_overall.append([cat, pct(mean(vals))])
    doc.paragraph("Category-Level Trend", style="Heading2")
    doc.table(table_rows(["Category", "Mean score across model-scenes"], category_overall), widths=[3200, 3200])

    doc.paragraph("Hardest Caption Scenes", style="Heading2")
    doc.table(
        table_rows(
            ["Scene", "Mean weighted score", "Best model", "Best score", "Models present"],
            [
                [
                    shorten_scene(str(r["scene"]), 58),
                    pct(float(r["mean_weighted_score"])),
                    r["best_model"],
                    pct(float(r["best_score"])),
                    r["num_models"],
                ]
                for r in scene_difficulty[:5]
            ],
        ),
        widths=[4300, 2100, 2600, 1400, 1400],
    )

    doc.paragraph("Video Quality and Consistency Results", style="Heading1")
    doc.paragraph(
        "The results.txt files are available for four models. WAN2.2-5B is not included here because no matching results.txt was present. FVD alone favors LTX-13B, but the aggregate rank across FVD, objective quality, subjective quality, scene consistency, and agent consistency favors LTX-2B."
    )
    doc.image(fvd_bar, width_in=7.0, caption="Raw FVD. Lower is better.")
    doc.image(fvd_grouped, width_in=7.0, caption="Quality and consistency metrics from results.txt.")
    doc.table(
        table_rows(
            ["Rank", "Model", "Avg rank", "FVD", "Objective", "Subjective", "Scene cons.", "Agent cons."],
            [
                [
                    r["rank"],
                    r["model"],
                    f"{float(r['avg_rank']):.2f}",
                    f"{float(r['raw_fvd']):.2f}",
                    f"{float(r['objective_quality']):.3f}",
                    f"{float(r['subjective_quality']):.3f}",
                    f"{float(r['scene_consistency']):.3f}",
                    f"{float(r['agent_consistency']):.3f}",
                ]
                for r in fvd_rank
            ],
        ),
        widths=[750, 2700, 1200, 1450, 1300, 1400, 1400, 1400],
    )

    doc.paragraph("Ego-Trajectory Adherence", style="Heading1")
    doc.paragraph(
        "The ego-trajectory metric compares GT ego motion against UniDepth-estimated generated motion after converting both to a shared [forward, lateral-left] coordinate frame. No best-fit rotation or scale is applied by default, so forward/backward and left/right sign errors remain visible. The default ranking metric combines total travelled-distance error with endpoint-vector error."
    )
    doc.image(traj_bar, width_in=7.0, caption="Common-scene ego-trajectory ranking. Lower is better.")
    doc.table(
        table_rows(
            ["Rank", "Model", "Scenes", "Combined rel. error", "Distance rel.", "Endpoint rel.", "Endpoint error m", "Forward error m", "Lateral error m"],
            [
                [
                    r["rank"],
                    r["model"],
                    r["num_scenes"],
                    f"{float(r['mean_combined_endpoint_distance_rel_error']):.3f}",
                    f"{float(r['mean_distance_abs_rel_error']):.3f}",
                    f"{float(r['mean_endpoint_rel_error']):.3f}",
                    f"{float(r['mean_endpoint_error_m']):.2f}",
                    f"{float(r['mean_forward_error_m']):.2f}",
                    f"{float(r['mean_lateral_left_error_m']):.2f}",
                ]
                for r in traj_common
            ],
        ),
        widths=[700, 2500, 900, 1600, 1300, 1300, 1450, 1450, 1450],
    )
    doc.paragraph("All-Available Trajectory Coverage", style="Heading2")
    doc.paragraph(
        "The all-available trajectory ranking has uneven scene counts but preserves the same ordering at the top. This is useful as a coverage diagnostic, while the common-scene table above is the fairer model-to-model comparison."
    )
    doc.table(
        table_rows(
            ["Rank", "Model", "Scenes", "Combined rel. error", "Endpoint error m"],
            [
                [
                    r["rank"],
                    r["model"],
                    r["num_scenes"],
                    f"{float(r['mean_combined_endpoint_distance_rel_error']):.3f}",
                    f"{float(r['mean_endpoint_error_m']):.2f}",
                ]
                for r in traj_all
            ],
        ),
        widths=[800, 3300, 1000, 2200, 1800],
    )
    doc.image(traj_ltx, width_in=7.0, caption="LTX-2B versus LTX-13B per-scene trajectory error.")
    doc.paragraph(
        "The biggest LTX-13B weakness is not a uniform loss on every scene. It is a pair of near-static night scenes where the GT ego motion is roughly 0.2 to 0.3 m, while LTX-13B produces 11 to 16 m of apparent motion. LTX-2B stays near stationary on those scenes, which heavily improves its aggregate trajectory score."
    )
    doc.image(traj_paths, width_in=7.0, caption="GT versus predicted trajectory shape for the near-static cloudy night scene.")

    doc.paragraph("Qualitative Visual References", style="Heading1")
    doc.paragraph(
        "The report uses repository assets directly: GT condition frames from drivinggen_input_dataset and generated video frames from benchmark_results. These thumbnails are not used as metrics; they are included to make the failure modes easier to inspect."
    )
    doc.image(contact_static, width_in=7.2, caption="Near-static night scene where LTX-13B shows large trajectory drift.")
    doc.image(contact_hard, width_in=7.2, caption="SE+000237, one of the hardest caption-following scenes.")

    doc.paragraph("Interpretation", style="Heading1")
    doc.bullet("LTX-2B dominates the combined picture because it is stable across all three axes measured so far: caption following, quality/consistency, and ego-trajectory adherence.")
    doc.bullet("LTX-13B still has strengths: it has the best raw FVD and wins or remains close on some trajectory scenes. Its average drops because a small number of low-motion cases fail badly.")
    doc.bullet("NVIDIA Cosmos 2.5-2B deserves attention for caption fidelity. It is top on common caption scenes and performs well on difficult dynamic captions, but missing coverage prevents it from being the best complete model here.")
    doc.bullet("Vista-2B and WAN2.2-5B are not uniformly bad, but their aggregate performance is less stable. Vista has strong objective quality in results.txt but weaker subjective/agent consistency and trajectory behavior.")

    doc.paragraph("Method Notes and Caveats", style="Heading1")
    doc.bullet("Only 10 of the planned 400 samples are represented. The report should be treated as an early benchmark readout.")
    doc.bullet("Caption following depends on the caption_eval pipeline and its verifier mix. Category trends are more reliable than any single claim outcome.")
    doc.bullet("Trajectory rankings use UniDepth-estimated generated ego trajectories, so they measure both generated video adherence and downstream SLAM recoverability.")
    doc.bullet("FVD/quality results.txt was not available for WAN2.2-5B in the current benchmark tree, so the quality aggregate ranking covers four models.")
    doc.bullet("For trajectory evaluation, the common-scene view is the primary fair comparison; all-available ranking is included only as a coverage-aware supplement.")

    doc.paragraph("Generated Files", style="Heading1")
    doc.table(
        table_rows(
            ["Artifact", "Path"],
            [
                ["DOCX report", str(REPORT_PATH.relative_to(REPO_ROOT))],
                ["Report charts/assets", str(ASSET_DIR.relative_to(REPO_ROOT))],
                ["Trajectory ranking CSV", "benchmark_results/ego_trajectory_distance_ranking/model_ranking.csv"],
                ["Trajectory scene metrics CSV", "benchmark_results/ego_trajectory_distance_ranking/scene_metrics.csv"],
            ],
        ),
        widths=[2600, 6500],
    )

    doc.save(REPORT_PATH)


def main() -> int:
    build_report()
    print(f"Wrote {REPORT_PATH}")
    print(f"Assets: {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
