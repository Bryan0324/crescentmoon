"""Figures for the report: reward curves, confidence curves, method comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

__all__ = [
    "draw_detections",
    "plot_method_comparison",
    "plot_confidence_curves",
    "plot_training_curve",
    "plot_attack_example",
    "plot_search_progress",
]

PALETTE = {"random": "#7f8fa6", "greedy": "#e1a13b", "ppo": "#3b7dd8"}


def _save(fig, out_path: str | Path, tight: bool = True) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _colour(name: str) -> str:
    return PALETTE.get(name.split("/")[0], "#5d6d7e")


def draw_detections(image_rgb: np.ndarray, detections, highlight: str | None = None) -> np.ndarray:
    """Overlay detection boxes + confidences on a copy of the image."""
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        colour = (255, 70, 70) if (highlight and det.cls_name == highlight) else (70, 200, 255)
        draw.rectangle(det.bbox, outline=colour, width=3)
        label = f"{det.cls_name} {det.confidence:.2f}"
        x1, y1 = det.bbox[0], det.bbox[1]
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label), y1], fill=colour)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


def plot_method_comparison(summaries: Sequence[dict[str, Any]], out_path: str | Path, title: str = ""):
    """Grouped bars: success rate / confidence drop / mean reward / episode length."""
    labels = [s.get("label", s["method"]) for s in summaries]
    metrics = [
        ("attack_success_rate", "Attack success rate", None),
        ("mean_confidence_drop", "Mean confidence drop", None),
        ("mean_reward", "Mean episode reward", None),
        ("mean_episode_length", "Mean episode length", None),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, (key, name, _) in zip(axes, metrics):
        values = [s.get(key, 0.0) for s in summaries]
        colours = [_colour(s["method"]) for s in summaries]
        bars = ax.bar(labels, values, color=colours)
        ax.set_title(name, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    if title:
        fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(top=0.84)
    return _save(fig, out_path)


def plot_confidence_curves(curves: dict[str, list[list[float]]], out_path: str | Path, baseline: float | None = None, title: str = ""):
    """Mean victim confidence over an episode, one line per method."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, episodes in curves.items():
        if not episodes:
            continue
        length = max(len(e) for e in episodes)
        padded = np.array(
            [e + [e[-1]] * (length - len(e)) if e else [np.nan] * length for e in episodes],
            dtype=float,
        )
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        xs = np.arange(1, length + 1)
        ax.plot(xs, mean, label=name, color=_colour(name), linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=_colour(name), alpha=0.15)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="#2c3e50", linewidth=1.2, label="baseline (clean)")
    ax.set_xlabel("step")
    ax.set_ylabel("victim confidence on target")
    ax.grid(alpha=0.25)
    ax.legend()
    if title:
        ax.set_title(title)
    return _save(fig, out_path)


def plot_training_curve(curve: dict[str, list], out_path: str | Path, title: str = "PPO training"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.asarray(curve.get("timesteps", []), dtype=float)
    ys = np.asarray(curve.get("returns", []), dtype=float)
    if len(ys):
        ax.plot(xs, ys, color="#b9c6d6", linewidth=1, label="episode return")
        window = max(1, len(ys) // 20)
        smooth = np.convolve(ys, np.ones(window) / window, mode="valid")
        ax.plot(xs[window - 1 :], smooth, color=PALETTE["ppo"], linewidth=2.2, label=f"moving avg ({window})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(title)
    return _save(fig, out_path)


def plot_attack_example(
    clean_rgb: np.ndarray,
    attacked_rgb: np.ndarray,
    clean_conf: float,
    attacked_conf: float,
    out_path: str | Path,
    title: str = "",
):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.6))
    for ax, img, caption in (
        (axes[0], clean_rgb, f"original scene\nvictim confidence = {clean_conf:.3f}"),
        (axes[1], attacked_rgb, f"attacked scene\nvictim confidence = {attacked_conf:.3f}"),
    ):
        ax.imshow(img)
        ax.set_title(caption, fontsize=11)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _save(fig, out_path, tight=False)
    return _save(fig, out_path)


def plot_search_progress(series: dict[str, Iterable[float]], out_path: str | Path, title: str = ""):
    """Best-so-far victim confidence as the search budget is spent (Stage 1)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, values in series.items():
        values = np.asarray(list(values), dtype=float)
        best_so_far = np.minimum.accumulate(values)
        ax.plot(np.arange(1, len(values) + 1), best_so_far, label=name, color=_colour(name), linewidth=2)
    ax.set_xlabel("queries to the environment (episodes)")
    ax.set_ylabel("best (lowest) victim confidence found")
    ax.grid(alpha=0.25)
    ax.legend()
    if title:
        ax.set_title(title)
    return _save(fig, out_path)
