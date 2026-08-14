#!/usr/bin/env python3
"""Render a calm, lightly-perspectival GitHub contribution height field.

The script has no runtime dependency beyond Pillow.  With no arguments it uses
deterministic demo data; --data accepts either CSV (date,count) or JSON.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter


TAU = math.tau
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RenderConfig:
    # Output
    width: int = 1000
    height: int = 360
    supersample: int = 2
    duration_seconds: float = 10.0
    fps: float = 12.5
    gif_colors: int = 128

    # Contribution field
    columns: int = 53
    rows: int = 7
    center_x: float = 500.0
    back_y: float = 143.0
    column_pitch: float = 15.0
    row_pitch: float = 22.0
    back_scale: float = 0.92
    front_scale: float = 1.015

    # Front-facing tile geometry
    cell_width: float = 10.4
    top_depth_x: float = 2.7       # max horizontal convergence at the field edges
    top_depth_y: float = 2.6
    top_back_width_scale: float = 0.93
    base_heights: tuple[float, ...] = (1.8, 4.2, 7.8, 12.4, 18.0)
    max_height_variation: float = 2.5

    # Wave field
    event_interval: float = 1.4
    wave_speed: float = 8.4       # contribution columns per second
    wave_lifetime: float = 2.85
    wave_band_width: float = 0.88
    wave_y_ratio: float = 0.40    # lower = more perspective-compressed ellipse
    wave_height: float = 13.5
    glow_strength: float = 0.72
    breath_height: float = 0.55

    # Dark UI palette, ordered like GitHub contribution levels
    background: RGB = (6, 11, 18)
    tile_faces: tuple[RGB, ...] = (
        (18, 27, 34),
        (10, 57, 39),
        (4, 101, 50),
        (22, 158, 70),
        (74, 224, 116),
    )
    wave_color: RGB = (73, 224, 132)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return float(value >= edge1)
    x = clamp((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def mix_color(a: RGB, b: RGB, amount: float) -> RGB:
    t = clamp(amount)
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def scale_color(color: RGB, factor: float) -> RGB:
    return tuple(round(clamp(channel * factor, 0, 255)) for channel in color)  # type: ignore[return-value]


def demo_contributions(config: RenderConfig, seed: int) -> list[list[int]]:
    """Create organic, repeatable GitHub-like activity without network access."""
    rng = random.Random(seed)
    counts = [[0 for _ in range(config.columns)] for _ in range(config.rows)]
    clusters = [(8, 2.0, 3.5), (21, 4.2, 5.5), (34, 1.7, 4.2), (45, 3.8, 3.0)]

    for column in range(config.columns):
        seasonal = 0.10 + 0.08 * (1.0 + math.sin(column * 0.30 - 1.0))
        cluster_energy = sum(
            amplitude * math.exp(-0.5 * ((column - center) / spread) ** 2)
            for center, amplitude, spread in clusters
        )
        weekly_focus = rng.uniform(0.70, 1.28)
        for row in range(config.rows):
            # GitHub rows are Sunday through Saturday (Mon/Wed/Fri are labelled).
            weekday_bias = (0.54, 0.88, 1.10, 1.22, 1.18, 1.06, 0.72)[row]
            activity = (seasonal + 0.105 * cluster_energy) * weekday_bias * weekly_focus
            activity += 0.13 if 24 <= column <= 28 and row in (1, 2, 3) else 0.0
            if rng.random() < clamp(activity, 0.05, 0.82):
                raw = 1.0 + 8.5 * activity + rng.gammavariate(1.5, 1.6)
                counts[row][column] = max(1, round(raw))
    return counts


def _records_to_grid(
    records: Iterable[tuple[date, int]], config: RenderConfig
) -> list[list[int]]:
    items = sorted(records, key=lambda item: item[0])
    grid = [[0 for _ in range(config.columns)] for _ in range(config.rows)]
    if not items:
        return grid

    latest = items[-1][0]
    # GitHub weeks run Sunday -> Saturday.  A rolling year normally occupies
    # 53 columns because its first and last weeks are only partially visible.
    final_saturday = latest + timedelta(days=(5 - latest.weekday()) % 7)
    first_sunday = final_saturday - timedelta(days=config.columns * 7 - 1)
    for day, count in items:
        offset = (day - first_sunday).days
        if 0 <= offset < config.columns * 7:
            github_row = (day.weekday() + 1) % 7  # Python Mon=0; GitHub Sun=0.
            grid[github_row][offset // 7] = max(0, int(count))
    return grid


def _parse_day(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def load_contributions(path: Path, config: RenderConfig) -> list[list[int]]:
    """Load JSON matrix/records or a CSV containing date and count columns."""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            records = [(_parse_day(row["date"]), int(row["count"])) for row in rows]
        return _records_to_grid(records, config)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("contributions", payload.get("data", payload))

    if (
        isinstance(payload, list)
        and len(payload) == config.rows
        and all(isinstance(row, list) for row in payload)
    ):
        widths = {len(row) for row in payload}
        if len(widths) != 1 or next(iter(widths)) not in (config.columns - 1, config.columns):
            raise ValueError(
                f"JSON matrix must be {config.rows} x {config.columns} "
                f"(or legacy {config.rows} x {config.columns - 1})"
            )
        matrix = [[max(0, int(value)) for value in row] for row in payload]
        if len(matrix[0]) == config.columns - 1:
            matrix = [[0, *row] for row in matrix]
        return matrix

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        records = [
            (_parse_day(str(item["date"])), int(item.get("count", item.get("contributionCount", 0))))
            for item in payload
        ]
        return _records_to_grid(records, config)

    raise ValueError(
        f"Expected a {config.rows}x{config.columns} JSON matrix or records with date/count fields"
    )


def contribution_levels(counts: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[float]]]:
    peak = max((value for row in counts for value in row), default=0)
    levels: list[list[int]] = []
    strengths: list[list[float]] = []
    denominator = math.log1p(peak) if peak else 1.0
    for row in counts:
        level_row: list[int] = []
        strength_row: list[float] = []
        for value in row:
            strength = math.log1p(value) / denominator if value > 0 else 0.0
            level = min(4, max(1, math.ceil(strength * 4.0))) if value > 0 else 0
            level_row.append(level)
            strength_row.append(strength)
        levels.append(level_row)
        strengths.append(strength_row)
    return levels, strengths


@dataclass(frozen=True)
class WaveEvent:
    time: float
    column: float
    row: float
    energy: float


class ContributionWaveRenderer:
    def __init__(self, counts: Sequence[Sequence[int]], config: RenderConfig, seed: int = 19):
        self.config = config
        self.counts = counts
        self.levels, self.strengths = contribution_levels(counts)
        self.rng = random.Random(seed)
        self.events = self._make_events(seed + 101)
        self.particles = self._make_particles(seed + 211)
        self.background = self._make_background()

    @property
    def frame_count(self) -> int:
        return round(self.config.duration_seconds * self.config.fps)

    def _make_events(self, seed: int) -> list[WaveEvent]:
        rng = random.Random(seed)
        count = max(1, round(self.config.duration_seconds / self.config.event_interval))
        # Spread centers through the field; fixed centers make the loop deterministic.
        anchors = [0.13, 0.72, 0.42, 0.86, 0.27, 0.59, 0.08, 0.78, 0.48, 0.93]
        events = []
        for index in range(count):
            fraction = anchors[index % len(anchors)]
            column = 4.0 + fraction * (self.config.columns - 9.0)
            column += rng.uniform(-1.5, 1.5)
            row = rng.uniform(1.1, self.config.rows - 1.3)
            events.append(
                WaveEvent(
                    time=index * self.config.duration_seconds / count,
                    column=column,
                    row=row,
                    energy=rng.uniform(0.86, 1.08),
                )
            )
        return events

    def _make_particles(self, seed: int) -> list[tuple[float, float, float, float]]:
        rng = random.Random(seed)
        particles = []
        for _ in range(42):
            x = rng.uniform(36.0, self.config.width - 36.0)
            y = rng.uniform(28.0, self.config.height * 0.77)
            size = rng.choice((0.45, 0.55, 0.7, 0.9))
            phase = rng.random() * TAU
            particles.append((x, y, size, phase))
        return particles

    def _row_scale(self, row: float) -> float:
        amount = row / max(1, self.config.rows - 1)
        return self.config.back_scale + (self.config.front_scale - self.config.back_scale) * amount

    def _cell_position(self, column: float, row: float) -> tuple[float, float, float]:
        scale = self._row_scale(row)
        centered_column = column - (self.config.columns - 1) * 0.5
        x = self.config.center_x + centered_column * self.config.column_pitch * scale
        y = self.config.back_y + row * self.config.row_pitch
        return x, y, scale

    def _make_background(self) -> Image.Image:
        config = self.config
        image = Image.new("RGB", (config.width, config.height), config.background)
        pixels = image.load()
        for y in range(config.height):
            vertical = y / max(1, config.height - 1)
            for x in range(config.width):
                # A restrained blue-green aura behind the field and a dark vignette.
                nx = (x - config.width * 0.50) / (config.width * 0.58)
                ny = (y - config.height * 0.58) / (config.height * 0.72)
                aura = math.exp(-2.7 * (nx * nx + ny * ny))
                edge = clamp((nx * nx + (ny * 0.78) ** 2) * 0.34)
                grain = ((x * 17 + y * 29) % 19) / 19.0 - 0.5
                r = config.background[0] + 1.2 * aura - 2.2 * edge + grain * 0.28
                g = config.background[1] + 8.0 * aura - 2.5 * edge + grain * 0.35
                b = config.background[2] + 10.5 * aura - 2.0 * edge + grain * 0.55
                pixels[x, y] = (
                    round(clamp(r, 0, 255)),
                    round(clamp(g, 0, 255)),
                    round(clamp(b, 0, 255)),
                )

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for x in range(44, config.width, 64):
            draw.line((x, 26, x, config.height - 22), fill=(57, 89, 102, 7), width=1)
        for y in range(38, config.height, 52):
            draw.line((28, y, config.width - 28, y), fill=(57, 89, 102, 6), width=1)

        back_half = config.column_pitch * config.columns * config.back_scale * 0.5
        front_half = config.column_pitch * config.columns * config.front_scale * 0.5
        field = [
            (config.center_x - back_half - 12, config.back_y - 10),
            (config.center_x + back_half + 12, config.back_y - 10),
            (config.center_x + front_half + 18, config.back_y + (config.rows - 1) * config.row_pitch + 12),
            (config.center_x - front_half - 18, config.back_y + (config.rows - 1) * config.row_pitch + 12),
        ]
        draw.polygon(field, fill=(8, 28, 30, 34), outline=(69, 118, 111, 22))
        return Image.alpha_composite(image.convert("RGBA"), overlay)

    def _active_waves(self, time_value: float) -> list[tuple[WaveEvent, float, float]]:
        active = []
        for event in self.events:
            age = (time_value - event.time) % self.config.duration_seconds
            if age < self.config.wave_lifetime:
                fade = 1.0 - smoothstep(
                    self.config.wave_lifetime * 0.66,
                    self.config.wave_lifetime,
                    age,
                )
                attack = smoothstep(0.0, 0.14, age)
                active.append((event, age, fade * attack))
        return active

    def _wave_response(
        self, column: int, row: int, active: Sequence[tuple[WaveEvent, float, float]]
    ) -> tuple[float, float]:
        config = self.config
        responses: list[float] = []
        for event, age, envelope in active:
            dx = column - event.column
            dy_screen = (row - event.row) * config.row_pitch
            dy_as_columns = dy_screen / (config.column_pitch * config.wave_y_ratio)
            distance = math.hypot(dx, dy_as_columns)
            radius = age * config.wave_speed
            ring = math.exp(-0.5 * ((distance - radius) / config.wave_band_width) ** 2)
            core = math.exp(-0.5 * (distance / 1.45) ** 2) * math.exp(-age / 0.48)
            responses.append(clamp((0.92 * ring + 0.42 * core) * envelope * event.energy))
        if not responses:
            return 0.0, 0.0
        peak = max(responses)
        lift = min(1.22, peak + 0.15 * sum(response for response in responses if response != peak))
        return lift, peak

    def _draw_particles(self, image: Image.Image, time_value: float) -> None:
        scale = self.config.supersample
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for x, y, size, phase in self.particles:
            shimmer = 0.5 + 0.5 * math.sin(TAU * time_value / self.config.duration_seconds + phase)
            alpha = round(5 + 13 * shimmer)
            radius = max(1, round(size * scale))
            cx, cy = round(x * scale), round(y * scale)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(95, 161, 145, alpha))
        image.alpha_composite(layer)

    def _draw_wave_rings(
        self, image: Image.Image, active: Sequence[tuple[WaveEvent, float, float]]
    ) -> None:
        config = self.config
        ss = config.supersample
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        crisp = Image.new("RGBA", image.size, (0, 0, 0, 0))
        crisp_draw = ImageDraw.Draw(crisp)
        for event, age, envelope in active:
            center_x, center_y, center_scale = self._cell_position(event.column, event.row)
            radius_x = age * config.wave_speed * config.column_pitch * center_scale
            radius_y = radius_x * config.wave_y_ratio
            if radius_x < 1.0:
                continue
            box = tuple(
                round(value * ss)
                for value in (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                )
            )
            glow_alpha = round(42 * envelope * event.energy * config.glow_strength)
            line_alpha = round(74 * envelope * event.energy)
            glow_draw.ellipse(box, outline=(*config.wave_color, glow_alpha), width=max(2, round(4.5 * ss)))
            crisp_draw.ellipse(box, outline=(*config.wave_color, line_alpha), width=max(1, round(0.78 * ss)))

        glow = glow.filter(ImageFilter.GaussianBlur(4.2 * ss))
        image.alpha_composite(glow)
        image.alpha_composite(crisp)

    def _draw_tiles(
        self,
        image: Image.Image,
        time_value: float,
        active: Sequence[tuple[WaveEvent, float, float]],
    ) -> None:
        config = self.config
        ss = config.supersample
        shadows = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadows)
        cell_glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(cell_glow)

        geometry: list[tuple[int, int, float, float, float, float, float]] = []
        for row in range(config.rows):
            for column in range(config.columns):
                x, y, scale = self._cell_position(column, row)
                level = self.levels[row][column]
                strength = self.strengths[row][column]
                wave_lift, wave_peak = self._wave_response(column, row, active)
                base = config.base_heights[level]
                base += strength * config.max_height_variation
                breath = config.breath_height * math.sin(
                    TAU * time_value / config.duration_seconds + column * 0.075 + row * 0.31
                )
                height = max(1.2, base + breath + config.wave_height * wave_lift * (0.78 + 0.22 * strength))
                width = config.cell_width * scale
                geometry.append((row, column, x, y, scale, height, wave_peak))

                shadow_draw.ellipse(
                    (
                        round((x - width * 0.62) * ss),
                        round((y - 1.0) * ss),
                        round((x + width * 0.82) * ss),
                        round((y + 4.2) * ss),
                    ),
                    fill=(0, 0, 0, 46),
                )
                if wave_peak > 0.035:
                    alpha = round(64 * wave_peak * config.glow_strength)
                    glow_draw.rounded_rectangle(
                        (
                            round((x - width * 0.8) * ss),
                            round((y - height - 4.0) * ss),
                            round((x + width * 0.9) * ss),
                            round((y + 3.0) * ss),
                        ),
                        radius=max(1, round(2.2 * ss)),
                        fill=(*config.wave_color, alpha),
                    )

        shadows = shadows.filter(ImageFilter.GaussianBlur(2.2 * ss))
        cell_glow = cell_glow.filter(ImageFilter.GaussianBlur(4.0 * ss))
        image.alpha_composite(shadows)
        image.alpha_composite(cell_glow)

        draw = ImageDraw.Draw(image, "RGBA")
        for row, column, x, y, scale, height, wave_peak in geometry:
            level = self.levels[row][column]
            width = config.cell_width * scale
            field_half_width = config.column_pitch * (config.columns - 1) * 0.5 * scale
            horizontal_position = clamp(
                (x - config.center_x) / max(1.0, field_half_width), -1.0, 1.0
            )
            # Every tile converges toward the same central vanishing point:
            # left tiles recede right, right tiles recede left, center tiles stay neutral.
            depth_x = -horizontal_position * config.top_depth_x * scale
            depth_y = config.top_depth_y * scale
            left, right = x - width * 0.5, x + width * 0.5
            top = y - height
            back_width = width * config.top_back_width_scale
            back_left = x + depth_x - back_width * 0.5
            back_right = x + depth_x + back_width * 0.5
            back_top = top - depth_y
            back_base = y - depth_y

            front = config.tile_faces[level]
            front = mix_color(front, config.wave_color, wave_peak * 0.38)
            top_color = mix_color(scale_color(front, 1.22), (117, 238, 164), wave_peak * 0.28)
            side_color = scale_color(front, 0.66)
            outline = mix_color(scale_color(front, 1.33), config.wave_color, wave_peak * 0.45)

            def points(values: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
                return [(round(px * ss), round(py * ss)) for px, py in values]

            # The face stays almost screen-aligned.  Only a narrow top/right reveal
            # supplies the lightweight volume cue.
            if depth_x > 0.08:
                draw.polygon(
                    points(
                        [
                            (right, top),
                            (back_right, back_top),
                            (back_right, back_base),
                            (right, y),
                        ]
                    ),
                    fill=(*side_color, 255),
                )
            elif depth_x < -0.08:
                draw.polygon(
                    points(
                        [
                            (left, top),
                            (back_left, back_top),
                            (back_left, back_base),
                            (left, y),
                        ]
                    ),
                    fill=(*side_color, 255),
                )
            draw.rounded_rectangle(
                (round(left * ss), round(top * ss), round(right * ss), round(y * ss)),
                radius=max(1, round(0.75 * ss)),
                fill=(*front, 255),
                outline=(*outline, 122),
                width=max(1, round(0.45 * ss)),
            )
            draw.polygon(
                points(
                    [
                        (left, top),
                        (right, top),
                        (back_right, back_top),
                        (back_left, back_top),
                    ]
                ),
                fill=(*top_color, 255),
                outline=(*outline, 126),
            )

    def render_frame(self, frame_index: int) -> Image.Image:
        config = self.config
        time_value = frame_index * config.duration_seconds / self.frame_count
        size = (config.width * config.supersample, config.height * config.supersample)
        image = self.background.resize(size, Image.Resampling.BICUBIC)
        self._draw_particles(image, time_value)
        active = self._active_waves(time_value)
        self._draw_wave_rings(image, active)
        self._draw_tiles(image, time_value, active)
        return image.convert("RGB").resize(
            (config.width, config.height), Image.Resampling.LANCZOS
        )

    def _global_palette(self) -> Image.Image:
        sample_count = 8
        thumb_size = (self.config.width // 4, self.config.height // 4)
        sheet = Image.new("RGB", (thumb_size[0] * sample_count, thumb_size[1]))
        for index in range(sample_count):
            frame_index = round(index * self.frame_count / sample_count) % self.frame_count
            sample = self.render_frame(frame_index).resize(thumb_size, Image.Resampling.LANCZOS)
            sheet.paste(sample, (index * thumb_size[0], 0))
        return sheet.quantize(colors=self.config.gif_colors, method=Image.Quantize.MEDIANCUT)

    def save_gif(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        palette = self._global_palette()
        frames: list[Image.Image] = []
        for index in range(self.frame_count):
            frame = self.render_frame(index)
            frames.append(
                frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            )
            if (index + 1) % 10 == 0 or index + 1 == self.frame_count:
                print(f"Rendered {index + 1:>3}/{self.frame_count} frames", end="\r", flush=True)
        print()

        frame_duration = max(20, round((1000.0 / self.config.fps) / 10.0) * 10)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            disposal=1,
            optimize=True,
            comment=b"Contribution Wave - generated with Pillow",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("contribution-wave.gif"))
    parser.add_argument("--data", type=Path, help="CSV date,count or JSON matrix/records")
    parser.add_argument("--seed", type=int, default=19, help="Demo data and layout seed")
    parser.add_argument("--duration", type=float, default=10.0, help="Loop duration in seconds")
    parser.add_argument("--fps", type=float, default=12.5, help="Frames per second")
    parser.add_argument("--width", type=int, default=1000, help="Output width")
    parser.add_argument("--height", type=int, default=360, help="Output height")
    parser.add_argument("--preview", type=Path, help="Also save a representative PNG frame")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width < 640 or args.height < 260:
        raise ValueError("Use at least 640x260 so the 52x7 field remains legible")

    scale_x = args.width / 1000.0
    scale_y = args.height / 360.0
    config = replace(
        RenderConfig(),
        width=args.width,
        height=args.height,
        duration_seconds=args.duration,
        fps=args.fps,
        center_x=500.0 * scale_x,
        back_y=143.0 * scale_y,
        column_pitch=15.0 * scale_x,
        row_pitch=22.0 * scale_y,
        cell_width=10.4 * scale_x,
        top_depth_x=2.7 * scale_x,
        top_depth_y=2.6 * scale_y,
        base_heights=tuple(value * scale_y for value in RenderConfig().base_heights),
        max_height_variation=2.5 * scale_y,
        wave_height=13.5 * scale_y,
    )

    counts = load_contributions(args.data, config) if args.data else demo_contributions(config, args.seed)
    renderer = ContributionWaveRenderer(counts, config, seed=args.seed)
    renderer.save_gif(args.output)
    if args.preview:
        renderer.render_frame(round(renderer.frame_count * 0.42)).save(args.preview)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Saved {args.output} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
