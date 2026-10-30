"""Assemble the reproducible one-minute submission video from Genesis recordings."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (960, 540)
FONT_REGULAR = Path(r"C:\Windows\Fonts\BIZ-UDGothicR.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\BIZ-UDGothicB.ttc")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.exists():
        raise FileNotFoundError(f"Japanese font not found: {path}")
    return ImageFont.truetype(str(path), size=size)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text(((FRAME_SIZE[0] - width) / 2, y), text, font=font, fill=fill)
    return y + box[3] - box[1]


def create_card(path: Path, title: str, lines: list[str], accent: tuple[int, int, int]) -> None:
    image = Image.new("RGB", FRAME_SIZE, (11, 20, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, FRAME_SIZE[1]), fill=accent)
    draw.rectangle((72, 116, 888, 120), fill=accent)

    y = 55
    y = _centered_text(draw, y, title, _font(46, bold=True), (245, 248, 252)) + 72
    for line in lines:
        y = _centered_text(draw, y, line, _font(27), (213, 222, 235)) + 28

    footer = "Genesis 1.2.1 / CPU / seed 21"
    _centered_text(draw, 488, footer, _font(18), (140, 158, 180))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_overlay(
    path: Path,
    title: str,
    detail: str,
    robot_detail: str,
    accent: tuple[int, int, int],
) -> None:
    image = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 565, 112), radius=10, fill=(11, 20, 32, 218))
    draw.rectangle((18, 18, 27, 112), fill=accent + (255,))
    draw.text((44, 32), title, font=_font(27, bold=True), fill=(245, 248, 252, 255))
    draw.text((44, 75), detail, font=_font(17), fill=(210, 221, 234, 255))
    draw.rounded_rectangle((18, 462, 620, 522), radius=9, fill=(11, 20, 32, 218))
    draw.text(
        (38, 478),
        robot_detail,
        font=_font(18, bold=True),
        fill=(245, 248, 252, 255),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_ffmpeg(arguments: list[str]) -> None:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "warning"]
    subprocess.run(command + arguments, check=True)


def encode_card(image_path: Path, output_path: Path, duration: int) -> None:
    run_ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-t",
            str(duration),
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "12800",
            str(output_path),
        ]
    )


def normalize_video(input_path: Path, overlay_path: Path, output_path: Path) -> None:
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-i",
            str(overlay_path),
            "-filter_complex",
            "[0:v][1:v]overlay=0:0:format=auto",
            "-an",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "12800",
            str(output_path),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final AI goalkeeper comparison video")
    parser.add_argument(
        "--ai",
        type=Path,
        default=PROJECT_ROOT / "videos" / "goalkeeper_ai_raw.mp4",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "videos" / "goalkeeper_baseline_raw.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "videos" / "PAI_final_goalkeeper.mp4",
    )
    args = parser.parse_args()
    for input_path in (args.ai, args.baseline):
        if not input_path.exists():
            parser.error(f"input video not found: {input_path}")

    working = PROJECT_ROOT / "tmp" / "submission-video"
    working.mkdir(parents=True, exist_ok=True)
    cards = [
        (
            "title",
            "AIロボットゴールキーパー",
            ["非線形なボール軌道を予測して迎撃", "Physical AI 応用1 最終課題"],
            (41, 151, 255),
            6,
        ),
        (
            "method",
            "仕組み",
            ["4時点の座標を小型MLPへ入力", "ゴール通過位置を予測し、直動ロボットを制御"],
            (129, 104, 255),
            7,
        ),
        (
            "ai",
            "AI予測",
            ["同一条件の10球に挑戦", "赤: 予測地点 / 青: ゴールキーパー"],
            (32, 201, 151),
            4,
        ),
        (
            "baseline",
            "ベースライン: 等速予測",
            ["AIと同じ10球で比較", "スピンと抗力による曲がりを考慮しない"],
            (245, 158, 11),
            4,
        ),
        (
            "summary",
            "比較結果",
            ["10球: AI 10成功 / 等速予測 1成功", "検証MAE: AI 0.078 m / 等速予測 0.388 m"],
            (32, 201, 151),
            10,
        ),
    ]

    encoded_cards: dict[str, Path] = {}
    for name, title, lines, accent, duration in cards:
        image_path = working / f"{name}.png"
        video_path = working / f"card_{name}.mp4"
        create_card(image_path, title, lines, accent)
        encode_card(image_path, video_path, duration)
        encoded_cards[name] = video_path

    ai_normalized = working / "ai.mp4"
    baseline_normalized = working / "baseline.mp4"
    ai_overlay = working / "ai-overlay.png"
    baseline_overlay = working / "baseline-overlay.png"
    create_overlay(
        ai_overlay,
        "AI: MLP軌道予測",
        "入力: 4時点のXY座標 → 出力: 赤い予測地点",
        "ロボット: 1自由度直動関節をPD位置制御",
        (32, 201, 151),
    )
    create_overlay(
        baseline_overlay,
        "ベースライン: 等速予測",
        "AIなし / 抗力とスピンによる曲がりを考慮しない",
        "ロボット: 同じ1自由度直動関節をPD位置制御",
        (245, 158, 11),
    )
    normalize_video(args.ai, ai_overlay, ai_normalized)
    normalize_video(args.baseline, baseline_overlay, baseline_normalized)

    segments = [
        encoded_cards["title"],
        encoded_cards["method"],
        encoded_cards["ai"],
        ai_normalized,
        encoded_cards["baseline"],
        baseline_normalized,
        encoded_cards["summary"],
    ]
    concat_file = working / "segments.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segments),
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    )
    print(args.output)


if __name__ == "__main__":
    main()
