#!/usr/bin/env python3
"""
Video Export Module
===================

Model: minimax-m2.5 (opencode)
Created: 2026-02-18
Version: 1.0

Export fractal animations as MP4 or WebM video.

Usage:
    from animation.video_export import create_video, create_video_from_frames
    
    # From frames
    create_video("output/frames/", "output/video.mp4", fps=30)
    
    # With audio
    create_video_with_audio("output/frames/", "output/video.mp4", 
                           audio_path="music.mp3", fps=30)
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def create_video(
    frame_dir: str,
    output_path: str,
    fps: int = 30,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 23,
    pixel_format: str = "yuv420p"
) -> bool:
    """
    Create video from image frames using ffmpeg.
    
    Args:
        frame_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec (libx264, libvpx for webm, etc.)
        preset: Encoding preset (ultrafast, fast, medium, slow)
        crf: Constant Rate Factor (quality, lower = better)
        pixel_format: Output pixel format
    
    Returns:
        True if successful
    """
    if not check_ffmpeg():
        logger.error("ffmpeg not found. Install with: apt install ffmpeg")
        return False
    
    frame_path = Path(frame_dir)
    if not frame_path.exists():
        logger.error(f"Frame directory not found: {frame_dir}")
        return False
    
    # Get all PNG files sorted
    frames = sorted(frame_path.glob("*.png"))
    if not frames:
        logger.error(f"No PNG frames found in {frame_dir}")
        return False
    
    logger.info(f"Creating video from {len(frames)} frames...")
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", str(frame_path / "frame_%04d.png"),  # Input pattern
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", pixel_format,
        str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr}")
        return False
    
    logger.info(f"✓ Created video: {output_path}")
    return True


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: int = 30,
    codec: str = "libx264"
) -> bool:
    """
    Create video from list of frame paths.
    
    Args:
        frame_paths: List of frame image paths
        output_path: Output video file
        fps: Frames per second
        codec: Video codec
    
    Returns:
        True if successful
    """
    if not check_ffmpeg():
        logger.error("ffmpeg not found")
        return False
    
    if not frame_paths:
        logger.error("No frames provided")
        return False
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory with sequential frames
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Copy frames with sequential names
        for i, src in enumerate(frame_paths):
            dst = temp_path / f"frame_{i:04d}.png"
            shutil.copy2(src, dst)
        
        # Create video
        return create_video(str(temp_dir), output_path, fps, codec)
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def create_video_with_audio(
    frame_dir: str,
    output_path: str,
    audio_path: str,
    fps: int = 30,
    video_codec: str = "libx264",
    audio_codec: str = "aac"
) -> bool:
    """
    Create video with audio track.
    
    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        audio_path: Audio file path (mp3, wav, etc.)
        fps: Video frame rate
        video_codec: Video codec
        audio_codec: Audio codec
    
    Returns:
        True if successful
    """
    if not check_ffmpeg():
        logger.error("ffmpeg not found")
        return False
    
    frame_path = Path(frame_dir)
    audio_path = Path(audio_path)
    
    if not frame_path.exists():
        logger.error(f"Frame directory not found: {frame_dir}")
        return False
    
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return False
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get duration of audio for accurate video length
    duration_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        audio_duration = float(result.stdout.strip()) if result.returncode == 0 else None
    except:
        audio_duration = None
    
    logger.info(f"Creating video with audio ({len(list(frame_path.glob('*.png')))} frames)...")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frame_path / "frame_%04d.png"),
        "-i", str(audio_path),
        "-c:v", video_codec,
        "-c:a", audio_codec,
        "-shortest",  # Finish when shortest input ends
        "-pix_fmt", "yuv420p",
        str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr}")
        return False
    
    logger.info(f"✓ Created video with audio: {output_path}")
    return True


def create_gif_with_ffmpeg(
    frame_dir: str,
    output_path: str,
    fps: int = 15,
    loop: int = 0,
    palette: bool = True
) -> bool:
    """
    Create optimized GIF using ffmpeg with palette generation.
    
    Args:
        frame_dir: Directory containing frames
        output_path: Output GIF path
        fps: Frame rate
        loop: Number of loops (0 = infinite)
        palette: Use palette optimization for better quality
    
    Returns:
        True if successful
    """
    if not check_ffmpeg():
        logger.error("ffmpeg not found")
        return False
    
    frame_path = Path(frame_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if palette:
        # Two-pass: generate palette then create GIF
        palette_file = output_file.parent / "palette.png"
        
        # Generate palette
        cmd1 = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(frame_path / "frame_%04d.png"),
            "-vf", "palettegen",
            str(palette_file)
        ]
        
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Palette generation failed: {result.stderr}")
            return False
        
        # Create GIF with palette
        cmd2 = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(frame_path / "frame_%04d.png"),
            "-i", str(palette_file),
            "-lavfi", f"paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop", str(loop),
            str(output_file)
        ]
        
        result = subprocess.run(cmd2, capture_output=True, text=True)
        
        # Clean up palette
        if palette_file.exists():
            palette_file.unlink()
            
    else:
        # Simple GIF without palette optimization
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(frame_path / "frame_%04d.png"),
            "-loop", str(loop),
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"GIF creation failed: {result.stderr}")
        return False
    
    logger.info(f"✓ Created GIF: {output_path}")
    return True


def get_video_info(video_path: str) -> dict:
    """Get video file information."""
    if not os.path.exists(video_path):
        return {}
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            video_stream = next((s for s in data.get("streams", []) 
                               if s.get("codec_type") == "video"), {})
            
            return {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "width": video_stream.get("width", 0),
                "height": video_stream.get("height", 0),
                "codec": video_stream.get("codec_name", ""),
                "fps": eval(video_stream.get("r_frame_rate", "0/1"))
            }
    except:
        pass
    
    return {}


# Quick usage function
def quick_video(
    frame_dir: str,
    output_path: str = "output/video.mp4",
    fps: int = 30,
    with_audio: str = None
) -> bool:
    """Quick video creation from frames."""
    if with_audio:
        return create_video_with_audio(frame_dir, output_path, with_audio, fps)
    else:
        return create_video(frame_dir, output_path, fps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Export")
    parser.add_argument("frames", help="Frame directory")
    parser.add_argument("-o", "--output", default="output/video.mp4", help="Output path")
    parser.add_argument("--fps", type=int, default=30, help="FPS")
    parser.add_argument("--audio", help="Audio file to add")
    
    args = parser.parse_args()
    
    if args.audio:
        create_video_with_audio(args.frames, args.output, args.audio, args.fps)
    else:
        create_video(args.frames, args.output, args.fps)