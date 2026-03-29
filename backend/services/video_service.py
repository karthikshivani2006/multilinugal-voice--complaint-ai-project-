"""
Video processing service using moviepy for evidence analysis.
MoviePy is optional - if not available, basic Gemini analysis still works.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MOVIEPY_AVAILABLE = False
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    logger.info("moviepy not installed - video audio extraction will be skipped")


class VideoService:
    """Handle video processing for evidence files."""

    @staticmethod
    def is_available() -> bool:
        """Check if moviepy is available."""
        return MOVIEPY_AVAILABLE

    @staticmethod
    def extract_audio_from_video(video_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[str]:
        """
        Extract audio from video file with optional time range.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds (optional, for subclipping)
            end_time: End time in seconds (optional, for subclipping)

        Returns:
            Path to extracted audio WAV file, or None if extraction fails
        """
        if not MOVIEPY_AVAILABLE:
            logger.warning("moviepy not available - cannot extract audio from video")
            return None

        try:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                logger.error(f"Video file not found: {video_path}")
                return None

            # Load video clip
            clip = VideoFileClip(str(video_path))

            try:
                # Optional: subclip if time range specified
                if start_time is not None and end_time is not None:
                    clip = clip.subclipped(start_time, end_time)

                # Check if video has audio
                if clip.audio is None:
                    logger.warning(f"Video has no audio track: {video_path}")
                    clip.close()
                    return None

                # Extract audio to temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    audio_path = tmp_audio.name

                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                logger.info(f"Audio extracted to: {audio_path}")
                return audio_path

            finally:
                clip.close()

        except Exception as e:
            logger.error(f"Audio extraction failed for {video_path}: {e}")
            return None

    @staticmethod
    def get_video_duration(video_path: str) -> Optional[float]:
        """
        Get video duration in seconds.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds, or None if unavailable
        """
        if not MOVIEPY_AVAILABLE:
            return None

        try:
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()
            return duration
        except Exception as e:
            logger.error(f"Failed to get video duration for {video_path}: {e}")
            return None

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dict with duration, fps, size info
        """
        if not MOVIEPY_AVAILABLE:
            return {}

        try:
            clip = VideoFileClip(str(video_path))
            info = {
                "duration": clip.duration,
                "fps": clip.fps,
                "width": clip.w,
                "height": clip.h,
                "has_audio": clip.audio is not None,
            }
            clip.close()
            return info
        except Exception as e:
            logger.error(f"Failed to get video info for {video_path}: {e}")
            return {}


video_service = VideoService()
