"""
YouTube transcript fetching and processing using yt-dlp.
"""

import asyncio
import logging
import tempfile
import json
import re
import time
from typing import Optional, List, Dict, Any
import yt_dlp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from utils import clean_transcript_text, Timer, REQUEST_COUNT
from cache import VideoContext
from cache import Transcript


logger = logging.getLogger(__name__)


class TranscriptError(Exception):
    """Custom exception for transcript-related errors."""

    pass


class VideoTooLongError(TranscriptError):
    """Raised when video is too long (> 3 hours)."""

    pass


class TranscriptUnavailableError(TranscriptError):
    """Raised when transcript is not available for the video."""

    pass


def get_yt_dlp_options():
    """Get yt-dlp options optimized for transcript extraction."""
    return {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "skip_download": True,
        "no_check_certificates": True,
        # Don't select any format since we're not downloading
        "format": "best[height<=144]/worst",  # Fallback to lowest quality if needed
        "ignore_errors": False,
        # User agent to avoid blocks
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        # Add some delay to avoid rate limiting
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        # Retry configuration
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 3,
    }


@retry(
    retry=retry_if_exception_type(
        (ConnectionError, TimeoutError, yt_dlp.DownloadError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def fetch_video_info_with_retry(video_id: str) -> Dict[str, Any]:
    """
    Fetch video info and subtitles with retry logic.

    Args:
        video_id: YouTube video ID

    Returns:
        Video info dictionary with subtitles

    Raises:
        TranscriptUnavailableError: If transcript is not available
        VideoTooLongError: If video is too long
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Configure yt-dlp options without format selection
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "skip_download": True,
            "no_check_certificates": True,
            # User agent to avoid blocks
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
            },
            # Retry configuration
            "retries": 2,
            "fragment_retries": 2,
            "extractor_retries": 2,
        }

        # Run yt-dlp in thread pool since it's synchronous
        def extract_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        info = await asyncio.get_event_loop().run_in_executor(None, extract_info)

        if not info:
            raise TranscriptUnavailableError("Could not extract video information")

        # Check video duration (yt-dlp gives duration in seconds)
        duration = info.get("duration", 0)
        if duration and duration > 10800:  # 3 hours in seconds
            raise VideoTooLongError(f"Video is too long: {duration/3600:.1f} hours")

        return info

    except yt_dlp.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg or "Video unavailable" in error_msg:
            raise TranscriptUnavailableError("Video is private or unavailable")
        elif "No video" in error_msg or "not found" in error_msg:
            raise TranscriptUnavailableError("Video not found")
        else:
            logger.error(f"yt-dlp error fetching video {video_id}: {e}")
            raise TranscriptError(f"Failed to fetch video info: {error_msg}")

    except Exception as e:
        logger.error(f"Unexpected error fetching video info for {video_id}: {e}")
        raise TranscriptError(f"Failed to fetch video info: {str(e)}")


async def extract_subtitles_with_ytdlp(video_id: str) -> List[Dict[str, Any]]:
    """
    Extract subtitles using yt-dlp subtitle download.

    Args:
        video_id: YouTube video ID

    Returns:
        List of subtitle entries
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Configure yt-dlp for subtitle extraction only - no format selection
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "skip_download": True,
            "no_check_certificates": True,
            "extract_flat": False,
            # User agent to avoid blocks
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
            },
            # Retry configuration
            "retries": 2,
            "fragment_retries": 2,
            "extractor_retries": 2,
        }

        subtitle_data = []

        def download_subs():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        # Extract info to get available subtitles
        info = await asyncio.get_event_loop().run_in_executor(None, download_subs)

        if not info:
            raise TranscriptUnavailableError("Could not extract video information")

        # Get subtitles from info
        subtitles = info.get("subtitles", {})
        automatic_captions = info.get("automatic_captions", {})

        # Prefer manual subtitles, fall back to automatic
        available_subs = subtitles or automatic_captions

        if not available_subs:
            raise TranscriptUnavailableError("No subtitles available for this video")

        # Try to get English subtitles first
        sub_lang = None
        for lang in ["en", "en-US", "en-GB"]:
            if lang in available_subs:
                sub_lang = lang
                break

        # If no English, get first available
        if not sub_lang:
            sub_lang = next(iter(available_subs.keys()))
            logger.info(f"Using subtitles in language: {sub_lang}")

        # Get the subtitle URL
        sub_formats = available_subs[sub_lang]

        # Find JSON3 format or best available
        sub_url = None
        for fmt in sub_formats:
            if fmt.get("ext") == "json3":
                sub_url = fmt.get("url")
                break

        if not sub_url and sub_formats:
            # Fallback to first available format
            sub_url = sub_formats[0].get("url")

        if not sub_url:
            raise TranscriptUnavailableError("Could not get subtitle download URL")

        # Download subtitle content
        def download_subtitle_content():
            import urllib.request

            # Create request with proper headers
            req = urllib.request.Request(
                sub_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req) as response:
                return response.read().decode("utf-8")

        subtitle_content = await asyncio.get_event_loop().run_in_executor(
            None, download_subtitle_content
        )

        # Parse subtitle content
        if sub_url.endswith(".json3") or "json" in sub_url:
            # Parse JSON3 format
            try:
                data = json.loads(subtitle_content)
                events = data.get("events", [])

                for event in events:
                    segs = event.get("segs", [])
                    start_time = event.get("tStartMs", 0) / 1000  # Convert to seconds
                    duration = event.get("dDurationMs", 0) / 1000

                    text_parts = []
                    for seg in segs:
                        if "utf8" in seg:
                            text_parts.append(seg["utf8"])

                    if text_parts:
                        subtitle_data.append(
                            {
                                "text": "".join(text_parts),
                                "start": start_time,
                                "duration": duration,
                            }
                        )

            except json.JSONDecodeError:
                # Fallback: treat as plain text
                lines = subtitle_content.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip():
                        subtitle_data.append(
                            {
                                "text": line.strip(),
                                "start": i * 5,  # Approximate timing
                                "duration": 5,
                            }
                        )
        else:
            # Parse other formats (VTT, SRT, etc.)
            lines = subtitle_content.strip().split("\n")
            current_time = 0

            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("WEBVTT")
                    and "-->" not in line
                    and not line.isdigit()
                ):
                    # Remove formatting tags
                    line = re.sub(r"<[^>]+>", "", line)
                    if line:
                        subtitle_data.append(
                            {"text": line, "start": current_time, "duration": 3}
                        )
                        current_time += 3

        return subtitle_data

    except Exception as e:
        logger.error(f"Error extracting subtitles with yt-dlp: {e}")
        raise TranscriptError(f"Failed to extract subtitles: {str(e)}")


def process_transcript_entries(transcript_data: List[Dict[str, Any]]) -> str:
    """
    Process raw transcript entries into clean text.

    Args:
        transcript_data: Raw transcript data

    Returns:
        Cleaned transcript text
    """
    if not transcript_data:
        return ""

    # Extract text from transcript entries
    text_parts = []
    for entry in transcript_data:
        text = entry.get("text", "").strip()
        if text:
            text_parts.append(text)

    # Join all text parts
    full_text = " ".join(text_parts)

    # Clean the text
    cleaned_text = clean_transcript_text(full_text)

    return cleaned_text


async def get_video_transcript(video_id: str) -> Transcript:
    """
    Get and process transcript for a YouTube video using yt-dlp.

    Args:
        video_id: YouTube video ID

    Returns:
        Transcript object with text, language, and timestamp

    Raises:
        TranscriptError: If transcript cannot be fetched or processed
    """
    with Timer("transcript_fetch"):
        try:
            logger.info(f"Fetching transcript for video: {video_id}")

            # First, get video info to check accessibility and duration
            video_info = await asyncio.wait_for(
                fetch_video_info_with_retry(video_id), timeout=15.0
            )

            # Try to extract subtitles
            transcript_data = await asyncio.wait_for(
                extract_subtitles_with_ytdlp(video_id), timeout=30.0
            )

            if not transcript_data:
                REQUEST_COUNT.labels(command="transcript", status="empty").inc()
                raise TranscriptUnavailableError("No subtitle data found")

            # Process the transcript
            cleaned_text = process_transcript_entries(transcript_data)

            if not cleaned_text.strip():
                REQUEST_COUNT.labels(
                    command="transcript", status="empty_after_cleaning"
                ).inc()
                raise TranscriptUnavailableError("Transcript is empty after cleaning")

            # Determine language from video info
            language = "en"  # Default to English

            # Create transcript object
            transcript = Transcript(
                text=cleaned_text,
                lang=language,
                timestamp=asyncio.get_event_loop().time(),
            )

            REQUEST_COUNT.labels(command="transcript", status="success").inc()
            logger.info(
                f"Successfully fetched transcript: {len(cleaned_text)} characters"
            )

            return transcript

        except asyncio.TimeoutError:
            REQUEST_COUNT.labels(command="transcript", status="timeout").inc()
            logger.error(f"Timeout fetching transcript for video: {video_id}")
            raise TranscriptError("Timeout while fetching transcript")

        except (TranscriptUnavailableError, VideoTooLongError) as e:
            REQUEST_COUNT.labels(command="transcript", status="unavailable").inc()
            logger.warning(f"Transcript unavailable for video {video_id}: {e}")
            raise

        except TranscriptError:
            REQUEST_COUNT.labels(command="transcript", status="error").inc()
            raise

        except Exception as e:
            REQUEST_COUNT.labels(command="transcript", status="error").inc()
            logger.error(
                f"Unexpected error processing transcript for video {video_id}: {e}"
            )
            raise TranscriptError(f"Failed to process transcript: {str(e)}")


async def validate_video_accessibility(video_id: str) -> bool:
    """
    Quick check if a video is accessible and has transcripts.

    Args:
        video_id: YouTube video ID

    Returns:
        True if video appears to have transcripts, False otherwise
    """
    try:
        # Quick check using yt-dlp
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "skip_download": True,
            "no_check_certificates": True,
            # User agent to avoid blocks
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
            },
            # Retry configuration
            "retries": 1,
            "fragment_retries": 1,
            "extractor_retries": 1,
        }

        def check_video():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    # Check if subtitles or automatic captions are available
                    subtitles = info.get("subtitles", {})
                    automatic_captions = info.get("automatic_captions", {})
                    return bool(subtitles or automatic_captions)
                return False

        result = await asyncio.get_event_loop().run_in_executor(None, check_video)
        return result

    except Exception as e:
        logger.debug(f"Video {video_id} accessibility check failed: {e}")
        return False


async def get_video_context(
    video_id: str, include_summary: bool = False
) -> VideoContext:
    """
    Get comprehensive video context including metadata, transcript, and optionally summary.

    Args:
        video_id: YouTube video ID
        include_summary: Whether to include pre-generated summary (if available)

    Returns:
        VideoContext object with all available video information

    Raises:
        TranscriptError: If video context cannot be fetched or processed
    """
    with Timer("video_context_fetch"):
        try:
            logger.info(f"Fetching comprehensive video context for: {video_id}")

            # Get video info and transcript data
            video_info = await asyncio.wait_for(
                fetch_video_info_with_retry(video_id), timeout=15.0
            )

            transcript_data = await asyncio.wait_for(
                extract_subtitles_with_ytdlp(video_id), timeout=30.0
            )

            if not transcript_data:
                raise TranscriptUnavailableError("No subtitle data found")

            # Process the transcript
            cleaned_text = process_transcript_entries(transcript_data)

            if not cleaned_text.strip():
                raise TranscriptUnavailableError("Transcript is empty after cleaning")

            # Create transcript object
            transcript = Transcript(
                text=cleaned_text,
                lang=video_info.get("language", "en"),
                timestamp=asyncio.get_event_loop().time(),
            )

            # Extract comprehensive metadata
            current_time = time.time()

            # Format upload date
            upload_date = video_info.get("upload_date")
            if upload_date and len(upload_date) >= 8:
                # Convert YYYYMMDD to YYYY-MM-DD
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

            # Create comprehensive video context
            video_context = VideoContext(
                # Video metadata
                video_id=video_id,
                title=video_info.get("title", "Unknown Title"),
                description=video_info.get("description"),
                duration=video_info.get("duration"),
                upload_date=upload_date,
                uploader=video_info.get("uploader"),
                uploader_id=video_info.get("uploader_id"),
                channel=video_info.get("channel"),
                channel_id=video_info.get("channel_id"),
                view_count=video_info.get("view_count"),
                like_count=video_info.get("like_count"),
                comment_count=video_info.get("comment_count"),
                categories=video_info.get("categories"),
                tags=video_info.get("tags"),
                thumbnail=video_info.get("thumbnail"),
                webpage_url=video_info.get(
                    "webpage_url", f"https://www.youtube.com/watch?v={video_id}"
                ),
                # Transcript data
                transcript=transcript,
                transcript_entries=transcript_data,  # Raw transcript with timing
                # AI-generated content (optional)
                summary=None,  # Will be filled by handlers if requested
                summary_timestamp=None,
                # Cache metadata
                cached_at=current_time,
                cache_version="1.0",
            )

            REQUEST_COUNT.labels(command="video_context", status="success").inc()
            logger.info(
                f"Successfully fetched video context: {video_context['title']} "
                f"({len(cleaned_text)} chars transcript, {video_context.get('duration', 0)/60:.1f}min duration)"
            )

            return video_context

        except asyncio.TimeoutError:
            REQUEST_COUNT.labels(command="video_context", status="timeout").inc()
            logger.error(f"Timeout fetching video context for: {video_id}")
            raise TranscriptError("Timeout while fetching video context")

        except (TranscriptUnavailableError, VideoTooLongError) as e:
            REQUEST_COUNT.labels(command="video_context", status="unavailable").inc()
            logger.warning(f"Video context unavailable for {video_id}: {e}")
            raise

        except TranscriptError:
            REQUEST_COUNT.labels(command="video_context", status="error").inc()
            raise

        except Exception as e:
            REQUEST_COUNT.labels(command="video_context", status="error").inc()
            logger.error(f"Unexpected error fetching video context for {video_id}: {e}")
            raise TranscriptError(f"Failed to fetch video context: {str(e)}")
