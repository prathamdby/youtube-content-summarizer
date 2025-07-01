"""
Utility functions for URL validation, logging, and metrics.
"""

import re
import logging
import time
from typing import Optional
from urllib.parse import urlparse, parse_qs
from prometheus_client import Counter, Histogram, start_http_server


# Prometheus metrics
REQUEST_COUNT = Counter(
    "telegram_bot_requests_total", "Total requests", ["command", "status"]
)
PROCESSING_TIME = Histogram(
    "telegram_bot_processing_seconds", "Processing time", ["operation"]
)
GEMINI_TOKENS = Counter("gemini_tokens_total", "Total Gemini tokens used", ["type"])


def setup_logging() -> None:
    """Configure JSON-structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various YouTube URL formats.

    Args:
        url: YouTube URL

    Returns:
        Video ID if valid, None otherwise
    """
    if not url or not isinstance(url, str):
        return None

    # Only accept HTTPS URLs for security
    if not url.startswith("https://"):
        return None

    try:
        parsed = urlparse(url)

        # Check if it's a valid YouTube domain
        if parsed.netloc not in [
            "www.youtube.com",
            "youtube.com",
            "youtu.be",
            "m.youtube.com",
        ]:
            return None

        # Handle youtu.be short URLs
        if parsed.netloc == "youtu.be":
            video_id = parsed.path[1:]  # Remove leading slash
            if len(video_id) == 11:  # YouTube video IDs are 11 characters
                return video_id

        # Handle youtube.com URLs
        elif "youtube.com" in parsed.netloc:
            if parsed.path == "/watch":
                query_params = parse_qs(parsed.query)
                video_ids = query_params.get("v")
                if video_ids and len(video_ids[0]) == 11:
                    return video_ids[0]

            # Handle embedded URLs like /embed/VIDEO_ID
            elif parsed.path.startswith("/embed/"):
                video_id = parsed.path[7:]  # Remove '/embed/'
                if len(video_id) == 11:
                    return video_id

    except Exception:
        return None

    return None


def sanitize_text(text: str) -> str:
    """
    Sanitize text to prevent Markdown injection while preserving readability.

    Args:
        text: Raw text input

    Returns:
        Sanitized text safe for Telegram
    """
    if not text:
        return ""

    # Escape Telegram MarkdownV2 special characters
    special_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")

    return text


def escape_markdown_v2(text: str) -> str:
    """
    Escape MarkdownV2 special characters while preserving intentional formatting.

    This function escapes special characters but preserves **bold** formatting
    that's already correctly formatted.

    Args:
        text: Text that may contain MarkdownV2 formatting

    Returns:
        Text with special characters escaped but formatting preserved
    """
    if not text:
        return ""

    # Characters that need escaping in MarkdownV2
    # We'll handle * separately to preserve **bold** formatting
    chars_to_escape = [
        "_",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]

    # First escape all special chars except *
    for char in chars_to_escape:
        text = text.replace(char, f"\\{char}")

    # Now handle * - we want to preserve **text** but escape standalone *
    # Replace single * that aren't part of **text** patterns
    import re

    # This regex finds * that aren't part of **text** patterns
    # It looks for * that are either:
    # 1. At start/end of string and not followed/preceded by *
    # 2. Not surrounded by other * characters
    def escape_single_asterisk(match):
        return "\\*"

    # Find standalone asterisks (not part of **bold** formatting)
    # This preserves **bold** while escaping single *
    text = re.sub(r"(?<!\*)\*(?!\*)", escape_single_asterisk, text)

    return text


def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text by removing music/sound markers and merging short lines.

    Args:
        text: Raw transcript text

    Returns:
        Cleaned transcript text
    """
    if not text:
        return ""

    # Remove common sound markers
    markers = [
        r"\[Music\]",
        r"\[Applause\]",
        r"\[Laughter\]",
        r"\[Sound\]",
        r"\[music\]",
        r"\[applause\]",
        r"\[laughter\]",
        r"\[sound\]",
        r"\(Music\)",
        r"\(Applause\)",
        r"\(Laughter\)",
        r"\(Sound\)",
        r"\(music\)",
        r"\(applause\)",
        r"\(laughter\)",
        r"\(sound\)",
    ]

    for marker in markers:
        text = re.sub(marker, "", text)

    # Split into lines and filter out very short ones
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Merge very short lines (< 10 characters) with the next line
    merged_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]

        # If current line is short and there's a next line, merge them
        if len(current_line) < 10 and i + 1 < len(lines):
            current_line += " " + lines[i + 1]
            i += 2
        else:
            i += 1

        merged_lines.append(current_line)

    # Join with newlines and clean up extra whitespace
    result = "\n".join(merged_lines)
    result = re.sub(r"\n\s*\n", "\n\n", result)  # Remove extra newlines
    result = re.sub(r" +", " ", result)  # Remove extra spaces

    return result.strip()


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for text.
    Uses approximation of ~4 characters per token.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics server."""
    try:
        start_http_server(port)
        logging.getLogger(__name__).info(f"Metrics server started on port {port}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to start metrics server: {e}")


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        PROCESSING_TIME.labels(operation=self.operation).observe(duration)

        logger = logging.getLogger(__name__)
        if exc_type is None:
            logger.info(f"Operation {self.operation} completed in {duration:.2f}s")
        else:
            logger.error(
                f"Operation {self.operation} failed after {duration:.2f}s: {exc_val}"
            )
