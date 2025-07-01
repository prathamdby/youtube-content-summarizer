"""
In-memory LRU cache with TTL eviction for storing transcripts.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, TypedDict, Union
from collections import OrderedDict


logger = logging.getLogger(__name__)


class Transcript(TypedDict):
    """Type definition for cached transcript data."""

    text: str
    lang: str
    timestamp: float


class VideoContext(TypedDict):
    """Type definition for comprehensive cached video data."""

    # Video metadata
    video_id: str
    title: str
    description: Optional[str]
    duration: Optional[float]  # in seconds
    upload_date: Optional[str]
    uploader: Optional[str]
    uploader_id: Optional[str]
    channel: Optional[str]
    channel_id: Optional[str]
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    categories: Optional[list]
    tags: Optional[list]
    thumbnail: Optional[str]
    webpage_url: str

    # Transcript data
    transcript: Transcript
    transcript_entries: Optional[list]  # Raw transcript with timing

    # AI-generated content
    summary: Optional[str]
    summary_timestamp: Optional[float]

    # Cache metadata
    cached_at: float
    cache_version: str


class LRUDict:
    """LRU Dictionary with TTL support."""

    def __init__(self, max_size: int = 25, ttl: int = 7200):  # 2 hours default TTL
        self.max_size = max_size
        self.ttl = ttl
        self.data: OrderedDict[int, Dict[str, Any]] = OrderedDict()

    def _is_expired(self, item: Dict[str, Any]) -> bool:
        """Check if an item has expired based on TTL."""
        return time.time() - item["created_at"] > self.ttl

    def _evict_expired(self) -> None:
        """Remove expired items."""
        current_time = time.time()
        expired_keys = [
            key
            for key, value in self.data.items()
            if current_time - value["created_at"] > self.ttl
        ]
        for key in expired_keys:
            del self.data[key]
            logger.debug(f"Evicted expired cache entry: {key}")

    def get(self, key: int) -> Optional[Union[Transcript, VideoContext]]:
        """Get item from cache, moving it to end (most recently used)."""
        self._evict_expired()

        if key not in self.data:
            return None

        # Move to end (most recently used)
        value = self.data.pop(key)

        # Check if expired
        if self._is_expired(value):
            logger.debug(f"Cache entry expired: {key}")
            return None

        # Re-insert at end
        self.data[key] = value

        # Support both old and new format
        if "video_context" in value:
            return value["video_context"]
        else:
            # Backward compatibility - return transcript
            return value["transcript"]

    def put(self, key: int, transcript: Transcript) -> None:
        """Put transcript item in cache, evicting LRU if at capacity."""
        self._evict_expired()

        # Remove if exists (to update position)
        if key in self.data:
            del self.data[key]

        # Add new item
        self.data[key] = {"transcript": transcript, "created_at": time.time()}

        # Evict LRU if over capacity
        while len(self.data) > self.max_size:
            oldest_key = next(iter(self.data))
            del self.data[oldest_key]
            logger.debug(f"Evicted LRU cache entry: {oldest_key}")

        logger.debug(f"Cached transcript for key: {key}")

    def put_video_context(self, key: int, video_context: VideoContext) -> None:
        """Put comprehensive video context in cache, evicting LRU if at capacity."""
        self._evict_expired()

        # Remove if exists (to update position)
        if key in self.data:
            del self.data[key]

        # Add new item
        self.data[key] = {"video_context": video_context, "created_at": time.time()}

        # Evict LRU if over capacity
        while len(self.data) > self.max_size:
            oldest_key = next(iter(self.data))
            del self.data[oldest_key]
            logger.debug(f"Evicted LRU cache entry: {oldest_key}")

        logger.debug(
            f"Cached video context for key: {key} (video: {video_context.get('title', 'Unknown')})"
        )

    def remove(self, key: int) -> bool:
        """Remove item from cache."""
        if key in self.data:
            del self.data[key]
            logger.debug(f"Removed cache entry: {key}")
            return True
        return False

    def size(self) -> int:
        """Get current cache size."""
        self._evict_expired()
        return len(self.data)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.data.clear()
        logger.debug("Cleared all cache entries")


class TranscriptCache:
    """
    Global transcript cache manager.
    Maintains per-chat LRU caches with TTL eviction.
    """

    def __init__(
        self, max_chats: int = 1000, chat_cache_size: int = 25, ttl: int = 7200
    ):
        self.max_chats = max_chats
        self.chat_cache_size = chat_cache_size
        self.ttl = ttl
        self.caches: Dict[int, LRUDict] = {}
        self._chat_access: OrderedDict[int, float] = (
            OrderedDict()
        )  # Track chat access times
        self._cleanup_task: Optional[asyncio.Task] = None

    def _get_or_create_chat_cache(self, chat_id: int) -> LRUDict:
        """Get or create cache for a specific chat."""
        if chat_id not in self.caches:
            # Evict oldest chat if at capacity
            while len(self.caches) >= self.max_chats:
                oldest_chat = next(iter(self._chat_access))
                self._remove_chat_cache(oldest_chat)

            self.caches[chat_id] = LRUDict(self.chat_cache_size, self.ttl)
            logger.debug(f"Created new cache for chat: {chat_id}")

        # Update access time
        self._chat_access[chat_id] = time.time()
        self._chat_access.move_to_end(chat_id)

        return self.caches[chat_id]

    def _remove_chat_cache(self, chat_id: int) -> None:
        """Remove cache for a specific chat."""
        if chat_id in self.caches:
            del self.caches[chat_id]

        if chat_id in self._chat_access:
            del self._chat_access[chat_id]

        logger.debug(f"Removed cache for chat: {chat_id}")

    def put(self, chat_id: int, message_id: int, transcript: Transcript) -> None:
        """Store transcript in cache."""
        cache = self._get_or_create_chat_cache(chat_id)
        cache.put(message_id, transcript)

    def put_video_context(
        self, chat_id: int, message_id: int, video_context: VideoContext
    ) -> None:
        """Store comprehensive video context in cache."""
        cache = self._get_or_create_chat_cache(chat_id)
        cache.put_video_context(message_id, video_context)

    def get(
        self, chat_id: int, message_id: int
    ) -> Optional[Union[Transcript, VideoContext]]:
        """Retrieve transcript or video context from cache."""
        if chat_id not in self.caches:
            return None

        cache = self.caches[chat_id]
        return cache.get(message_id)

    def get_transcript(self, chat_id: int, message_id: int) -> Optional[Transcript]:
        """Retrieve just the transcript from cache (backward compatibility)."""
        cached_data = self.get(chat_id, message_id)

        if cached_data is None:
            return None

        # If it's a VideoContext, extract the transcript
        if isinstance(cached_data, dict) and "transcript" in cached_data:
            return cached_data["transcript"]
        elif isinstance(cached_data, dict) and "text" in cached_data:
            # It's a Transcript object
            return cached_data
        else:
            return None

    def get_video_context(
        self, chat_id: int, message_id: int
    ) -> Optional[VideoContext]:
        """Retrieve comprehensive video context from cache."""
        cached_data = self.get(chat_id, message_id)

        if cached_data is None:
            return None

        # Only return if it's a VideoContext
        if isinstance(cached_data, dict) and "video_id" in cached_data:
            return cached_data
        else:
            return None

    def remove(self, chat_id: int, message_id: int) -> bool:
        """Remove specific transcript from cache."""
        if chat_id not in self.caches:
            return False

        cache = self.caches[chat_id]
        return cache.remove(message_id)

    def clear_chat(self, chat_id: int) -> None:
        """Clear all transcripts for a specific chat."""
        if chat_id in self.caches:
            self.caches[chat_id].clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = sum(cache.size() for cache in self.caches.values())

        return {
            "total_chats": len(self.caches),
            "total_entries": total_entries,
            "max_chats": self.max_chats,
            "chat_cache_size": self.chat_cache_size,
            "ttl_seconds": self.ttl,
        }

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started cache cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped cache cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries and inactive chats."""
        while True:
            try:
                current_time = time.time()

                # Clean up expired entries in all caches
                for chat_id, cache in list(self.caches.items()):
                    cache._evict_expired()

                    # Remove empty caches
                    if cache.size() == 0:
                        self._remove_chat_cache(chat_id)

                # Remove inactive chats (no access for 4 hours)
                inactive_threshold = current_time - (4 * 3600)
                inactive_chats = [
                    chat_id
                    for chat_id, last_access in self._chat_access.items()
                    if last_access < inactive_threshold
                ]

                for chat_id in inactive_chats:
                    self._remove_chat_cache(chat_id)

                # Log stats periodically
                stats = self.get_stats()
                if stats["total_entries"] > 0:
                    logger.info(f"Cache stats: {stats}")

                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)  # Sleep for 1 minute on error


# Global cache instance
transcript_cache = TranscriptCache()
