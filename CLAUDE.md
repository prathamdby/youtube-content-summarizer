# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Bot
- **Local development**: `python bot.py`
- **Docker development**: `docker-compose up -d`
- **Docker logs**: `docker-compose logs -f`
- **Docker status**: `docker-compose ps`

### Testing and Validation
No formal test suite exists. Use these manual tests:
```bash
# Test URL validation
python -c "from utils import extract_video_id; print(extract_video_id('https://youtu.be/dQw4w9WgXcQ'))"

# Test transcript cleaning
python -c "from utils import clean_transcript_text; print(clean_transcript_text('[Music] Hello world [Applause]'))"
```

### Monitoring and Health
- **Health check**: `curl http://localhost:8080/health`
- **Metrics**: `curl http://localhost:8000/metrics` (Prometheus format)
- **Ping**: `curl http://localhost:8080/ping`

## Architecture Overview

This is a Python-based Telegram bot that summarizes YouTube videos using Google's Gemini AI. The architecture follows a modular async design:

### Core Components
- **bot.py**: Main entry point with `YouTubeSummarizerBot` class, handles application lifecycle
- **handlers.py**: Telegram command/message handlers (`/start`, `/summarize`, auto-summary, Q&A)
- **youtube.py**: YouTube transcript extraction using yt-dlp with metadata collection
- **gemini.py**: Google Gemini AI integration with chunking for large transcripts (25k token chunks)
- **cache.py**: Multi-level LRU caching system (`transcript_cache` with VideoContext storage)
- **utils.py**: Utilities for validation, logging, metrics, and text processing

### Key Architectural Patterns
- **Async/await throughout**: All I/O operations are non-blocking
- **Semaphore-based concurrency**: Max 5 concurrent Gemini API calls to prevent rate limits
- **Context-aware caching**: Videos stored with full metadata (title, duration, uploader, etc.)
- **Error resilience**: Comprehensive error handling with exponential backoff
- **Production monitoring**: Prometheus metrics, health checks, structured JSON logging

### Data Flow
1. User sends YouTube URL → URL validation (utils.py)
2. Video metadata extraction → yt-dlp (youtube.py)
3. Transcript fetching → subtitle processing (youtube.py)
4. Text chunking → Gemini summarization (gemini.py)
5. VideoContext caching → LRU with TTL (cache.py)
6. Follow-up Q&A → Context retrieval from cache

## Environment Setup

Required environment variables:
- `BOT_TOKEN`: Telegram bot token from @BotFather
- `GEMINI_API_KEY`: Google Gemini API key from AI Studio

Create `.env` file from `.env.example` template.

## Key Implementation Details

### Gemini API Integration
**CRITICAL**: This project uses the **new Google GenAI SDK** (`google-genai` package), not the deprecated `google-generativeai`. When working with Gemini API code:

- **Import**: `from google import genai` and `from google.genai import types`
- **Client**: `client = genai.Client(api_key="...")` 
- **Generation**: `client.models.generate_content(model=..., contents=..., config=...)`
- **Configuration**: `types.GenerateContentConfig(temperature=..., safety_settings=...)`
- **Safety Settings**: All safety filters are disabled using `threshold='BLOCK_NONE'` for all harm categories
- **No Token Limits**: The `max_output_tokens` parameter has been removed to allow natural response lengths

### Caching System
- **Global cache**: 1000 chats max, 25 videos per chat
- **TTL**: 2 hours for video context
- **Storage**: Complete VideoContext objects with metadata and transcript

### Rate Limiting
- **Gemini API**: 5 concurrent requests via semaphore
- **Retry logic**: Exponential backoff for API limits and timeouts

### Content Processing
- **Video limits**: Max 3 hours duration
- **Transcript cleaning**: Removes music tags, applause, repeated whitespace
- **Chunking**: 25k token chunks with map-reduce for large videos
- **No artificial token limits**: Gemini generates responses of natural length

### Security Features
- **Input sanitization**: URL validation, content safety filters disabled per requirements
- **Container security**: Non-root user, resource limits
- **HTTPS only**: Only HTTPS YouTube URLs accepted

## Deployment

The bot is designed for single-instance deployment with Docker. Key considerations:
- Memory limit: 512MB (256MB reserved)
- CPU limit: 0.5 cores (0.25 reserved) 
- Health checks every 30s
- Log rotation (10MB, 3 files)
- Graceful shutdown handling

## Working with Gemini API

When modifying Gemini integration:

1. **Always use the correct SDK**: `google-genai` package, never `google-generativeai`
2. **Client pattern**: Create client once in gemini.py module and reuse
3. **Safety settings**: All categories set to `BLOCK_NONE` - don't add token limits
4. **Error handling**: Handle `MAX_TOKENS`, `SAFETY`, and API errors specifically
5. **Async patterns**: Use `run_in_executor` for sync Gemini calls in async context
6. **Chunking strategy**: 25k token chunks for large transcripts with map-reduce

### Common Gemini Tasks

- **Adding safety settings**: Use `types.SafetySetting(category='...', threshold='BLOCK_NONE')`
- **Modifying prompts**: Update `_build_single_summary_prompt()` and related methods
- **Handling responses**: Check for `response.text` first, then `response.candidates[0].content.parts[0].text`
- **Error recovery**: Look for `finish_reason` to understand why generation stopped

## Development Patterns

When adding new features:
1. Use async/await patterns consistently
2. Add appropriate error handling with logging
3. Update Prometheus metrics if adding new operations
4. Consider cache invalidation for data changes
5. Test with various YouTube URL formats
6. Follow the established VideoContext pattern for data storage

When debugging:
1. Check Docker logs: `docker-compose logs -f`
2. Monitor metrics endpoint for performance issues
3. Verify health endpoint responds correctly
4. Check Gemini API quota and rate limits
5. Look for `finish_reason` in Gemini responses to understand truncation/blocking

## Video Context Architecture

The `VideoContext` TypedDict in cache.py defines the comprehensive data structure used throughout the application:

- **Metadata**: Title, duration, uploader, view counts, categories, tags
- **Transcript**: Cleaned text and timing entries
- **AI Content**: Generated summary with timestamp
- **Caching**: TTL-based expiration and LRU eviction

This centralized data model ensures consistency across all components and enables rich context-aware Q&A functionality.