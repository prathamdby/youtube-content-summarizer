# YouTube Content Summarizer Bot

A fast, efficient Telegram bot that provides AI-powered summaries of YouTube videos using Google's Gemma 3 27B model. The bot fetches video transcripts and generates comprehensive summaries, with support for follow-up Q&A.

## Features

- 🎥 **Video Summarization**: Get comprehensive summaries of YouTube videos with metadata
- 🤖 **AI-Powered**: Uses Google Gemma 3 27B for high-quality summaries
- 💬 **Interactive Q&A**: Ask follow-up questions about video content with context awareness
- ⚡ **Fast Processing**: Async architecture with concurrent request handling
- 📊 **Rich Metadata**: Extracts video title, duration, uploader, view count, and more
- 📈 **Smart Caching**: Video context caching with comprehensive metadata storage
- 🔒 **Security**: Input sanitization and content safety filters
- 🚀 **Production Ready**: Docker deployment with health checks and resource management

## Architecture

```
bot.py (main entry point)
├── handlers.py      # Telegram command & message handlers
├── youtube.py       # YouTube transcript fetching via yt-dlp
├── gemini.py        # Google Gemini AI integration with chunking
├── cache.py         # In-memory LRU cache with VideoContext storage
└── utils.py         # Utilities for validation, logging, metrics
```

## Tech Stack

- **Python 3.12** with asyncio for concurrent processing
- **python-telegram-bot v20.7** for Telegram Bot API integration
- **yt-dlp 2024.12.13** for robust YouTube subtitle extraction
- **google-generativeai 0.8.3** official SDK for Gemma 3 27B
- **In-memory LRU caching** with TTL for video context storage
- **Prometheus metrics** for monitoring and observability
- **Docker** deployment with health checks and resource limits

## Quick Start

### Prerequisites

1. **Telegram Bot Token**: Get from [@BotFather](https://t.me/botfather)
2. **Google Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Local Development

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd youtube-content-summarizer
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:

   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run the bot**:
   ```bash
   python bot.py
   ```

### Docker Deployment

1. **Clone and configure**:

   ```bash
   git clone <repository-url>
   cd youtube-content-summarizer
   cp env.example .env
   # Edit .env with your API keys
   ```

2. **Deploy with Docker Compose**:

   ```bash
   docker-compose up -d
   ```

3. **Check status**:
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

## Usage

### Basic Commands

- `/start` - Welcome message and introduction
- `/help` - Detailed help and usage instructions
- `/summarize <YouTube URL>` - Generate video summary
- `/stats` - Show bot statistics and cache status

### Example Usage

1. **Summarize a video**:

   ```
   /summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

2. **Ask follow-up questions**:
   After receiving a summary, reply to the bot's message with your question:
   ```
   What are the main points discussed about machine learning?
   ```

### Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

## Configuration

### Environment Variables

| Variable         | Description                        | Required |
| ---------------- | ---------------------------------- | -------- |
| `BOT_TOKEN`      | Telegram bot token from @BotFather | Yes      |
| `GEMINI_API_KEY` | Google Gemini API key              | Yes      |

### Limits and Constraints

- **Video Length**: Maximum 3 hours
- **URL Security**: Only HTTPS YouTube URLs accepted
- **Transcript Requirement**: Videos must have captions/subtitles
- **Cache TTL**: 2 hours for video context storage
- **Concurrent Requests**: 5 simultaneous Gemini API calls
- **Cache Limits**: 1000 chats max, 25 videos per chat

## Monitoring

### Health Checks

- **Health endpoint**: `http://localhost:8080/health`
- **Ping endpoint**: `http://localhost:8080/ping`

### Metrics (Prometheus)

Available at `http://localhost:8000`:

- `telegram_bot_requests_total` - Request counter by command and status
- `telegram_bot_processing_seconds` - Processing time histogram by operation
- `gemini_tokens_total` - Token usage counter by type (input/output)

### Logs

JSON-structured logs with the following information:

- Timestamp and log level
- Module and operation context
- Processing metrics and errors
- Cache statistics

## Error Handling

The bot handles various error scenarios gracefully:

- **Invalid URLs**: Clear error messages with format examples
- **Missing Transcripts**: Informative messages about caption availability
- **Rate Limits**: Exponential backoff with user notifications
- **Content Safety**: Gemini safety filter integration
- **Timeouts**: Graceful timeout handling with retry logic
- **YouTube Blocking**: Uses yt-dlp with proper headers to bypass bot detection

## Performance

### Processing Pipeline

1. **URL Validation** (< 1ms)
2. **Video Info Extraction** (1-3 seconds) - Uses yt-dlp for metadata
3. **Transcript Fetch** (2-5 seconds) - Subtitle extraction with timing
4. **Text Cleaning** (< 100ms)
5. **AI Summarization** (5-15 seconds) - Gemma 3 27B
6. **Video Context Caching** (< 10ms) - Comprehensive metadata storage

### Optimization Features

- **Async I/O**: Non-blocking operations throughout
- **Semaphore Limits**: Prevents API rate limit violations (5 concurrent)
- **LRU Caching**: Efficient memory usage with TTL and video context storage
- **Chunking**: Handles large transcripts via map-reduce (25k token chunks)
- **Connection Pooling**: Reused HTTP connections
- **Smart Retry Logic**: Exponential backoff for rate limits and timeouts

## Development

### Project Structure

```
youtube-content-summarizer/
├── bot.py                 # Main application entry point
├── handlers.py            # Telegram bot handlers
├── youtube.py             # YouTube transcript processing
├── gemini.py              # Google Gemini AI integration
├── cache.py               # In-memory caching system
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Deployment configuration
├── env.example           # Environment template
└── README.md             # This file
```

### Key Components

- **VideoContext Storage**: Comprehensive video metadata with transcript data
- **yt-dlp Integration**: Robust YouTube data extraction bypassing restrictions
- **Gemma 3 27B**: Latest model with improved performance
- **Multi-level Caching**: Per-chat LRU caches with global chat management
- **Context-aware Q&A**: Uses video metadata for better question answering
- **Production Monitoring**: Health checks, metrics, and structured logging

### Dependencies

```
python-telegram-bot[all]==20.7
yt-dlp==2024.12.13
google-generativeai==0.8.3
aiohttp==3.9.1
tenacity==8.2.3
prometheus-client==0.19.0
python-dotenv==1.0.0
```

### Testing

Run basic functionality tests:

```bash
# Test URL validation
python -c "from utils import extract_video_id; print(extract_video_id('https://youtu.be/dQw4w9WgXcQ'))"

# Test transcript cleaning
python -c "from utils import clean_transcript_text; print(clean_transcript_text('[Music] Hello world [Applause]'))"
```

## Deployment

### Production Considerations

1. **Resource Limits**: Set appropriate memory/CPU limits
2. **Monitoring**: Deploy with metrics collection
3. **Logging**: Centralized log aggregation
4. **Health Checks**: Load balancer integration
5. **Secrets Management**: Secure API key storage

### Scaling

The bot is designed for single-instance deployment with:

- In-memory caching (no external dependencies)
- Concurrent request handling via semaphores
- Graceful degradation under high load

For higher scale requirements, consider:

- Multiple bot instances with load balancing
- External caching (Redis) for shared state
- Database storage for persistent analytics

## Troubleshooting

### Common Issues

1. **"Transcript unavailable"**: Video doesn't have captions or subtitles
2. **"Rate limit exceeded"**: Temporary API limits, retry automatically with backoff
3. **"Video too long"**: Videos over 3 hours aren't supported
4. **"Invalid URL"**: Only HTTPS YouTube URLs accepted
5. **"Video is private or unavailable"**: Video requires special access or is deleted
6. **"Processing timeout"**: Large videos may take longer, automatic retry logic applies

### Debug Information

Check logs for detailed error information:

```bash
docker-compose logs -f youtube-summarizer-bot
```

Monitor metrics:

```bash
curl http://localhost:8080/health
curl http://localhost:8000/metrics
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Pratham Dubey

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs for error details
3. Ensure API keys are valid and have sufficient quota
