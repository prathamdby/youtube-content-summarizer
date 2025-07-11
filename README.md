# YouTube Content Summarizer Bot

A Telegram bot that summarizes YouTube videos using Google's Gemini AI, with follow-up Q&A capabilities.

## Features

- 🎥 **Video Summarization**: Get comprehensive summaries of YouTube videos
- 🤖 **AI-Powered Q&A**: Ask follow-up questions about video content
- 🔄 **Auto-Summarization**: Automatically summarizes YouTube links in group chats
- 📊 **Rich Context**: Extracts video metadata, transcript, and generates intelligent summaries
- 🛡️ **Bot Detection Bypass**: Uses Cloudflare WARP proxy for reliable YouTube access
- 🐳 **Docker Ready**: Production-ready containerized deployment

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Telegram Bot Token ([Get from BotFather](https://t.me/BotFather))
- Google Gemini API Key ([Get from Google AI Studio](https://aistudio.google.com/app/apikey))

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/youtube-content-summarizer.git
cd youtube-content-summarizer
```

2. Create a `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Start the services:

```bash
docker-compose up -d
```

The bot will automatically set up a Cloudflare WARP proxy to bypass YouTube's bot detection.

### Usage

1. **Start the bot**: Send `/start` to your bot
2. **Summarize a video**: Send `/summarize https://www.youtube.com/watch?v=VIDEO_ID`
3. **Ask questions**: Reply to any summary message with your question
4. **Auto-summarization**: Just paste YouTube links in group chats (if enabled)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │ ── │  WARP Proxy     │ ── │   YouTube API   │
│   (Port 8080)   │    │  (Port 1080)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│   Gemini AI     │                          │   Transcript    │
│   (Summarizer)  │                          │   Extraction    │
└─────────────────┘                          └─────────────────┘
```

## Proxy Configuration

The bot uses [Cloudflare WARP](https://github.com/cmj2002/warp-docker) to bypass YouTube's bot detection:

- **Automatic Setup**: WARP proxy starts automatically with Docker Compose
- **Reliable Access**: Uses Cloudflare's infrastructure for stable YouTube access
- **No Manual Configuration**: Works out of the box with no additional setup needed

## Monitoring

The bot includes built-in observability:

- **Health Checks**: Available at `http://localhost:8080/health`
- **Metrics**: Prometheus metrics at `http://localhost:8000/metrics`
- **Structured Logging**: JSON-formatted logs for easy parsing

## Commands

| Command            | Description                           |
| ------------------ | ------------------------------------- |
| `/start`           | Show welcome message and bot features |
| `/help`            | Display detailed usage instructions   |
| `/summarize <URL>` | Summarize a YouTube video             |
| `/stats`           | Show bot statistics and cache info    |

## Supported YouTube URLs

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

## Limitations

- Videos must have captions/subtitles available
- Maximum video length: 3 hours
- Only HTTPS URLs are accepted for security
- Some videos may be blocked due to content policies

## Development

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export BOT_TOKEN=your_bot_token
export GEMINI_API_KEY=your_gemini_key
export PROXY_URL=socks5://localhost:1080  # If using WARP locally
```

3. Run the bot:

```bash
python bot.py
```

### Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Check code quality
python -m flake8 .
python -m black .
```

## Configuration

### Environment Variables

| Variable         | Description                  | Required | Default         |
| ---------------- | ---------------------------- | -------- | --------------- |
| `BOT_TOKEN`      | Telegram bot token           | Yes      | -               |
| `GEMINI_API_KEY` | Google Gemini API key        | Yes      | -               |
| `PROXY_URL`      | Proxy URL for YouTube access | No       | Auto-configured |

### Docker Compose Configuration

The `docker-compose.yml` file includes:

- **WARP Proxy Service**: Handles YouTube access through Cloudflare
- **Bot Service**: Main application with dependency on WARP
- **Health Checks**: Automatic service monitoring
- **Resource Limits**: Memory and CPU constraints
- **Logging**: JSON-formatted logs with rotation

## Troubleshooting

### Common Issues

1. **"Transcript Unavailable"**: Video doesn't have captions
2. **"Video Too Long"**: Video exceeds 3-hour limit
3. **"Rate Limit"**: Temporary Gemini API limitation
4. **"Network Error"**: WARP proxy connection issues

### WARP Proxy Issues

If YouTube access fails:

1. Check WARP container status:

```bash
docker-compose logs warp
```

2. Test WARP connectivity:

```bash
docker-compose exec warp python3 -c "
import urllib.request
proxy = urllib.request.ProxyHandler({'https': 'socks5://127.0.0.1:1080'})
opener = urllib.request.build_opener(proxy)
req = urllib.request.Request('https://cloudflare.com/cdn-cgi/trace')
response = opener.open(req, timeout=10)
print(response.read().decode('utf-8'))
"
```

3. Restart WARP service:

```bash
docker-compose restart warp
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Cloudflare WARP Docker](https://github.com/cmj2002/warp-docker) for proxy solution
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube transcript extraction
- [Google Gemini](https://ai.google.dev/) for AI summarization
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for Telegram integration
