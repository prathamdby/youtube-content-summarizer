# YouTube Content Summarizer Bot Plan

## 1. Tech Stack

- **Python 3.12**
- **python-telegram-bot v20+** (Bot API 9.0)
- **yt-dlp** for transcript extraction
- **google-generativeai** (official Google GenAI package)
- **Gemma 3 27B** (2 M-token window)
- **asyncio** + **aiohttp** for async I/O; **tenacity** for retries
- **In-memory** LRU caches (no external datastore)

## 2. User Flow & Commands

| Step | User Action                     | Bot Reaction                                 |
| ---- | ------------------------------- | -------------------------------------------- |
| 1    | `/summarize <YouTube URL>`      | validate URL → enqueue "job"                 |
| 2    |                                 | send "Fetching transcript…" progress message |
| 3    |                                 | fetch & clean transcript                     |
| 4    |                                 | stream to Gemma 3 27B → summary              |
| 5    |                                 | edit progress message with summary + hint    |
| 6    | user replies to summary message | retrieve transcript from cache → Gemini Q&A  |

## 3. Architecture (all async, single process)

```
bot.py
├── handlers.py      # command & message routers
├── youtube.py       # transcript fetch & clean
├── gemini.py        # Google GenAI wrapper, chunking, backoff
├── cache.py         # LRUDicts + TTL eviction
└── utils.py         # validation, logging, metrics
```

**Core objects**

```python
Transcript = TypedDict(
    {
        "text": str,
        "lang": str,
        "timestamp": float,
    }
)

# cache per chat_id → LRU of message_id → Transcript
Cache: Dict[int, LRUDict[int, Transcript]]
```

## 4. Processing Pipeline

1. **URL validation** → extract `video_id`.
2. **Transcript retrieval**
   - extract subtitles using `yt-dlp`
   - abort with "Transcript unavailable" if no captions exist
3. **Cleaning**
   - strip markers (`[Music]`, `[Applause]`), merge short lines
4. **Chunking & Summarization**
   - ≤ 120 k tokens → single Gemini call via `google.generativeai`
   - else map-reduce: ≤ 25 k-token chunks → per-chunk summary → final synthesis
5. **Caching**
   - key = `chat_id + summary_message_id`
   - per-chat LRU size 25, TTL 2 h; background prune
6. **Follow-up Q&A**
   - on `reply_to_message`: load transcript → prompt system=transcript + user question → Gemini
7. **Concurrency limits**
   - `asyncio.Semaphore(5)` per external API
   - exponential back-off on 429 (max 60 s)

## 5. Robustness Tactics

- **Stateless restarts**: RAM-only; users resend `/summarize` on restart
- **Early validation**: catch bad URLs/commands
- **Timeouts**: 10 s fetch, 60 s Gemini; cancel on timeout
- **Graceful degradation**: if transcript unavailable, inform user (no fallback API)
- **Memory ceilings**: cap total cache size (e.g. 50 MB); LRU eviction

## 6. Security & Abuse

- accept only HTTPS YouTube links
- sanitize inputs to prevent Markdown/HTML injection
- reject videos > 3 h
- honor Gemini safety filters; relay sanitized errors

## 7. Deployment

- Docker (Alpine + Python slim)
- ENV: `BOT_TOKEN`, `GEMINI_API_KEY`
- run under systemd or Docker Compose with auto-restart
- expose `/ping` health endpoint

## 8. Observability

- JSON logs (chat_id, step, latency)
- Prometheus metrics: request count, failures, token usage
- optional Sentry for uncaught exceptions

## 9. Testing Matrix

| Layer       | Tests                                                   |
| ----------- | ------------------------------------------------------- |
| Unit        | URL parsing, transcript clean, chunking, cache eviction |
| Integration | end-to-end summary (mock Gemini)                        |
| Load        | 100 parallel `/summarize` honoring semaphores           |
| Resilience  | kill mid-job; ensure no dangling tasks                  |

## 10. Future Enhancements (no-DB)

1. multi-language auto-translate captions
2. `/tl <lang>` for summaries in target language
3. inline button "Toggle detail" for shorter/longer summaries
4. optional in-memory SQLite for ad-hoc indexing

---

**Deliverable**: single repo (\~600 LOC), zero external state, production-ready with Bot API 9.0 & Gemma 3 27B via Google GenAI package.

## 11. Google GenAI Integration Details

**Setup & Configuration**

```python
import google.generativeai as genai

# Configure the client
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemma-3-27b-it')
```

**Key Benefits**

- **Built-in retry logic** and error handling
- **Streaming support** for real-time responses
- **Automatic rate limiting** and backoff
- **Type-safe** request/response handling
- **Simplified** token counting and content generation

**Usage Pattern**

```python
async def generate_summary(transcript: str) -> str:
    response = await model.generate_content_async(
        f"Summarize this YouTube transcript:\n\n{transcript}",
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1000,
        )
    )
    return response.text
```
