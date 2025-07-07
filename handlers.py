"""
Telegram bot command and message handlers.
"""

import asyncio
import logging
import re
from typing import Optional
from telegram import Update, Message
from telegram.ext import ContextTypes
from telegram.constants import ParseMode, ChatAction, ChatType

from utils import (
    extract_video_id,
    sanitize_text,
    escape_markdown_v2,
    Timer,
    REQUEST_COUNT,
)
from youtube import (
    get_video_transcript,
    get_video_context,
    TranscriptError,
    TranscriptUnavailableError,
    VideoTooLongError,
)
from gemini import gemini_client, GeminiError, GeminiSafetyError, GeminiRateLimitError
from cache import transcript_cache, VideoContext


logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    REQUEST_COUNT.labels(command="start", status="received").inc()

    welcome_message = """🎥 **YouTube Content Summarizer Bot**

I can help you get quick summaries of YouTube videos\\!

**How to use:**
• Send `/summarize <YouTube URL>` to get a video summary
• After getting a summary, reply to my message with any question about the video content

**Example:**
`/summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ`

**Features:**
✅ Works with any public YouTube video with captions
✅ Handles videos up to 3 hours long
✅ Ask follow\\-up questions about the content
✅ Fast and accurate AI\\-powered summaries

**Note:** Only HTTPS YouTube URLs are supported for security\\."""

    try:
        await update.message.reply_text(
            welcome_message, parse_mode=ParseMode.MARKDOWN_V2
        )
        REQUEST_COUNT.labels(command="start", status="success").inc()
    except Exception as e:
        logger.error(f"Error sending start message: {e}")
        REQUEST_COUNT.labels(command="start", status="error").inc()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    REQUEST_COUNT.labels(command="help", status="received").inc()

    help_message = """🤖 **Help \\- YouTube Content Summarizer**

**Commands:**
• `/start` \\- Show welcome message
• `/help` \\- Show this help message
• `/summarize <URL>` \\- Summarize a YouTube video
• `/stats` \\- Show bot statistics

**How to summarize a video:**
1\\. Copy a YouTube video URL
2\\. Send `/summarize <URL>` to the bot
3\\. Wait for the summary \\(usually 10\\-30 seconds\\)
4\\. Reply to the summary message to ask questions about the video

**Supported URL formats:**
• `https://www.youtube.com/watch?v=VIDEO_ID`
• `https://youtu.be/VIDEO_ID`
• `https://m.youtube.com/watch?v=VIDEO_ID`

**Limitations:**
• Videos must have captions/subtitles available
• Maximum video length: 3 hours
• Only HTTPS URLs are accepted
• Some videos may be blocked due to content policies

**Tips:**
• Reply to summary messages to ask specific questions
• The bot remembers recent summaries for follow\\-up questions
• Summaries are structured with key points and takeaways"""

    try:
        await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN_V2)
        REQUEST_COUNT.labels(command="help", status="success").inc()
    except Exception as e:
        logger.error(f"Error sending help message: {e}")
        REQUEST_COUNT.labels(command="help", status="error").inc()


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command."""
    REQUEST_COUNT.labels(command="stats", status="received").inc()

    try:
        cache_stats = transcript_cache.get_stats()

        stats_message = f"""📊 **Bot Statistics**

**Cache Status:**
• Active chats: {cache_stats['total_chats']}
• Cached videos: {cache_stats['total_entries']}
• Cache TTL: {cache_stats['ttl_seconds'] // 3600} hours

**Memory Usage:**
• Max chats: {cache_stats['max_chats']}
• Videos per chat: {cache_stats['chat_cache_size']}

**Features:**
• ✅ Comprehensive video metadata
• ✅ Transcript with timing data
• ✅ AI\\-generated summaries
• ✅ Follow\\-up Q&A support

*Note: All data is stored in\\-memory and cleared on bot restart\\.*"""

        await update.message.reply_text(stats_message, parse_mode=ParseMode.MARKDOWN_V2)
        REQUEST_COUNT.labels(command="stats", status="success").inc()
    except Exception as e:
        logger.error(f"Error sending stats message: {e}")
        REQUEST_COUNT.labels(command="stats", status="error").inc()


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summarize command."""
    REQUEST_COUNT.labels(command="summarize", status="received").inc()

    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a YouTube URL\\.\n\n"
            "**Usage:** `/summarize <YouTube URL>`\n\n"
            "**Example:** `/summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="summarize", status="invalid_args").inc()
        return

    url = " ".join(context.args)
    video_id = extract_video_id(url)

    if not video_id:
        await update.message.reply_text(
            "❌ Invalid YouTube URL\\.\n\n"
            "Please provide a valid HTTPS YouTube URL\\.\n\n"
            "**Supported formats:**\n"
            "• `https://www.youtube.com/watch?v=VIDEO_ID`\n"
            "• `https://youtu.be/VIDEO_ID`\n"
            "• `https://m.youtube.com/watch?v=VIDEO_ID`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="summarize", status="invalid_url").inc()
        return

    # Send initial processing message
    processing_message = await update.message.reply_text(
        "🔄 **Processing your request\\.\\.\\.**\n\n"
        "Fetching transcript from YouTube\\.\\.\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        with Timer("full_summarization"):
            # Show typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Step 1: Fetch comprehensive video context
            try:
                video_context = await get_video_context(video_id)
            except VideoTooLongError as e:
                await processing_message.edit_text(
                    f"❌ **Video Too Long**\n\n"
                    f"This video is too long to process \\(max: 3 hours\\)\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(command="summarize", status="video_too_long").inc()
                return

            except TranscriptUnavailableError as e:
                await processing_message.edit_text(
                    f"❌ **Transcript Unavailable**\n\n"
                    f"This video doesn't have captions/subtitles available\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(command="summarize", status="no_transcript").inc()
                return

            except TranscriptError as e:
                await processing_message.edit_text(
                    f"❌ **Transcript Error**\n\n"
                    f"Failed to fetch video transcript\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(
                    command="summarize", status="transcript_error"
                ).inc()
                return

            # Step 2: Update progress message
            await processing_message.edit_text(
                "🤖 **Generating summary\\.\\.\\.**\n\n"
                "Analyzing transcript with AI\\.\\.\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )

            # Show typing indicator for AI processing
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Step 3: Generate summary
            try:
                summary = await gemini_client.generate_summary(
                    video_context["transcript"]["text"]
                )
            except GeminiSafetyError as e:
                await processing_message.edit_text(
                    f"❌ **Content Safety Block**\n\n"
                    f"The AI safety filters blocked this content\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(command="summarize", status="safety_block").inc()
                return

            except GeminiRateLimitError as e:
                await processing_message.edit_text(
                    f"⏱️ **Rate Limit**\n\n"
                    f"AI service is temporarily busy\\. Please try again in a few minutes\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(command="summarize", status="rate_limit").inc()
                return

            except GeminiError as e:
                await processing_message.edit_text(
                    f"❌ **AI Processing Error**\n\n"
                    f"Failed to generate summary\\.\n\n"
                    f"Error: {sanitize_text(str(e))}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                REQUEST_COUNT.labels(command="summarize", status="ai_error").inc()
                return

            # Step 4: Format and send final response with rich metadata
            video_title = escape_markdown_v2(video_context["title"])
            video_channel = video_context.get("channel") or video_context.get(
                "uploader", "Unknown"
            )
            video_duration = video_context.get("duration", 0)
            duration_str = (
                f"{int(video_duration // 60)}:{int(video_duration % 60):02d}"
                if video_duration
                else "Unknown"
            )

            # Create header with metadata in MarkdownV2 format
            header_message = f"""✅ **YouTube Video Summary**

🎥 **{video_title}**
📺 **Channel:** {escape_markdown_v2(video_channel)}
⏱️ **Duration:** {duration_str}

"""

            # Create footer with tip in MarkdownV2 format
            footer_message = """

💡 **Tip:** Reply to this message to ask questions about the video content\\!"""

            # Escape the AI-generated summary for MarkdownV2 compatibility
            escaped_summary = escape_markdown_v2(summary)

            # Combine header + escaped summary + footer
            final_message = header_message + escaped_summary + footer_message

            final_msg = await processing_message.edit_text(
                final_message, parse_mode=ParseMode.MARKDOWN_V2
            )

            # Step 5: Store comprehensive video context with summary
            # Add summary to video context
            video_context["summary"] = summary
            video_context["summary_timestamp"] = asyncio.get_event_loop().time()

            # Cache the complete video context for follow-up questions
            transcript_cache.put_video_context(
                chat_id=update.effective_chat.id,
                message_id=final_msg.message_id,
                video_context=video_context,
            )

            REQUEST_COUNT.labels(command="summarize", status="success").inc()
            logger.info(
                f"Successfully summarized video {video_id} for chat {update.effective_chat.id}"
            )

    except asyncio.TimeoutError:
        await processing_message.edit_text(
            "⏱️ **Timeout**\n\n" "The request timed out\\. Please try again\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="summarize", status="timeout").inc()
    except Exception as e:
        logger.error(f"Unexpected error in summarize command: {e}")
        await processing_message.edit_text(
            f"❌ **Unexpected Error**\n\n"
            f"An unexpected error occurred\\.\n\n"
            f"Error: {sanitize_text(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="summarize", status="unexpected_error").inc()


async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle replies to summary messages for Q&A."""
    REQUEST_COUNT.labels(command="reply", status="received").inc()

    # Check if this is a reply to a bot message
    if (
        not update.message.reply_to_message
        or update.message.reply_to_message.from_user.id != context.bot.id
    ):
        return

    replied_message_id = update.message.reply_to_message.message_id
    chat_id = update.effective_chat.id
    user_question = update.message.text

    if not user_question or len(user_question.strip()) < 3:
        await update.message.reply_text(
            "❓ Please ask a specific question about the video content\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="invalid_question").inc()
        return

    # Retrieve video context or transcript from cache
    cached_data = transcript_cache.get(chat_id, replied_message_id)

    if not cached_data:
        await update.message.reply_text(
            "❌ **Video Data Not Found**\n\n"
            "The video data for this summary is no longer available in cache\\. "
            "Please run `/summarize` again with the video URL\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="transcript_not_found").inc()
        return

    # Extract transcript text based on data type
    if isinstance(cached_data, dict) and "video_id" in cached_data:
        # New video context format
        transcript_text = cached_data["transcript"]["text"]
        video_title = cached_data.get("title", "Unknown Video")
    elif isinstance(cached_data, dict) and "text" in cached_data:
        # Old transcript format (backward compatibility)
        transcript_text = cached_data["text"]
        video_title = "Video"
    else:
        await update.message.reply_text(
            "❌ **Invalid Data Format**\n\n"
            "The cached data format is not recognized\\. "
            "Please run `/summarize` again with the video URL\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="invalid_format").inc()
        return

    # Send thinking message
    thinking_message = await update.message.reply_text(
        "🤔 **Thinking\\.\\.\\.**\n\n" "Analyzing your question\\.\\.\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        # Generate answer with video context if available
        if isinstance(cached_data, dict) and "video_id" in cached_data:
            # Use rich video context for better answers
            answer = await gemini_client.answer_question_with_context(
                cached_data, user_question
            )
        else:
            # Fallback to transcript-only for backward compatibility
            answer = await gemini_client.answer_question(transcript_text, user_question)

            # Format and send response with header/footer in MarkdownV2 and answer in plain text
        header_message = f"""💬 **Answer about {escape_markdown_v2(video_title)}:**

"""

        footer_message = f"""

❓ **Your question:** {sanitize_text(user_question)}"""

        # Escape the AI-generated answer for MarkdownV2 compatibility
        escaped_answer = escape_markdown_v2(answer)

        # Combine header + escaped answer + footer
        response_message = header_message + escaped_answer + footer_message

        await thinking_message.edit_text(
            response_message, parse_mode=ParseMode.MARKDOWN_V2
        )

        REQUEST_COUNT.labels(command="reply", status="success").inc()
        logger.info(f"Successfully answered question for chat {chat_id}")

    except GeminiSafetyError as e:
        await thinking_message.edit_text(
            f"❌ **Content Safety Block**\n\n"
            f"The AI safety filters blocked this content\\.\n\n"
            f"Error: {sanitize_text(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="safety_block").inc()

    except GeminiRateLimitError as e:
        await thinking_message.edit_text(
            f"⏱️ **Rate Limit**\n\n"
            f"AI service is temporarily busy\\. Please try again in a few minutes\\.\n\n"
            f"Error: {sanitize_text(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="rate_limit").inc()

    except GeminiError as e:
        await thinking_message.edit_text(
            f"❌ **AI Processing Error**\n\n"
            f"Failed to generate answer\\.\n\n"
            f"Error: {sanitize_text(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="ai_error").inc()

    except Exception as e:
        logger.error(f"Unexpected error in reply handler: {e}")
        await thinking_message.edit_text(
            f"❌ **Unexpected Error**\n\n"
            f"An unexpected error occurred\\.\n\n"
            f"Error: {sanitize_text(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        REQUEST_COUNT.labels(command="reply", status="unexpected_error").inc()


async def auto_summary_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Automatically summarize YouTube links posted in group chats."""
    if update.effective_chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    message_text = update.message.text or ""

    # Find potential URLs in the message
    urls = re.findall(r"https://\S+", message_text)
    video_id: Optional[str] = None
    for url in urls:
        cleaned = url.rstrip(".,")
        vid = extract_video_id(cleaned)
        if vid:
            video_id = vid
            break

    if not video_id:
        return

    REQUEST_COUNT.labels(command="autosummary", status="received").inc()

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        video_context = await get_video_context(video_id)
        summary = await gemini_client.generate_brief_summary(
            video_context["transcript"]["text"]
        )

        escaped_summary = escape_markdown_v2(summary)
        reply_msg = await update.message.reply_text(
            f"🎬 **Quick Summary:**\n\n{escaped_summary}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

        # Store context for follow-up questions
        video_context["summary"] = summary
        video_context["summary_timestamp"] = asyncio.get_event_loop().time()
        transcript_cache.put_video_context(
            chat_id=update.effective_chat.id,
            message_id=reply_msg.message_id,
            video_context=video_context,
        )

        REQUEST_COUNT.labels(command="autosummary", status="success").inc()

    except Exception as e:
        logger.error(f"Auto summary failed: {e}")
        REQUEST_COUNT.labels(command="autosummary", status="error").inc()


async def handle_unknown_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle unknown commands."""
    REQUEST_COUNT.labels(command="unknown", status="received").inc()

    unknown_message = """❓ **Unknown Command**

I don't recognize that command\\. Here's what I can do:

**Available commands:**
• `/start` \\- Get started
• `/help` \\- Show help
• `/summarize <URL>` \\- Summarize a YouTube video
• `/stats` \\- Show bot statistics

**Tip:** Use `/help` to see detailed usage instructions\\."""

    try:
        await update.message.reply_text(
            unknown_message, parse_mode=ParseMode.MARKDOWN_V2
        )
        REQUEST_COUNT.labels(command="unknown", status="success").inc()
    except Exception as e:
        logger.error(f"Error sending unknown command message: {e}")
        REQUEST_COUNT.labels(command="unknown", status="error").inc()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors that occur during bot operation."""
    logger.error(f"Exception while handling an update: {context.error}")

    # Try to send a user-friendly error message if update contains a message
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ **Oops\\!**\n\n"
                "Something went wrong while processing your request\\. Please try again\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

    REQUEST_COUNT.labels(command="error", status="handled").inc()
