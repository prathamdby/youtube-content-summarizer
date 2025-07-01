"""
YouTube Content Summarizer Telegram Bot
Main entry point for the bot application.
"""

import os
import asyncio
import logging
import signal
from typing import Optional
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from utils import setup_logging, start_metrics_server
from handlers import (
    start_command,
    help_command,
    stats_command,
    summarize_command,
    handle_reply,
    handle_unknown_command,
    error_handler,
)
from cache import transcript_cache
from gemini import gemini_client


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class YouTubeSummarizerBot:
    """Main bot application class."""

    def __init__(self, token: str):
        self.token = token
        self.application: Optional[Application] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the bot application and all components."""
        logger.info("Initializing YouTube Summarizer Bot...")

        # Initialize Telegram bot application
        self.application = Application.builder().token(self.token).build()

        # Add command handlers
        self.application.add_handler(CommandHandler("start", start_command))
        self.application.add_handler(CommandHandler("help", help_command))
        self.application.add_handler(CommandHandler("stats", stats_command))
        self.application.add_handler(CommandHandler("summarize", summarize_command))

        # Add message handlers
        # Handle replies to bot messages (for Q&A)
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & filters.REPLY & ~filters.COMMAND, handle_reply
            )
        )

        # Handle unknown commands
        self.application.add_handler(
            MessageHandler(filters.COMMAND, handle_unknown_command)
        )

        # Add error handler
        self.application.add_error_handler(error_handler)

        # Start cache cleanup task
        await transcript_cache.start_cleanup_task()

        logger.info("Bot initialization complete")

    async def start(self) -> None:
        """Start the bot and all background services."""
        if not self.application:
            await self.initialize()

        logger.info("Starting YouTube Summarizer Bot...")

        # Start metrics server
        start_metrics_server(port=8000)

        # Initialize the application
        await self.application.initialize()
        await self.application.start()

        # Start polling for updates
        await self.application.updater.start_polling(
            allowed_updates=["message"], drop_pending_updates=True
        )

        logger.info("Bot is running and polling for updates...")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the bot and cleanup resources."""
        logger.info("Shutting down YouTube Summarizer Bot...")

        if self.application:
            # Stop polling
            if self.application.updater.running:
                await self.application.updater.stop()

            # Stop application
            await self.application.stop()
            await self.application.shutdown()

        # Stop cache cleanup task
        await transcript_cache.stop_cleanup_task()

        # Signal shutdown complete
        self._shutdown_event.set()

        logger.info("Bot shutdown complete")

    def signal_handler(self, signum: int) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())


async def health_check_server():
    """Simple health check HTTP server."""
    from aiohttp import web

    async def health_check(request):
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "service": "youtube-summarizer-bot",
                "cache_stats": transcript_cache.get_stats(),
            }
        )

    app = web.Application()
    app.router.add_get("/ping", health_check)
    app.router.add_get("/health", health_check)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()

    logger.info("Health check server started on port 8080")
    return runner


async def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger.info("Starting YouTube Content Summarizer Bot")

    # Get configuration from environment
    bot_token = os.getenv("BOT_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not bot_token:
        logger.error("BOT_TOKEN environment variable is required")
        return 1

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable is required")
        return 1

    # Validate Gemini client initialization
    try:
        # Test that gemini client is properly initialized
        logger.info(f"Gemini client initialized with model: {gemini_client.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return 1

    # Create and start bot
    bot = YouTubeSummarizerBot(bot_token)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        bot.signal_handler(signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start health check server
        health_runner = await health_check_server()

        # Start the bot
        await bot.start()

        # Cleanup health check server
        await health_runner.cleanup()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    logger.info("Bot stopped successfully")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
