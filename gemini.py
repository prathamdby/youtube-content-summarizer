"""
Google Gemini AI integration with chunking and retry logic.
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from textwrap import dedent

from utils import estimate_tokens, Timer, GEMINI_TOKENS, sanitize_text


logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Custom exception for Gemini-related errors."""

    pass


class GeminiRateLimitError(GeminiError):
    """Raised when hitting Gemini rate limits."""

    pass


class GeminiSafetyError(GeminiError):
    """Raised when Gemini blocks content due to safety filters."""

    pass


class GeminiClient:
    """Async wrapper for Google GenAI with retry logic and chunking support."""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY environment variable is required")

        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    @retry(
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, GeminiRateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
    )
    async def _generate_content_with_retry(self, prompt: str) -> str:
        """Generate content with retry logic for rate limits and network errors."""
        async with self._semaphore:
            try:
                # Validate prompt is not empty
                if not prompt or not prompt.strip():
                    raise GeminiError("Prompt cannot be empty")

                with Timer("gemini_generation"):
                    # Disable all safety settings as requested
                    safety_settings = [
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_CIVIC_INTEGRITY",
                            threshold="BLOCK_NONE",
                        ),
                    ]

                    config = types.GenerateContentConfig(
                        temperature=0.7,
                        safety_settings=safety_settings,
                    )

                    # Ensure prompt is properly formatted and log for debugging
                    cleaned_prompt = prompt.strip()
                    logger.debug(
                        f"Sending prompt to Gemini (length: {len(cleaned_prompt)})"
                    )

                    # Use async execution with run_in_executor for sync API
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.model_name,
                            contents=cleaned_prompt,
                            config=config,
                        ),
                    )

                    # Check if response was blocked or empty
                    generated_text = None
                    finish_reason = None

                    if hasattr(response, "text") and response.text:
                        generated_text = response.text
                    elif hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, "finish_reason"):
                            finish_reason = candidate.finish_reason

                        if hasattr(candidate, "content"):
                            if (
                                hasattr(candidate.content, "parts")
                                and candidate.content.parts
                            ):
                                generated_text = candidate.content.parts[0].text
                            elif hasattr(candidate.content, "text"):
                                generated_text = candidate.content.text
                        elif hasattr(candidate, "text"):
                            generated_text = candidate.text

                    # Handle MAX_TOKENS finish reason specifically
                    if finish_reason and "MAX_TOKENS" in str(finish_reason):
                        if generated_text and generated_text.strip():
                            logger.warning(
                                f"Response truncated due to MAX_TOKENS limit. Partial response received."
                            )
                            return generated_text.strip()
                        else:
                            logger.error(
                                f"MAX_TOKENS reached but no content generated. Consider reducing input size or increasing max_output_tokens."
                            )
                            raise GeminiError(
                                "Response truncated due to token limit - try reducing input size or increasing max_output_tokens"
                            )

                    if not generated_text or not generated_text.strip():
                        # Enhanced safety check
                        if finish_reason and "SAFETY" in str(finish_reason):
                            raise GeminiSafetyError(
                                f"Content blocked by safety filters: {finish_reason}"
                            )
                        else:
                            logger.error(
                                f"No text found in response. Response: {response}"
                            )
                            raise GeminiError(
                                "Empty response from Gemini - no text content found"
                            )

                    # Track token usage
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        GEMINI_TOKENS.labels(type="input").inc(
                            response.usage_metadata.prompt_token_count or 0
                        )
                        GEMINI_TOKENS.labels(type="output").inc(
                            response.usage_metadata.candidates_token_count or 0
                        )

                    return generated_text.strip()

            except APIError as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if (
                    "rate limit" in error_str
                    or "quota" in error_str
                    or "too many requests" in error_str
                ):
                    logger.warning(f"Rate limit hit: {e}")
                    raise GeminiRateLimitError(f"Rate limit exceeded: {e}")

                # Handle safety blocks
                if "safety" in error_str or "blocked" in error_str:
                    logger.warning(f"Content safety block: {e}")
                    raise GeminiSafetyError(f"Content blocked by safety filters: {e}")

                # Handle other API errors
                if "api key" in error_str or "authentication" in error_str:
                    raise GeminiError(f"Authentication error: {e}")

                logger.error(f"Gemini generation error: {e}")
                raise GeminiError(f"Generation failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise GeminiError(f"Unexpected error: {e}")

    def _chunk_text(self, text: str, max_tokens: int = 25000) -> List[str]:
        """
        Split text into chunks that fit within token limits.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        if estimate_tokens(text) <= max_tokens:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, start new chunk
            if (
                current_chunk
                and estimate_tokens(current_chunk + "\n\n" + paragraph) > max_tokens
            ):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If any chunk is still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if estimate_tokens(chunk) <= max_tokens:
                final_chunks.append(chunk)
            else:
                # Split by sentences
                sentences = chunk.split(". ")
                sub_chunk = ""

                for sentence in sentences:
                    if (
                        sub_chunk
                        and estimate_tokens(sub_chunk + ". " + sentence) > max_tokens
                    ):
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                    else:
                        if sub_chunk:
                            sub_chunk += ". " + sentence
                        else:
                            sub_chunk = sentence

                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())

        return final_chunks

    async def generate_summary(self, transcript: str, max_tokens: int = 120000) -> str:
        """
        Generate summary of transcript with automatic chunking if needed.

        Args:
            transcript: Full transcript text
            max_tokens: Maximum tokens for single request (default 120k)

        Returns:
            Generated summary

        Raises:
            GeminiError: If generation fails
        """
        try:
            logger.info(f"Generating summary for transcript ({len(transcript)} chars)")

            # Estimate tokens
            estimated_tokens = estimate_tokens(transcript)
            logger.info(f"Estimated tokens: {estimated_tokens}")

            # If transcript fits in single request, generate summary directly
            if estimated_tokens <= max_tokens:
                prompt = self._build_single_summary_prompt(transcript)
                return await self._generate_content_with_retry(prompt)

            # For large transcripts, use map-reduce approach
            else:
                return await self._generate_summary_map_reduce(transcript)

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise GeminiError(f"Failed to generate summary: {str(e)}")

    def _build_single_summary_prompt(self, transcript: str) -> str:
        """Build optimized single summary prompt using modern prompt design principles."""
        return dedent(
            f"""<system>
You are an expert YouTube Content Analyst specializing in extracting maximum value from video transcripts across all domains. Your role is to transform lengthy video content into actionable, comprehensive summaries that save viewers time while preserving all essential insights.
</system>

<objective>
Analyze the provided YouTube transcript and create a structured, comprehensive summary that captures the complete value proposition of the video for viewers who need to understand the content quickly.
</objective>

<instructions>
STEP 1: Content Analysis
- Read the entire transcript to understand the overarching narrative and key themes
- Identify the primary subject matter, purpose, and target audience
- Note critical segments, major transitions, and emphasis points
- Filter out technical artifacts (timestamps, audio cues like "[Music]", "[Applause]")

STEP 2: Information Extraction  
- Extract actionable insights, expert knowledge, and unique perspectives
- Identify concrete data points, statistics, and evidence presented
- Note recurring themes and concepts that receive emphasis
- Capture any practical steps, recommendations, or conclusions

STEP 3: Summary Construction
- Structure response using the EXACT format specified below
- Prioritize substance and actionable value over superficial details
- Ensure each section provides genuine utility to someone who hasn't watched
- Focus on information that viewers can remember and potentially implement

CRITICAL CONSTRAINTS:
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)
- Length limit: Under 3000 characters total to fit Telegram message limits
- Content focus: Substantive information only (ignore filler, small talk, transitions)
- Writing style: Natural conversational tone as if explaining to a colleague
</instructions>

<output_format>
🎯 Main Topic
[Single clear sentence describing the video's fundamental subject and purpose]

🔑 Key Points
• [First major insight - specific, actionable, valuable to viewers]
• [Second major insight - distinct from first, includes relevant details]  
• [Third major insight - different aspect, focuses on practical value]
• [Fourth insight if applicable - maintains focus on highest-value content]
• [Fifth insight if applicable - ensures comprehensive coverage without redundancy]

💡 Key Takeaways
[2-3 sentences summarizing what viewers should remember and potentially act upon after watching this video. Focus on the "so what" - why this matters and how it can be applied.]

📌 Context & Background
[Relevant background information, broader context, or important references that enhance understanding and help viewers connect this content to related topics or applications.]

VERIFICATION: Ensure total response is under 3000 characters while maintaining comprehensiveness and value.
</output_format>

<context>
<transcript>
{transcript}
</transcript>
</context>

Begin your analysis:"""
        ).strip()

    async def _generate_summary_map_reduce(self, transcript: str) -> str:
        """
        Generate summary using map-reduce for large transcripts.

        Args:
            transcript: Full transcript text

        Returns:
            Final summary
        """
        logger.info("Using map-reduce approach for large transcript")

        # Step 1: Chunk the transcript
        chunks = self._chunk_text(transcript, max_tokens=25000)
        logger.info(f"Split transcript into {len(chunks)} chunks")

        # Step 2: Generate summary for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = self._build_chunk_summary_prompt(chunk, i + 1, len(chunks))

            try:
                summary = await self._generate_content_with_retry(prompt)
                chunk_summaries.append(summary)
                logger.info(f"Generated summary for chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i+1}: {e}")
                chunk_summaries.append(
                    f"[Summary unavailable for this segment due to error: {str(e)}]"
                )

        # Step 3: Combine chunk summaries into final summary
        return await self._generate_final_synthesis(chunk_summaries)

    def _build_chunk_summary_prompt(
        self, chunk: str, chunk_num: int, total_chunks: int
    ) -> str:
        """Build optimized chunk summary prompt using modern prompt design principles."""
        return dedent(
            f"""<system>
You are an expert content analyst specializing in extracting key insights from video transcript segments. Your role is to identify and summarize the most valuable information from each portion of a larger video.
</system>

<objective>
Analyze this specific segment (Part {chunk_num} of {total_chunks}) of a YouTube video transcript and extract the most important information, insights, and valuable content discussed in this portion.
</objective>

<instructions>
STEP 1: Segment Analysis
- Focus exclusively on substantive content in this specific segment
- Ignore timestamps, speaker changes, and audio cues ("[Music]", "[Applause]")
- Identify main topics, arguments, and insights discussed in this portion

STEP 2: Key Information Extraction
- Extract important facts, expert insights, and actionable information
- Note significant details that contribute to the video's overall value
- Capture any data points, examples, or practical recommendations
- Identify concepts that will be important for understanding the complete video

STEP 3: Context-Aware Summary
- Remember this is part of a larger video - maintain context
- Focus on information that will be crucial for the final comprehensive summary
- Don't miss important details that might seem minor but are contextually significant

CRITICAL CONSTRAINTS:
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)
- Length limit: Under 3000 characters to fit Telegram message limits
- Content focus: High-value information only (skip filler, transitions, tangents)
- Writing style: Natural conversational tone as if briefing a colleague
</instructions>

<context>
<segment_info>
Current Segment: {chunk_num} of {total_chunks}
</segment_info>

<transcript_segment>
{chunk}
</transcript_segment>
</context>

Provide your focused analysis of the key information in this segment:"""
        ).strip()

    async def _generate_final_synthesis(self, chunk_summaries: List[str]) -> str:
        """Generate final synthesis from chunk summaries using modern prompt design principles."""
        combined_summaries = "\n\n".join(
            [
                f"**Segment {i+1}:** {summary}"
                for i, summary in enumerate(chunk_summaries)
            ]
        )

        prompt = dedent(
            f"""<system>
You are an expert YouTube content synthesis specialist who excels at creating comprehensive, cohesive summaries from multiple content segments. Your role is to weave together segment analyses into a unified, valuable summary that captures the complete video's worth.
</system>

<objective>
Synthesize these segment-by-segment summaries into one comprehensive, cohesive summary that captures the full value proposition of the complete YouTube video for viewers who need to understand the entire content efficiently.
</objective>

<instructions>
STEP 1: Integration Analysis
- Identify overarching themes and concepts that span multiple segments
- Recognize how different segments build upon each other and connect
- Map the logical flow and progression of ideas throughout the video
- Note recurring concepts and how they develop across segments

STEP 2: Content Prioritization and Synthesis
- Extract the most valuable insights from across all segments
- Eliminate redundancy while preserving all important details and nuances
- Focus on information that provides genuine, actionable value to viewers
- Ensure no critical insights are lost in the synthesis process

STEP 3: Comprehensive Summary Construction
- Use the EXACT output format specified below
- Ensure the summary reads as a cohesive whole, not disconnected parts
- Make each section substantive, actionable, and comprehensive
- Focus on what viewers need to know and can act upon

CRITICAL CONSTRAINTS:
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)
- Length limit: Under 3000 characters total to fit Telegram message limits
- Content focus: Complete video value (not just segment highlights)
- Writing style: Natural conversational tone as if summarizing for a colleague
- Completeness: Must capture the full video's value proposition
</instructions>

<output_format>
🎯 Main Topic
[Clear, comprehensive description of what this entire video covers and its primary purpose]

🔑 Key Points
• [Most important insight or information from the complete video]
• [Second key point that adds distinct value and covers different aspects]
• [Third key point covering another important dimension]
• [Fourth point if needed - ensure comprehensive coverage without redundancy]
• [Fifth point if needed - maintain focus on highest-value content for viewers]

💡 Key Takeaways
[2-3 sentences capturing what viewers should remember and potentially implement from the entire video. Focus on actionable outcomes and practical applications.]

📌 Context & Background
[Relevant context, background information, broader implications, or connections to related topics discussed throughout the video]

VERIFICATION: Ensure total response is under 3000 characters while maintaining comprehensiveness and capturing the complete video's value.
</output_format>

<context>
<segment_summaries>
{combined_summaries}
</segment_summaries>
</context>

Begin your comprehensive synthesis:"""
        ).strip()

        return await self._generate_content_with_retry(prompt)

    async def generate_brief_summary(
        self, transcript: str, max_tokens: int = 120000
    ) -> str:
        """Generate a concise 2-3 sentence summary of a transcript."""
        try:
            estimated_tokens = estimate_tokens(transcript)

            if estimated_tokens <= max_tokens:
                prompt = self._build_brief_summary_prompt(transcript)
                return await self._generate_content_with_retry(prompt)

            # For large transcripts, summarize in chunks then synthesize
            chunks = self._chunk_text(transcript, max_tokens=25000)
            chunk_summaries = []
            for chunk in chunks:
                p = self._build_brief_summary_prompt(chunk)
                chunk_summaries.append(await self._generate_content_with_retry(p))

            synthesis_prompt = self._build_brief_synthesis_prompt(
                "\n\n".join(chunk_summaries)
            )
            return await self._generate_content_with_retry(synthesis_prompt)

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Brief summary generation failed: {e}")
            raise GeminiError(f"Failed to generate brief summary: {str(e)}")

    def _build_brief_summary_prompt(self, transcript: str) -> str:
        """Prompt for concise summary of a transcript."""
        return dedent(
            f"""
        Summarize the following YouTube transcript in no more than three sentences.
        Use PLAIN TEXT with emojis and no markdown formatting.
        Keep your response under 300 characters to fit within Telegram's message limits.

        <transcript>
        {transcript}
        </transcript>

        **Concise summary (PLAIN TEXT WITH EMOJIS ONLY, under 300 characters):**
        """
        ).strip()

    def _build_brief_synthesis_prompt(self, summaries: str) -> str:
        """Prompt to combine chunk summaries into a concise overview."""
        return dedent(
            f"""
        Combine these short summaries into one concise overview (max three sentences).
        Keep your response under 300 characters to fit within Telegram's message limits.

        <summaries>
        {summaries}
        </summaries>

        **Final concise summary (PLAIN TEXT WITH EMOJIS ONLY, under 300 characters):**
        """
        ).strip()

    async def answer_question(self, transcript: str, question: str) -> str:
        """
        Answer a question about the transcript content.

        Args:
            transcript: Full transcript text
            question: User's question

        Returns:
            Answer to the question

        Raises:
            GeminiError: If generation fails
        """
        try:
            logger.info(f"Answering question about transcript: {question[:100]}...")

            # Sanitize the question
            safe_question = sanitize_text(question)

            # For Q&A, we can use larger context since it's more focused
            if estimate_tokens(transcript) > 100000:
                # If transcript is very large, summarize it first for context
                context_summary = await self.generate_summary(transcript)
                prompt = self._build_summary_qa_prompt(context_summary, safe_question)
            else:
                prompt = self._build_transcript_qa_prompt(transcript, safe_question)

            answer = await self._generate_content_with_retry(prompt)
            logger.info("Generated answer for user question")

            return answer

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise GeminiError(f"Failed to answer question: {str(e)}")

    def _build_transcript_qa_prompt(self, transcript: str, question: str) -> str:
        """Build optimized Q&A prompt for full transcript using modern design principles."""
        return dedent(
            f"""<system>
You are an expert YouTube Video Intelligence Analyst with exceptional ability to answer questions about video content with precision and helpfulness. Your role is to analyze video transcripts and provide accurate, comprehensive answers based exclusively on the content presented.
</system>

<objective>
Answer the user's specific question about this YouTube video by thoroughly analyzing the transcript and providing the most accurate, helpful response possible based solely on the video content.
</objective>

<instructions>
STEP 1: Question Analysis
- Carefully parse what the user is asking and identify the specific information they need
- Determine the scope of information required (specific facts, general concepts, examples, etc.)
- Assess whether this requires direct quotes, summarization, or interpretation

STEP 2: Content Search and Analysis
- Systematically search through the transcript for all relevant information
- Identify specific details, examples, explanations, or data points that address the question
- Note the context around relevant information to ensure accurate interpretation
- Collect supporting evidence and examples when available

STEP 3: Response Construction
- Provide a direct, clear answer based exclusively on video content
- Include specific details, examples, and relevant context when available
- Structure the response to be comprehensive yet concise
- Acknowledge any limitations in the available information

CRITICAL CONSTRAINTS:
- Information source: EXCLUSIVELY the provided transcript (NO external knowledge)
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)
- Length limit: Under 2500 characters to fit Telegram message limits with headers/footers
- Content focus: Direct answers with supporting details from the video
- Writing style: Natural conversational tone as if explaining to a colleague

RESPONSE GUIDELINES:
- If information is clearly available: Provide comprehensive answer with specific details and examples
- If information is partially available: Share what's available and clearly indicate limitations
- If information is missing: State clearly that the topic isn't covered and explain what the video does discuss
</instructions>

<context>
<transcript>
{transcript}
</transcript>

<user_question>
{question}
</user_question>
</context>

Provide your precise answer based on the video content:"""
        ).strip()

    def _build_summary_qa_prompt(self, summary: str, question: str) -> str:
        """Build optimized Q&A prompt for summary-based questions using modern design principles."""
        return dedent(
            f"""<system>
You are an expert YouTube Video Intelligence Analyst specializing in answering questions based on video summaries. Your role is to provide accurate, helpful answers using the available summary information while being transparent about the scope and limitations of the source material.
</system>

<objective>
Answer the user's specific question about this YouTube video using the provided summary. Deliver the most accurate and helpful response possible based on the summary content, while clearly indicating when information may be limited by the summary format.
</objective>

<instructions>
STEP 1: Question Analysis  
- Parse the user's question to understand exactly what information they're seeking
- Identify whether the question asks for specific details, general concepts, or practical applications
- Determine if the question can be fully or partially answered from the summary

STEP 2: Summary Analysis
- Systematically review the summary for all information relevant to the question
- Extract specific details, insights, and examples that address the user's inquiry
- Note any context or background information that supports the answer

STEP 3: Response Construction
- Provide a direct, clear answer based on the summary content
- Include relevant details and examples when available in the summary
- Be transparent about any limitations due to the summary format
- Structure the response to be comprehensive yet concise

CRITICAL CONSTRAINTS:
- Information source: EXCLUSIVELY the provided summary (NO external knowledge)
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)
- Length limit: Under 2500 characters to fit Telegram message limits with headers/footers
- Content focus: Direct answers with supporting details from the summary
- Writing style: Natural conversational tone as if explaining to a colleague

RESPONSE GUIDELINES:
- If summary contains clear answer: Provide comprehensive response with available details and context
- If summary has partial information: Share what's available and acknowledge limitations clearly
- If information is missing: State that the specific details aren't in the summary and explain what information is available
</instructions>

<context>
<video_summary>
{summary}
</video_summary>

<user_question>
{question}
</user_question>
</context>

Provide your answer based on the summary content:"""
        ).strip()

    async def answer_question_with_context(
        self, video_context: dict, question: str
    ) -> str:
        """
        Answer a question using comprehensive video context including metadata.

        Args:
            video_context: Complete video context with metadata and transcript
            question: User's question

        Returns:
            Answer to the question using all available context

        Raises:
            GeminiError: If generation fails
        """
        try:
            logger.info(f"Answering question with full context: {question[:100]}...")

            # Sanitize the question
            safe_question = sanitize_text(question)

            # Extract video metadata
            title = video_context.get("title", "Unknown")
            channel = video_context.get("channel") or video_context.get(
                "uploader", "Unknown"
            )
            description = video_context.get("description", "")[
                :500
            ]  # Limit description length
            duration = video_context.get("duration", 0)
            upload_date = video_context.get("upload_date", "Unknown")
            categories = video_context.get("categories", [])
            tags = video_context.get("tags", [])
            view_count = video_context.get("view_count")

            # Format duration
            duration_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration
                else "Unknown"
            )

            # Get transcript text
            transcript_text = video_context["transcript"]["text"]

            # Build comprehensive context prompt
            metadata_context = f"""Video Title: {title}
Channel: {channel}
Duration: {duration_str}
Upload Date: {upload_date}"""

            if view_count:
                metadata_context += f"\nView Count: {view_count:,}"

            if categories:
                metadata_context += f"\nCategories: {', '.join(categories[:3])}"

            if tags and len(tags) > 0:
                metadata_context += f"\nTags: {', '.join(tags[:5])}"

            if description:
                metadata_context += f"\nDescription (excerpt): {description}..."

            # For large transcripts, use summary instead
            if estimate_tokens(transcript_text) > 80000:
                # Use pre-generated summary if available, otherwise generate one
                if video_context.get("summary"):
                    content_context = f"Video Summary:\n{video_context['summary']}"
                else:
                    summary = await self.generate_summary(transcript_text)
                    content_context = f"Video Summary:\n{summary}"
            else:
                content_context = f"Video Transcript:\n{transcript_text}"

            prompt = self._build_context_qa_prompt(
                {
                    "title": title,
                    "channel": channel,
                    "description": description,
                    "duration_str": duration_str,
                    "upload_date": upload_date,
                    "categories": categories[:3],
                    "tags": tags[:5],
                    "view_count": view_count,
                },
                content_context,
                safe_question,
            )

            answer = await self._generate_content_with_retry(prompt)
            logger.info("Generated context-aware answer for user question")

            return answer

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Context-aware question answering failed: {e}")
            raise GeminiError(f"Failed to answer question with context: {str(e)}")

    def _build_context_qa_prompt(
        self, context_data: Dict[str, Any], content_context: str, question: str
    ) -> str:
        """Build comprehensive context-aware Q&A prompt using modern design principles."""
        # Build metadata section
        metadata_lines = [
            f"Title: {context_data['title']}",
            f"Channel: {context_data['channel']}",
            f"Duration: {context_data['duration_str']}",
            f"Upload Date: {context_data['upload_date']}",
        ]

        if context_data["view_count"]:
            metadata_lines.append(f"View Count: {context_data['view_count']:,}")

        if context_data["categories"]:
            metadata_lines.append(
                f"Categories: {', '.join(context_data['categories'])}"
            )

        if context_data["tags"]:
            metadata_lines.append(f"Tags: {', '.join(context_data['tags'])}")

        if context_data["description"]:
            metadata_lines.append(f"Description: {context_data['description']}...")

        metadata_section = "\n".join(metadata_lines)

        return dedent(
            f"""<system>
You are an expert YouTube Video Intelligence Analyst with comprehensive ability to answer questions about videos using all available information sources. Your role is to provide accurate, helpful answers by strategically utilizing video metadata, summaries, and contextual information to best serve the user's needs.
</system>

<objective>
Answer the user's specific question about this YouTube video using the optimal combination of available information sources (metadata, summary, and context). Provide the most accurate and comprehensive response possible while being transparent about your information sources and any limitations.
</objective>

<instructions>
STEP 1: Question Classification and Analysis
- Analyze the user's question to determine the type of information needed
- Classify whether this is a metadata question (title, creator, duration), content question (what's discussed), or analytical question (insights, applications)
- Determine which information sources will be most relevant and reliable

STEP 2: Multi-Source Information Gathering
- For metadata questions: Prioritize video metadata as the primary source
- For content questions: Use the summary as the primary source with metadata for context
- For analytical questions: Combine summary insights with metadata context
- Systematically extract all relevant information from the appropriate sources

STEP 3: Integrated Response Construction
- Synthesize information from multiple sources into a coherent, comprehensive answer
- Provide specific details and examples when available
- Structure the response to directly address the user's question
- Be transparent about which sources provided the information and any limitations

CRITICAL CONSTRAINTS:
- Information sources: ONLY the provided metadata, summary, and available context (NO external knowledge)
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)  
- Length limit: Under 2500 characters to fit Telegram message limits with headers/footers
- Content focus: Direct answers using the best available information sources
- Writing style: Natural conversational tone as if explaining to a colleague

SOURCE PRIORITIZATION GUIDELINES:
- Basic video info (title, creator, duration): Use metadata as authoritative source
- Content and topics discussed: Use summary as primary source with metadata context
- Detailed insights and analysis: Rely primarily on summary content
- Missing information: Clearly state what isn't available and explain what information is provided
</instructions>

<context>
<video_metadata>
{metadata_section}
</video_metadata>

<video_content>
{content_context}
</video_content>

<user_question>
{question}
</user_question>
</context>

Provide your comprehensive answer using the best available information sources:"""
        ).strip()

    def _build_general_context_qa_prompt(
        self, metadata: Dict[str, Any], summary: str, question: str
    ) -> str:
        """Build optimized general Q&A prompt using modern design principles."""
        return dedent(
            f"""<system>
You are an expert YouTube Video Intelligence Analyst with comprehensive ability to answer questions about videos using all available information sources. Your role is to provide accurate, helpful answers by strategically utilizing video metadata, summaries, and contextual information to best serve the user's needs.
</system>

<objective>
Answer the user's specific question about this YouTube video using the optimal combination of available information sources (metadata, summary, and context). Provide the most accurate and comprehensive response possible while being transparent about your information sources and any limitations.
</objective>

<instructions>
STEP 1: Question Classification and Analysis
- Analyze the user's question to determine the type of information needed
- Classify whether this is a metadata question (title, creator, duration), content question (what's discussed), or analytical question (insights, applications)
- Determine which information sources will be most relevant and reliable

STEP 2: Multi-Source Information Gathering
- For metadata questions: Prioritize video metadata as the primary source
- For content questions: Use the summary as the primary source with metadata for context
- For analytical questions: Combine summary insights with metadata context
- Systematically extract all relevant information from the appropriate sources

STEP 3: Integrated Response Construction
- Synthesize information from multiple sources into a coherent, comprehensive answer
- Provide specific details and examples when available
- Structure the response to directly address the user's question
- Be transparent about which sources provided the information and any limitations

CRITICAL CONSTRAINTS:
- Information sources: ONLY the provided metadata, summary, and available context (NO external knowledge)
- Output format: PLAIN TEXT with emojis only (NO markdown, bold, asterisks, or special formatting)  
- Length limit: Under 2500 characters to fit Telegram message limits with headers/footers
- Content focus: Direct answers using the best available information sources
- Writing style: Natural conversational tone as if explaining to a colleague

SOURCE PRIORITIZATION GUIDELINES:
- Basic video info (title, creator, duration): Use metadata as authoritative source
- Content and topics discussed: Use summary as primary source with metadata context
- Detailed insights and analysis: Rely primarily on summary content
- Missing information: Clearly state what isn't available and explain what information is provided
</instructions>

<context>
<video_metadata>
Title: {metadata.get('title', 'Not available')}
Creator: {metadata.get('creator', 'Not available')}
Duration: {metadata.get('duration', 'Not available')}
View Count: {metadata.get('view_count', 'Not available')}
Upload Date: {metadata.get('upload_date', 'Not available')}
</video_metadata>

<video_summary>
{summary}
</video_summary>

<user_question>
{question}
</user_question>
</context>

Provide your comprehensive answer using the best available information sources:"""
        ).strip()


# Global Gemini client instance
gemini_client = GeminiClient()
