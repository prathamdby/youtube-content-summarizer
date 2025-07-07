"""
Google Gemini AI integration with chunking and retry logic.
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai

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
        self, api_key: Optional[str] = None, model_name: str = "gemma-3-27b-it"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY environment variable is required")

        # Configure the client
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(model_name=self.model_name)

        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    @retry(
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, GeminiRateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
    )
    async def _generate_content_with_retry(self, prompt: str, **kwargs) -> str:
        """Generate content with retry logic for rate limits and network errors."""
        async with self._semaphore:
            try:
                with Timer("gemini_generation"):
                    generation_config = genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=kwargs.get("max_output_tokens", 1000),
                        top_p=0.8,
                        top_k=40,
                    )

                    # Use async generation
                    response = await self.model.generate_content_async(
                        prompt, generation_config=generation_config
                    )

                    # Check if response was blocked
                    if not response.text:
                        if (
                            response.prompt_feedback
                            and response.prompt_feedback.block_reason
                        ):
                            raise GeminiSafetyError(
                                f"Content blocked: {response.prompt_feedback.block_reason}"
                            )
                        else:
                            raise GeminiError("Empty response from Gemini")

                    # Track token usage
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        GEMINI_TOKENS.labels(type="input").inc(
                            response.usage_metadata.prompt_token_count or 0
                        )
                        GEMINI_TOKENS.labels(type="output").inc(
                            response.usage_metadata.candidates_token_count or 0
                        )

                    return response.text.strip()

            except Exception as e:
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
            max_tokens: Maximum tokens for single request (default 120k for Gemma 3 27B)

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
                return await self._generate_content_with_retry(
                    prompt, max_output_tokens=1000
                )

            # For large transcripts, use map-reduce approach
            else:
                return await self._generate_summary_map_reduce(transcript)

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise GeminiError(f"Failed to generate summary: {str(e)}")

    def _build_single_summary_prompt(self, transcript: str) -> str:
        """Build optimized single summary prompt using advanced design principles."""
        return dedent(
            f"""
        <role>
        You are an expert YouTube content analyst and summarization specialist with deep expertise in extracting key insights from video content across all domains including education, entertainment, technology, business, and more.
        </role>

        <task>
        Your mission is to analyze a YouTube video transcript and create a comprehensive, well-structured summary that captures the essence and value of the content for viewers.
        </task>

        <instructions>
        **IMPORTANT:** Follow this exact thinking and analysis process:

        1. **Content Analysis Phase**
           - First, read through the entire transcript to understand the overall narrative and structure
           - Identify the primary topic, theme, and purpose of the video
           - Note any key segments, transitions, or important discussions

        2. **Information Extraction Phase**
           - Extract the most valuable and actionable information
           - Identify recurring themes and emphasis points
           - Note any expert insights, data points, or unique perspectives shared

        3. **Summary Construction Phase**
           - Structure your response using the EXACT format specified below
           - Ensure each section provides genuine value to someone who hasn't watched the video
           - Focus on substance over superficial details

        **ALWAYS ignore technical elements** like timestamps, speaker labels, or audio cues (e.g., "[Music]", "[Applause]") - focus exclusively on substantive content.

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.
        </instructions>

        <output_format>
        **CRITICAL:** Structure your summary using this EXACT format with PLAIN TEXT only:

        🎯 Main Topic
        [One clear sentence describing what this video is fundamentally about]

        🔑 Key Points
        • [First major point or insight - be specific and actionable]
        • [Second major point or insight - include relevant details]
        • [Third major point or insight - focus on value for viewers]
        • [Fourth point if applicable - maintain focus on most important content]
        • [Fifth point if applicable - ensure each point adds distinct value]

        💡 Key Takeaways
        [2-3 sentences summarizing what viewers should remember and potentially act upon after watching this video]

        📌 Context & Background
        [Relevant background information, references, or context that enhances understanding of the content]
        </output_format>

        <transcript>
        {transcript}
        </transcript>

        **Your analysis and summary (PLAIN TEXT WITH EMOJIS ONLY):**
        """
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
                summary = await self._generate_content_with_retry(
                    prompt, max_output_tokens=500
                )
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
        """Build optimized chunk summary prompt."""
        return dedent(
            f"""
        <role>
        You are an expert content analyst specializing in extracting key insights from video transcript segments.
        </role>

        <task>
        Analyze this segment of a YouTube video transcript (Part {chunk_num} of {total_chunks}) and extract the most important information and insights discussed in this specific portion.
        </task>

        <instructions>
        **IMPORTANT:** Your analysis should:

        1. **Focus on substance** - Ignore timestamps, speaker changes, and audio cues
        2. **Extract key information** - Identify main points, important details, and valuable insights
        3. **Maintain context** - Remember this is part of a larger video
        4. **Be comprehensive** - Don't miss important information that might be critical for the overall summary

        **ALWAYS prioritize:** Important facts, expert insights, actionable information, key arguments, and significant details that contribute to the video's overall value.

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO asterisks, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.
        </instructions>

        <transcript_segment>
        {chunk}
        </transcript_segment>

        **Provide a focused summary of the key information in this segment (PLAIN TEXT WITH EMOJIS ONLY):**
        """
        ).strip()

    async def _generate_final_synthesis(self, chunk_summaries: List[str]) -> str:
        """Generate final synthesis from chunk summaries with enhanced prompt."""
        combined_summaries = "\n\n".join(
            [
                f"**Segment {i+1}:** {summary}"
                for i, summary in enumerate(chunk_summaries)
            ]
        )

        prompt = dedent(
            f"""
        <role>
        You are an expert YouTube content synthesis specialist who excels at creating comprehensive summaries from multiple content segments.
        </role>

        <task>
        Synthesize these segment-by-segment summaries into one cohesive, comprehensive summary that captures the full value of the complete YouTube video.
        </task>

        <instructions>
        **CRITICAL:** Follow this synthesis process:

        1. **Integration Analysis**
           - Identify overarching themes that span multiple segments
           - Recognize how different segments build upon each other
           - Note the logical flow and progression of ideas

        2. **Content Prioritization**
           - Extract the most valuable insights from all segments
           - Eliminate redundancy while preserving important details
           - Focus on information that provides genuine value to viewers

        3. **Final Structure Creation**
           - Use the EXACT output format specified below
           - Ensure the summary reads as a cohesive whole, not disconnected parts
           - Make each section substantive and actionable

        **ALWAYS ensure** your final summary captures the complete video's value and could stand alone as a comprehensive overview.

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO asterisks, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.
        </instructions>

        <output_format>
        **CRITICAL:** Structure your summary using this EXACT format with PLAIN TEXT only:

        🎯 Main Topic
        [Clear, comprehensive description of what this entire video covers]

        🔑 Key Points
        • [Most important insight or information from the complete video]
        • [Second key point that adds distinct value]
        • [Third key point covering different aspect]
        • [Fourth point if needed - ensure comprehensive coverage]
        • [Fifth point if needed - maintain focus on highest-value content]

        💡 Key Takeaways
        [2-3 sentences capturing what viewers should remember and potentially implement from the entire video]

        📌 Context & Background
        [Relevant context, background information, or broader implications discussed in the video]
        </output_format>

        <segment_summaries>
        {combined_summaries}
        </segment_summaries>

        **Your comprehensive synthesis (PLAIN TEXT WITH EMOJIS ONLY):**
        """
        ).strip()

        return await self._generate_content_with_retry(prompt, max_output_tokens=1000)

    async def generate_brief_summary(self, transcript: str, max_tokens: int = 120000) -> str:
        """Generate a concise 2-3 sentence summary of a transcript."""
        try:
            estimated_tokens = estimate_tokens(transcript)

            if estimated_tokens <= max_tokens:
                prompt = self._build_brief_summary_prompt(transcript)
                return await self._generate_content_with_retry(prompt, max_output_tokens=300)

            # For large transcripts, summarize in chunks then synthesize
            chunks = self._chunk_text(transcript, max_tokens=25000)
            chunk_summaries = []
            for chunk in chunks:
                p = self._build_brief_summary_prompt(chunk)
                chunk_summaries.append(
                    await self._generate_content_with_retry(p, max_output_tokens=150)
                )

            synthesis_prompt = self._build_brief_synthesis_prompt("\n\n".join(chunk_summaries))
            return await self._generate_content_with_retry(synthesis_prompt, max_output_tokens=300)

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

        <transcript>
        {transcript}
        </transcript>

        **Concise summary (PLAIN TEXT WITH EMOJIS ONLY):**
        """
        ).strip()

    def _build_brief_synthesis_prompt(self, summaries: str) -> str:
        """Prompt to combine chunk summaries into a concise overview."""
        return dedent(
            f"""
        Combine these short summaries into one concise overview (max three sentences).

        <summaries>
        {summaries}
        </summaries>

        **Final concise summary (PLAIN TEXT WITH EMOJIS ONLY):**
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

            answer = await self._generate_content_with_retry(
                prompt, max_output_tokens=800
            )
            logger.info("Generated answer for user question")

            return answer

        except (GeminiSafetyError, GeminiRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise GeminiError(f"Failed to answer question: {str(e)}")

    def _build_transcript_qa_prompt(self, transcript: str, question: str) -> str:
        """Build optimized Q&A prompt for full transcript."""
        return dedent(
            f"""
        <role>
        You are an expert YouTube video analyst with exceptional ability to answer questions about video content accurately and helpfully.
        </role>

        <task>
        Answer a user's question about a YouTube video by analyzing the complete transcript and providing a precise, informative response.
        </task>

        <instructions>
        **IMPORTANT:** Follow this analysis process:

        1. **Question Understanding**
           - Carefully analyze what the user is asking
           - Identify the specific type of information they need
           - Determine the scope and depth of answer required

        2. **Content Analysis**
           - Search through the transcript for relevant information
           - Identify all content that relates to the question
           - Note specific details, examples, or explanations provided

        3. **Response Construction**
           - Provide a direct, clear answer based on the video content
           - Include specific details and examples when available
           - Acknowledge limitations if the information isn't fully covered

        **ALWAYS base your response** exclusively on the video content. **NEVER add** external information not present in the transcript.

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO asterisks, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.

        <if_information_available>
        If the transcript contains clear information to answer the question:
        - Provide a comprehensive, detailed answer
        - Include specific examples or details mentioned
        - Quote or reference relevant parts when helpful
        </if_information_available>

        <if_information_unclear>
        If the answer isn't completely clear from the transcript:
        - Provide what information is available
        - Clearly indicate what aspects are unclear or not covered
        - Suggest what additional information would be needed
        </if_information_unclear>

        <if_information_missing>
        If the transcript doesn't contain information to answer the question:
        - Clearly state that the information isn't covered in this video
        - Briefly explain what topics the video does cover instead
        - Acknowledge the limitation directly and helpfully
        </if_information_missing>
        </instructions>

        <transcript>
        {transcript}
        </transcript>

        <user_question>
        {question}
        </user_question>

        **Your precise, helpful answer based on the video content (PLAIN TEXT WITH EMOJIS ONLY):**
        """
        ).strip()

    def _build_summary_qa_prompt(self, summary: str, question: str) -> str:
        """Build optimized Q&A prompt for summary-based answers."""
        return dedent(
            f"""
        <role>
        You are an expert video content analyst specializing in answering questions based on comprehensive video summaries.
        </role>

        <task>
        Answer a user's question using a detailed summary of a YouTube video. Provide the most accurate and helpful response possible given the summary information available.
        </task>

        <instructions>
        **IMPORTANT:** Follow this structured approach:

        1. **Question Analysis**
           - Understand exactly what information the user needs
           - Identify if this is about main topics, specific details, or general insights

        2. **Summary Review**
           - Carefully examine the summary for relevant information
           - Note any details, examples, or insights that address the question
           - Assess the completeness of available information

        3. **Response Delivery**
           - Provide the most complete answer possible from the summary
           - Be clear about the source of your information
           - Acknowledge any limitations due to working from summary rather than full content

        **CRITICAL:** Base your answer exclusively on the provided summary. 

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO asterisks, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.

        <if_summary_contains_answer>
        When the summary clearly addresses the question:
        - Provide a comprehensive answer based on the summary
        - Reference specific points from the summary
        - Include relevant details and context
        </if_summary_contains_answer>

        <if_summary_partially_addresses>
        When the summary only partially addresses the question:
        - Share what information is available from the summary
        - Clearly indicate that this is based on a summary
        - Note that more details might be available in the full video
        </if_summary_partially_addresses>

        <if_summary_lacks_info>
        When the summary doesn't address the question:
        - Clearly state the limitation
        - Briefly describe what the video summary does cover
        - Suggest that the user might need the full video for this specific information
        </if_summary_lacks_info>
        </instructions>

        <video_summary>
        {summary}
        </video_summary>

        <user_question>
        {question}
        </user_question>

        **Your answer based on the video summary (PLAIN TEXT WITH EMOJIS ONLY):**
        """
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

            answer = await self._generate_content_with_retry(
                prompt, max_output_tokens=800
            )
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
        """Build comprehensive context-aware Q&A prompt."""
        # Build metadata section
        metadata_lines = [
            f"**Title:** {context_data['title']}",
            f"**Channel:** {context_data['channel']}",
            f"**Duration:** {context_data['duration_str']}",
            f"**Upload Date:** {context_data['upload_date']}",
        ]

        if context_data["view_count"]:
            metadata_lines.append(f"**View Count:** {context_data['view_count']:,}")

        if context_data["categories"]:
            metadata_lines.append(
                f"**Categories:** {', '.join(context_data['categories'])}"
            )

        if context_data["tags"]:
            metadata_lines.append(f"**Tags:** {', '.join(context_data['tags'])}")

        if context_data["description"]:
            metadata_lines.append(f"**Description:** {context_data['description']}...")

        metadata_section = "\n".join(metadata_lines)

        return dedent(
            f"""
        <role>
        You are an expert YouTube video intelligence analyst with comprehensive knowledge across all domains. You excel at answering questions by leveraging both video metadata and content to provide the most accurate and complete responses possible.
        </role>

        <task>
        Answer a user's question about a YouTube video using ALL available information including detailed metadata and video content. Provide the most comprehensive and accurate answer possible.
        </task>

        <instructions>
        **CRITICAL:** Follow this comprehensive analysis process:

        1. **Question Classification**
           - Determine if this is about metadata (title, channel, views, etc.)
           - Identify if this requires content analysis (topics discussed, specific information)
           - Assess if both metadata and content are needed for a complete answer

        2. **Information Integration**
           - Use video metadata for context and basic information
           - Leverage video content for substantive answers about topics discussed
           - Combine both sources when they provide complementary information

        3. **Response Optimization**
           - Prioritize accuracy by citing your information sources
           - Provide specific details and examples when available
           - Give comprehensive answers that fully address the user's need

        **ALWAYS use ALL available information** to provide the most complete answer possible.

        **CRITICAL OUTPUT REQUIREMENT:** Your response must be in PLAIN TEXT format only. Use emojis for visual appeal but NO markdown formatting, NO bold text, NO asterisks, NO special characters for formatting. Write naturally as if you're explaining to someone in a conversation.

        <metadata_questions>
        For questions about basic video information (title, creator, duration, etc.):
        - Use the metadata as the primary source
        - Reference the content if it provides additional context
        - Provide accurate, specific details
        </metadata_questions>

        <content_questions>
        For questions about topics, information, or content discussed:
        - Use the video content as the primary source
        - Reference metadata for additional context when relevant
        - Provide detailed answers with specific examples
        </content_questions>

        <comprehensive_questions>
        For questions requiring both metadata and content:
        - Integrate information from both sources seamlessly
        - Provide a holistic answer that addresses all aspects
        - Cite your sources appropriately
        </comprehensive_questions>

        <if_complete_info_available>
        When you have complete information to answer the question:
        - Provide a comprehensive, detailed response
        - Include specific examples and details
        - Reference both metadata and content as appropriate
        </if_complete_info_available>

        <if_partial_info_available>
        When you have partial information:
        - Share all available relevant information
        - Clearly indicate what you can determine from the available data
        - Note what additional information might be needed
        </if_partial_info_available>

        <if_insufficient_info>
        When the available information is insufficient:
        - Clearly state the limitation
        - Explain what information you do have access to
        - Suggest how the user might find the specific information they need
        </if_insufficient_info>
        </instructions>

        <video_metadata>
        {metadata_section}
        </video_metadata>

        <video_content>
        {content_context}
        </video_content>

        <user_question>
        {question}
        </user_question>

        **Your comprehensive answer using all available information (PLAIN TEXT WITH EMOJIS ONLY):**
        """
        ).strip()


# Global Gemini client instance
gemini_client = GeminiClient()
