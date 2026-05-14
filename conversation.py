"""
Conversation Engine: Orchestrates intent detection, retrieval, and LLM generation.
Stateless design — all context comes from the request.
"""

import json
import logging
import os
import re
from typing import Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai

from app.models.schemas import (
    AssessmentRecommendation,
    ChatRequest,
    ChatResponse,
    Message,
)
from app.prompts.templates import (
    build_chat_messages,
    format_catalog_context,
    CATALOG_CONTEXT_TEMPLATE,
)
from app.retriever.hybrid_retriever import get_retriever

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "gemini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash"

MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1


def _extract_query_for_retrieval(messages: list[Message]) -> str:
    """
    Extract the most relevant query string for retrieval.
    Combines recent user messages for context.
    """
    user_messages = [m.content for m in messages if m.role == "user"]
    # Use last 2 user messages to capture refinements
    recent = user_messages[-2:] if len(user_messages) >= 2 else user_messages
    return " ".join(recent)


def _count_turns(messages: list[Message]) -> int:
    """Count how many full turns (user+assistant pairs) have occurred."""
    return sum(1 for m in messages if m.role == "user")


def _call_openai(system_prompt: str, messages: list[dict]) -> str:
    """Call OpenAI GPT-4o-mini."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, messages: list[dict]) -> str:
    """Call Google Gemini Flash."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )

    # Convert messages to Gemini format
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(
        messages[-1]["content"] if messages else "Hello",
        generation_config={"temperature": 0.1, "max_output_tokens": 1000},
    )
    return response.text


def _call_llm(system_prompt: str, messages: list[dict]) -> str:
    """Route LLM call based on configured provider."""
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return _call_gemini(system_prompt, messages)
    elif OPENAI_API_KEY:
        return _call_openai(system_prompt, messages)
    else:
        raise ValueError(
            "No LLM API key configured. Set OPENAI_API_KEY or GEMINI_API_KEY."
        )


def _parse_llm_response(raw: str) -> dict:
    """Parse JSON response from LLM, with fallback handling."""
    # Strip markdown code fences if present
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Fallback
    logger.warning(f"Failed to parse LLM response as JSON: {raw[:200]}")
    return {
        "intent": "clarify",
        "reply": "Could you tell me more about the role you're hiring for?",
        "recommendation_names": [],
    }


def _validate_and_build_recommendations(
    names: list[str],
    catalog_assessments: list[dict],
) -> list[AssessmentRecommendation]:
    """
    HALLUCINATION PREVENTION: Validate recommended names against retrieved catalog.
    Only return assessments that exist in the retrieved catalog.
    """
    # Build lookup from retrieved catalog
    catalog_lookup = {a["name"].lower(): a for a in catalog_assessments}

    # Also load full catalog for validation
    retriever = get_retriever()
    full_catalog = retriever.get_all()
    full_lookup = {a["name"].lower(): a for a in full_catalog}

    validated = []
    for name in names:
        name_lower = name.lower().strip()

        # Check retrieved catalog first (preferred)
        if name_lower in catalog_lookup:
            a = catalog_lookup[name_lower]
            validated.append(
                AssessmentRecommendation(
                    name=a["name"],
                    url=a.get("url", ""),
                    test_type=a.get("test_type", "K"),
                )
            )
        # Fallback: check full catalog
        elif name_lower in full_lookup:
            a = full_lookup[name_lower]
            validated.append(
                AssessmentRecommendation(
                    name=a["name"],
                    url=a.get("url", ""),
                    test_type=a.get("test_type", "K"),
                )
            )
        # Fuzzy match as last resort
        else:
            best_match = None
            for key in full_lookup:
                if name_lower in key or key in name_lower:
                    best_match = full_lookup[key]
                    break
            if best_match:
                logger.info(f"Fuzzy matched '{name}' → '{best_match['name']}'")
                validated.append(
                    AssessmentRecommendation(
                        name=best_match["name"],
                        url=best_match.get("url", ""),
                        test_type=best_match.get("test_type", "K"),
                    )
                )
            else:
                logger.warning(
                    f"Assessment '{name}' not found in catalog — dropping to prevent hallucination"
                )

    # Cap at MAX_RECOMMENDATIONS
    return validated[:MAX_RECOMMENDATIONS]


def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversation processing function.

    Flow:
    1. Extract query from conversation history
    2. Retrieve relevant assessments (hybrid search)
    3. Build LLM prompt with retrieved context
    4. Call LLM to determine intent and generate response
    5. Validate recommendations against catalog
    6. Return structured response
    """
    messages = request.messages
    turn_number = _count_turns(messages)

    # Step 1: Build retrieval query from conversation
    query = _extract_query_for_retrieval(messages)
    logger.info(f"Retrieval query: '{query}' (turn {turn_number})")

    # Step 2: Hybrid retrieval
    retriever = get_retriever()
    retrieved = retriever.retrieve(query, top_k=10)
    logger.info(f"Retrieved {len(retrieved)} assessments")

    # Step 3: Build catalog context for prompt
    catalog_text = format_catalog_context(retrieved)
    catalog_context = CATALOG_CONTEXT_TEMPLATE.format(catalog_items=catalog_text)

    # Step 4: Build LLM messages
    system_prompt, chat_messages = build_chat_messages(
        [m.model_dump() for m in messages],
        catalog_context,
        turn_number,
    )

    # Step 5: Call LLM
    try:
        raw_response = _call_llm(system_prompt, chat_messages)
        parsed = _parse_llm_response(raw_response)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ChatResponse(
            reply="I'm having trouble processing your request. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    intent = parsed.get("intent", "clarify")
    reply = parsed.get("reply", "")
    rec_names = parsed.get("recommendation_names", [])

    # Step 6: Validate and build recommendations (hallucination prevention)
    recommendations = []
    if intent in ("recommend", "refine", "compare") and rec_names:
        recommendations = _validate_and_build_recommendations(rec_names, retrieved)

    # Determine end of conversation
    end_of_conversation = (
        intent == "refuse"
        or turn_number >= 8
        or (intent == "recommend" and bool(recommendations))
    )

    # Don't end conversation prematurely if recommendations are empty
    if not recommendations and intent == "recommend":
        end_of_conversation = False

    logger.info(
        f"Intent: {intent}, Recommendations: {len(recommendations)}, "
        f"End: {end_of_conversation}"
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
