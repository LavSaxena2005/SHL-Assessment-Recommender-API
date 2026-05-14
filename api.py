from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import json

# Load catalog
with open("catalog.json", "r") as f:
    catalog = json.load(f)

# Create router
router = APIRouter()

# Request models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# Health route
@router.get("/health")
def health():
    return {"status": "ok"}


# Chat route
@router.post("/chat")
def chat(request: ChatRequest):

    user_message = request.messages[-1].content.lower()

    recommendations = []

    # Clarification
    if len(user_message.split()) < 3:

        return {
            "reply": "What role or skills are you hiring for?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # Java
    if "java" in user_message:

        for item in catalog:

            if "Java" in item["skills"]:

                recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"]
                })

        reply = (
            "I recommend Java assessments for evaluating "
            "programming and coding skills."
        )

    # Python
    elif "python" in user_message:

        for item in catalog:

            if "Python" in item["skills"]:

                recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"]
                })

        reply = (
            "I recommend Python assessments for technical evaluation."
        )

    # Personality
    elif "personality" in user_message:

        for item in catalog:

            if "Personality" in item["skills"]:

                recommendations.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"]
                })

        reply = (
            "I recommend personality assessments "
            "for behavioral evaluation."
        )

    # Compare
    elif "compare" in user_message:

        reply = (
            "Java assessments evaluate programming ability, "
            "while personality assessments measure behavioral traits."
        )

    # Fallback
    else:

        reply = (
            "Please specify the role, skills, or type of "
            "assessment needed."
        )

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False
    }