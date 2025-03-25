# utils/helpers.py
import hashlib
from typing import List, Dict, Any

def generate_document_id(content: str) -> str:
    """Generate a unique ID for a document based on its content."""
    return hashlib.md5(content.encode()).hexdigest()

def format_chat_history(history: List[Dict[str, Any]]) -> str:
    """Format chat history for context window."""
    formatted = []
    for message in history:
        role = message["role"]
        content = message["content"]
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)