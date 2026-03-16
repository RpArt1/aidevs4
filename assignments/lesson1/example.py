import json
import os
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Dict, Any

# --- Mocks for the external variables/functions from your JS ---
RESPONSES_API_ENDPOINT = os.getenv("RESPONSES_API_ENDPOINT", "https://api.example.com/v1/responses")
AI_API_KEY = os.getenv("AI_API_KEY", "your-api-key")
EXTRA_API_HEADERS: Dict[str, str] = {}

def resolve_model_for_provider(model_name: str) -> str:
    return model_name  # Replace with actual implementation

def extract_response_text(data: Dict[str, Any]) -> str:
    return data.get("text", "")  # Replace with actual implementation based on your API
# ---------------------------------------------------------------

MODEL = resolve_model_for_provider("gpt-5.4")

PERSON_SCHEMA = {
    "type": "json_schema",
    "name": "person",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"],
                "description": "Full name of the person. Use null if not mentioned."
            },
            "age": {
                "type": ["number", "null"],
                "description": "Age in years. Use null if not mentioned or unclear."
            },
            "occupation": {
                "type": ["string", "null"],
                "description": "Job title or profession. Use null if not mentioned."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of skills, technologies, or competencies. Empty array if none mentioned."
            }
        },
        "required": ["name", "age", "occupation", "skills"],
        "additionalProperties": False
    }
}

def extract_person(text: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}",
        **EXTRA_API_HEADERS
    }

    payload = {
        "model": MODEL,
        "input": f'Extract person information from: "{text}"',
        "text": {"format": PERSON_SCHEMA}
    }

    req = urllib.request.Request(
        url=RESPONSES_API_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        # Handle non-200 HTTP responses
        try:
            error_data = json.loads(e.read().decode("utf-8"))
            error_msg = error_data.get("error", {}).get("message", f"Request failed with status {e.code}")
        except json.JSONDecodeError:
            error_msg = f"Request failed with status {e.code}"
        raise RuntimeError(error_msg)
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")

    # Catch internal API errors masked as 200 OK
    if "error" in response_data:
        error_msg = response_data["error"].get("message", "Unknown API error")
        raise RuntimeError(error_msg)

    output_text = extract_response_text(response_data)
    
    if not output_text:
        raise ValueError("Missing text output in API response")

    return json.loads(output_text)

def main():
    text = "John is 30 years old and works as a software engineer. He is skilled in JavaScript, Python, and React."
    
    try:
        person = extract_person(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Using .get() and 'or' mimics the ?? (nullish coalescing) operator in JS
    name = person.get("name") or "unknown"
    age = person.get("age") or "unknown"
    occupation = person.get("occupation") or "unknown"
    skills = person.get("skills", [])
    
    skills_str = ", ".join(skills) if skills else "none"

    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Occupation: {occupation}")
    print(f"Skills: {skills_str}")

if __name__ == "__main__":
    main()