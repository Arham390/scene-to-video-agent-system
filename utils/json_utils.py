"""
JSON utilities — robust extraction of JSON from LLM responses.
"""

import json
import re


def extract_json_from_llm_response(text: str) -> dict:
    """
    Strip markdown code fences and extract valid JSON from an LLM response.
    Falls back to a minimal valid structure if parsing fails.
    """
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    print("[json_utils] ⚠️  Could not parse JSON from LLM response. Returning minimal structure.")
    return {}

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
