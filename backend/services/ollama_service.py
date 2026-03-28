"""
Ollama-based LLM service for voice-mode AI chat responses.
Uses a local Ollama instance (llama2) as a drop-in alternative to Gemini.
"""

import json
import requests

from utils.constants import COMPLAINT_TYPES, REQUIRED_COMPLAINT_FIELDS, SUPPORTED_LANGUAGES


OLLAMA_BASE_URL = "http://localhost:11434"
# Prefer smaller/faster models first; fall back to larger ones
OLLAMA_MODEL_PREFERENCE = ["phi3:mini", "phi3", "phi", "mistral", "llama2", "codellama"]


class OllamaService:
    def __init__(self) -> None:
        self.base_url = OLLAMA_BASE_URL
        self.model = self._pick_model()

    def _pick_model(self) -> str:
        """Return the fastest available model from the preference list."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            print(f"[ollama] Available models: {available}")
            for preferred in OLLAMA_MODEL_PREFERENCE:
                for avail in available:
                    if preferred in avail:
                        print(f"[ollama] Selected model: {avail}")
                        return avail
            return available[0] if available else "llama2"
        except Exception:
            return "llama2"

    def _generate(self, prompt: str, retries: int = 2) -> str:
        import time
        for i in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=180,  # 3 minutes — CPU inference can be slow
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as e:
                err_str = str(e).lower()
                if i < retries - 1 and "timeout" not in err_str:
                    time.sleep(1)
                    continue
                print(f"Ollama API Error: {e}")
                return ""
        return ""

    def _safe_json_parse(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        # Try to extract JSON from surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def detect_language(self, text: str) -> str:
        prompt = (
            "Detect the language of the user text from this list only: "
            f"{', '.join(SUPPORTED_LANGUAGES)}. "
            "Return only one language name, no extra words.\n\n"
            f"Text: {text}"
        )
        guess = (self._generate(prompt) or "English").strip()
        return guess if guess in SUPPORTED_LANGUAGES else "English"

    def translate_text(self, text: str, target_language: str) -> str:
        if not text.strip():
            return text
        prompt = (
            f"Translate the following text to {target_language}. "
            "Keep meaning exact, simple, and natural.\n\n"
            f"Text: {text}"
        )
        return (self._generate(prompt) or "").strip()

    def generate_complaint_chat_reply(
        self,
        user_language: str,
        user_message: str,
        conversation_messages: list[dict],
        collected_fields: dict,
    ) -> dict:
        prompt = f"""You are a helpful cyber crime complaint assistant using voice.

Your job:
1. Extract ALL complaint details from the user's natural speech in one go
2. Do NOT ask about fields the user already mentioned
3. Acknowledge briefly what you understood, then ask for the NEXT missing field only
4. Be conversational and brief (voice-friendly, no long lists)
5. Always respond in: {user_language}

Field extraction rules & mappings:
- Name -> `full_name`
- Phone/number -> `phone_number`
- Email -> `email`
- Money lost / amount -> `amount_lost`
- UPI -> `suspect_vpa`
- Suspect's phone -> `suspect_phone`
- When it happened -> `date_time`
- App/website used -> `platform`
- What happened -> `description` AND `complaint_type`

Required fields: {REQUIRED_COMPLAINT_FIELDS}
Optional: amount_lost, transaction_id, suspect_details, suspect_vpa, suspect_phone, suspect_bank_account, platform
Complaint types: {COMPLAINT_TYPES}

Example Input: "i am rohit my number is 76 and email is 67@gmail and i have issue with payemnt"
Example Output JSON:
{{
  "assistant_response": "Thanks Rohit, I have your email and phone noted. To help with the payment issue, can you tell me which platform or app you used?",
  "intent": "file_complaint",
  "field_updates": {{"full_name": "Rohit", "phone_number": "76", "email": "67@gmail", "description": "issue with payment", "complaint_type": "Financial Fraud"}},
  "next_required_field": "platform",
  "missing_fields": ["platform", "address", "date_time"]
}}

Conversation:
{json.dumps(conversation_messages[-8:], ensure_ascii=False)}

Already collected:
{json.dumps(collected_fields, ensure_ascii=False)}

User said: "{user_message}"

Return ONLY valid JSON, no extra text:
"""
        response_text = self._generate(prompt)
        payload = self._safe_json_parse(response_text or "")
        if not payload:
            return {
                "assistant_response": self.translate_text("What can I help you with?", user_language),
                "intent": "general",
                "field_updates": {},
                "next_required_field": None,
                "missing_fields": REQUIRED_COMPLAINT_FIELDS,
            }
        return payload

    def transcribe_audio(self, file_path: str, mime_type: str) -> str:
        """Ollama doesn't support audio natively — return empty."""
        return ""


ollama_service = OllamaService()
