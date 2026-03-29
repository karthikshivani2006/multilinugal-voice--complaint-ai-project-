import json
import logging
import tempfile
from pathlib import Path

import google.generativeai as genai

from config import settings
from services.video_service import video_service
from utils.constants import COMPLAINT_TYPES, REQUIRED_COMPLAINT_FIELDS, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(self._pick_model_name())

    def _pick_model_name(self) -> str:
        preferred = [
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
        try:
            available = [m.name.split("models/")[-1] for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            for candidate in preferred:
                if candidate in available:
                    return candidate
            if available:
                return available[0]
        except Exception as e:
            print(f"Error picking model: {e}")
            pass
        return "gemini-1.5-flash-latest"

    def _generate_with_retry(self, prompt: str | list, retries: int = 3) -> str:
        import time
        for i in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text or ""
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate_limit" in err_str or "quota" in err_str) and i < retries - 1:
                    time.sleep(2 ** i) # Exponential backoff
                    continue
                print(f"Gemini API Error: {e}")
                return ""
        return ""

    def _safe_json_parse(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _extract_first_json_object(self, text: str) -> dict:
        payload = self._safe_json_parse(text)
        if payload:
            return payload

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return self._safe_json_parse(text[start : end + 1])
        return {}

    def detect_language(self, text: str) -> str:
        prompt = (
            "Detect the language of the user text from this list only: "
            f"{', '.join(SUPPORTED_LANGUAGES)}. "
            "Return only one language name, no extra words.\n\n"
            f"Text: {text}"
        )
        response_text = self._generate_with_retry(prompt)
        guess = (response_text or "English").strip()
        return guess if guess in SUPPORTED_LANGUAGES else "English"

    def translate_text(self, text: str, target_language: str) -> str:
        if not text.strip():
            return text
        prompt = (
            f"Translate the following text to {target_language}. "
            "Keep meaning exact, simple, and natural for Indian users.\n\n"
            f"Text: {text}"
        )
        response_text = self._generate_with_retry(prompt)
        return (response_text or "").strip()

    def to_english(self, text: str) -> str:
        return self.translate_text(text, "English")

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
        response_text = self._generate_with_retry(prompt)
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

    def analyze_evidence(self, file_path: str, mime_type: str) -> str:
        """
        Analyze evidence file (document, image, or video) using Gemini.
        For videos, optionally extract audio and transcribe.
        IMPORTANT: Always returns a string (possibly empty) for extracted_text field.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Evidence file not found: {file_path}")
            return f"[File not found: {path.name}]"

        filename = path.name
        file_size_mb = path.stat().st_size / (1024 * 1024)
        summary = ""
        
        # Try Gemini analysis if API key is valid
        if settings.gemini_api_key and settings.gemini_api_key != "dev_key_placeholder":
            prompt = (
                "Extract all meaningful text and cyber-crime-relevant details from this file. "
                "Return plain text summary with names, accounts, UTR, links, timestamps, and suspicious indicators if present. "
                "Be concise but thorough. If the document contains important information, extract it."
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Uploading evidence file to Gemini (attempt {attempt + 1}/{max_retries}): {filename}")
                    uploaded = genai.upload_file(path=str(path), mime_type=mime_type)
                    logger.info(f"File uploaded successfully: {uploaded.uri}")
                    
                    logger.info(f"Generating content from uploaded file: {filename}")
                    response = self.model.generate_content([prompt, uploaded], stream=False)
                    summary = (response.text or "").strip()
                    
                    if summary:
                        logger.info(f"✅ Gemini analysis completed for {filename}: {len(summary)} chars extracted")
                        break
                    else:
                        logger.warning(f"Gemini returned empty response for {filename}, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2 ** attempt)
                            continue
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"❌ Gemini analysis error for {filename} (attempt {attempt + 1}): {err_msg}")
                    
                    # Check if it's a rate limit or temporary error
                    if any(x in err_msg.lower() for x in ["429", "rate", "quota", "timeout", "temporarily"]):
                        if attempt < max_retries - 1:
                            import time
                            wait_time = 2 ** attempt
                            logger.info(f"Rate limit detected, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    # On final attempt, use enhanced fallback
                    if attempt == max_retries - 1:
                        summary = f"[Evidence: {filename} ({mime_type}) - {file_size_mb:.1f}MB - Gemini analysis unavailable]"
                        logger.warning(f"Using fallback summary for {filename}: {summary}")
            
            # If still empty, create better fallback
            if not summary:
                summary = f"[Evidence uploaded: {filename} ({mime_type})]"
                logger.warning(f"No text extracted from {filename}, using basic fallback")
        else:
            logger.info(f"⚠️  Gemini API key not configured, using fallback for {file_path}")
            # Use informative fallback summary
            summary = f"[Evidence: {filename} ({mime_type})]"

        # Try audio extraction for video files (moviepy is optional)
        if (mime_type or "").startswith("video/"):
            logger.info(f"📹 Processing video evidence: {filename}")
            
            try:
                # Get video info
                video_info = video_service.get_video_info(file_path)
                if video_info:
                    duration = video_info.get('duration', 0)
                    has_audio = video_info.get('has_audio', False)
                    logger.info(f"   Duration: {duration:.1f}s, Audio: {has_audio}")
                    
                    # Extract audio - try full video first, then shortened if long
                    if video_service.is_available() and has_audio:
                        # For very long videos (>5 mins), extract first 5 mins
                        if duration > 300:
                            logger.info(f"   Video is {duration:.1f}s long, extracting first 300s for analysis...")
                            audio_path = video_service.extract_audio_from_video(file_path, start_time=0, end_time=300)
                        else:
                            audio_path = video_service.extract_audio_from_video(file_path)
                        
                        if audio_path:
                            try:
                                logger.info(f"   Transcribing extracted audio...")
                                transcription = self.transcribe_audio(str(audio_path), "audio/wav")
                                if transcription:
                                    summary = f"{summary}\n\n📝 Audio Transcription:\n{transcription}".strip()
                                    logger.info(f"   ✅ Audio transcription added ({len(transcription)} chars)")
                            except Exception as e:
                                logger.warning(f"   ❌ Transcription failed: {e}")
                    else:
                        logger.info(f"   ℹ️  Video audio extraction skipped (available={video_service.is_available()}, has_audio={has_audio})")
            except Exception as e:
                logger.warning(f"Video processing failed: {e}")

        # Ensure we always return something meaningful
        final_summary = summary.strip() if summary else f"[Evidence file: {filename}]"
        logger.debug(f"Final extracted text for {filename}: {len(final_summary)} chars")
        return final_summary

    def analyze_id_proof(self, file_path: str, mime_type: str) -> dict:
        """Extract identity information from ID proof with retry logic and detailed extraction status."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"ID proof file not found: {file_path}")
            return {
                "document_type": "",
                "name": "",
                "id_number": "",
                "dob": "",
                "address": "",
                "phone": "",
                "email": "",
                "confidence": "LOW",
                "extraction_status": "FILE_NOT_FOUND",
                "missing_fields": ["name", "phone", "email", "document_type"],
                "extracted_text": f"[ID Proof: File not found - {path.name}]",
            }

        prompt = (
            "Read this Indian government ID proof and extract fields into JSON only. "
            "Possible document types: Aadhaar, PAN, Voter ID, Driving License, Passport, Other. "
            "For each field, mark null/empty if not clearly visible. "
            "Return strict JSON with keys: "
            "document_type, name, id_number, dob, address, phone, email, confidence, notes, extracted_text."
        )

        max_retries = 3
        extracted_result = None
        text = ""

        for attempt in range(max_retries):
            try:
                logger.info(f"📋 ID proof analysis (attempt {attempt + 1}/{max_retries}): {path.name}")
                
                uploaded = genai.upload_file(path=str(path), mime_type=mime_type)
                response = self.model.generate_content([prompt, uploaded])
                text = (response.text or "").strip()

                if not text:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)
                    continue

                payload = self._extract_first_json_object(text)
                if payload and isinstance(payload, dict):
                    extracted_result = payload
                    logger.info(f"✅ ID proof extraction successful")
                    break
                elif attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

            except Exception as e:
                err_msg = str(e).lower()
                logger.error(f"❌ ID proof analysis error (attempt {attempt + 1}): {e}")

                if any(x in err_msg for x in ["429", "rate", "quota", "timeout"]) and attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt == max_retries - 1:
                    logger.warning(f"All {max_retries} attempts failed for ID proof extraction")

        # Process extracted result
        if extracted_result and isinstance(extracted_result, dict):
            name = (extracted_result.get("name") or "").strip()
            phone = (extracted_result.get("phone") or "").strip()
            email = (extracted_result.get("email") or "").strip()
            doc_type = (extracted_result.get("document_type") or "").strip()

            missing_fields = []
            if not name:
                missing_fields.append("name")
            if not phone:
                missing_fields.append("phone")
            if not email:
                missing_fields.append("email")

            return {
                "document_type": doc_type,
                "name": name,
                "id_number": (extracted_result.get("id_number") or "").strip(),
                "dob": (extracted_result.get("dob") or "").strip(),
                "address": (extracted_result.get("address") or "").strip(),
                "phone": phone,
                "email": email,
                "confidence": (extracted_result.get("confidence") or "MEDIUM").strip(),
                "extraction_status": "PARTIAL_SUCCESS" if missing_fields else "SUCCESS",
                "missing_fields": missing_fields,
                "extracted_text": (extracted_result.get("extracted_text") or text).strip(),
            }

        # Fallback: Document couldn't be clearly read
        return {
            "document_type": "",
            "name": "",
            "id_number": "",
            "dob": "",
            "address": "",
            "phone": "",
            "email": "",
            "confidence": "LOW",
            "extraction_status": "UNCLEAR",
            "missing_fields": ["name", "phone", "email", "document_type"],
            "extracted_text": self.analyze_evidence(file_path=file_path, mime_type=mime_type),
        }

    def transcribe_audio(self, file_path: str, mime_type: str) -> str:
        """Transcribe spoken audio/video to text using Gemini multimodal."""
        prompt = (
            "Transcribe the spoken content in this audio/video file exactly as spoken. "
            "Return only the transcription text, nothing else. "
            "If the audio is in a regional Indian language, transcribe it faithfully in that language."
        )

        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Audio file not found: {file_path}")
            return ""

        filename = path.name
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🎙️  Uploading audio for transcription: {filename} (attempt {attempt + 1}/{max_retries})")
                uploaded = genai.upload_file(path=str(path), mime_type=mime_type)
                logger.info(f"   Uploaded: {uploaded.uri}")
                
                logger.info(f"🔄 Generating transcription...")
                response = self.model.generate_content([prompt, uploaded], stream=False)
                text = (response.text or "").strip()
                
                if text:
                    logger.info(f"   ✅ Transcription completed: {len(text)} chars")
                    return text
                else:
                    logger.warning(f"   ⚠️  Transcription returned empty response")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    return ""
                    
            except Exception as e:
                err_msg = str(e)
                logger.error(f"   ❌ Transcription error: {err_msg}")
                
                if any(x in err_msg.lower() for x in ["429", "rate", "quota", "timeout"]):
                    if attempt < max_retries - 1:
                        import time
                        wait_time = 2 ** attempt
                        logger.info(f"   Rate limit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                if attempt == max_retries - 1:
                    logger.error(f"   Transcription failed after {max_retries} attempts")
                    return ""
        
        return ""


gemini_service = GeminiService()
