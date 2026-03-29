from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import re
import os
from datetime import datetime

from database import get_db
from schemas.chat import ChatRequest, ChatResponse
from services.conversation_service import conversation_service
from services.gemini_service import gemini_service
from services.ollama_service import ollama_service
from services.translation_service import translation_service
from services.validation_service import validation_service
from utils.constants import COMPLAINT_TYPES, REQUIRED_COMPLAINT_FIELDS, SUPPORTED_LANGUAGES, UNKNOWN_INPUT_HINTS


router = APIRouter(prefix="/chat", tags=["Chat"])


OPTIONAL_CHAT_FIELDS = [
    "amount_lost",
    "transaction_id",
    "suspect_details",
    "suspect_vpa",
    "suspect_phone",
    "suspect_bank_account",
]
ALLOWED_CHAT_FIELDS = set(REQUIRED_COMPLAINT_FIELDS + OPTIONAL_CHAT_FIELDS)

BASE_CHAT_FLOW_FIELDS = [
    "full_name",
    "phone_number",
    "email",
    "address",
    "complaint_type",
    "date_time",
    "description",
    "platform",
    "suspect_details",
]

FINANCIAL_COMPLAINT_TYPES = {
    "UPI / Payment Fraud",
    "Credit/Debit Card Fraud",
    "Internet Banking Fraud",
    "OTP Fraud",
    "E-Commerce Fraud / Fake Delivery",
    "Loan / Insurance Fraud",
    "SIM Swap Fraud",
    "Aadhaar Misuse",
}

TYPE_BASED_FIELDS = {
    "financial": [
        "amount_lost",
        "transaction_id",
        "suspect_vpa",
        "suspect_phone",
        "suspect_bank_account",
    ],
}

FIELD_LABELS = {
    "full_name": "full name",
    "phone_number": "phone number",
    "email": "email",
    "address": "address",
    "complaint_type": "incident type",
    "date_time": "incident date and time",
    "description": "incident description",
    "platform": "platform",
    "amount_lost": "amount lost",
    "transaction_id": "transaction ID / UTR",
    "suspect_details": "suspect details",
    "suspect_vpa": "suspect UPI / VPA",
    "suspect_phone": "suspect phone number",
    "suspect_bank_account": "suspect bank account number",
}

FIELD_PROMPTS = {
    "full_name": "Please share your full name.",
    "phone_number": "Please share your 10-digit phone number.",
    "email": "Please share your email address.",
    "address": "Please share your complete address.",
    "complaint_type": "Please tell me the type of incident.",
    "date_time": "Please share the date and time of the incident.",
    "description": "Please explain what happened in detail.",
    "platform": "Please share the platform where this happened.",
    "amount_lost": "Please share the amount lost. If no money was lost, enter 0.",
    "transaction_id": "Please share the transaction ID / UTR. If unavailable, enter N/A.",
    "suspect_details": "Please share suspect details like name, profile, link, or contact information.",
    "suspect_vpa": "Please share suspect UPI / VPA ID (for example: scammer@upi). If unavailable, enter N/A.",
    "suspect_phone": "Please share suspect phone number. If unavailable, enter N/A.",
    "suspect_bank_account": "Please share suspect bank account number. If unavailable, enter N/A.",
}

COMPLAINT_TYPE_KEYWORDS = {
    "UPI / Payment Fraud": ["upi", "payment", "money fraud", "money lost", "gpay", "phonepe", "paytm"],
    "OTP Fraud": ["otp", "verification code"],
    "Phishing / Vishing / Smishing": ["phishing", "vishing", "smishing", "fake link", "suspicious link"],
    "Social Media Hacking": ["instagram hacked", "facebook hacked", "account hacked", "social media hacked"],
    "Defamation / Fake Profiles": ["fake profile", "fake account", "impersonation", "defamation"],
    "Credit/Debit Card Fraud": ["card fraud", "credit card", "debit card"],
    "Internet Banking Fraud": ["net banking", "internet banking", "banking fraud"],
    "Job Fraud / Employment Scam": ["job scam", "employment scam", "fake job"],
    "E-Commerce Fraud / Fake Delivery": ["ecommerce", "fake delivery", "online order fraud"],
    "Other": ["other", "unknown type", "not sure"],
}


def _translate_safe(text: str, language: str) -> str:
    if language == "English":
        return text
    translated = ""
    # Prefer local Ollama translation so multilingual responses still work
    # even when Gemini is unavailable in local/dev environments.
    if ollama_service.is_available():
        translated = ollama_service.translate_text(text, language)

    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    gemini_enabled = bool(gemini_api_key) and not gemini_api_key.startswith("dev_dummy")
    if not translated and gemini_enabled:
        translated = translation_service.translate(text, language)
    return translated.strip() if translated and translated.strip() else text


def _is_dont_know_input(message: str) -> bool:
    text = _normalize(message).lower()
    if not text:
        return False

    dont_know_markers = {
        "i don't know",
        "i dont know",
        "dont know",
        "don't know",
        "not sure",
        "no idea",
        "unknown",
        "nahi pata",
        "pata nahi",
        "mujhe nahi pata",
        "తెలియదు",
        "నాకు తెలియదు",
        "naaku teliyadu",
        "naku teliyadu",
        "teliyadu",
        "theriyadhu",
        "enakku theriyadhu",
        "ಗೊತ್ತಿಲ್ಲ",
        "enikku ariyilla",
        "maloom nahi",
    }

    return any(marker in text for marker in dont_know_markers)


def _is_help_request(message: str) -> bool:
    text = _normalize(message).lower()
    if not text:
        return False

    help_markers = {
        "explain",
        "guide",
        "how to",
        "example",
        "what should i",
        "help me",
        "samjhao",
        "samjha",
        "ela",
        "cheppu",
        "detail",
        "detailed",
    }
    return any(marker in text for marker in help_markers)


def _is_greeting(message: str) -> bool:
    text = _normalize(message).lower()
    if not text:
        return False
    return bool(re.search(r"\b(hi|hello|hey|hii|helo|good morning|good evening|namaste|నమస్తే|నమస్కారం)\b", text))


def _append_unknown_hint(response_text: str, next_field: str | None, language: str, user_message: str) -> str:
    if not (_is_dont_know_input(user_message) or _is_help_request(user_message)):
        return response_text

    hint = UNKNOWN_INPUT_HINTS.get(next_field or "")
    bonus = _translate_safe(
        "If you do not know any field, do not worry. You can enter approximate details, write in your own language, and use N/A where not applicable.",
        language,
    )

    if hint:
        translated_hint = _translate_safe(hint, language)
        return f"{response_text}\n\n{translated_hint}\n\n{bonus}"

    generic_help = _translate_safe(
        "I will guide you step by step. Share whichever detail you know. For unknown details, you can say skip. "
        "Example format: name, phone, email, address, incident type, date/time, what happened, and platform.",
        language,
    )
    return f"{response_text}\n\n{generic_help}\n\n{bonus}"


def _spoken_numbers_to_digits(message: str) -> str:
    word_map = {
        "zero": "0", "oh": "0", "o": "0",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }
    tokens = re.findall(r"[a-zA-Z]+", (message or "").lower())
    digits = "".join(word_map[token] for token in tokens if token in word_map)
    return digits


def _extract_phone_digits(message: str) -> str:
    text = _normalize(message)
    numeric = re.sub(r"\D", "", text)
    spoken = _spoken_numbers_to_digits(text)
    digits = numeric if len(numeric) >= len(spoken) else spoken
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits


def _normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _looks_like_noise(value: str) -> bool:
    text = _normalize(value)
    if not text:
        return True

    only_symbols = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if not only_symbols:
        return True

    # Repeated single character (for example: "aaaaaa", "111111")
    if len(set(only_symbols.lower())) == 1 and len(only_symbols) >= 5:
        return True

    return False


def _is_valid_datetime(value: str) -> bool:
    text = _normalize(value)
    if not text:
        return False

    # Accept ISO-like datetime values from browser inputs
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except ValueError:
        pass

    # Accept natural-language datetime phrases only if they include numbers
    return bool(re.search(r"\d", text)) and len(text) >= 6


def _is_valid_field_value(field: str, value: str) -> bool:
    text = _normalize(value)
    lowered = text.lower()
    if _looks_like_noise(text):
        return False

    if lowered in {"n/a", "na", "none", "skip", "unknown", "not available"}:
        return field in {
            "email",
            "address",
            "date_time",
            "platform",
            "amount_lost",
            "transaction_id",
            "suspect_details",
            "suspect_vpa",
            "suspect_phone",
            "suspect_bank_account",
        }

    if field == "full_name":
        lowered = text.lower()
        if re.search(r"\b(file|register|submit|raise)\b.*\b(complaint|report)\b|\bcomplaint\b", lowered):
            return False
        if len(text) < 3:
            return False
        if re.search(r"\d", text):
            return False
        return True

    if field == "phone_number":
        return validation_service.validate_phone(text)

    if field == "email":
        return validation_service.validate_email(text)

    if field == "address":
        return len(text) >= 6 and len(text.split()) >= 2

    if field == "complaint_type":
        return _normalize_complaint_type(text) is not None

    if field == "date_time":
        return _is_valid_datetime(text)

    if field == "description":
        return len(text) >= 10 and len(text.split()) >= 3

    if field == "platform":
        return len(text) >= 2

    if field == "transaction_id":
        return validation_service.validate_utr(text)

    if field == "suspect_phone":
        return validation_service.validate_phone(text)

    return len(text) >= 2


def _infer_value_for_field(field: str | None, message: str) -> str | None:
    if not field:
        return None

    text = _normalize(message)
    lowered = text.lower()
    if not text:
        return None

    optional_field_markers = {
        "email",
        "address",
        "date_time",
        "platform",
        "amount_lost",
        "transaction_id",
        "suspect_details",
        "suspect_vpa",
        "suspect_phone",
        "suspect_bank_account",
    }
    no_value_markers = {
        "n/a",
        "na",
        "none",
        "skip",
        "not available",
        "dont have",
        "don't have",
        "do not have",
        "no email",
        "i dont have",
        "i don't have",
        "i do not have",
        "dont have email",
        "don't have email",
        "do not have email",
        "without email",
        "mail id ledu",
        "email ledu",
        "mail nahi hai",
        "email nahi hai",
        "no",
    }

    if field in optional_field_markers and any(marker in lowered for marker in no_value_markers):
        return "N/A"

    if field == "phone_number":
        digits = _extract_phone_digits(text)
        if len(digits) >= 10:
            candidate = digits[-10:]
            return candidate if _is_valid_field_value(field, candidate) else None
        return text if _is_valid_field_value(field, text) else None

    if field == "email":
        match = re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", text)
        if match:
            candidate = match.group(0)
            return candidate if _is_valid_field_value(field, candidate) else None
        return text if _is_valid_field_value(field, text) else None

    if field == "complaint_type":
        return _normalize_complaint_type(text)

    return text if _is_valid_field_value(field, text) else None


def _infer_or_buffer_expected_field(field: str | None, message: str, meta: dict) -> str | None:
    if not field:
        return None

    inferred = _infer_value_for_field(field, message)
    if inferred:
        if field == "phone_number":
            meta.pop("partial_phone_digits", None)
        return inferred

    if field != "phone_number":
        return None

    digits_now = _extract_phone_digits(message)
    if not digits_now:
        return None

    buffered = str(meta.get("partial_phone_digits") or "") + digits_now
    buffered = re.sub(r"\D", "", buffered)

    if len(buffered) >= 10:
        candidate = buffered[-10:]
        if _is_valid_field_value("phone_number", candidate):
            meta.pop("partial_phone_digits", None)
            return candidate

    meta["partial_phone_digits"] = buffered
    return None


def _sanitize_llm_updates(raw_updates: dict) -> dict:
    clean: dict[str, str] = {}
    for key, value in (raw_updates or {}).items():
        if key not in ALLOWED_CHAT_FIELDS:
            continue
        if value is None:
            continue
        value_text = _normalize(str(value))
        if not value_text:
            continue
        clean[key] = value_text
    return clean


def _public_fields(collected_fields: dict | None) -> dict:
    data = dict(collected_fields or {})
    data.pop("__meta__", None)
    return data


def _normalize_complaint_type(value: str) -> str | None:
    text = _normalize(value).lower()
    if not text:
        return None

    for complaint_type in COMPLAINT_TYPES:
        if complaint_type.lower() == text:
            return complaint_type

    for complaint_type, keywords in COMPLAINT_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return complaint_type

    return None


def _next_field_prompt(field: str | None, language: str) -> str:
    if not field:
        return ""
    if field == "complaint_type":
        prompt = (
            "Please tell me the type of incident. Example: UPI / Payment Fraud, "
            "Phishing / Vishing / Smishing, Social Media Hacking, Defamation / Fake Profiles, OTP Fraud."
        )
    else:
        prompt = FIELD_PROMPTS.get(field)
    if not prompt:
        return ""
    return _translate_safe(prompt, language)


def _chat_flow_fields(collected_fields: dict | None) -> list[str]:
    public = _public_fields(collected_fields or {})
    complaint_type = _normalize(public.get("complaint_type") or "")
    fields = list(BASE_CHAT_FLOW_FIELDS)

    if complaint_type in FINANCIAL_COMPLAINT_TYPES:
        fields.extend(TYPE_BASED_FIELDS["financial"])

    return fields


def _missing_chat_fields(collected_fields: dict | None) -> list[str]:
    public = _public_fields(collected_fields or {})
    ordered_fields = _chat_flow_fields(public)
    return [field for field in ordered_fields if not _normalize(str(public.get(field) or ""))]


@router.post("", response_model=ChatResponse)
async def chat_with_assistant(payload: ChatRequest, db: Session = Depends(get_db)):
    import asyncio

    session = conversation_service.get_or_create(db, payload.session_id)

    if not payload.language and not session.messages:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        return ChatResponse(
            session_id=payload.session_id,
            detected_language="English",
            response=f"Please select your language. Supported languages: {supported}",
            intent="general",
            next_field=None,
            collected_fields={},
            missing_fields=BASE_CHAT_FLOW_FIELDS,
        )

    language = payload.language or session.selected_language
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    existing_public_fields = _public_fields(session.collected_fields)
    meta = dict((session.collected_fields or {}).get("__meta__") or {})
    retry_counts = dict(meta.get("retry_counts") or {})
    expected_field = meta.get("expected_field")

    conversation_service.add_message(db, session, role="user", content=payload.message)

    # Greeting-first UX: start with welcome + ask how to help.
    if not existing_public_fields and _is_greeting(payload.message):
        greet = _translate_safe(
            "Hello! What can I do for you today? If you want to file a cyber complaint, just tell me and I will collect details step by step.",
            language,
        )
        meta["expected_field"] = None
        meta["retry_counts"] = retry_counts
        conversation_service.add_message(db, session, role="assistant", content=greet)
        conversation_service.update_after_chat(db, session, language, "general", {"__meta__": meta})
        return ChatResponse(
            session_id=payload.session_id,
            detected_language=language,
            response=greet,
            intent="general",
            next_field=None,
            collected_fields=_public_fields(session.collected_fields),
            missing_fields=_missing_chat_fields(existing_public_fields),
        )

    # Try Ollama first (in a thread so it doesn't block the event loop).
    # If Ollama times out or is unavailable, fall back to Gemini.
    llm_payload = None
    if ollama_service.is_available():
        print(f"[chat] Trying Ollama ({ollama_service.model}) ...")
        try:
            llm_payload = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama_service.generate_complaint_chat_reply,
                    language,
                    payload.message,
                    session.messages,
                    session.collected_fields or {},
                ),
                timeout=4,  # fail fast so UI voice flow does not hang
            )
        except asyncio.TimeoutError:
            print("[chat] Ollama timed out — falling back to Gemini")
            llm_payload = None
        except Exception as e:
            print(f"[chat] Ollama error: {e} — falling back to Gemini")
            llm_payload = None

    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    gemini_enabled = bool(gemini_api_key) and not gemini_api_key.startswith("dev_dummy")

    if llm_payload is None and gemini_enabled:
        print("[chat] Using Gemini")
        try:
            llm_payload = await asyncio.wait_for(
                asyncio.to_thread(
                    gemini_service.generate_complaint_chat_reply,
                    language,
                    payload.message,
                    session.messages,
                    session.collected_fields or {},
                ),
                timeout=10,
            )
        except Exception as gemini_error:
            print(f"[chat] Gemini error: {gemini_error}")
            merged_public = dict(existing_public_fields)
            candidate_field = expected_field
            if not candidate_field:
                current_missing = _missing_chat_fields(merged_public)
                candidate_field = current_missing[0] if current_missing else None

            inferred_value = _infer_or_buffer_expected_field(candidate_field, payload.message, meta)
            if candidate_field and inferred_value:
                merged_public[candidate_field] = inferred_value

            missing_fields = _missing_chat_fields(merged_public)
            next_field = missing_fields[0] if missing_fields else None
            fallback_prompt = _next_field_prompt(next_field, language)
            if not fallback_prompt and not next_field:
                fallback_prompt = _translate_safe(
                    "All required complaint details are captured. You can now upload evidence/ID proof and submit the complaint.",
                    language,
                )
            if not fallback_prompt:
                fallback_prompt = _translate_safe(
                    "AI service is temporarily busy. Please continue and share your complaint details.",
                    language,
                )
            fallback_prompt = _append_unknown_hint(fallback_prompt, next_field, language, payload.message)

            meta["expected_field"] = next_field
            meta["retry_counts"] = retry_counts
            conversation_service.add_message(db, session, role="assistant", content=fallback_prompt)
            update_payload = {"__meta__": meta}
            if candidate_field and inferred_value:
                update_payload[candidate_field] = inferred_value
            conversation_service.update_after_chat(db, session, language, "general", update_payload)

            return ChatResponse(
                session_id=payload.session_id,
                detected_language=language,
                response=fallback_prompt,
                intent="general",
                next_field=next_field,
                collected_fields=merged_public,
                missing_fields=missing_fields,
            )

    if llm_payload is None and not gemini_enabled:
        merged_public = dict(existing_public_fields)
        candidate_field = expected_field
        if not candidate_field:
            current_missing = _missing_chat_fields(merged_public)
            candidate_field = current_missing[0] if current_missing else None

        inferred_value = _infer_or_buffer_expected_field(candidate_field, payload.message, meta)
        if candidate_field and inferred_value:
            merged_public[candidate_field] = inferred_value

        missing_fields = _missing_chat_fields(merged_public)
        next_field = missing_fields[0] if missing_fields else None
        fallback_prompt = _next_field_prompt(next_field, language)
        if not fallback_prompt and not next_field:
            fallback_prompt = _translate_safe(
                "All required complaint details are captured. You can now upload evidence/ID proof and submit the complaint.",
                language,
            )
        if not fallback_prompt:
            fallback_prompt = _translate_safe(
                "Please continue and share your complaint details clearly.",
                language,
            )
        fallback_prompt = _append_unknown_hint(fallback_prompt, next_field, language, payload.message)

        meta["expected_field"] = next_field
        meta["retry_counts"] = retry_counts
        conversation_service.add_message(db, session, role="assistant", content=fallback_prompt)
        update_payload = {"__meta__": meta}
        if candidate_field and inferred_value:
            update_payload[candidate_field] = inferred_value
        conversation_service.update_after_chat(db, session, language, "general", update_payload)

        return ChatResponse(
            session_id=payload.session_id,
            detected_language=language,
            response=fallback_prompt,
            intent="general",
            next_field=next_field,
            collected_fields=merged_public,
            missing_fields=missing_fields,
        )

    raw_updates = llm_payload.get("field_updates", {})
    sanitized_updates = _sanitize_llm_updates(raw_updates)

    # If we asked for a specific field, only accept that field to prevent random updates.
    candidate_updates: dict[str, str]
    if expected_field:
        candidate_updates = {}
        if expected_field in sanitized_updates:
            candidate_updates[expected_field] = sanitized_updates[expected_field]
    else:
        candidate_updates = sanitized_updates

    valid_updates: dict[str, str] = {}
    for key, value in candidate_updates.items():
        normalized_value = _normalize(value)
        if _looks_like_noise(normalized_value):
            continue
        if key == "complaint_type":
            mapped_type = _normalize_complaint_type(normalized_value)
            if not mapped_type:
                continue
            valid_updates[key] = mapped_type
            continue
        valid_updates[key] = normalized_value

    # If extractor missed the expected field, infer directly from user message.
    if expected_field and expected_field not in valid_updates:
        inferred_value = _infer_or_buffer_expected_field(expected_field, payload.message, meta)
        if inferred_value:
            valid_updates[expected_field] = inferred_value

    # When expected field is still missing, ask for that field again without strict per-turn validation.
    if expected_field and expected_field not in valid_updates:
        prompt = f"Please share your {FIELD_LABELS.get(expected_field, expected_field)}."

        assistant_text = _translate_safe(prompt, language)
        intent = llm_payload.get("intent")
        next_field = expected_field
        assistant_text = _append_unknown_hint(assistant_text, next_field, language, payload.message)

        merged_public = dict(existing_public_fields)
        missing_fields = _missing_chat_fields(merged_public)
        if expected_field not in missing_fields:
            missing_fields.insert(0, expected_field)

        meta["expected_field"] = expected_field
        meta["retry_counts"] = retry_counts

        update_payload = {"__meta__": meta}
        conversation_service.add_message(db, session, role="assistant", content=assistant_text)
        conversation_service.update_after_chat(db, session, language, intent, update_payload)

        return ChatResponse(
            session_id=payload.session_id,
            detected_language=language,
            response=assistant_text,
            intent=intent,
            next_field=next_field,
            collected_fields=_public_fields(session.collected_fields),
            missing_fields=missing_fields,
        )

    # Expected field was captured; clear retry metadata.
    if expected_field and expected_field in valid_updates:
        retry_counts.pop(expected_field, None)
        if expected_field == "phone_number":
            meta.pop("partial_phone_digits", None)

    field_updates = valid_updates
    intent = llm_payload.get("intent")
    merged_public = dict(existing_public_fields)
    merged_public.update(field_updates)
    missing_fields = _missing_chat_fields(merged_public)

    # Keep complaint collection deterministic and stable for users:
    # always follow REQUIRED_COMPLAINT_FIELDS order.
    next_field = missing_fields[0] if missing_fields else None

    meta["expected_field"] = next_field
    meta["retry_counts"] = retry_counts
    field_updates_with_meta = dict(field_updates)
    field_updates_with_meta["__meta__"] = meta

    assistant_text = llm_payload.get("assistant_response") or _translate_safe(
        "What can I help you with?",
        language,
    )

    if next_field:
        assistant_text = f"{assistant_text}\n\n{_next_field_prompt(next_field, language)}"
    else:
        assistant_text = _translate_safe(
            "All required complaint details are captured. You can now upload evidence/ID proof and submit the complaint.",
            language,
        )

    assistant_text = _append_unknown_hint(assistant_text, next_field, language, payload.message)

    # Avoid forced post-translation in fast mode; LLM prompt already requests target language.

    conversation_service.add_message(db, session, role="assistant", content=assistant_text)
    conversation_service.update_after_chat(db, session, language, intent, field_updates_with_meta)

    return ChatResponse(
        session_id=payload.session_id,
        detected_language=language,
        response=assistant_text,
        intent=intent,
        next_field=next_field,
        collected_fields=_public_fields(session.collected_fields),
        missing_fields=missing_fields,
    )

