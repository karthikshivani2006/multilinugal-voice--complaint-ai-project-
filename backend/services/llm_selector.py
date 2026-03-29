"""
LLM Service with Gemini primary and Ollama fallback.
Provides intelligent version selection with automatic fallback.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Track which LLM is active
ACTIVE_LLM = None


def get_llm_service():
    """
    Get appropriate LLM service with intelligent fallback.
    - Primary: Gemini (powerful, cloud-based)
    - Fallback: Ollama (local, always available)
    """
    global ACTIVE_LLM
    
    try:
        from services.gemini_service import gemini_service
        from config import settings
        
        # Check if Gemini API is properly configured
        if settings.gemini_api_key and settings.gemini_api_key != "dev_key_placeholder":
            ACTIVE_LLM = "gemini-2.0-flash"
            logger.info("✅ Using Gemini 2.0 Flash (primary)")
            return gemini_service
        else:
            logger.warning("⚠️  Gemini API key not configured, falling back to Ollama")
    except Exception as e:
        logger.warning(f"⚠️  Gemini service initialization failed: {e}, falling back to Ollama")
    
    try:
        from services.ollama_service import ollama_service
        ACTIVE_LLM = "ollama-llama2"
        logger.info("✅ Using Ollama Llama2 (fallback)")
        return ollama_service
    except Exception as e:
        logger.error(f"❌ Both Gemini and Ollama failed: {e}")
        raise RuntimeError("No LLM service available! Configure Gemini API key or Ollama.")


def get_active_llm_name() -> str:
    """Get the name of the currently active LLM."""
    global ACTIVE_LLM
    if ACTIVE_LLM is None:
        # Initialize by calling get_llm_service
        get_llm_service()
    return ACTIVE_LLM or "unknown"


# Export the intelligent LLM selector
llm_service = get_llm_service()
