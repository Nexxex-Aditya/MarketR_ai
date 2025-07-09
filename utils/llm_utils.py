import os
from dotenv import load_dotenv
import openai
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Load environment variables (consider moving this to a central config load in orchestrator)
load_dotenv()

# --- LLM Client Abstraction (conceptual) ---
# In a real-world scenario, you'd have different client implementations
# for OpenAI, Gemini, HuggingFace, etc.

def get_llm_client(llm_config: Dict[str, Any]):
    provider = llm_config.get('provider', 'openai').lower()
    api_key = llm_config.get('api_key', os.getenv('OPENAI_API_KEY'))

    if not api_key:
        logger.warning(f"API key not found for LLM provider: {provider}")

    if provider == 'openai':
        openai.api_key = api_key
        return openai.ChatCompletion # Return the module/class that handles the API call
    elif provider == 'gemini':
        # Placeholder for Gemini integration
        # from google.generativeai import GenerativeModel # example import
        # return GenerativeModel(model_name=llm_config.get('model', 'gemini-pro'))
        logger.warning("Gemini integration not fully implemented in this demo.")
        raise NotImplementedError("Gemini client not implemented.")
    # Add more providers as needed
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def call_llm(prompt: str, llm_config: Dict[str, Any], temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Call the specified LLM with a prompt and return the response text.
    """
    try:
        client = get_llm_client(llm_config)
        model_name = llm_config.get('model', 'gpt-4') # Default to gpt-4 for OpenAI

        if llm_config.get('provider', 'openai').lower() == 'openai':
            response = client.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        elif llm_config.get('provider').lower() == 'gemini':
            # Example for Gemini, adjust based on actual API
            # response = client.generate_content(prompt, temperature=temperature, max_tokens=max_tokens)
            # return response.text.strip()
            raise NotImplementedError("Gemini client not implemented.")
        else:
            raise ValueError("Unsupported LLM provider for direct call.")

    except NotImplementedError as e:
        logger.error(f"LLM functionality not implemented for selected provider: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return f"Error: {e}" 