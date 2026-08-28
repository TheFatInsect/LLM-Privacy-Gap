"""
LLM Client for Data Collection Scripts

Standalone LLM client for Reddit relevance filtering and other data collection tasks.
Configured with OrcaRouter by default.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs (HTTP request success messages)
logging.getLogger("httpx").setLevel(logging.WARNING)


# =============================================================================
# Configuration - OrcaRouter API
# =============================================================================

DEFAULT_CONFIG = {
    "provider": "orcarouter",
    "base_url": "https://api.orcarouter.ai/v1",
    "api_key_env": "ORCAROUTER_API_KEY",
    "model": "openai/gpt-5",
    "temperature": 0.1,
    "max_tokens": 2000,
    "timeout": 120.0,
}


# =============================================================================
# Response Data Class
# =============================================================================

@dataclass
class LLMResponse:
    """Standardized LLM response container."""
    content: str
    parsed_json: Optional[Dict] = None
    model: str = ""
    usage: Dict[str, int] = None
    latency_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """
    Simplified LLM client for data collection tasks.
    Pre-configured for OrcaRouter's OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: float = None,
        base_url: str = None,
        api_key: str = None,
        max_retries: int = 3,
    ):
        """
        Initialize LLM client with OrcaRouter defaults.
        
        Args:
            model: Model name (default: openai/gpt-5)
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Max response tokens (default: 2000)
            timeout: Request timeout in seconds (default: 120)
            base_url: API base URL (default: OrcaRouter)
            api_key: API key (default: from ORCAROUTER_API_KEY)
            max_retries: Retry attempts on failure
        """
        self.model = model or os.environ.get("ORCAROUTER_MODEL", DEFAULT_CONFIG["model"])
        self.temperature = temperature if temperature is not None else DEFAULT_CONFIG["temperature"]
        self.max_tokens = max_tokens or DEFAULT_CONFIG["max_tokens"]
        self.timeout = timeout or DEFAULT_CONFIG["timeout"]
        self.base_url = base_url or os.environ.get("ORCAROUTER_BASE_URL", DEFAULT_CONFIG["base_url"])
        self.api_key = api_key or os.environ.get(DEFAULT_CONFIG["api_key_env"])
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Missing OrcaRouter API key. Set the ORCAROUTER_API_KEY environment variable."
            )
        
        # Initialize OpenAI client
        self._client = None
        self._init_client()
        
        logger.info(f"LLMClient initialized: model={self.model}, base_url={self.base_url[:30]}...")
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str = "text",
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            response_format: "text" or "json".
            
        Returns:
            LLMResponse object with content and metadata.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.perf_counter()
                
                response = self._client.chat.completions.create(**request_kwargs)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                content = response.choices[0].message.content
                
                # Parse JSON if requested
                parsed_json = None
                if response_format == "json":
                    try:
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON: {e}")
                        parsed_json = self._extract_json(content)
                
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                
                return LLMResponse(
                    content=content,
                    parsed_json=parsed_json,
                    model=response.model,
                    usage=usage,
                    latency_ms=latency_ms,
                    success=True,
                )
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Retry on transient errors
                if any(x in error_str for x in ["rate", "timeout", "connection", "server"]):
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error
                    break
        
        # All retries failed
        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error(f"LLM request failed after {self.max_retries} attempts: {error_msg}")
        
        return LLMResponse(
            content="",
            success=False,
            error_message=error_msg,
        )
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Try to extract JSON from text that might have extra content."""
        import re
        
        # Try to find JSON block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_str)
                except:
                    continue
        
        return None


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = LLMClient()
    
    response = client.chat_completion(
        system_prompt="You are a helpful assistant. Respond in JSON format.",
        user_prompt="Say hello and include a 'message' field.",
        response_format="json"
    )
    
    print(f"Success: {response.success}")
    print(f"Content: {response.content}")
    print(f"Parsed JSON: {response.parsed_json}")
    print(f"Usage: {response.usage}")
