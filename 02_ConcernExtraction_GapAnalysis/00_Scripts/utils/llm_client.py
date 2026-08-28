# LLM Client
# Unified interface for LLM API calls with async/concurrent support

import os
import httpx
import json
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


class LLMClient:
    """
    Unified LLM client for the Gap Analysis pipeline.
    Supports OpenAI-compatible APIs with concurrent requests.
    """
    
    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0,  # Default to 0 for reproducibility
        max_tokens: int = 999999,
        timeout: int = 120,  # Increased default timeout
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai-compatible",
        max_concurrent: int = 70  # Max concurrent requests
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
            api_key: Optional API key (defaults to env var)
            base_url: Optional base URL for API
            provider: OpenAI-compatible API provider name
            max_concurrent: Maximum concurrent requests
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.provider = provider

        if self.provider.lower() == "orcarouter" and not api_key:
            raise ValueError(
                "Missing OrcaRouter API key. Set the ORCAROUTER_API_KEY environment variable."
            )
        
        # Store credentials for thread-local clients
        self._api_key = api_key
        self._base_url = base_url
        
        # Thread-local storage for clients
        self._local = threading.local()
        
        # Initialize main thread client
        self._init_client()
        
        # Token tracking for cost estimation (thread-safe)
        self._token_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def _init_client(self):
        """Initialize OpenAI client for current thread."""
        client_kwargs = {}
        if self._api_key:
            client_kwargs['api_key'] = self._api_key
        if self._base_url:
            client_kwargs['base_url'] = self._base_url
        
        # Handle SOCKS proxy issues with httpx
        # httpx doesn't support socks:// out of the box. 
        # If we detect a socks proxy, we try to use http:// scheme (common for mixed proxies like Clash/v2ray)
        try:
            proxy_env_vars = ['https_proxy', 'HTTPS_PROXY', 'http_proxy', 'HTTP_PROXY', 'all_proxy', 'ALL_PROXY']
            proxy_url = next((os.environ.get(k) for k in proxy_env_vars if os.environ.get(k)), None)
            
            if proxy_url and ('socks5://' in proxy_url or 'socks://' in proxy_url):
                # Replace socks scheme with http
                new_proxy = proxy_url.replace('socks5://', 'http://').replace('socks://', 'http://')
                # Create custom http client
                client_kwargs['http_client'] = httpx.Client(proxy=new_proxy)
        except Exception as e:
            # If explicit proxy handling fails, fallback to default behavior (let OpenAI/httpx handle it)
            # This ensures we don't break things if something unexpected happens
            print(f"Warning: Attempt to fix proxy configuration failed: {e}")

        self._local.client = OpenAI(**client_kwargs)
    
    @property
    def client(self) -> OpenAI:
        """Get thread-local client instance."""
        if not hasattr(self._local, 'client'):
            self._init_client()
        return self._local.client
    
    def _update_tokens(self, input_tokens: int, output_tokens: int):
        """Thread-safe token counter update."""
        with self._token_lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            Dict with 'content', 'usage', and 'success' keys
        """
        last_error = None
        error_type = "Unknown"
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                
                # Track token usage (thread-safe)
                usage = response.usage
                if usage:
                    self._update_tokens(usage.prompt_tokens, usage.completion_tokens)
                
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0
                    }
                }
            
            except ConnectionError as e:
                last_error = e
                error_type = "ConnectionError"
            except TimeoutError as e:
                last_error = e
                error_type = "TimeoutError"
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
            
            # Retry with exponential backoff
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        
        return {
            "success": False,
            "error": f"{error_type}: {str(last_error)}",
            "content": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Send a chat request expecting JSON response.
        
        Args:
            messages: List of message dicts
            max_retries: Number of retry attempts
        
        Returns:
            Parsed JSON response or error dict
        """
        response = self.chat(
            messages=messages,
            response_format={"type": "json_object"},
            max_retries=max_retries
        )
        
        if not response["success"]:
            return response
        
        try:
            response["parsed"] = json.loads(response["content"])
        except json.JSONDecodeError as e:
            response["success"] = False
            response["error"] = f"JSON parse error: {str(e)}"
            response["parsed"] = None
        
        return response
    
    def chat_json_batch(
        self,
        batch: List[tuple],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chat requests concurrently.
        
        Args:
            batch: List of (item_id, messages) tuples
            max_workers: Max concurrent workers (default: self.max_concurrent)
            progress_callback: Optional callback(completed, total) for progress updates
        
        Returns:
            List of (item_id, response) tuples in same order as input
        """
        if max_workers is None:
            max_workers = self.max_concurrent
        
        results = [None] * len(batch)
        completed = 0
        
        def process_one(idx: int, item_id: Any, messages: List[Dict[str, str]]) -> tuple:
            """Process a single request."""
            response = self.chat_json(messages)
            return idx, item_id, response
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_one, idx, item_id, messages): idx
                for idx, (item_id, messages) in enumerate(batch)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx, item_id, response = future.result()
                results[idx] = (item_id, response)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(batch))
        
        return results
    
    def get_cost_estimate(
        self,
        input_price_per_1k: float = 0.001,
        output_price_per_1k: float = 0.008
    ) -> Dict[str, float]:
        """
        Get estimated cost based on tracked token usage.
        
        Args:
            input_price_per_1k: Price per 1K input tokens
            output_price_per_1k: Price per 1K output tokens
        
        Returns:
            Dict with cost breakdown
        """
        input_cost = (self.total_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_price_per_1k
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    
    def reset_token_tracking(self):
        """Reset token tracking counters."""
        with self._token_lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LLMClient':
        """
        Create LLMClient from pipeline configuration.
        
        Args:
            config: Pipeline configuration dict
        
        Returns:
            Configured LLMClient instance
        """
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'openai-compatible')
        api_key_env = llm_config.get('api_key_env')
        api_key = llm_config.get('api_key')
        if not api_key and api_key_env:
            api_key = os.environ.get(api_key_env)

        return cls(
            model=llm_config.get(
                'model',
                'openai/gpt-5' if provider.lower() == 'orcarouter' else 'gpt-5'
            ),
            temperature=llm_config.get('temperature', 0),  # Default to 0 for reproducibility
            max_tokens=llm_config.get('max_tokens', 999999),
            timeout=llm_config.get('timeout', 120),  # Increased default
            api_key=api_key,
            base_url=llm_config.get('base_url'),
            provider=provider,
            max_concurrent=llm_config.get('max_concurrent', 70)
        )
