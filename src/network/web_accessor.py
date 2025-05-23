# src/network/web_accessor.py
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urljoin
import json
# from playwright.async_api import async_playwright # Commented out
import hashlib
import time # For rate limiter

# Assuming EventBus and Event will be imported from src.event_system
from src.event_system.event_bus import EventBus, Event, EventPriority # Adjusted import
# Assuming memory_system will be an instance of InfiniteMemorySystem or similar
# from src.memory.infinite_memory import InfiniteMemorySystem # For type hint if needed

class RateLimiter:
    """速率限制器"""
    def __init__(self, default_delay: float = 1.0):
        self.limits: Dict[str, float] = {} # Stores last access time for a domain
        self.default_delay: float = default_delay  # Default delay in seconds
        self._lock = asyncio.Lock() # Lock for thread-safe access to self.limits

    async def acquire(self, domain: str):
        """获取访问许可"""
        async with self._lock:
            last_access_time = self.limits.get(domain, 0)
            now = time.monotonic() # Using monotonic clock
            
            wait_time = 0
            if now - last_access_time < self.default_delay:
                wait_time = self.default_delay - (now - last_access_time)
            
            if wait_time > 0:
                print(f"RateLimiter: Waiting {wait_time:.2f}s for domain {domain}")
                await asyncio.sleep(wait_time)
            
            self.limits[domain] = time.monotonic()

class WebAccessor:
    """增强的网络访问器"""
    def __init__(self, event_bus: EventBus, memory_system: Any, default_user_agent: Optional[str] = None):
        self.event_bus = event_bus
        self.memory_system = memory_system # Should be an instance of InfiniteMemorySystem or similar
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright_context = None # Placeholder for playwright browser context
        self.browser = None # Placeholder for playwright browser
        self.rate_limiter = RateLimiter(default_delay=1.0) # Default 1s delay between requests to the same domain

        self.default_user_agent = default_user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        print(f"WebAccessor initialized. Playwright features are currently disabled (commented out).")

    async def initialize(self):
        """初始化网络访问器"""
        self.session = aiohttp.ClientSession(headers={'User-Agent': self.default_user_agent})
        print("aiohttp.ClientSession initialized for WebAccessor.")
        
        # Initialize Playwright (commented out to avoid immediate dependency)
        # try:
        #     pw_instance = await async_playwright().start()
        #     # Using chromium, can be firefox or webkit
        #     self.browser = await pw_instance.chromium.launch(headless=True) 
        #     self.playwright_context = await self.browser.new_context(
        #         user_agent=self.default_user_agent
        #     )
        #     print("Playwright initialized successfully (Chromium).")
        # except Exception as e:
        #     print(f"Playwright initialization failed: {e}. Dynamic content fetching with browser will be unavailable.")
        #     self.browser = None
        #     self.playwright_context = None
        pass


    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves content from memory system acting as a cache."""
        if hasattr(self.memory_system, 'retrieve'):
            # Assuming short-term memory is used for caching
            # The key needs to be specific to avoid collision if memory_system is shared
            # For simplicity, using the key directly. A prefix might be better.
            cache_results = await self.memory_system.retrieve(query=cache_key, memory_types=['short_term'], top_k=1)
            if cache_results:
                cached_data = cache_results[0]
                # Ensure it's not just a placeholder or unrelated item
                if cached_data.get('id') == cache_key and 'web_content' in cached_data:
                    print(f"Cache hit for {cache_key}")
                    return cached_data['web_content'] # Assuming content is stored under 'web_content'
        return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl_seconds: int = 3600):
        """Caches the result in the memory system."""
        if hasattr(self.memory_system, 'store'):
            # Store the actual web content under a specific field
            await self.memory_system.store(
                memory_type='short_term', 
                data={'web_content': result, 'retrieved_at': datetime.now().isoformat()}, # Storing the entire result dict
                memory_id=cache_key, # Use cache_key as the ID in memory
                # Pass expiry in hours for store method if it supports it like the example memory system
                # expiry_hours = ttl_seconds / 3600 
            )
            print(f"Cached result for {cache_key}")

    async def fetch_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """智能获取URL内容"""
        options = options or {}
        request_hash = hashlib.md5(f"{url}{json.dumps(options, sort_keys=True)}".encode()).hexdigest()
        cache_key = f"web_cache_{request_hash}"

        if not options.get('force_refresh'):
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
            
        domain = urlparse(url).netloc
        if not domain:
             return {"success": False, "error": "Invalid URL (missing domain)", "url": url, "status": -1}
        await self.rate_limiter.acquire(domain)
        
        response_data: Dict[str, Any] = {"url": url, "success": False, "status": -1}

        try:
            use_browser = await self._requires_browser(url, options)
            if use_browser:
                if self.playwright_context:
                    # result = await self._fetch_with_browser(url, options) # Actual call
                    print(f"Placeholder: _fetch_with_browser would be called for {url}. Playwright is disabled.")
                    response_data.update({"error": "Playwright not enabled/initialized", "content": "Dynamic fetch unavailable."})
                else:
                    print(f"Warning: Browser fetch required for {url} but Playwright is not available. Falling back to aiohttp.")
                    response_data = await self._fetch_with_session(url, options)
            else:
                response_data = await self._fetch_with_session(url, options)
                
            if response_data.get("success"):
                await self._cache_result(cache_key, response_data)
            
            # Send event regardless of success to log the attempt
            event_type = "web.fetched" if response_data.get("success") else "web.fetch_failed"
            event_data = {
                "url": url,
                "status": response_data.get('status'),
                "success": response_data.get('success', False),
                "content_type": response_data.get('content_type'),
                "error": response_data.get('error')
            }
            await self.event_bus.emit(Event(type=event_type, data=event_data, source="WebAccessor"))
            
            return response_data
            
        except Exception as e:
            print(f"Exception during fetch_url for {url}: {e}")
            error_response = {"success": False, "error": str(e), "url": url, "status": -2}
            await self.event_bus.emit(Event(type="web.fetch_failed", data=error_response, source="WebAccessor"))
            return error_response

    async def _requires_browser(self, url: str, options: Dict[str, Any]) -> bool:
        """Determines if fetching the URL likely requires a full browser (JS execution)."""
        if options.get('force_browser'): return True
        if options.get('force_simple_request'): return False
        # Simple heuristic: if it's not a common content type, or if JS rendering is suspected.
        # This is a placeholder. More sophisticated logic (e.g. based on URL patterns, past failures) can be added.
        if any(ext in url.lower() for ext in ['.js', '.json', '.xml', '.css', '.txt', '.md']):
            return False # Usually these don't need full browser
        if options.get('wait_for_selector') or options.get('execute_script'):
            return True # These options imply browser usage
        print(f"Placeholder: _requires_browser check for {url} (defaulting to False).")
        return False # Default to simple fetch

    async def _fetch_with_session(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches content using aiohttp.ClientSession."""
        if not self.session:
            return {"success": False, "error": "aiohttp.ClientSession not initialized.", "url": url, "status": -3}
        
        headers = options.get('headers', {})
        try:
            async with self.session.get(url, headers=headers, timeout=options.get('timeout', 30)) as response:
                content_type = response.headers.get('Content-Type', '')
                
                if "text/html" in content_type or "application/json" in content_type or "text/plain" in content_type:
                    text_content = await response.text()
                    content_to_return = text_content
                    # Basic parsing for HTML to extract main text using BeautifulSoup
                    if "text/html" in content_type and options.get("extract_main_text", True):
                        soup = BeautifulSoup(text_content, 'html.parser')
                        # Remove script and style elements
                        for script_or_style in soup(["script", "style"]):
                            script_or_style.decompose()
                        # Get text
                        text = soup.get_text()
                        # Break into lines and remove leading/trailing space on each
                        lines = (line.strip() for line in text.splitlines())
                        # Break multi-headlines into a line each
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        # Drop blank lines
                        content_to_return = '\n'.join(chunk for chunk in chunks if chunk)
                elif "image/" in content_type or "application/pdf" in content_type: # Handle binary
                    content_to_return = await response.read() # bytes
                else: # Other binary or unknown types
                    content_to_return = await response.read()

                return {
                    "success": response.status >= 200 and response.status < 300,
                    "status": response.status,
                    "url": str(response.url), # Final URL after redirects
                    "content": content_to_return,
                    "content_type": content_type,
                    "headers": dict(response.headers)
                }
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"aiohttp.ClientError: {str(e)}", "url": url, "status": -4}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timed out.", "url": url, "status": -5}
        except Exception as e: # Catch any other exceptions
            return {"success": False, "error": f"Generic error in _fetch_with_session: {str(e)}", "url": url, "status": -6}


    async def _fetch_with_browser(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """(Placeholder) 使用浏览器获取动态内容. Playwright logic commented out."""
        if not self.playwright_context: # Should be self.browser or self.playwright_context
            return {"success": False, "error": "Playwright context not initialized.", "url": url, "status": -7}

        print(f"Placeholder: Simulating fetch with Playwright for URL: {url}")
        # page = await self.playwright_context.new_page()
        # try:
        #     if options.get('user_agent'): # This should be set at context level ideally
        #         await page.set_extra_http_headers({'User-Agent': options['user_agent']})
            
        #     response = await page.goto(url, wait_until=options.get('wait_until', 'networkidle'), timeout=options.get('timeout', 60000))
            
        #     if options.get('wait_for_selector'):
        #         await page.wait_for_selector(options['wait_for_selector'], timeout=options.get('selector_timeout', 30000))
                
        #     if options.get('execute_script'):
        #         script_result = await page.evaluate(options['execute_script'])
        #     else:
        #         script_result = None
                
        #     content = await page.content() # Full HTML content
        #     main_text = None
        #     if options.get("extract_main_text", True):
        #         # Using BeautifulSoup on browser content for consistency, or Playwright's locators
        #         soup = BeautifulSoup(content, 'html.parser')
        #         for script_or_style in soup(["script", "style"]): script_or_style.decompose()
        #         text = soup.get_text()
        #         lines = (line.strip() for line in text.splitlines())
        #         chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        #         main_text = '\n'.join(chunk for chunk in chunks if chunk)

        #     screenshot_bytes = None
        #     if options.get('screenshot'):
        #         screenshot_bytes = await page.screenshot(full_page=options.get('full_page_screenshot', True))
                
        #     return {
        #         "success": response.status >= 200 and response.status < 300 if response else False,
        #         "url": page.url,
        #         "content": main_text if main_text else content, # Prefer main text if extracted
        #         "full_html": content if main_text else None, # Provide full HTML if main text was different
        #         "status": response.status if response else -1,
        #         "content_type": response.headers.get('content-type', 'text/html') if response else 'text/html',
        #         "screenshot_base64": base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else None,
        #         "script_result": script_result,
        #         "headers": response.headers if response else None
        #     }
        # except Exception as e:
        #     return {"success": False, "error": f"Playwright error: {str(e)}", "url": url, "status": -8}
        # finally:
        #     if 'page' in locals() and page: await page.close()
        return {"success": False, "error": "Playwright functionality is currently a placeholder.", "url": url, "content": "Dynamic fetch placeholder."}


    async def search_web(self, query: str, search_engine: str = "duckduckgo", num_results: int = 5) -> List[Dict[str, Any]]:
        """(Placeholder) 执行网络搜索"""
        print(f"Placeholder: Performing web search for '{query}' using '{search_engine}' (top {num_results} results).")
        
        # In a real implementation, this would use APIs or web scraping for search engines.
        # Example structure for results:
        # results = [
        #     {"title": "Example Title 1", "url": "http://example.com/page1", "snippet": "Description of page 1..."},
        #     {"title": "Example Title 2", "url": "http://example.com/page2", "snippet": "Description of page 2..."},
        # ]
        
        # Placeholder results:
        results = []
        for i in range(num_results):
            results.append({
                "title": f"Placeholder Search Result {i+1} for '{query}'",
                "url": f"http://example.com/search_placeholder_{search_engine}_{i+1}",
                "snippet": f"This is a placeholder snippet for result {i+1} from {search_engine} regarding '{query}'.",
                "summary": "Placeholder summary (would fetch and summarize URL content)." 
            })

        # The original design had a section to fetch summaries for each result.
        # This can be very time-consuming. Making it optional or a secondary step.
        # For now, the 'summary' is just a placeholder.
        # If actual fetching is needed here, it should use self.fetch_url carefully.

        await self.event_bus.emit(Event(
            type="web.searched",
            data={"query": query, "engine": search_engine, "num_results_requested": num_results, "results_returned_count": len(results)},
            source="WebAccessor"
        ))
        return results
        
    async def monitor_changes(self, url: str, interval: int = 3600, options: Optional[Dict[str, Any]] = None):
        """(Placeholder) 监控网页变化"""
        print(f"Placeholder: Starting to monitor changes for URL: {url} every {interval}s.")
        previous_hash = None
        
        while True: # This creates an infinite loop; manage task cancellation in main agent
            try:
                fetch_options = options or {}
                fetch_options['force_refresh'] = True # Always get fresh copy for monitoring
                result = await self.fetch_url(url, fetch_options)
                
                if result.get('success') and isinstance(result.get('content'), (str, bytes)):
                    current_content = result['content']
                    # Ensure content is bytes for hashlib
                    if isinstance(current_content, str):
                        current_content_bytes = current_content.encode('utf-8', 'ignore')
                    else:
                        current_content_bytes = current_content
                    
                    content_hash = hashlib.md5(current_content_bytes).hexdigest()
                    
                    if previous_hash and content_hash != previous_hash:
                        print(f"Change detected for {url}! Old hash: {previous_hash}, New hash: {content_hash}")
                        await self.event_bus.emit(Event(
                            type="web.changed",
                            data={
                                "url": url,
                                "previous_hash": previous_hash,
                                "new_hash": content_hash,
                                # Optionally include some part of content if small, or diff
                                "content_preview": result['content'][:200] if isinstance(result['content'], str) else "(binary content)"
                            },
                            source="WebAccessor"
                        ))
                    else:
                        print(f"No change detected for {url} (hash: {content_hash}).")
                        
                    previous_hash = content_hash
                else:
                    print(f"Failed to fetch {url} for monitoring or content not suitable for hashing: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"Error during monitor_changes loop for {url}: {e}")
                # Potentially emit an event for monitoring error
                await self.event_bus.emit(Event(
                    type="web.monitor_error",
                    data={"url": url, "error": str(e)},
                    source="WebAccessor"
                ))
            
            print(f"Monitor: Sleeping for {interval}s for {url}.")
            await asyncio.sleep(interval)

    async def close(self):
        """Closes sessions and browser if they exist."""
        if self.session and not self.session.closed:
            await self.session.close()
            print("aiohttp.ClientSession closed.")
        # if self.browser:
        #     await self.browser.close()
        #     print("Playwright browser closed.")
        # if hasattr(async_playwright, '_instance') and async_playwright._instance: # Check if playwright was started
        #     await async_playwright().stop()
        #     print("Playwright stopped.")
        pass # Playwright cleanup is commented out

# Helper for datetime if not available elsewhere
from datetime import datetime
