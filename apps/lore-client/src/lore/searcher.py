import asyncio
import logging
import random
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

class SearXNGSearcher:
    """Manages SearXNG instances and performs searches with fallback rotation.
    """
    
    def __init__(self):
        self.instances: List[str] = []
        self.current_index = 0
        self.client = httpx.AsyncClient(timeout=10.0)
        
    async def discover_instances(self):
        """Fetches public instances from searx.space and filters for high performance.
        """
        logger.info("Fetching SearXNG instances from searx.space...")
        try:
            response = await self.client.get("https://searx.space/data/instances.json")
            response.raise_for_status()
            data = response.json()
            
            instances_data = data.get("instances", {})
            valid_instances = []
            
            for url, info in instances_data.items():
                # Filter for instances that are public, have no error, and are online
                if info.get("network_type") != "public":
                    continue
                if info.get("error") is not None:
                    continue
                
                # Check timing if available (prefer faster instances)
                timing = info.get("timing", {}).get("search", {}).get("average", 999)
                if timing > 2.0: # Skip slow instances (> 2 seconds)
                    continue
                    
                valid_instances.append((url, timing))
            
            # Sort by timing (fastest first)
            valid_instances.sort(key=lambda x: x[1])
            self.instances = [url for url, _ in valid_instances]
            
            # If list is empty, fallback to some known defaults
            if not self.instances:
                logger.warning("No high-performance instances found on searx.space. Using defaults.")
                self.instances = [
                    "https://searx.be/",
                    "https://searxng.site/",
                    "https://paulgo.io/",
                ]
            
            # Shuffle slightly to avoid everyone hitting the same top instance
            # but keep the top ones mostly at the front.
            # Let's just use the top 10 and shuffle them.
            top_n = min(10, len(self.instances))
            top_instances = self.instances[:top_n]
            random.shuffle(top_instances)
            self.instances[:top_n] = top_instances
            
            logger.info(f"Discovered {len(self.instances)} valid SearXNG instances.")
            
        except Exception as e:
            logger.error(f"Failed to discover instances: {e}. Using fallback defaults.")
            self.instances = [
                "https://searx.be/",
                "https://searxng.site/",
                "https://paulgo.io/",
            ]
            
    def _get_next_instance(self) -> str:
        if not self.instances:
            raise RuntimeError("No SearXNG instances available.")
        
        url = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return url

    async def search(self, query: str, num_results: int = 10) -> List[dict]:
        """Performs a search using rotating instances.
        """
        if not self.instances:
            await self.discover_instances()
            
        attempts = 0
        max_attempts = min(5, len(self.instances))
        
        while attempts < max_attempts:
            instance_url = self._get_next_instance()
            logger.info(f"Attempting search on {instance_url} (Attempt {attempts + 1}/{max_attempts})")
            
            try:
                # SearXNG JSON API endpoint is usually /search or /
                # We need to specify format=json
                search_url = instance_url.rstrip("/") + "/search"
                
                # Some instances might not support JSON unless enabled in settings,
                # but public ones usually do if they are listed as working.
                params = {
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": "zh-CN", # Prioritize Chinese results as per objective
                }
                
                response = await self.client.get(search_url, params=params, timeout=5.0)
                
                if response.status_code == 429:
                    logger.warning(f"Rate limited by {instance_url}, switching...")
                    attempts += 1
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                logger.info(f"Successfully retrieved {len(results)} results from {instance_url}")
                return results[:num_results]
                
            except (httpx.HTTPError, ValueError) as e:
                logger.warning(f"Failed to search on {instance_url}: {e}. Trying next instance.")
                attempts += 1
                await asyncio.sleep(0.5) # Short pause before retry
                
        raise RuntimeError("All attempted SearXNG instances failed.")

    async def close(self):
        await self.client.aclose()
