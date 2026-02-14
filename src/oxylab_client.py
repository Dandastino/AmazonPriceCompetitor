import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OXYLABS_BASE_URL = "https://realtime.oxylabs.io/v1/queries"
DEFAULT_TIMEOUT = 30
RATE_LIMIT_DELAY = 0.1
SEARCH_STRATEGIES = ["featured", "price_asc", "price_desc", "rating_desc", "avg_rating_desc"]


class OxylabsClient:
    """Client for Oxylabs API with error handling and rate limiting."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """Initialize Oxylabs client with credentials from environment."""
        self.timeout = timeout
        self.username = os.getenv("OXYLAB_USERNAME")
        self.password = os.getenv("OXYLAB_PASSWORD")

        if not self.username or not self.password:
            raise ValueError(
                "Oxylabs credentials not found. Set OXYLAB_USERNAME and OXYLAB_PASSWORD in .env"
            )

        logger.info("OxylabsClient initialized")

    def post_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post query to Oxylabs API with error handling.

        Args:
            payload: Query payload dict

        Returns:
            JSON response from API

        Raises:
            ValueError: Invalid credentials
            ConnectionError: Network issue
            TimeoutError: Request timeout
            Exception: API error
        """
        try:
            logger.debug(f"Posting query to Oxylabs: {payload.get('source')}")
            response = requests.post(
                OXYLABS_BASE_URL,
                auth=(self.username, self.password),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Oxylabs API request successful")
            return response.json()

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to Oxylabs API: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"Oxylabs API request timed out (>{self.timeout}s). Check internet."
            logger.error(error_msg)
            raise TimeoutError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"Oxylabs API error {e.response.status_code}: {e.response.text[:200]}"
            logger.error(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Unexpected error in Oxylabs request: {e}", exc_info=True)
            raise

    @staticmethod
    def extract_content(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product content from API response."""
        if not isinstance(payload, dict):
            return {}

        # Try results array first (Oxylabs standard)
        if "results" in payload and isinstance(payload["results"], list) and payload["results"]:
            first = payload["results"][0]
            if isinstance(first, dict) and "content" in first:
                content = first["content"]
                return content if content else {}

        # Fallback to direct content
        if "content" in payload:
            return payload.get("content", {})

        return {}

    @staticmethod
    def normalize_product(content: Dict[str, Any], asin: str = "", domain: str = "", geo: str = "") -> Dict[str, Any]:
        """
        Normalize product data from API response.

        Args:
            content: Raw product content from API
            asin: Product ASIN (fallback if not in content)
            domain: Amazon domain
            geo: Geographic location

        Returns:
            Normalized product dict
        """
        if not isinstance(content, dict):
            content = {}

        category_path = []
        if content.get("category_path") and isinstance(content["category_path"], list):
            category_path = [str(cat).strip() for cat in content["category_path"] if cat]

        normalized = {
            "asin": content.get("asin") or asin,
            "url": content.get("url", ""),
            "brand": content.get("brand", ""),
            "price": content.get("price"),
            "stock": content.get("stock", ""),
            "title": content.get("title", ""),
            "rating": content.get("rating"),
            "images": content.get("images", []),
            "categories": content.get("category", []) or content.get("categories", []),
            "category_path": category_path,
            "currency": content.get("currency", "$"),
            "buybox": content.get("buybox", []),
            "product_overview": content.get("product_overview", []),
            "amazon_domain": domain,
            "amazon_geo_location": geo,
        }

        return normalized

    def scrape_product_details(self, asin: str, geo_location: str, domain: str) -> Optional[Dict[str, Any]]:
        """
        Scrape product details for a single ASIN.

        Args:
            asin: Product ASIN
            geo_location: Geographic location (e.g., 'us')
            domain: Amazon domain (e.g., 'com')

        Returns:
            Normalized product dict or None on failure
        """
        try:
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            payload = {
                "source": "amazon_product",
                "query": asin,
                "geo_location": geo_location,
                "domain": domain,
                "parse": True,
            }

            logger.info(f"Scraping product: {asin} for {domain}/{geo_location}")
            raw = self.post_query(payload)
            content = self.extract_content(raw)
            normalized = self.normalize_product(content, asin, domain, geo_location)

            if not normalized.get("title"):
                logger.warning(f"No title found for ASIN {asin}")

            logger.debug(f"Successfully scraped {asin}")
            return normalized

        except ValueError as e:
            logger.error(f"Validation error for {asin}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error scraping product {asin}: {e}", exc_info=True)
            raise

    @staticmethod
    def clean_product_name(title: str) -> str:
        """Clean product title for search queries."""
        if not title or not isinstance(title, str):
            return ""

        cleaned = title.strip()
        if not cleaned:
            return ""

        # Split on common delimiters and keep left part
        parts = re.split(r"\s*[-|:]\s*", cleaned, maxsplit=1)
        return parts[0].strip()

    @staticmethod
    def extract_search_results(content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search result items from API response."""
        if not isinstance(content, dict):
            return []

        items = []

        # Oxylabs search response structure
        if "results" in content and isinstance(content["results"], dict):
            results = content["results"]
            if "organic" in results and isinstance(results["organic"], list):
                items.extend(results["organic"])
            if "paid" in results and isinstance(results["paid"], list):
                items.extend(results["paid"])

        # Fallback structure
        elif "product" in content and isinstance(content.get("product"), list):
            items.extend(content["product"])

        logger.debug(f"Extracted {len(items)} search result items")
        return items

    @staticmethod
    def normalize_search_result(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single search result."""
        if not isinstance(item, dict):
            return None

        asin = item.get("asin") or item.get("product_asin")
        title = item.get("title")

        if not (asin and title):
            return None

        return {
            "asin": asin,
            "title": title,
            "categories": item.get("categories"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "url": item.get("url", ""),
            "images": item.get("images", []),
        }

    def search_competitors(
        self,
        query_title: str,
        domain: str,
        categories: List[str],
        pages: int = 1,
        geo_location: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search for competitor products.

        Args:
            query_title: Product title to search
            domain: Amazon domain
            categories: Category filters
            pages: Number of pages to search
            geo_location: Geographic location

        Returns:
            List of normalized search results
        """
        try:
            if not query_title or not isinstance(query_title, str):
                logger.warning(f"Invalid query_title: {query_title}")
                return []

            if pages < 1 or pages > 5:
                logger.warning(f"Pages {pages} out of range, clamping to 1-5")
                pages = max(1, min(pages, 5))

            st.write("🔎 Searching for competitors")
            search_title = self.clean_product_name(query_title)
            results = []
            seen_asins = set()

            total_requests = len(SEARCH_STRATEGIES) * pages
            progress_bar = st.progress(0)
            progress_text = st.empty()
            req_count = 0

            logger.info(f"Starting competitor search: '{search_title}' in {domain}/{geo_location}")

            for strategy_idx, sort_by in enumerate(SEARCH_STRATEGIES):
                for page in range(1, pages + 1):
                    try:
                        payload = {
                            "source": "amazon_search",
                            "query": search_title,
                            "domain": domain,
                            "geo_location": geo_location,
                            "categories": categories,
                            "sort_by": sort_by,
                            "page": page,
                            "parse": True,
                        }

                        if categories and categories[0]:
                            payload["refinement"] = {"category": categories[0]}

                        req_count += 1
                        progress_text.write(f"Searching [{sort_by}, page {page}]...")
                        progress_bar.progress(min(req_count / total_requests, 1.0))

                        content = self.extract_content(self.post_query(payload))
                        items = self.extract_search_results(content)

                        for item in items:
                            normalized = self.normalize_search_result(item)
                            if normalized and normalized["asin"] not in seen_asins:
                                seen_asins.add(normalized["asin"])
                                results.append(normalized)

                        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting

                    except Exception as e:
                        logger.warning(f"Error in search strategy {sort_by}, page {page}: {e}")
                        progress_text.write(f"⚠️ Error in strategy {sort_by}: {str(e)[:50]}")
                        continue

            progress_bar.empty()
            progress_text.empty()

            st.write(f"✅ Found {len(results)} competitors")
            logger.info(f"Competitor search complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching competitors: {e}", exc_info=True)
            st.error(f"Error searching competitors: {str(e)}")
            return []

    def scrape_multiple_products(
        self, asins: List[str], geo_location: str, domain: str
    ) -> List[Dict[str, Any]]:
        """
        Scrape details for multiple products.

        Args:
            asins: List of product ASINs
            geo_location: Geographic location
            domain: Amazon domain

        Returns:
            List of normalized product dicts
        """
        if not asins or not isinstance(asins, list):
            logger.warning(f"Invalid asins input: {asins}")
            return []

        st.write(f"🔎 Scraping {len(asins)} products from {geo_location} ({domain})")

        products = []
        progress_text = st.empty()
        progress_bar = st.progress(0)

        logger.info(f"Starting scraping of {len(asins)} products")

        for idx, asin in enumerate(asins, 1):
            try:
                progress_text.write(f"Processing {idx}/{len(asins)}: {asin}")
                progress_bar.progress(idx / len(asins))

                product = self.scrape_product_details(asin, geo_location, domain)
                products.append(product)
                progress_text.write(f"✅ {product.get('title', asin)[:50]}")

                time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error scraping {asin}: {e}")
                progress_text.write(f"❌ Error: {asin} - {str(e)[:40]}")
                continue

        progress_text.empty()
        progress_bar.empty()

        st.write(f"✅ Scraped {len(products)}/{len(asins)} products successfully")
        logger.info(f"Scraping complete: {len(products)}/{len(asins)} products")
        return products


# Backward-compatible function interfaces
_client = None


def get_client() -> OxylabsClient:
    """Get or create global Oxylabs client."""
    global _client
    if _client is None:
        _client = OxylabsClient()
    return _client


def scrape_product_details(asin: str, geo_location: str, domain: str) -> Optional[Dict[str, Any]]:
    """Scrape product details (uses OxylabsClient)."""
    return get_client().scrape_product_details(asin, geo_location, domain)


def search_competitors(
    query_title: str, domain: str, categories: List[str], pages: int = 1, geo_location: str = ""
) -> List[Dict[str, Any]]:
    """Search competitors (uses OxylabsClient)."""
    return get_client().search_competitors(query_title, domain, categories, pages, geo_location)


def scrape_multiple_products(asins: List[str], geo_location: str, domain: str) -> List[Dict[str, Any]]:
    """Scrape multiple products (uses OxylabsClient)."""
    return get_client().scrape_multiple_products(asins, geo_location, domain)


