import os
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.core import MongoDB
from src.utils import extract_category_names, get_logger, handle_exception

load_dotenv()

logger = get_logger(__name__)


class ProductTypeDetector:
    """Detect and match product types for accurate competitor filtering."""
    
    # Product type keywords - organized by category
    PRODUCT_TYPES = {
        'shampoo': ['shampoo', 'shampooing', 'hair wash', 'hairwash'],
        'conditioner': ['conditioner', 'conditioning', 'cond.'],
        'cream': ['cream', 'creams', 'hair cream', 'treatment cream'],
        'mask': ['mask', 'hair mask', 'deep treatment'],
        'oil': ['oil', 'hair oil', 'serum oil', 'argan oil'],
        'spray': ['spray', 'hair spray', 'setting spray', 'texture spray'],
        'gel': ['gel', 'hair gel', 'styling gel'],
        'paste': ['paste', 'hair paste', 'styling paste'],
        'mousse': ['mousse', 'hair mousse', 'foam'],
        'lotion': ['lotion', 'hair lotion', 'leave-in'],
        'serum': ['serum', 'hair serum', 'smoothing serum'],
        'balm': ['balm', 'hair balm', 'styling balm'],
        'wax': ['wax', 'hair wax', 'styling wax'],
        'pomade': ['pomade', 'hair pomade'],
        'toner': ['toner', 'hair toner', 'color toner'],
        'shampoo bar': ['shampoo bar', 'solid shampoo'],
        'conditioner bar': ['conditioner bar', 'solid conditioner'],
        'rinse': ['rinse', 'hair rinse', 'final rinse'],
        'essence': ['essence', 'hair essence'],
    }
    
    @classmethod
    def extract_product_type(cls, title: str) -> Optional[str]:
        """
        Extract product type from title.
        
        Returns:
            Product type (e.g., 'shampoo', 'conditioner') or None if unclear
        """
        if not title:
            return None
        
        title_lower = title.lower()
        
        # Check each product type and its keywords
        for product_type, keywords in cls.PRODUCT_TYPES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return product_type
        
        return None
    
    @classmethod
    def normalize_product_type(cls, product_type: Optional[str]) -> Optional[str]:
        """
        Normalize product type variations.
        Example: 'shampooing' -> 'shampoo'
        """
        if not product_type:
            return None
        
        # Direct match
        if product_type in cls.PRODUCT_TYPES:
            return product_type
        
        # Check if it's in keyword list and map back to main type
        product_type_lower = product_type.lower()
        for main_type, keywords in cls.PRODUCT_TYPES.items():
            if product_type_lower in keywords:
                return main_type
        
        return None
    
    @classmethod
    def get_product_type(cls, product_data: Dict[str, Any]) -> Optional[str]:
        """
        Get normalized product type from product data.
        """
        title = product_data.get('title', '')
        product_type = cls.extract_product_type(title)
        return cls.normalize_product_type(product_type)
    
    @classmethod
    def same_product_type(cls, product1: Dict[str, Any], product2: Dict[str, Any]) -> bool:
        """
        Check if two products are the same type.
        """
        type1 = cls.get_product_type(product1)
        type2 = cls.get_product_type(product2)
        
        # Both have types and they match
        if type1 and type2:
            return type1 == type2
        
        # If one or both have no detected type, return False (require explicit match)
        return False


class UnitPriceNormalizer:
    """Normalize prices by unit volume/weight for fair comparison."""
    
    # Size patterns to extract from product titles
    SIZE_PATTERNS = {
        # Weight: grams to kg
        r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\b': ('weight_g', 1),
        r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram?)\b': ('weight_g', 1000),
        r'(\d+(?:\.\d+)?)\s*(?:oz|ounces?)\b': ('weight_g', 28.35),
        r'(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?)\b': ('weight_g', 453.6),
        
        # Volume: ml to liters
        r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter?)\b': ('volume_ml', 1),
        r'(\d+(?:\.\d+)?)\s*(?:l|liter?|litre?)\b': ('volume_ml', 1000),
        r'(\d+(?:\.\d+)?)\s*(?:fl\s?oz|fluid ounces?)\b': ('volume_ml', 29.57),
        r'(\d+(?:\.\d+)?)\s*(?:pint?)\b': ('volume_ml', 473.2),
    }
    
    # Standard unit: 100g or 100ml
    STANDARD_UNIT_VALUE = 100
    
    @classmethod
    def extract_size(cls, title: str) -> Optional[Tuple[str, float, str]]:
        """
        Extract size from product title.
        
        Returns:
            Tuple of (unit_type, size_in_standard_units, original_size_str)
            Example: ('weight', 500, '500g') means 500g = 5 × 100g standard units
        """
        if not title:
            return None
        
        title_lower = title.lower()
        
        for pattern, (unit_type, multiplier) in cls.SIZE_PATTERNS.items():
            match = re.search(pattern, title_lower)
            if match:
                try:
                    size_value = float(match.group(1))
                    size_normalized = size_value * multiplier  # Convert to base unit (g or ml)
                    original_size = match.group(0)
                    
                    # Determine if weight or volume for display
                    if 'weight' in unit_type:
                        return ('weight', size_normalized, original_size, 'g')
                    else:
                        return ('volume', size_normalized, original_size, 'ml')
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @classmethod
    def normalize_price(cls, price: float, size_info: Optional[Tuple[str, float, str, str]]) -> Dict[str, Any]:
        """
        Normalize price to per 100g or 100ml.
        
        Returns dict with:
        - absolute_price: original price
        - normalized_price: price per 100 units
        - size_info: size extracted from title
        - price_per_unit: human-readable price per unit
        """
        if not price or price <= 0:
            return {
                'absolute_price': price,
                'normalized_price': None,
                'size_info': size_info,
                'price_per_unit': 'N/A',
                'has_size_data': False
            }
        
        if not size_info:
            return {
                'absolute_price': price,
                'normalized_price': None,
                'size_info': None,
                'price_per_unit': 'N/A',
                'has_size_data': False
            }
        
        try:
            unit_type, size_value, original_size, unit_symbol = size_info
            
            # Calculate price per 100 units (100g or 100ml)
            normalized_price = (price / size_value) * cls.STANDARD_UNIT_VALUE
            
            # Create human-readable format
            price_per_unit = f"€{normalized_price:.2f}/100{unit_symbol}"
            
            return {
                'absolute_price': price,
                'normalized_price': normalized_price,
                'size_info': size_info,
                'price_per_unit': price_per_unit,
                'has_size_data': True,
                'original_size': original_size
            }
        except (ValueError, ZeroDivisionError):
            logger.warning(f"Failed to normalize price. Price: {price}, Size info: {size_info}")
            return {
                'absolute_price': price,
                'normalized_price': None,
                'size_info': size_info,
                'price_per_unit': 'N/A',
                'has_size_data': False
            }
    
    @classmethod
    def get_comparison_price(cls, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract price and size from product, return comprehensive price info.
        """
        price = product_data.get('price')
        title = product_data.get('title', '')
        
        size_info = cls.extract_size(title)
        normalized = cls.normalize_price(price, size_info)
        
        return normalized


class CompetitorInsights(BaseModel):
    asin: str
    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    rating: Optional[float] = None
    key_points: List[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    summary: str
    positioning: str
    top_competitors: List[CompetitorInsights]
    recommendations: List[str]


class RateLimiter:
    """Rate limiter to prevent API quota exhaustion."""
    
    def __init__(self, calls_per_minute: int = 3):
        """Initialize rate limiter."""
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit exceeded."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old calls outside the window
        self.call_times = [t for t in self.call_times if t > cutoff]
        
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = (self.call_times[0] - cutoff).total_seconds() + 1
            logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            self.call_times = []
        
        self.call_times.append(now)


def validate_api_key() -> bool:
    """Validate OpenAI API key at startup."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not set in environment")
        return False
    if len(api_key) < 20:
        logger.error("❌ OPENAI_API_KEY appears invalid (too short)")
        return False
    logger.info("✅ OpenAI API key validated")
    return True


class CompetitorAnalyzer:
    """Analyzes competitors using LLM with robust error handling."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """Initialize analyzer with LLM config."""
        self.model = model
        self.temperature = temperature
        self.db = MongoDB()
        self.rate_limiter = RateLimiter(calls_per_minute=3)
        self.cache: Dict[str, tuple] = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour
        
        # Validate API key
        if not validate_api_key():
            logger.warning("⚠️ API key validation failed - operations may fail")

    def _get_cache_key(self, asin: str) -> str:
        """Generate cache key for competitor analysis."""
        return f"analysis_{asin}"
    
    def _get_cached_analysis(self, asin: str) -> Optional[str]:
        """Retrieve cached analysis if available and not expired."""
        key = self._get_cache_key(asin)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                logger.info(f"📦 Using cached analysis for {asin}")
                return result
            else:
                del self.cache[key]
        return None
    
    def _cache_analysis(self, asin: str, result: str) -> None:
        """Cache analysis result."""
        key = self._get_cache_key(asin)
        self.cache[key] = (result, datetime.now().timestamp())
    
    def _find_competitors_by_similarity(self, product: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find competitors by category, product type, and price similarity."""
        try:
            all_products = self.db.get_all_products()
            if not all_products:
                return []
            
            target_asin = product.get("asin")
            target_price_info = UnitPriceNormalizer.get_comparison_price(product)
            target_normalized_price = target_price_info.get('normalized_price') or product.get("price", 0)
            target_categories = product.get("categories", [])
            target_domain = product.get("amazon_domain", "com")
            target_type = ProductTypeDetector.get_product_type(product)
            
            logger.info(f"🔍 Searching competitors for {target_asin} (Type: {target_type})")
            
            competitors = []
            
            for p in all_products:
                # Skip self
                if p.get("asin") == target_asin:
                    continue
                
                # Filter by domain (same marketplace)
                if p.get("amazon_domain") != target_domain:
                    continue
                
                # 🔴 PRODUCT TYPE FILTER - Skip if different product type
                competitor_type = ProductTypeDetector.get_product_type(p)
                same_type = ProductTypeDetector.same_product_type(product, p)
                
                if target_type and competitor_type and not same_type:
                    # Both have types but they don't match - skip this competitor
                    continue
                
                # Get normalized price for competitor
                comp_price_info = UnitPriceNormalizer.get_comparison_price(p)
                comp_normalized_price = comp_price_info.get('normalized_price') or p.get("price", 0)
                
                # Calculate similarity score
                score = 0
                
                # Product type match (very high weight - must be same type!)
                if same_type:
                    score += 70
                    logger.debug(f"✅ Same product type: {target_type}")
                
                # Category match (heavy weight)
                if target_categories and p.get("categories"):
                    target_cat_names = extract_category_names(target_categories)
                    p_cat_names = extract_category_names(p.get("categories", []))
                    common_cats = set(target_cat_names) & set(p_cat_names)
                    if common_cats:
                        score += 50
                
                # Price proximity using NORMALIZED prices (moderate weight)
                if target_normalized_price > 0 and comp_normalized_price > 0:
                    price_diff = abs(target_normalized_price - comp_normalized_price) / target_normalized_price
                    if price_diff < 0.5:  # Within 50% of normalized price
                        score += 30
                
                # Brand similarity
                if product.get("brand") and p.get("brand") == product.get("brand"):
                    score += 20
                
                if score > 0:
                    # Store competitor with price info
                    competitors.append({
                        "product": p,
                        "score": score,
                        "price_info": comp_price_info,
                        "product_type": competitor_type
                    })
            
            # Sort by score and return top N
            competitors.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"📊 Found {len(competitors)} potential competitors, returning top {limit}")
            
            formatted = []
            for comp_item in competitors[:limit]:
                c = comp_item["product"]
                price_info = comp_item["price_info"]
                
                formatted.append({
                    "asin": c.get("asin", ""),
                    "title": c.get("title", ""),
                    "price": c.get("price"),
                    "currency": c.get("currency", ""),
                    "rating": c.get("rating"),
                    "amazon_domain": c.get("amazon_domain", ""),
                    "price_info": price_info,  # Add normalized price info
                    "price_per_unit": price_info.get('price_per_unit', 'N/A'),
                    "product_type": comp_item.get("product_type", "unknown")
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Error finding competitors: {e}")
            return []
    

    def analyze_competitors(self, asin: str) -> str:
        """Analyze competitors for a given ASIN with caching and rate limiting."""
        try:
            # Check cache first
            cached = self._get_cached_analysis(asin)
            if cached:
                return cached
            
            # Validate input
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            # Fetch product data
            product = self.db.get_product(asin)
            if not product:
                logger.warning(f"Product not found for ASIN: {asin}")
                return self._fallback_response(asin)

            # Get competitors linked to this product
            competitors = self.db.get_competitors(asin)
            if not competitors:
                logger.warning(f"No competitor data for ASIN: {asin}")
                fallback = self._create_response_string(
                    AnalysisOutput(
                        summary=f"Product {asin} found but lacks competitor data.",
                        positioning="Unable to position without competitors.",
                        top_competitors=[],
                        recommendations=["Scrape competitors for this product before running analysis."]
                    )
                )
                self._cache_analysis(asin, fallback)
                return fallback

            # Rate limit before LLM call
            self.rate_limiter.wait_if_needed()
            
            # Build and invoke LLM chain with fallback
            try:
                result = self._invoke_llm(product, competitors)
            except Exception as e:
                logger.warning(f"Primary LLM failed: {e}. Attempting fallback...")
                result = self._invoke_llm_fallback(product, competitors)

            # Format and cache response
            response = self._create_response_string(result)
            self._cache_analysis(asin, response)
            
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"Error: Invalid input - {e}"
        except Exception as e:
            logger.error(f"Unexpected error analyzing competitors: {e}", exc_info=True)
            return f"Error analyzing competitors: {str(e)}"

    def _invoke_llm(self, product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> AnalysisOutput:
        """Invoke LLM with error handling and optional caching."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import PydanticOutputParser
        except ImportError as e:
            raise ImportError(f"Missing LangChain dependencies: {e}")

        try:
            parser = PydanticOutputParser(pydantic_object=AnalysisOutput)

            template = (
                "You are an expert e-commerce market strategist specializing in Amazon product analysis. "
                "Analyze the target product against its competitors to identify market positioning and opportunities.\n\n"
                
                "=== TARGET PRODUCT ===\n"
                "Title: {product_title}\n"
                "Brand: {brand}\n"
                "Price: {price_display}\n"
                "Rating: {rating}/5.0\n"
                "Categories: {categories}\n"
                "Region: Amazon {amazon_domain}\n\n"
                
                "=== COMPETITOR LANDSCAPE ===\n"
                "{competitors}\n\n"
                
                "=== IMPORTANT: UNIT PRICE ANALYSIS ===\n"
                "⚠️ CRITICAL: When comparing prices, always consider normalized unit prices (e.g., price per 100g, 100ml).\n"
                "A product might have a higher absolute price but lower unit price, making it MORE competitive.\n"
                "Example: 500ml @ €5 (€1/100ml) vs 250ml @ €1.80 (€0.72/100ml) - The first is actually cheaper per unit!\n\n"
                
                "=== ANALYSIS INSTRUCTIONS ===\n"
                "1. SUMMARY (2-3 sentences): Provide a high-level market overview. Compare unit prices, NOT just absolute prices.\n"
                "2. POSITIONING (2-3 sentences): Analyze competitive differentiation on unit price, rating, and value proposition.\n"
                "3. TOP COMPETITORS: List 3-5 most relevant competitors with unit price comparison.\n"
                "4. RECOMMENDATIONS (3-5 actionable items):\n"
                "   - Unit price optimization (size strategy - larger bottles often have lower per-unit cost)\n"
                "   - Quality/rating improvement opportunities\n"
                "   - Market gaps or underserved sizes\n"
                "   - Positioning strategies based on unit value\n"
                "   - Bundle or size variation recommendations\n\n"
                
                "{format_instructions}"
            )

            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "product_title", "brand", "price_display", "rating",
                    "categories", "amazon_domain", "competitors"
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            llm = ChatOpenAI(
                model=self.model, 
                temperature=self.temperature,
                timeout=30,
                max_retries=2
            )
            chain = prompt | llm | parser

            # Get normalized price for target product
            target_price_info = UnitPriceNormalizer.get_comparison_price(product)
            price_display = product.get("currency", "$") + str(product.get("price", "N/A"))
            if target_price_info.get('has_size_data'):
                price_display += f" ({target_price_info['price_per_unit']})"
            
            # Format competitors with unit price info
            competitors_formatted = []
            for comp in competitors:
                comp_str = f"• {comp['asin']} | {comp['title'][:60]} | {comp['currency']}{comp['price']}"
                if comp.get('price_info', {}).get('has_size_data'):
                    comp_str += f" ({comp['price_per_unit']})"
                comp_str += f" | ⭐ {comp['rating']}"
                competitors_formatted.append(comp_str)
            
            competitors_text = "\n".join(competitors_formatted) if competitors_formatted else "No competitors found"

            # Handle categories
            categories = product.get("categories", [])
            category_str = ", ".join(extract_category_names(categories)) if categories else "N/A"

            result = chain.invoke({
                "product_title": product.get("title", "Unknown"),
                "brand": product.get("brand", "N/A"),
                "price_display": price_display,
                "rating": product.get("rating", "N/A"),
                "categories": category_str,
                "amazon_domain": product.get("amazon_domain", "com"),
                "competitors": competitors_text,
            })

            return result

        except ValidationError as e:
            logger.error(f"LLM output validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise
    
    def _invoke_llm_fallback(self, product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> AnalysisOutput:
        """Fallback to gpt-3.5-turbo if primary model fails."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import PydanticOutputParser
        except ImportError as e:
            raise ImportError(f"Missing LangChain dependencies: {e}")

        logger.info("🔄 Falling back to gpt-3.5-turbo (cheaper model)...")
        
        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
        template = (
            "Analyze this Amazon product vs competitors. Provide: summary, positioning, top 3 competitors, "
            "and 3 recommendations. Product: {product_title} ({currency}{price}, rating {rating}). "
            "Competitors: {competitors}\n\n{format_instructions}"
        )

        prompt = PromptTemplate(
            template=template,
            input_variables=["product_title", "currency", "price", "rating", "competitors"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, timeout=30, max_retries=1)
        chain = prompt | llm | parser

        result = chain.invoke({
            "product_title": product.get("title", "Unknown"),
            "currency": product.get("currency", "$"),
            "price": product.get("price", "N/A"),
            "rating": product.get("rating", "N/A"),
            "competitors": str(competitors)[:500]  # Truncate for cheaper model
        })

        return result

    def _create_response_string(self, result: AnalysisOutput) -> str:
        """Format analysis output as readable string."""
        lines = [
            "Summary:\n" + result.summary,
            "\nPositioning:\n" + result.positioning,
            "\nTop Competitors:"
        ]

        for c in result.top_competitors[:5]:
            pts = "; ".join(c.key_points) if c.key_points else "No additional info"
            currency = c.currency or "$"
            price_str = f"{currency} {c.price:.2f}" if c.price else "N/A"
            rating_str = f"{c.rating:.1f}" if c.rating else "N/A"
            lines.append(
                f"  • {c.asin} | {c.title[:40]} | {price_str} | ⭐ {rating_str}"
            )

        if result.recommendations:
            lines.append("\nRecommendations:")
            for rec in result.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def _fallback_response(self, asin: str) -> str:
        """Return fallback response when product/data not found."""
        return (
            f"Unable to analyze competitors for ASIN {asin}.\n"
            "Possible reasons:\n"
            "  • Product not found in database\n"
            "  • No competitors available\n"
            "Please ensure the product is in the database and has competitor data."
        )


# Backward-compatible function for existing code
def analyze_competitors(asin: str) -> str:
    """Analyze competitors (uses CompetitorAnalyzer class)."""
    analyzer = CompetitorAnalyzer()
    return analyzer.analyze_competitors(asin)