import os
import logging
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.db import Database

load_dotenv()

logger = logging.getLogger(__name__)


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


class CompetitorAnalyzer:
    """Analyzes competitors using LLM with robust error handling."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """Initialize analyzer with LLM config."""
        self.model = model
        self.temperature = temperature
        self.db = Database()

    def format_competitors(self, parent_asin: str) -> List[Dict[str, Any]]:
        """Extract competitor data from database with error handling."""
        try:
            comps = self.db.search_products({"parent_asin": parent_asin})
            if not comps:
                logger.warning(f"No competitors found for {parent_asin}")
                return []

            formatted = []
            for c in comps:
                try:
                    formatted.append({
                        "asin": c.get("asin", ""),
                        "title": c.get("title", ""),
                        "price": c.get("price"),
                        "currency": c.get("currency", ""),
                        "rating": c.get("rating"),
                        "amazon_domain": c.get("amazon_domain", "")
                    })
                except Exception as e:
                    logger.error(f"Error formatting competitor {c.get('asin')}: {e}")
                    continue
            return formatted
        except Exception as e:
            logger.error(f"Error fetching competitors: {e}")
            return []

    def analyze_competitors(self, asin: str) -> str:
        """Analyze competitors for a given ASIN."""
        try:
            # Validate input
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            # Fetch product data
            product = self.db.get_product(asin)
            if not product:
                logger.warning(f"Product not found for ASIN: {asin}")
                return self._fallback_response(asin)

            # Get competitors
            competitors = self.format_competitors(asin)
            if not competitors:
                logger.warning(f"No competitor data for ASIN: {asin}")
                return self._create_response_string(
                    AnalysisOutput(
                        summary=f"Product {asin} found but lacks competitor data.",
                        positioning="Unable to position without competitors.",
                        top_competitors=[],
                        recommendations=["Collect more competitor data."]
                    )
                )

            # Build and invoke LLM chain
            result = self._invoke_llm(product, competitors)

            # Format and return response
            return self._create_response_string(result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"Error: Invalid input - {e}"
        except Exception as e:
            logger.error(f"Unexpected error analyzing competitors: {e}", exc_info=True)
            return f"Error analyzing competitors: {str(e)}"

    def _invoke_llm(self, product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> AnalysisOutput:
        """Invoke LLM with error handling."""
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
                "Price: {currency} {price}\n"
                "Rating: {rating}/5.0\n"
                "Categories: {categories}\n"
                "Region: Amazon {amazon_domain}\n\n"
                
                "=== COMPETITOR LANDSCAPE ===\n"
                "{competitors}\n\n"
                
                "=== ANALYSIS INSTRUCTIONS ===\n"
                "1. SUMMARY (2-3 sentences): Provide a high-level market overview including market size, trend, and the target product's position.\n"
                "2. POSITIONING (2-3 sentences): Analyze competitive differentiation. Consider price positioning (premium/mid-tier/budget), "
                "quality indicators (ratings), and unique value propositions.\n"
                "3. TOP COMPETITORS: List 3-5 most relevant competitors with key competitive advantages (e.g., lower price, higher rating, brand recognition).\n"
                "4. RECOMMENDATIONS (3-5 actionable items):\n"
                "   - Price optimization strategies (if applicable)\n"
                "   - Quality/rating improvement opportunities\n"
                "   - Market gaps or underserved niches\n"
                "   - Positioning strategies to stand out\n"
                "   - Customer satisfaction or feature enhancements\n\n"
                
                "=== CRITICAL NOTES ===\n"
                "• Always display prices with their correct currency symbol (EUR €, GBP £, JPY ¥, etc.)\n"
                "• When comparing, normalize to same currency context\n"
                "• Focus on actionable insights, not generic statements\n"
                "• Prioritize competitors' differentiation factors (price, quality, reviews)\n"
                "• If ratings are similar, emphasize other competitive factors\n\n"
                "{format_instructions}"
            )

            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "product_title", "brand", "price", "currency", "rating",
                    "categories", "amazon_domain", "competitors"
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            llm = ChatOpenAI(model=self.model, temperature=self.temperature)
            chain = prompt | llm | parser

            # Handle categories - they can be dicts or strings
            categories = product.get("categories", [])
            if categories:
                if isinstance(categories[0], dict):
                    category_str = ", ".join([cat.get("name", str(cat)) for cat in categories])
                else:
                    category_str = ", ".join(categories)
            else:
                category_str = "N/A"

            result = chain.invoke({
                "product_title": product.get("title", "Unknown"),
                "brand": product.get("brand", "N/A"),
                "price": product.get("price", "N/A"),
                "currency": product.get("currency", "$"),
                "rating": product.get("rating", "N/A"),
                "categories": category_str,
                "amazon_domain": product.get("amazon_domain", "com"),
                "competitors": competitors,
            })

            return result

        except ValidationError as e:
            logger.error(f"LLM output validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _create_response_string(self, result: AnalysisOutput) -> str:
        """Format analysis output as readable string."""
        lines = [
            "Summary:\n" + result.summary,
            "\nPositioning:\n" + result.positioning,
            "\nTop Competitors:"
        ]

        for c in result.top_competitors[:5]:
            pts = "; ".join(c.key_points) if c.key_points else "N/A"
            currency = c.currency or "$"
            price_str = f"{currency} {c.price:.2f}" if c.price else "N/A"
            rating_str = f"{c.rating:.1f}" if c.rating else "N/A"
            lines.append(
                f"  • {c.asin} | {c.title} | {price_str} | ⭐ {rating_str} | {pts}"
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