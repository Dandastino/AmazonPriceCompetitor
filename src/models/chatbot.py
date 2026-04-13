import logging
import time
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.core import MongoDB
from src.utils import format_price, format_rating, get_logger

logger = get_logger(__name__)


class ProductChatbot:
    """Chatbot for answering questions about Amazon products using RAG."""

    _context_cache: Optional[str] = None
    _context_cache_ts: float = 0.0
    _CONTEXT_TTL: float = 300.0  # seconds (5 minutes)

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize chatbot with LLM and database."""
        self.model = model
        self.temperature = temperature
        self.db = MongoDB()
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

    def get_product_context(self, query: str) -> str:
        """Get relevant product context from database (cached for 5 minutes)."""
        now = time.monotonic()
        if ProductChatbot._context_cache is not None and (now - ProductChatbot._context_cache_ts) < ProductChatbot._CONTEXT_TTL:
            logger.debug("Using cached product context")
            return ProductChatbot._context_cache

        try:
            products = self.db.get_all_products()
            
            if not products:
                return "No products found in the database."
            
            # Remove duplicates by ASIN
            seen_asins = set()
            unique_products = []
            for prod in products:
                asin = prod.get("asin")
                if asin and asin not in seen_asins:
                    seen_asins.add(asin)
                    unique_products.append(prod)
            
            # Format product data for context (show ALL unique products)
            context_parts = [f"TOTAL PRODUCTS IN DATABASE: {len(unique_products)}\n"]
            for prod in unique_products:
                asin = prod.get("asin", "Unknown")
                title = prod.get("title", "Unknown")
                brand = prod.get("brand", "N/A")
                price = prod.get("price", "N/A")
                currency = prod.get("currency", "$")
                rating = prod.get("rating", "N/A")
                stock = prod.get("stock", "Unknown")
                domain = prod.get("amazon_domain", "com")

                price_str = format_price(price, currency=currency, min_value=None)
                rating_str = format_rating(rating, style="fraction")
                
                context_parts.append(
                    f"ASIN: {asin} | Title: {title[:50]} | Brand: {brand} | "
                    f"Price: {price_str} | Rating: {rating_str} | Stock: {stock} | Domain: amazon.{domain}"
                )

            result = "\n".join(context_parts)
            ProductChatbot._context_cache = result
            ProductChatbot._context_cache_ts = time.monotonic()
            return result

        except Exception as e:
            logger.error(f"Error getting product context: {e}")
            return "Error retrieving product data."

    def get_competitor_context(self, parent_asin: str) -> str:
        """Get competitor information for a specific product."""
        try:
            competitors = self.db.get_competitors(parent_asin)
            
            if not competitors:
                return f"No competitors found for ASIN {parent_asin}."
            
            context_parts = [f"Competitors for {parent_asin}:"]
            for comp in competitors[:5]:  # Top 5 competitors
                asin = comp.get("asin", "Unknown")
                title = comp.get("title", "Unknown")
                price = comp.get("price", "N/A")
                currency = comp.get("currency", "$")
                rating = comp.get("rating", "N/A")

                price_str = format_price(price, currency=currency, min_value=None)
                rating_str = format_rating(rating, style="fraction")
                
                context_parts.append(
                    f"- {title[:60]}...\n"
                    f"  ASIN: {asin} | Price: {price_str} | Rating: {rating_str}"
                )
            
            return "\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Error getting competitor context: {e}")
            return "Error retrieving competitor data."

    def chat(self, user_message: str, chat_history: List[tuple] = None) -> str:
        """Process user message and return chatbot response."""
        try:
            product_context = self.get_product_context(user_message)
            
            messages = []
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=human_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            system_prompt = (
                "You are Alex, a helpful AI assistant for the Amazon Competitor Analysis Tool.\n\n"
                "You have FULL access to the user's product database. ALL their products and data are listed below:\n\n"
                "=== USER'S PRODUCT DATABASE ===\n"
                f"{product_context}\n\n"
                "=== TOOL STRUCTURE & NAVIGATION ===\n"
                "The tool has 4 main tabs:\n"
                "1. All Products: View all scraped products, browse inventory, delete products.\n"
                "2. Competitors: Select a product and view its competitors ranked by score. Shows your product vs top competitors and a full sorted list.\n"
                "3. Analysis: In-depth analysis tools with two modes (radio selector at the top):\n"
                "   • Review Analysis & Sentiment: Analyze customer reviews — sentiment breakdown, top positive/negative aspects, narrative report.\n"
                "   • AI Competitor Insights: Chat-based AI analysis of competitor strategies and market positioning. Supports follow-up questions.\n"
                "4. AI Assistant: This tab — ask questions about your products, competitors, pricing, and get intelligent answers.\n\n"
                "=== SIDEBAR ===\n"
                "The sidebar ('Scrape Product') is how new products are imported:\n"
                "• Enter the ASIN (Amazon product ID, 10 characters — found in the product URL after /dp/).\n"
                "  Example: s → ASIN = B0CX23VSAS\n"
                "• Select the Domain (Amazon marketplace: com, it, de, fr, uk, es, ca, jp, br, in).\n"
                "• Click 'Scrape Product' to import.\n\n"
                "=== HOW TO USE THE TOOL ===\n"
                "Step 1 — SCRAPE: Use the sidebar to enter an ASIN + Domain and click 'Scrape Product'.\n"
                "Step 2 — VIEW: Go to 'All Products' to browse your imported products.\n"
                "Step 3 — FIND COMPETITORS: On any product card, click '🔎 Find Competitors' to load competing products.\n"
                "Step 4 — COMPARE: Go to 'Competitors' tab, select a product, and review the ranked competitor list.\n"
                "Step 5 — ANALYZE: Go to 'Analysis' tab and choose:\n"
                "   • 'Review Analysis & Sentiment' to study customer feedback and sentiment.\n"
                "   • 'AI Competitor Insights' to start an AI chat analysis of the competitive landscape.\n\n"
                "=== YOUR CAPABILITIES ===\n"
                "• Answer questions about any product in the database — price, rating, brand, stock, ASIN.\n"
                "• Compare products and identify best performers.\n"
                "• Explain competitor scores, rankings, and price differences.\n"
                "• Discuss review sentiment, customer pain points, and improvement opportunities.\n"
                "• Suggest pricing and positioning strategies.\n"
                "• Guide the user through any feature or workflow in the tool.\n\n"
                "=== GUIDELINES ===\n"
                "• Always reference specific data (ASIN, price, rating, brand) from the database when relevant.\n"
                "• When explaining features, name the exact tab or button to use.\n"
                "• Be concise, friendly, and actionable.\n"
                "• Use markdown for readability.\n\n"
                "Now answer the user's question helpfully, referencing the tool structure and product data when relevant."
            )
            
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                *[(msg.type, msg.content) for msg in messages],
                ("human", "{input}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({"input": user_message})
            
            return response.content
        
        except Exception as e:
            logger.error(f"Error in chatbot: {e}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

    def get_product_by_asin(self, asin: str) -> Optional[Dict[str, Any]]:
        """Get specific product details."""
        return self.db.get_product(asin)

    def search_products_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search products by keyword in title or brand."""
        try:
            all_products = self.db.get_all_products()
            keyword_lower = keyword.lower()
            
            matches = [
                p for p in all_products
                if keyword_lower in p.get("title", "").lower()
                or keyword_lower in p.get("brand", "").lower()
            ]
            
            return matches
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
