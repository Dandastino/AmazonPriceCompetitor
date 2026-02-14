import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.db import Database

logger = logging.getLogger(__name__)


class ProductChatbot:
    """Chatbot for answering questions about Amazon products using RAG."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize chatbot with LLM and database."""
        self.model = model
        self.temperature = temperature
        self.db = Database()
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

    def get_product_context(self, query: str) -> str:
        """Get relevant product context from database."""
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
                
                price_str = f"{currency} {price:.2f}" if isinstance(price, (int, float)) else "N/A"
                rating_str = f"{rating:.1f}/5.0" if isinstance(rating, (int, float)) else "N/A"
                
                context_parts.append(
                    f"ASIN: {asin} | Title: {title[:50]} | Brand: {brand} | "
                    f"Price: {price_str} | Rating: {rating_str} | Stock: {stock} | Domain: amazon.{domain}"
                )
            
            return "\n".join(context_parts)
        
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
                
                price_str = f"{currency} {price:.2f}" if isinstance(price, (int, float)) else "N/A"
                rating_str = f"{rating:.1f}/5.0" if isinstance(rating, (int, float)) else "N/A"
                
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
                "You are Alex, a helpful shopping assistant for the Amazon Competitor Analysis Tool.\n\n"
                "You have FULL access to the user's product database. ALL their products and data are listed below:\n\n"
                "=== USER'S PRODUCT DATABASE ===\n"
                f"{product_context}\n\n"
                "=== YOUR CAPABILITIES ===\n"
                "1. PRODUCT ANALYSIS:\n"
                "   • Answer questions about ANY product in their database\n"
                "   • Compare products by price, rating, brand, stock status\n"
                "   • Identify the best performing products\n"
                "   • Show product statistics and insights\n"
                "   • Filter or sort products by various criteria\n\n"
                "2. MARKET INSIGHTS:\n"
                "   • Discuss pricing trends across their products\n"
                "   • Identify stock availability issues\n"
                "   • Analyze rating patterns\n"
                "   • Suggest product recommendations\n\n"
                "3. TOOL GUIDANCE:\n"
                "   • Explain how to scrape new products (use sidebar)\n"
                "   • Guide on best product analysis\n"
                "   • Explain database management\n"
                "   • Help with navigation and features\n\n"
                "=== SCRAPING INPUTS (IMPORTANT) ===\n"
                "When the user asks how to scrape, explain these required inputs:\n"
                "• ASIN: Amazon product identifier (10 characters). Find it on the product page URL or details.\n"
                "  Example URL: https://www.amazon.com/dp/B0CX23VSAS -> ASIN = B0CX23VSAS\n"
                "• Geo/Region: Use a postal code (recommended) or leave it blank.\n"
                "  Avoid country names like 'Germany' and avoid using just 'de' as geo_location.\n"
                "• Domain: Amazon marketplace domain (com, uk, de, fr, it, es, ca, jp, br, in).\n\n"
                "If they need help finding this data, suggest:\n"
                "• Open the product page and copy the ASIN from the URL or product details.\n"
                "• If you only know the country, set Geo empty and select the matching domain (e.g., domain 'de').\n"
                "• If unsure, use Geo empty or a postal code, and Domain='com' as a default.\n\n"
                "=== IMPORTANT GUIDELINES ===\n"
                "• ALWAYS reference specific data from their product database\n"
                "• Be specific with ASIN, prices, ratings, and brands\n"
                "• If they ask about a product, check if it exists in their database\n"
                "• Provide actionable insights and clear recommendations\n"
                "• Use markdown formatting for readability\n"
                "• Be conversational and helpful\n\n"
                "Now answer the user's question using the product data provided above."
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
