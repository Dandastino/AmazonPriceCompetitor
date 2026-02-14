import logging
from typing import Tuple

import streamlit as st

from src.db import Database
from src.llm import analyze_competitors
from src.services import ProductService
from src.chatbot import ProductChatbot

logger = logging.getLogger(__name__)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Amazon Competitor Analysis",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .competitor-card {
            background-color: #ffffff;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 0.25rem;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 0.25rem;
            border-left: 4px solid #28a745;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 0.25rem;
            border-left: 4px solid #dc3545;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render page header with title and description."""
    st.markdown(
        """
        # 📊 Amazon Competitor Analysis
        #### Analyze your Amazon products and monitor the competitive landscape
        """
    )


def render_input_section() -> Tuple[str, str, str]:
    """Render input form for product and search parameters."""
    col1, col2, col3 = st.columns(3)

    with col1:
        asin = st.text_input(
            "🔍 Product ASIN",
            placeholder="e.g., B0CX23VSAS",
            help="Enter the Amazon ASIN of the product",
        ).strip()

    with col2:
        geo = st.text_input(
            "📍 Region/Zip Code",
            placeholder="e.g., us, de, 83980",
            help="Enter region code or postal code",
        ).strip()

    with col3:
        domain = st.selectbox(
            "🌐 Amazon Domain",
            ["com", "ca", "uk", "de", "fr", "it", "es", "br", "jp", "in"],
            help="Select the Amazon marketplace domain",
        )

    return asin, geo, domain


def render_product_card(product: dict, db: Database, index: int = 0):
    """Render a single product card with details and action buttons."""
    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])

        # Product image
        with col_img:
            try:
                images = product.get("images", [])
                if images and len(images) > 0:
                    st.image(images[0], width=250)
                else:
                    st.info("📷 No image available")
            except Exception as e:
                st.warning(f"Error loading image: {str(e)[:50]}")

        # Product information
        with col_info:
            # Title
            title = product.get("title", product.get("asin", "Unknown"))
            st.subheader(title[:80] + "..." if len(title) > 80 else title)

            # Metrics - adjusted column ratios for better spacing
            metric_cols = st.columns([1.2, 1, 1.5, 1.5])

            with metric_cols[0]:
                currency = product.get("currency", "$")
                price = product.get("price", "N/A")
                if isinstance(price, (int, float)):
                    price_str = f"{currency} {price:.2f}"
                else:
                    price_str = "N/A"
                st.markdown("**💰 Price**")
                st.markdown(f"<span style='font-size: 1.5em;'>{price_str}</span>", unsafe_allow_html=True)

            with metric_cols[1]:
                rating = product.get("rating", "N/A")
                if isinstance(rating, (int, float)):
                    rating_str = f"{rating:.1f}⭐"
                else:
                    rating_str = "N/A"
                st.markdown("**⭐ Rating**")
                st.markdown(f"<span style='font-size: 1.5em;'>{rating_str}</span>", unsafe_allow_html=True)

            with metric_cols[2]:
                brand = product.get("brand", "-")
                brand_str = brand[:25] if isinstance(brand, str) else "-"
                st.markdown("**🏢 Brand**")
                st.markdown(f"<span style='font-size: 1.5em;'>{brand_str}</span>", unsafe_allow_html=True)

            with metric_cols[3]:
                stock = product.get("stock", "Unknown")
                stock_str = str(stock)[:25]
                st.markdown("**📦 Stock**")
                st.markdown(f"<span style='font-size: 1.5em;'>{stock_str}</span>", unsafe_allow_html=True)

            # Additional info
            col_details = st.columns([1, 1])
            with col_details[0]:
                domain_info = f"amazon.{product.get('amazon_domain', 'com')}"
                st.caption(f"🌐 Domain: {domain_info}")

            with col_details[1]:
                geo_info = product.get("amazon_geo_location", "-")
                st.caption(f"📍 Geo: {geo_info}")

            # Category info
            categories = product.get("categories", [])
            if categories:
                cat_names = []
                
                first_cat = categories[0]
                if isinstance(first_cat, dict) and "ladder" in first_cat:
                    ladder_list = first_cat.get("ladder", [])
                    for item in ladder_list[:5]: 
                        name = item.get("name", "")
                        if name:
                            parts = [p.strip() for p in name.split('&')]
                            cat_names.extend(parts)
                else:
                    for cat in categories[:3]:
                        name = cat.get("name", str(cat)) if isinstance(cat, dict) else str(cat)
                        cat_names.append(name)

                if cat_names:
                    unique_cats = list(dict.fromkeys(cat_names))[:3]
                    cats_str = " • ".join(unique_cats)
                    st.caption(f"📂 Category: {cats_str}")
                    
                # Delete button
                if st.button(
                    "🗑️ Delete Product",
                    key=f"delete_{product['asin']}_{index}",
                    use_container_width=True,
                    type="secondary",
                ):
                    if db.delete_product(product["asin"]):
                        st.success(f"✅ Product {product['asin']} deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete product")


def render_products_section(db: Database):
    """Render all products with pagination (10 per page)."""
    products = db.get_all_products()

    if not products:
        st.info("👋 No products stored yet. Scrape a product to get started!")
        return

    st.divider()
    st.subheader(f"📦 Your Products ({len(products)})")

    # Pagination setup
    items_per_page = 10
    total_pages = (len(products) + items_per_page - 1) // items_per_page

    # Initialize page in session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    # Browser-style pagination controls
    col_left, col_right = st.columns([3, 1])
    with col_right:
        pagination_cols = st.columns([0.8, 0.8, 1.5])
        
        with pagination_cols[0]:
            if st.button("⬅️", use_container_width=True, disabled=(st.session_state["current_page"] == 1)):
                st.session_state["current_page"] = max(1, st.session_state["current_page"] - 1)
                st.rerun()
        
        with pagination_cols[1]:
            if st.button("➡️", use_container_width=True, disabled=(st.session_state["current_page"] == total_pages)):
                st.session_state["current_page"] = min(total_pages, st.session_state["current_page"] + 1)
                st.rerun()
        
        with pagination_cols[2]:
            st.markdown(f"<p style='text-align: center; margin-top: 0.5rem;'><b>{st.session_state['current_page']} / {total_pages}</b></p>", unsafe_allow_html=True)

    page = st.session_state["current_page"]

    # Calculate products for current page
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(products))

    st.caption(f"Showing products {start_idx + 1}–{end_idx} of {len(products)}")
    st.divider()

    # Display products for current page
    for idx in range(start_idx, end_idx):
        render_product_card(products[idx], db, idx)
    
    # Delete all data button at the bottom
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Delete All Data", use_container_width=True, type="secondary"):
            if st.session_state.get("delete_confirmed") is None:
                st.session_state["delete_confirmed"] = False
            
            if not st.session_state["delete_confirmed"]:
                st.session_state["delete_confirmed"] = True
                st.rerun()
    
    if st.session_state.get("delete_confirmed"):
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ Confirm Delete All", use_container_width=True, type="primary"):
                if db.delete_all_products():
                    st.success("✅ All data deleted successfully!")
                    st.session_state["delete_confirmed"] = None
                    st.session_state["selected_asin"] = None
                    st.rerun()
                else:
                    st.error("Failed to delete data")
        
        with col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["delete_confirmed"] = None
                st.rerun()


def render_product_analysis_section(db: Database):
    """Analyze all products and recommend the best one."""
    products = db.get_all_products()

    if not products:
        st.info("👋 No products to analyze yet. Scrape products first!")
        return

    st.divider()
    st.subheader("🏆 Best Product Comparison")

    if st.button("🚀 Analyze My Products", type="primary", use_container_width=True):
        with st.spinner("Analyzing your products..."):
            # Score products
            scored_products = []
            
            for product in products:
                score = 0
                details = []
                
                # Rating score (higher is better, max 5)
                rating = product.get("rating", 0)
                if isinstance(rating, (int, float)):
                    rating_score = (rating / 5) * 40  # 40 points
                    score += rating_score
                    details.append(f"Rating: ⭐ {rating:.1f}/5.0 ({rating_score:.0f} pts)")
                
                # Price score (lower is better)
                price = product.get("price", 0)
                if isinstance(price, (int, float)) and price > 0:
                    # Normalize price score (cheaper = higher score)
                    max_price = max([p.get("price", 0) for p in products if isinstance(p.get("price"), (int, float)) and p.get("price") > 0] or [1])
                    price_score = ((max_price - price) / max_price) * 30 if max_price > 0 else 0
                    score += price_score
                    currency = product.get("currency", "$")
                    details.append(f"Price: {currency} {price:.2f} ({price_score:.0f} pts)")
                
                # Brand quality (stock availability)
                stock = product.get("stock", "Unknown")
                stock_score = 0
                if isinstance(stock, str):
                    if stock.lower() in ["in stock", "available"]:
                        stock_score = 20
                        stock_status = "In Stock"
                    elif stock.lower() in ["out of stock"]:
                        stock_score = 5
                        stock_status = "Out of Stock"
                    else:
                        stock_score = 15
                        stock_status = stock
                else:
                    stock_score = 15
                    stock_status = str(stock)
                
                score += stock_score
                details.append(f"Stock: {stock_status} ({stock_score:.0f} pts)")
                
                scored_products.append({
                    "product": product,
                    "score": score,
                    "details": details
                })
            
            # Sort by score
            scored_products.sort(key=lambda x: x["score"], reverse=True)
            best_product = scored_products[0]
            
            # Display best product
            st.success(f"🏆 **Best Product: {best_product['product'].get('title', 'Unknown')[:60]}**")
            
            with st.container(border=True):
                col_best, col_metrics = st.columns([2, 1])
                
                with col_best:
                    st.subheader("Winner Details")
                    st.write(f"**ASIN:** {best_product['product'].get('asin')}")
                    st.write(f"**Brand:** {best_product['product'].get('brand', 'N/A')}")
                    st.write(f"**Domain:** amazon.{best_product['product'].get('amazon_domain', 'com')}")
                
                with col_metrics:
                    st.subheader("Score Breakdown")
                    for detail in best_product['details']:
                        st.write(detail)
                    st.metric("Total Score", f"{best_product['score']:.0f} pts")
            
            # Show all products ranked
            if len(scored_products) > 1:
                st.subheader("📊 Product Rankings")
                
                for idx, item in enumerate(scored_products[:10], 1):
                    with st.container(border=True):
                        col_rank, col_info, col_score = st.columns([0.5, 2, 1])
                        
                        with col_rank:
                            if idx == 1:
                                st.markdown("### 🥇")
                            elif idx == 2:
                                st.markdown("### 🥈")
                            elif idx == 3:
                                st.markdown("### 🥉")
                            else:
                                st.markdown(f"### #{idx}")
                        
                        with col_info:
                            title = item['product'].get('title', 'Unknown')[:50]
                            st.write(f"**{title}**")
                            for detail in item['details']:
                                st.caption(detail)
                        
                        with col_score:
                            st.metric("Score", f"{item['score']:.0f}")



def render_chatbot_section():
    """Render chatbot interface for product questions."""
    # Custom styling
    st.markdown("""
        <style>
        .bot-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .suggestion-btn {
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 20px;
            padding: 10px 15px;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Bot header
    st.markdown("""
        <div class="bot-header">
            <h2>🤖 Meet Alex - Your Shopping Assistant</h2>
            <p>Ask me anything about your products, pricing, competitors, or how to use this tool!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot in session state
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = ProductChatbot()
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Chat messages container
    st.divider()
    
    # Welcome message if no history
    if not st.session_state["chat_history"]:
        with st.chat_message("assistant", avatar="🤖"):
            st.write("""
            **Hello! I'm Alex, your shopping assistant!** 👋
            
            I can help you:
            - 📊 Analyze your product database
            - 💰 Compare prices and ratings
            - 🎯 Find your best performing products
            - 📚 Guide you on how to use this tool
            
            Just ask me anything!
            """)
    else:
        # Display chat history
        for human_msg, ai_msg in st.session_state["chat_history"]:
            with st.chat_message("user", avatar="👤"):
                st.write(human_msg)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(ai_msg)
    
    # Chat input - always at bottom
    if prompt := st.chat_input("Ask Alex anything... (e.g., What's my best product?)"):
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Alex is thinking..."):
                try:
                    response = st.session_state["chatbot"].chat(
                        prompt, 
                        st.session_state["chat_history"]
                    )
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state["chat_history"].append((prompt, response))
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chatbot error: {e}", exc_info=True)
    
    # Controls at the bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()
    
    with col2:
        if st.button("❓ Help & Tips", use_container_width=True, type="secondary"):
            st.session_state["show_help"] = not st.session_state.get("show_help", False)
    
    # Show help section
    if st.session_state.get("show_help", False):
        st.info(
            """
            ### 📚 How to Use Alex
            
            **Product Questions:**
            - "What products do I have in my database?"
            - "Which product has the highest rating?"
            - "Compare the prices of [brand name]"
            - "Tell me about ASIN [ASIN]"
            
            **Analysis Questions:**
            - "What's my best product?"
            - "Which product should I focus on?"
            
            **Tool Help:**
            - "How do I scrape a product?"
            - "What domains are supported?"
            """
        )


def main():
    """Main application entry point."""
    configure_page()
    render_header()

    # Initialize services
    db = Database()
    service = ProductService()

    # Sidebar for scraping
    with st.sidebar:
        st.header("📥 Scrape Product")
        asin, geo, domain = render_input_section()

        if st.button("🔍 Scrape Product", use_container_width=True, type="primary"):
            if not asin:
                st.error("Please enter an ASIN")
            else:
                with st.spinner("Scraping product..."):
                    try:
                        result = service.scrape_and_store_product(asin, geo, domain)
                        if result:
                            st.success("✅ Product scraped successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to scrape product")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Scraping error: {e}", exc_info=True)

        st.divider()
        st.info(
            "💡 **Tips:**\n"
            "- Enter a valid Amazon ASIN\n"
            "- Region affects pricing & availability\n"
            "- Domain selects the Amazon marketplace"
        )

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📦 Products", "🏆 Best Product", "💬 Chatbot"])
    
    with tab1:
        render_products_section(db)
    
    with tab2:
        render_product_analysis_section(db)
    
    with tab3:
        render_chatbot_section()

    # Footer
    st.divider()
    st.caption(
        "Amazon Competitor Analysis Tool | "
        "Powered by Oxylabs & OpenAI | "
        "[GitHub](https://github.com)"
    )


if __name__ == "__main__":
    main()