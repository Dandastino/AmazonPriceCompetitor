"""UI rendering components for the Amazon Competitor Analyzer."""

import logging
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import streamlit as st

from src.utils import extract_category_names, format_price, format_rating

logger = logging.getLogger(__name__)


class UIRenderer:
    """Handles all Streamlit UI rendering logic."""
    
    @staticmethod
    def configure_page() -> None:
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
            [data-testid="column"] > div > label {
                margin-bottom: 0.5rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_header() -> None:
        """Render page header with title and description."""
        st.markdown(
            """
            # 📊 Amazon Competitor Analysis
            #### Analyze your Amazon products and monitor the competitive landscape
            """
        )

    @staticmethod
    def render_input_section() -> Tuple[str, str, str]:
        """Render input form for product and search parameters."""
        col1, col2, col3 = st.columns([1.5, 1, 1], gap="medium")

        with col1:
            st.write("**🆔 ASIN**")
            asin = st.text_input(
                "ASIN",
                placeholder="B0CX23VSAS",
                help="Enter the Amazon ASIN of the product",
                label_visibility="collapsed",
                key="asin_input"
            ).strip()

        with col2:
            st.write("**📍 Zip Code**")
            geo = st.text_input(
                "Zip Code",
                placeholder="83980",
                help="Enter postal code",
                label_visibility="collapsed",
                key="zip_input"
            ).strip()

        with col3:
            st.write("**🌐 Domain**")
            domain = st.selectbox(
                "Domain",
                ["com", "ca", "mx", "br", "uk", "de", "fr", "it", "es", "nl", "be", "se", "pl", "tr", "jp", "in", "sg", "ae", "sa", "eg", "com.au", "co.za"],
                help="Select the Amazon marketplace domain",
                label_visibility="collapsed",
                key="domain_input"
            )

        return asin, geo, domain

    @staticmethod
    def create_product_options(products: List[Dict]) -> List[str]:
        """Create standardized product options for dropdowns."""
        return [f"{p.get('title', 'Unknown')[:60]} ({p.get('asin')})" for p in products]

    @staticmethod
    def get_selected_product(products: List[Dict], selected_option: str, product_options: List[str]) -> Dict:
        """Get product dict from selected dropdown option."""
        selected_index = product_options.index(selected_option)
        return products[selected_index]

    @staticmethod
    def render_competitor_card(competitor: Dict[str, Any], index: int = 0, score: Optional[float] = None) -> None:
        """Render a competitor card (no action buttons, just info)."""
        with st.container(border=True):
            col_img, col_info = st.columns([1, 3])

            # Product image
            with col_img:
                try:
                    images = competitor.get("images", [])
                    if images and len(images) > 0:
                        st.image(images[0], width=250)
                    else:
                        st.info("📷 No image available")
                except Exception as e:
                    st.warning(f"Error loading image: {str(e)[:50]}")

            # Product information
            with col_info:
                title = competitor.get("title", competitor.get("asin", "Unknown"))
                st.subheader(title[:80] + "..." if len(title) > 80 else title)

                metric_cols = st.columns([1.2, 1, 1.5, 1.5])

                with metric_cols[0]:
                    currency = competitor.get("currency", "$")
                    price = competitor.get("price", "N/A")
                    price_str = format_price(price, currency=currency, min_value=None)
                    st.markdown("**💰 Price**")
                    st.markdown(f"<span style='font-size: 1.5em;'>{price_str}</span>", unsafe_allow_html=True)

                with metric_cols[1]:
                    rating = competitor.get("rating", "N/A")
                    rating_str = format_rating(rating, style="star_suffix")
                    st.markdown("**⭐ Rating**")
                    st.markdown(f"<span style='font-size: 1.5em;'>{rating_str}</span>", unsafe_allow_html=True)

                with metric_cols[2]:
                    brand = competitor.get("brand", "-")
                    brand_str = brand[:25] if isinstance(brand, str) else "-"
                    st.markdown("**🏢 Brand**")
                    st.markdown(f"<span style='font-size: 1.5em;'>{brand_str}</span>", unsafe_allow_html=True)

                with metric_cols[3]:
                    stock = competitor.get("stock", "Unknown")
                    stock_str = str(stock)[:25]
                    st.markdown("**📦 Stock**")
                    st.markdown(f"<span style='font-size: 1.5em;'>{stock_str}</span>", unsafe_allow_html=True)

                st.caption(f"🔗 ASIN: {competitor.get('asin', 'N/A')}")

                if score is not None:
                    st.caption(f"🏅 Score: {score:.0f}/110")

    @staticmethod
    def render_product_card(product: Dict[str, Any], db, service, index: int = 0) -> None:
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
                title = product.get("title", product.get("asin", "Unknown"))
                st.subheader(title[:80] + "..." if len(title) > 80 else title)

                metric_cols = st.columns([1.2, 1, 1.5, 1.5])

                with metric_cols[0]:
                    currency = product.get("currency", "$")
                    price = product.get("price", "N/A")
                    price_str = format_price(price, currency=currency, min_value=None)
                    st.markdown("**💰 Price**")
                    st.markdown(f"<span style='font-size: 1.5em;'>{price_str}</span>", unsafe_allow_html=True)

                with metric_cols[1]:
                    rating = product.get("rating", "N/A")
                    rating_str = format_rating(rating, style="star_suffix")
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
                    cat_names = extract_category_names(
                        categories,
                        split_on_ampersand=True,
                        max_items=3,
                    )
                    if cat_names:
                        cats_str = " • ".join(cat_names)
                        st.caption(f"📂 Category: {cats_str}")

                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button(
                        "🔎 Find Competitors",
                        key=f"find_comp_{product['asin']}_{index}",
                        width='stretch',
                        type="primary",
                    ):
                        UIRenderer._handle_find_competitors(product, service, db)

                with col_btn2:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{product['asin']}_{index}",
                        width='stretch',
                        type="secondary",
                    ):
                        UIRenderer._handle_delete_product(product, db)

    @staticmethod
    def _handle_find_competitors(product: Dict, service, db) -> None:
        """Handle finding competitors for a product."""
        with st.spinner("Finding competitors..."):
            try:
                domain = product.get("amazon_domain", "com")
                geo = product.get("amazon_geo_location", "")
                results = service.fetch_and_store_competitors(
                    product["asin"], domain, geo, pages=2
                )
                if results:
                    st.success(f"✅ Stored {len(results)} competitors")
                    st.rerun()
                else:
                    st.warning("No competitors found")
            except Exception as e:
                st.error(f"Error finding competitors: {str(e)}")
                logger.error(f"Find competitors error: {e}", exc_info=True)

    @staticmethod
    def _handle_delete_product(product: Dict, db) -> None:
        """Handle deleting a product."""
        if db.delete_product(product["asin"]):
            st.success(f"✅ Product {product['asin']} deleted")
            st.rerun()
        else:
            st.error("Failed to delete product")

    @staticmethod
    def render_pagination(total_items: int, items_per_page: int = 10, key_prefix: str = "") -> int:
        """Render pagination controls and return current page."""
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        if f"page_{key_prefix}" not in st.session_state:
            st.session_state[f"page_{key_prefix}"] = 1

        col_left, col_right = st.columns([3, 1])
        with col_right:
            pagination_cols = st.columns([0.8, 0.8, 1.5])
            
            with pagination_cols[0]:
                if st.button("⬅️", key=f"prev_{key_prefix}", width='stretch', 
                           disabled=(st.session_state[f"page_{key_prefix}"] == 1)):
                    st.session_state[f"page_{key_prefix}"] = max(1, st.session_state[f"page_{key_prefix}"] - 1)
                    st.rerun()
            
            with pagination_cols[1]:
                if st.button("➡️", key=f"next_{key_prefix}", width='stretch', 
                           disabled=(st.session_state[f"page_{key_prefix}"] == total_pages)):
                    st.session_state[f"page_{key_prefix}"] = min(total_pages, st.session_state[f"page_{key_prefix}"] + 1)
                    st.rerun()
            
            with pagination_cols[2]:
                st.markdown(f"<p style='text-align: center; margin-top: 0.5rem;'><b>{st.session_state[f'page_{key_prefix}']} / {total_pages}</b></p>", unsafe_allow_html=True)

        return st.session_state[f"page_{key_prefix}"]
