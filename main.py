"""Amazon Competitor Analysis Tool - Main Application."""

import logging

import streamlit as st

from src.core import ProductService
from src.pages.ui_renderer import UIRenderer
from src.pages import products_page, competitors_page, tools_analysis, tools_assistant

logger = logging.getLogger(__name__)
ui = UIRenderer()


def _render_sidebar():
    """Render the sidebar with product scraping controls."""
    with st.sidebar:
        st.header("📥 Scrape Product")
        
        service = ProductService()
        asin, geo, domain = ui.render_input_section()

        if st.button("🔍 Scrape Product", width='stretch', type="primary"):
            if not asin:
                st.error("❌ Please enter an ASIN to scrape a product")
            else:
                with st.spinner("🔄 Scraping product from Amazon..."):
                    try:
                        result = service.scrape_and_store_product(asin, geo, domain)
                        if result:
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        logger.error(f"Scraping error: {e}", exc_info=True)

        st.divider()
        st.info(
            "💡 **Tips:**\n"
            "- Enter a valid Amazon ASIN\n"
            "- Region affects pricing & availability\n"
            "- Domain selects the Amazon marketplace"
        )


def _render_footer():
    """Render page footer."""
    st.divider()
    st.caption(
        "Amazon Competitor Analysis Tool | "
        "Powered by Oxylabs & OpenAI | "
        "[GitHub](https://github.com)"
    )


def main():
    """Main application entry point."""
    ui.configure_page()
    ui.render_header()

    _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 All Products",
        "🔍 Competitors",
        "📊 Analysis",
        "🤖 Assistant"
    ])
    
    with tab1:
        products_page.render()
    
    with tab2:
        competitors_page.render()
    
    with tab3:
        tools_analysis.render()
    
    with tab4:
        tools_assistant.render()

    _render_footer()


if __name__ == "__main__":
    main()
