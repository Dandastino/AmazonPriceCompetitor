"""Products management page."""

import streamlit as st
from src.core import MongoDB, ProductService
from src.pages.ui_renderer import UIRenderer

ui = UIRenderer()


def render():
    """Render all products with pagination."""
    db = MongoDB()
    service = ProductService()
    products = db.get_all_products()

    if not products:
        st.info("👋 No products stored yet. Scrape a product to get started!")
        return

    st.divider()
    st.subheader(f"📦 Your Products ({len(products)})")

    items_per_page = 10
    page = ui.render_pagination(len(products), items_per_page, key_prefix="products")
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(products))

    st.caption(f"Showing products {start_idx + 1}–{end_idx} of {len(products)}")
    st.divider()

    for idx in range(start_idx, end_idx):
        ui.render_product_card(products[idx], db, service, idx)
    
    st.divider()
    _render_delete_all_section(db)


def _render_delete_all_section(db):
    """Render delete all data confirmation UI."""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Delete All Data", width='stretch', type="secondary"):
            if st.session_state.get("delete_confirmed") is None:
                st.session_state["delete_confirmed"] = False
            
            if not st.session_state["delete_confirmed"]:
                st.session_state["delete_confirmed"] = True
                st.rerun()
    
    if st.session_state.get("delete_confirmed"):
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ Confirm Delete All", width='stretch', type="primary"):
                if db.delete_all_products():
                    st.success("✅ All data deleted successfully!")
                    st.session_state["delete_confirmed"] = None
                    st.session_state["selected_asin"] = None
                    st.rerun()
                else:
                    st.error("Failed to delete data")
        
        with col_cancel:
            if st.button("❌ Cancel", width='stretch'):
                st.session_state["delete_confirmed"] = None
                st.rerun()
