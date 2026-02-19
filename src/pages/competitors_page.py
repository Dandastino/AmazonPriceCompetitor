"""Competitors view page."""

import pandas as pd
import streamlit as st

from src.utils import format_price, format_rating
from src.core import AnalyticsEngine, MongoDB
from src.pages.ui_renderer import UIRenderer

ui = UIRenderer()
analytics = AnalyticsEngine()


def render():
    """Render competitors view with filtering and comparison."""
    db = MongoDB()
    st.subheader("🔍 Competitor Products Database")
    
    products = db.get_all_products()
    
    if not products:
        st.info("👋 No products yet. Scrape some products to get started!")
        return
    
    products_with_competitors = []
    for product in products:
        asin = product.get("asin")
        if asin:
            competitors = db.get_competitors(asin)
            if competitors:
                products_with_competitors.append({
                    "parent": product,
                    "competitors": competitors,
                    "count": len(competitors)
                })
    
    if not products_with_competitors:
        st.info("📊 No competitors found yet. Click '🔎 Find Competitors' on product cards!")
        return
    
    parent_options = [
        f"{item['parent'].get('title', 'Unknown')[:50]} ({item['count']} competitors)"
        for item in products_with_competitors
    ]
    
    selected = st.selectbox("Select a product to view its competitors:", parent_options, key="competitor_select")
    selected_index = parent_options.index(selected)
    selected_item = products_with_competitors[selected_index]
    
    parent = selected_item["parent"]
    competitors = selected_item["competitors"]
    
    st.divider()
    
    st.markdown("### 🎯 Your Product")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", format_price(parent.get("price"), currency=parent.get("currency", "$")))
        with col2:
            st.metric("Rating", format_rating(parent.get("rating"), style="star_suffix"))
        with col3:
            st.metric("Brand", parent.get("brand", "N/A"))
        with col4:
            st.metric("Stock", parent.get("stock", "Unknown"))
    
    # Sort all competitors by score (descending)
    scored_competitors = analytics.score_competitors(competitors)
    ordered_competitors = sorted(scored_competitors, key=lambda x: x.get("score", 0), reverse=True)

    st.divider()
    st.markdown(f"### 📊 All Competitors ({len(ordered_competitors)})")
    
    page = ui.render_pagination(len(ordered_competitors), items_per_page=10, key_prefix="competitors")
    start_idx = (page - 1) * 10
    end_idx = min(start_idx + 10, len(ordered_competitors))
    
    st.caption(f"Showing competitors {start_idx + 1}–{end_idx} of {len(ordered_competitors)}")
    st.divider()
    
    for idx in range(start_idx, end_idx):
        item = ordered_competitors[idx]
        competitor = item.get("competitor", {})
        score = item.get("score")
        ui.render_competitor_card(competitor, idx, score=score)
    
    st.divider()
    st.markdown("### 📥 Export Data")
    
    comparison_data = []
    for item in ordered_competitors:
        comp = item.get("competitor", {})
        comparison_data.append({
            "ASIN": comp.get("asin", "N/A"),
            "Title": comp.get("title", "Unknown"),
            "Price": format_price(comp.get("price"), currency=comp.get("currency", "$")),
            "Rating": format_rating(comp.get("rating"), style="number"),
            "Brand": comp.get("brand", "N/A"),
            "Stock": comp.get("stock", "Unknown")
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Competitor Data (CSV)",
            data=csv,
            file_name=f"competitors_{parent.get('asin')}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    st.divider()
