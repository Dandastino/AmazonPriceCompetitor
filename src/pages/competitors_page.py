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
    st.subheader("Competitors list")
    
    products = db.get_all_products()
    
    if not products:
        st.info("You must scrape some products before looking for competitors!")
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
        st.info("No competitor data available!")
        return
    
    parent_options = [
        f"{item['parent'].get('title', 'Unknown')[:50]} ({item['count']} competitors)"
        for item in products_with_competitors
    ]
    
    selected = st.selectbox("Select a product:", parent_options, key="competitor_select")
    selected_index = parent_options.index(selected)
    selected_item = products_with_competitors[selected_index]
    
    parent = selected_item["parent"]
    competitors = selected_item["competitors"]
            
    # Sort all competitors by score (descending)
    scored_competitors = analytics.score_competitors(competitors)
    ordered_competitors = sorted(scored_competitors, key=lambda x: x.get("score", 0), reverse=True)

    st.divider()
    st.markdown(f"### Total Competitors: {len(ordered_competitors)}")
    
    if "page_competitors" not in st.session_state:
        st.session_state["page_competitors"] = 1
    
    page = st.session_state["page_competitors"]
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
    st.markdown("### Export Data")
    
    # Pagination at the end
    ui.render_pagination(len(ordered_competitors), items_per_page=10, key_prefix="competitors")
    
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
            label="Download All Data in CSV",
            data=csv,
            file_name=f"competitors_{parent.get('asin')}.csv",
            mime="text/csv",
            width='stretch',
            type='secondary'
        )
    