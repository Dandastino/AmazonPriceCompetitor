"""Reviews and AI insights analysis page."""

import logging

import pandas as pd
import streamlit as st

from src.core import MongoDB, AnalyticsEngine, analyze_competitors
from src.models import ReviewAnalyzer
from src.pages.ui_renderer import UIRenderer
from src.core.visualization import (
    plot_pain_points_bar_chart,
    plot_sentiment_distribution,
    generate_complaint_word_cloud,
)

logger = logging.getLogger(__name__)
ui = UIRenderer()
analytics = AnalyticsEngine()


def render():
    """Main analysis page with tool selection."""
    st.subheader("📊 Analysis Tools")
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["📝 Review Analysis & Sentiment", "🎯 AI Competitor Insights", "💡 Price Analysis"],
        horizontal=True
    )
    
    st.divider()
    
    if "Review" in analysis_type:
        render_review_analysis()
    elif "Competitor" in analysis_type:
        render_ai_insights()
    else:
        render_analysis()


def render_review_analysis():
    """Render review analysis with sentiment analysis."""
    db = MongoDB()
    st.subheader("📝 Review Analysis & Sentiment Insights")
    
    products = db.get_all_products()
    
    if not products:
        st.info("👋 No products to analyze. Scrape products first!")
        return
    
    st.markdown("""
    This section allows you to:
    - 🔍 Scrape product reviews from Amazon
    - 🤖 Analyze sentiment using AI
    - 📊 Visualize pain points and gaps
    - ☁️ Generate word clouds from complaints
    """)
    
    st.divider()
    
    product_options = ui.create_product_options(products)
    
    selected = st.selectbox("Select product to analyze reviews:", product_options, key="review_product_select")
    
    product = ui.get_selected_product(products, selected, product_options)
    
    st.divider()
    st.markdown("### 🔍 Step 1: Scrape Reviews")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        max_reviews = st.slider("Max reviews to scrape:", 10, 200, 50, 10, key="max_reviews_slider")
    with col2:
        st.write("")
        st.write("")
        scrape_button = st.button("🚀 Scrape Reviews", type="primary", width='stretch')
    with col3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Cache", type="secondary", width='stretch'):
            if "review_data" in st.session_state:
                del st.session_state["review_data"]
            st.success("Cache cleared!")
            st.rerun()
    
    if scrape_button:
        with st.spinner("🔄 Scraping reviews from Amazon..."):
            try:
                from src.core import scrape_product_reviews
                
                asin = product.get("asin")
                domain = product.get("amazon_domain", "com")
                geo = product.get("amazon_geo_location", "")
                
                reviews = scrape_product_reviews(
                    asin=asin,
                    domain=domain,
                    geo_location=geo,
                    max_reviews=max_reviews,
                    show_progress=True
                )
                
                if reviews:
                    st.session_state["review_data"] = {
                        "product": product,
                        "reviews": reviews
                    }
                    st.success(f"✅ Scraped {len(reviews)} reviews successfully!")
                else:
                    st.warning("No reviews found for this product")
                    
            except Exception as e:
                st.error(f"Error scraping reviews: {str(e)}")
                logger.error(f"Review scraping error: {e}", exc_info=True)
    
    if "review_data" in st.session_state:
        review_data = st.session_state["review_data"]
        reviews = review_data["reviews"]
        
        st.divider()
        st.markdown(f"### 🤖 Step 2: Analyze Sentiment ({len(reviews)} reviews)")
        
        if st.button("🧠 Analyze Sentiment", type="primary", width='stretch'):
            with st.spinner("🤖 Analyzing sentiment with GPT-4o-mini..."):
                try:
                    analyzer = ReviewAnalyzer(analysis_mode="llm")
                    
                    result = analyzer.analyze_competitor_product(
                        asin=product.get("asin"),
                        product_title=product.get("title", "Unknown"),
                        geo_location=product.get("amazon_geo_location", ""),
                        domain=product.get("amazon_domain", "com"),
                        max_reviews=len(reviews),
                        show_progress=True
                    )
                    
                    if "error" not in result:
                        st.session_state["sentiment_analysis"] = result
                        st.success("✅ Sentiment analysis complete!")
                    else:
                        st.error(f"Analysis failed: {result['error']}")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Sentiment analysis error: {e}", exc_info=True)
    
    if "sentiment_analysis" in st.session_state:
        _render_sentiment_results(st.session_state["sentiment_analysis"])


def _render_sentiment_results(analysis_result):
    """Render sentiment analysis results."""
    gap_analysis = analysis_result.get("gap_analysis", {})
    analyzed_reviews = analysis_result.get("reviews", [])
    
    st.divider()
    st.markdown("### 📊 Step 3: View Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reviews Analyzed", analysis_result.get("total_reviews_analyzed", 0))
    with col2:
        sentiment_breakdown = gap_analysis.get("sentiment_breakdown", {})
        negative_pct = (sentiment_breakdown.get("negative", 0) / 
                      sum(sentiment_breakdown.values()) * 100) if sentiment_breakdown.values() else 0
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
    with col3:
        st.metric("Aspects Found", gap_analysis.get("total_aspects", 0))
    
    st.divider()
    
    if gap_analysis:
        plot_pain_points_bar_chart(gap_analysis, max_aspects=10)
        st.divider()
        plot_sentiment_distribution(gap_analysis)
        st.divider()
    
    if analyzed_reviews:
        generate_complaint_word_cloud(analyzed_reviews, min_rating=3.0, max_words=50)
        st.divider()
    
    with st.expander("📋 View Detailed Aspect Analysis"):
        top_negative = gap_analysis.get("top_negative_aspects", [])
        if top_negative:
            aspect_df = pd.DataFrame([
                {
                    "Aspect": a["aspect"].replace("_", " ").title(),
                    "Total Mentions": a["total_mentions"],
                    "Negative": a["negative_count"],
                    "Positive": a["positive_count"],
                    "Negative %": f"{a['negative_ratio']*100:.1f}%"
                }
                for a in top_negative[:15]
            ])
            st.dataframe(aspect_df, width='stretch')


def render_ai_insights():
    """Render AI-powered competitor insights."""
    db = MongoDB()
    st.subheader("🎯 AI-Powered Competitor Insights")
    
    products = db.get_all_products()
    
    if not products:
        st.info("👋 No products to analyze. Scrape products first!")
        return
    
    st.markdown("""
    Get strategic insights powered by GPT-4:
    - 🎯 Market positioning analysis
    - 💡 Competitive advantages & gaps
    - 📈 Price & value comparison
    - ✨ Actionable recommendations
    """)
    
    st.divider()
    
    product_options = ui.create_product_options(products)
    selected = st.selectbox("Select product for AI analysis:", product_options, key="ai_insights_select")
    product = ui.get_selected_product(products, selected, product_options)
    
    asin = product.get("asin")
    competitors = db.get_competitors(asin)
    
    if not competitors:
        st.warning(f"⚠️ No competitors found. Click '🔎 Find Competitors' on the product card first!")
        return
    
    st.info(f"📊 Found {len(competitors)} competitors for analysis")
    
    if st.button("🧠 Generate AI Insights", type="primary", width='stretch'):
        with st.spinner("🤖 Analyzing with AI... This may take 10-30 seconds"):
            try:
                analysis = analyze_competitors(asin)
                
                if analysis:
                    st.session_state["ai_analysis"] = {
                        "product": product,
                        "analysis": analysis,
                        "timestamp": pd.Timestamp.now()
                    }
                    st.success("✅ AI analysis complete!")
                else:
                    st.error("Failed to generate analysis")
                    
            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
                logger.error(f"AI analysis error: {e}", exc_info=True)
    
    if "ai_analysis" in st.session_state:
        ai_data = st.session_state["ai_analysis"]
        analysis = ai_data["analysis"]
        analyzed_product = ai_data["product"]
        
        st.divider()
        st.markdown(f"### 📄 Analysis Results")
        st.caption(f"Generated: {ai_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown(analysis)
        
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="📥 Download Analysis",
                data=analysis,
                file_name=f"ai_insights_{analyzed_product.get('asin')}.txt",
                mime="text/plain",
            )
        with col2:
            if st.button("🔄 Clear Results", type="secondary"):
                del st.session_state["ai_analysis"]
                st.rerun()


def render_analysis():
    """Render product analysis and pricing insights."""
    db = MongoDB()
    st.subheader("💡 Product Analysis")
    
    products = db.get_all_products()
    
    if not products:
        st.info("👋 No products yet. Scrape some products first!")
        return
    
    product_options = ui.create_product_options(products)
    selected = st.selectbox("Select a product to analyze:", product_options, key="analysis_product_select")
    product = ui.get_selected_product(products, selected, product_options)
    
    competitors = db.get_competitors(product.get("asin"))
    
    if not competitors:
        st.warning("⚠️ No competitors found. Use '🔎 Find Competitors' first to analyze pricing!")
        return
    
    st.divider()
    
    scored_competitors = analytics.score_competitors(competitors)
    competitor_stats = analytics.calculate_competitor_stats(competitors)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Competitors", competitor_stats["total"])
    with col2:
        st.metric("Avg Price", f"${competitor_stats['avg_price']:.2f}")
    with col3:
        st.metric("Avg Rating", f"{competitor_stats['avg_rating']:.1f}/5")
    with col4:
        st.metric("In Stock", f"{competitor_stats['in_stock_percentage']:.0f}%")
    
    st.divider()
    
    st.markdown("### 📈 Your Product Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = product.get("price", 0)
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        rating = product.get("rating", 0)
        st.metric("Rating", f"{rating:.1f}/5")
    
    with col3:
        reviews = product.get("reviews_count") or product.get("num_reviews") or 0
        st.metric("Reviews", int(reviews))
    
    st.divider()
    
    st.markdown("### 💡 Price Recommendation")
    
    avg_competitor_price = competitor_stats["avg_price"]
    min_competitive_price = min([c.get("price", 0) for c in competitors if isinstance(c.get("price"), (int, float)) and c.get("price") > 0] or [0])
    max_competitive_price = max([c.get("price", 0) for c in competitors if isinstance(c.get("price"), (int, float)) and c.get("price") > 0] or [0])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Competitor Price", f"${min_competitive_price:.2f}")
    with col2:
        st.metric("Avg Competitor Price", f"${avg_competitor_price:.2f}")
    with col3:
        st.metric("Max Competitor Price", f"${max_competitive_price:.2f}")
    
    st.info(
        f"💡 **Recommendation:** Consider pricing between ${min_competitive_price:.2f} and ${avg_competitor_price:.2f} "
        f"to stay competitive while maintaining good margins."
    )
