"""Reviews and AI insights analysis page."""

import logging

import streamlit as st

from src.core import MongoDB, analyze_competitors
from src.models import ReviewAnalyzer
from src.pages.ui_renderer import UIRenderer

logger = logging.getLogger(__name__)
ui = UIRenderer()


def render():
    """Main analysis page with tool selection."""
    st.subheader("Analysis Tools")

    analysis_type = st.radio(
        "What do you want to analyze?",
        ["Review Analysis & Sentiment", "AI Competitor Insights"],
        horizontal=True
    )

    st.divider()

    if "Review" in analysis_type:
        render_review_analysis()
    else:
        render_ai_insights()


def render_review_analysis():
    """Render review analysis with sentiment analysis."""
    db = MongoDB()
    st.subheader("Review Analysis & Sentiment Insights")

    products = db.get_all_products()

    if not products:
        st.info("No products to analyze. Scrape products first!")
        return

    product_options = ui.create_product_options(products)
    selected = st.selectbox("Select product to analyze reviews:", product_options, key="review_product_select")
    product = ui.get_selected_product(products, selected, product_options)

    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_button = st.button("Analyze Reviews", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", type="secondary", use_container_width=True):
            for key in ["review_analysis_result"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if analyze_button:
        asin = product.get("asin")
        domain = product.get("amazon_domain", "com")
        title = product.get("title", "Unknown")

        try:
            with st.status("Analyzing reviews...", expanded=True) as status:
                st.write("Fetching reviews from Amazon...")
                analyzer = ReviewAnalyzer(analysis_mode="llm")

                progress_bar = st.progress(0, text="Starting analysis...")

                result = analyzer.analyze_competitor_product(
                    asin=asin,
                    product_title=title,
                    domain=domain,
                    max_reviews=15,
                    show_progress=False,
                    progress_bar=progress_bar
                )

                if "error" in result:
                    status.update(label="Analysis failed", state="error")
                    st.error(result["error"])
                else:
                    gap_analysis = result.get("gap_analysis", {})
                    total_aspects = gap_analysis.get("total_aspects", 0)

                    if total_aspects == 0:
                        status.update(label="No aspects extracted", state="error")
                        st.error(
                            "Aspect extraction returned no data. "
                            "This usually means the OpenAI API key is missing/invalid, "
                            "or the reviews have no text content."
                        )
                    else:
                        progress_bar.progress(95, text="Generating report...")
                        st.write("Generating insights report...")
                        narrative = analyzer.generate_narrative_report(
                            gap_analysis,
                            result.get("reviews", [])
                        )
                        progress_bar.progress(100, text="Done!")
                        status.update(
                            label=f"Analysis complete — {result.get('total_reviews_analyzed', 0)} reviews analyzed",
                            state="complete",
                            expanded=False
                        )
                        st.session_state["review_analysis_result"] = {
                            "narrative": narrative,
                            "total_reviews": result.get("total_reviews_analyzed", 0),
                            "product_title": title,
                            "gap_analysis": gap_analysis,
                        }

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error(f"Review analysis error: {e}", exc_info=True)

    if "review_analysis_result" in st.session_state:
        data = st.session_state["review_analysis_result"]
        st.divider()
        st.caption(f"Based on **{data['total_reviews']}** reviews analyzed — {data['product_title']}")
        st.markdown(data["narrative"])

        gap = data.get("gap_analysis", {})
        if gap:
            with st.expander("Debug: raw aspect data"):
                col1, col2, col3 = st.columns(3)
                breakdown = gap.get("sentiment_breakdown", {})
                col1.metric("Total aspects", gap.get("total_aspects", 0))
                col2.metric("Positive mentions", breakdown.get("positive", 0))
                col3.metric("Negative mentions", breakdown.get("negative", 0))
                top_pos = gap.get("top_positive_aspects", [])
                top_neg = gap.get("top_negative_aspects", [])
                if top_pos:
                    st.write("**Top positive aspects:**", [f"{a['aspect']} ({a['positive_count']}+)" for a in top_pos[:5]])
                if top_neg:
                    st.write("**Top negative aspects:**", [f"{a['aspect']} ({a['negative_count']}-)" for a in top_neg[:5]])


def render_ai_insights():
    """Render AI-powered competitor insights as a conversational chat interface."""
    db = MongoDB()

    products = db.get_all_products()

    if not products:
        st.info("No products to analyze. Scrape products first!")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        product_options = ui.create_product_options(products)
        selected = st.selectbox(
            "Product",
            product_options,
            key="ai_insights_select",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("New chat", use_container_width=True):
            st.session_state.pop("ai_chat_history", None)
            st.session_state.pop("ai_chat_asin", None)
            st.rerun()

    product = ui.get_selected_product(products, selected, product_options)
    asin = product.get("asin")
    competitors = db.get_competitors(asin)

    if not competitors:
        st.warning("No competitors found. Click '🔎 Find Competitors' on the product card first!")
        return

    # Reset chat when product changes
    if st.session_state.get("ai_chat_asin") != asin:
        st.session_state.ai_chat_history = []
        st.session_state.ai_chat_asin = asin

    chat_history: list = st.session_state.setdefault("ai_chat_history", [])

    # Render existing chat messages
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # No conversation started yet — show start button
    if not chat_history:
        st.caption(f"**{len(competitors)}** competitors loaded — press the button to start.")
        if st.button("Generate Analysis", type="primary", use_container_width=True):
            with st.chat_message("assistant"):
                with st.spinner("Analyzing competitors..."):
                    try:
                        analysis = analyze_competitors(asin)
                    except Exception as e:
                        analysis = f"Analysis failed: {str(e)}"
                        logger.error(f"AI analysis error: {e}", exc_info=True)
                st.markdown(analysis)
            chat_history.append({"role": "assistant", "content": analysis})
        return

    # Download button for the initial analysis (compact, below chat)
    first_analysis = next((m["content"] for m in chat_history if m["role"] == "assistant"), None)
    if first_analysis:
        st.download_button(
            "Download analysis",
            data=first_analysis,
            file_name=f"ai_insights_{asin}.txt",
            mime="text/plain",
        )

    # Follow-up chat input
    if user_input := st.chat_input("Ask a follow-up question about this product..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.core.llm import CompetitorAnalyzer
                    analyzer = CompetitorAnalyzer()
                    response = analyzer.followup(
                        chat_history=chat_history[:-1],
                        question=user_input,
                        product=product,
                        competitors=competitors,
                    )
                except Exception as e:
                    response = f"Error processing your question: {str(e)}"
                    logger.error(f"Follow-up error: {e}", exc_info=True)
            st.markdown(response)
        chat_history.append({"role": "assistant", "content": response})


