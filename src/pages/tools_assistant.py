"""AI Assistant - Product Chatbot."""

import logging

import streamlit as st

from src.models import ProductChatbot
from src.core import MongoDB, AnalyticsEngine
from src.pages.ui_renderer import UIRenderer

logger = logging.getLogger(__name__)
ui = UIRenderer()
analytics = AnalyticsEngine()


def render():
    """Main assistant page with chatbot."""
    st.subheader("🤖 Alex - AI Assistant")
    
    st.divider()
    
    render_chatbot()


def render_chatbot():
    """Render chatbot interface for product questions."""
    db = MongoDB()
    st.subheader("💬 Ask Alex")
    
    products = db.get_all_products()
    
    if not products:
        st.info("👋 No products yet. Scrape some products first!")
        return
    
    product_options = ui.create_product_options(products)
    selected = st.selectbox("Select a product to ask questions about:", product_options, key="chatbot_product_select")
    product = ui.get_selected_product(products, selected, product_options)
    
    st.divider()
    st.info("Ask questions about this product, competitors, pricing strategies, etc.")

    product_key = product.get("asin") or product.get("title") or "default"
    history_map = st.session_state.setdefault("chat_history", {})
    messages = history_map.setdefault(product_key, [])

    col_clear, col_delete = st.columns([1, 1])
    with col_clear:
        if st.button("🧹 Clear chat", width='stretch', key=f"clear_chat_{product_key}"):
            history_map[product_key] = []
            st.rerun()
    with col_delete:
        if st.button("↩️ Delete last", width='stretch', key=f"delete_last_{product_key}"):
            if history_map[product_key]:
                history_map[product_key].pop()
                st.rerun()

    if messages:
        for idx, msg in enumerate(messages):
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))
                if st.button("🗑️ Delete", key=f"delete_msg_{product_key}_{idx}"):
                    history_map[product_key].pop(idx)
                    st.rerun()

    user_question = st.chat_input("Write a message...")
    if user_question:
        messages.append({"role": "user", "content": user_question})

        chat_history = []
        pending_user = None
        for msg in messages:
            if msg.get("role") == "user":
                pending_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and pending_user is not None:
                chat_history.append((pending_user, msg.get("content", "")))
                pending_user = None

        with st.spinner("🤖 Thinking..."):
            try:
                chatbot = ProductChatbot()
                answer = chatbot.chat(user_message=user_question, chat_history=chat_history)
                messages.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Chatbot error: {e}", exc_info=True)
