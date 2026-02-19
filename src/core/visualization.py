"""
Data Visualization Module for Review Analysis
Phase 4: Bar Charts, Radar Charts, and Word Clouds

This module provides visualization functions for:
1. Common pain points (Bar Chart)
2. Gap comparison (Radar Chart / Spider Chart)
3. Complaint word clouds
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

from src.utils import get_logger, handle_exception, UINotifier

logger = get_logger(__name__)


def plot_pain_points_bar_chart(
    gap_analysis: Dict[str, Any],
    title: str = "Common Pain Points (Negative Mentions)",
    max_aspects: int = 10
) -> None:
    """
    Phase 4: Create bar chart showing frequency of negative mentions.
    
    Args:
        gap_analysis: Gap analysis result dict
        title: Chart title
        max_aspects: Maximum number of aspects to show
    """
    try:
        top_negative = gap_analysis.get('top_negative_aspects', [])
        
        if not top_negative:
            UINotifier.warning("No negative aspects found to visualize")
            return
        
        # Prepare data
        aspects = []
        negative_counts = []
        negative_ratios = []
        
        for aspect_data in top_negative[:max_aspects]:
            aspect = aspect_data['aspect'].replace('_', ' ').title()
            aspects.append(aspect)
            negative_counts.append(aspect_data['negative_count'])
            negative_ratios.append(aspect_data['negative_ratio'] * 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Aspect': aspects,
            'Negative Mentions': negative_counts,
            'Negative %': negative_ratios
        })
        
        # Display using Streamlit
        st.subheader(title)
        
        # Bar chart
        st.bar_chart(df.set_index('Aspect')['Negative Mentions'])
        
        # Data table
        with st.expander("📊 View Detailed Data"):
            st.dataframe(df, width='stretch')
        
        logger.info(f"Plotted pain points bar chart with {len(aspects)} aspects")
        
    except Exception as e:
        handle_exception(logger, e, "Failed to create bar chart")


def plot_gap_radar_chart(
    your_product_scores: Dict[str, float],
    competitor_scores: Dict[str, float],
    your_product_name: str = "Your Product",
    competitor_name: str = "Competitor",
    max_aspects: int = 8
) -> None:
    """
    Phase 4: Create radar/spider chart comparing aspect scores.
    
    This visualizes the gap between your product and competitor
    across multiple aspects.
    
    Args:
        your_product_scores: Your product's aspect scores (0-1)
        competitor_scores: Competitor's aspect scores (0-1)
        your_product_name: Label for your product
        competitor_name: Label for competitor
        max_aspects: Maximum aspects to show
    """
    try:
        # Find common aspects
        common_aspects = list(
            set(your_product_scores.keys()) & set(competitor_scores.keys())
        )
        
        if not common_aspects:
            UINotifier.warning("No common aspects found between products")
            return
        
        # Limit to top aspects by total score
        aspect_totals = {
            aspect: your_product_scores[aspect] + competitor_scores[aspect]
            for aspect in common_aspects
        }
        top_aspects = sorted(
            aspect_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_aspects]
        
        aspects = [a[0].replace('_', ' ').title() for a in top_aspects]
        your_scores = [your_product_scores[a[0]] for a in top_aspects]
        competitor_scores_list = [competitor_scores[a[0]] for a in top_aspects]
        
        # Create radar chart using Plotly
        fig = go.Figure()
        
        # Add your product trace
        fig.add_trace(go.Scatterpolar(
            r=your_scores + [your_scores[0]],  # Close the polygon
            theta=aspects + [aspects[0]],
            fill='toself',
            name=your_product_name,
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Add competitor trace
        fig.add_trace(go.Scatterpolar(
            r=competitor_scores_list + [competitor_scores_list[0]],
            theta=aspects + [aspects[0]],
            fill='toself',
            name=competitor_name,
            line_color='#ff7f0e',
            fillcolor='rgba(255, 127, 14, 0.3)'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                )
            ),
            showlegend=True,
            title=f"Gap Map: {your_product_name} vs {competitor_name}",
            height=500
        )
        
        st.subheader("🎯 Gap Map (Radar Chart)")
        st.plotly_chart(fig, width='stretch')
        
        # Interpretation guide
        with st.expander("ℹ️ How to Read This Chart"):
            st.write("""
            - **Outer edge (1.0):** 100% positive sentiment
            - **Center (0.0):** 100% negative sentiment
            - **Middle (0.5):** Neutral sentiment
            
            **Blue area:** Your product's performance
            **Orange area:** Competitor's performance
            
            - Where blue extends beyond orange: Your strength
            - Where orange extends beyond blue: Competitor's strength
            - Small areas: Weakness/pain points for that product
            """)
        
        logger.info(f"Created radar chart comparing {len(aspects)} aspects")
        
    except Exception as e:
        handle_exception(logger, e, "Failed to create radar chart")


def generate_complaint_word_cloud(
    reviews: List[Dict[str, Any]],
    min_rating: float = 3.0,
    title: str = "Common Complaint Words (Reviews < 3 stars)",
    max_words: int = 50
) -> None:
    """
    Phase 4: Generate word cloud from low-rated reviews.
    
    Highlights the most common words in negative reviews.
    
    Args:
        reviews: List of review dicts with 'text' and 'rating'
        min_rating: Only include reviews below this rating
        title: Chart title
        max_words: Maximum words in cloud
    """
    try:
        from src.sentiment_analysis import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Filter low-rated reviews
        negative_reviews = [
            r for r in reviews 
            if r.get('rating') and float(r['rating']) < min_rating
        ]
        
        if not negative_reviews:
            UINotifier.warning(f"No reviews found with rating < {min_rating} stars")
            return
        
        st.subheader(title)
        st.write(f"Analyzing {len(negative_reviews)} low-rated reviews...")
        
        # Combine all negative review text
        all_text = " ".join([
            r.get('text', '') or r.get('cleaned_text', '')
            for r in negative_reviews
        ])
        
        if not all_text or len(all_text.strip()) < 20:
            st.warning("Not enough text content in negative reviews")
            return
        
        # Preprocess text
        cleaned_text = preprocessor.preprocess(all_text, remove_stopwords=True)
        
        if not cleaned_text or len(cleaned_text.strip()) < 20:
            st.warning("Insufficient text after preprocessing")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Reds',
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(cleaned_text)
        
        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        st.pyplot(fig)
        plt.close()
        
        # Show top words
        word_freq = wordcloud.words_
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        with st.expander("📝 Top 10 Complaint Words"):
            word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            word_df['Frequency'] = word_df['Frequency'].apply(lambda x: f"{x:.3f}")
            st.table(word_df)
        
        logger.info(f"Generated word cloud from {len(negative_reviews)} reviews")
        
    except ImportError:
        UINotifier.error("WordCloud package not installed. Run: pip install wordcloud")
    except Exception as e:
        handle_exception(logger, e, "Failed to generate word cloud")


def generate_aspect_word_clouds(
    gap_analysis: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    top_n_aspects: int = 3
) -> None:
    """
    Generate word clouds for specific negative aspects.
    
    Args:
        gap_analysis: Gap analysis result
        reviews: List of review dicts
        top_n_aspects: Number of top negative aspects to visualize
    """
    try:
        from src.sentiment_analysis import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        critical_gaps = gap_analysis.get('critical_gaps', [])
        
        if not critical_gaps:
            st.info("No critical gaps to visualize")
            return
        
        st.subheader("🔍 Word Clouds by Aspect")
        
        for gap in critical_gaps[:top_n_aspects]:
            aspect = gap['aspect']
            aspect_display = aspect.replace('_', ' ').title()
            
            # Extract sentences mentioning this aspect from reviews
            relevant_text = []
            for review in reviews:
                text = review.get('text', '') or review.get('cleaned_text', '')
                if aspect in text.lower():
                    # Get sentences with this aspect
                    sentences = preprocessor.tokenize_sentences(text)
                    for sent in sentences:
                        if aspect in sent.lower():
                            relevant_text.append(sent)
            
            if not relevant_text:
                continue
            
            combined_text = " ".join(relevant_text)
            cleaned = preprocessor.preprocess(combined_text, remove_stopwords=True)
            
            if len(cleaned.strip()) < 20:
                continue
            
            # Generate word cloud
            try:
                wordcloud = WordCloud(
                    width=600,
                    height=300,
                    background_color='white',
                    colormap='Reds',
                    max_words=30,
                    relative_scaling=0.5
                ).generate(cleaned)
                
                with st.expander(f"🔴 {aspect_display} ({gap['negative_count']} negative mentions)"):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    
                    if gap.get('common_complaints'):
                        st.write("**Common complaints:**")
                        for complaint in gap['common_complaints']:
                            st.write(f"- {complaint}")
            
            except Exception as e:
                logger.warning(f"Failed to create word cloud for {aspect}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error in aspect word clouds: {e}", exc_info=True)
        st.error(f"Failed to generate aspect word clouds: {str(e)}")


def plot_sentiment_distribution(
    gap_analysis: Dict[str, Any],
    title: str = "Overall Sentiment Distribution"
) -> None:
    """
    Create pie chart showing sentiment distribution.
    
    Args:
        gap_analysis: Gap analysis result
        title: Chart title
    """
    try:
        sentiment_breakdown = gap_analysis.get('sentiment_breakdown', {})
        
        if not sentiment_breakdown:
            st.warning("No sentiment data available")
            return
        
        # Prepare data
        labels = []
        values = []
        colors = []
        
        color_map = {
            'positive': '#28a745',
            'neutral': '#ffc107',
            'negative': '#dc3545'
        }
        
        for sentiment, count in sentiment_breakdown.items():
            if count > 0:
                labels.append(sentiment.capitalize())
                values.append(count)
                colors.append(color_map.get(sentiment, '#6c757d'))
        
        if not values:
            st.warning("No sentiment data to display")
            return
        
        # Create pie chart using Plotly
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        logger.info("Created sentiment distribution chart")
        
    except Exception as e:
        logger.error(f"Error creating sentiment pie chart: {e}", exc_info=True)
        st.error(f"Failed to create sentiment chart: {str(e)}")


def create_comparison_table(
    your_scores: Dict[str, float],
    competitor_scores: Dict[str, float],
    your_name: str = "Your Product",
    competitor_name: str = "Competitor"
) -> None:
    """
    Create side-by-side comparison table of aspect scores.
    
    Args:
        your_scores: Your product's scores
        competitor_scores: Competitor's scores
        your_name: Your product label
        competitor_name: Competitor label
    """
    try:
        # Find common aspects
        common_aspects = set(your_scores.keys()) & set(competitor_scores.keys())
        
        if not common_aspects:
            st.warning("No common aspects to compare")
            return
        
        # Prepare comparison data
        comparison_data = []
        for aspect in common_aspects:
            your_score = your_scores[aspect]
            comp_score = competitor_scores[aspect]
            gap = your_score - comp_score
            
            comparison_data.append({
                'Aspect': aspect.replace('_', ' ').title(),
                your_name: f"{your_score:.2f}",
                competitor_name: f"{comp_score:.2f}",
                'Gap': f"{gap:+.2f}",
                'Winner': your_name if gap > 0 else (competitor_name if gap < 0 else 'Tie')
            })
        
        # Sort by gap magnitude
        comparison_data.sort(key=lambda x: abs(float(x['Gap'])), reverse=True)
        
        df = pd.DataFrame(comparison_data)
        
        st.subheader("📊 Aspect Score Comparison")
        
        # Style the dataframe
        def highlight_winner(row):
            if row['Winner'] == your_name:
                return ['background-color: #d4edda'] * len(row)
            elif row['Winner'] == competitor_name:
                return ['background-color: #f8d7da'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = df.style.apply(highlight_winner, axis=1)
        st.dataframe(styled_df, width='stretch')
        
        st.caption(f"🟢 Green: {your_name} wins | 🔴 Red: {competitor_name} wins")
        
    except Exception as e:
        logger.error(f"Error creating comparison table: {e}", exc_info=True)
        st.error(f"Failed to create comparison table: {str(e)}")


def display_comprehensive_analysis(
    your_analysis: Dict[str, Any],
    competitor_analysis: Dict[str, Any],
    your_reviews: List[Dict[str, Any]],
    competitor_reviews: List[Dict[str, Any]],
    your_name: str = "Your Product",
    competitor_name: str = "Competitor"
) -> None:
    """
    Display comprehensive analysis with all visualizations.
    
    Phase 4: Complete visualization suite.
    
    Args:
        your_analysis: Your product's gap analysis
        competitor_analysis: Competitor's gap analysis
        your_reviews: Your product's reviews
        competitor_reviews: Competitor's reviews
        your_name: Your product label
        competitor_name: Competitor label
    """
    try:
        st.title("🎯 Comprehensive Gap Analysis & Visualization")
        st.write("---")
        
        # Get aspect scores
        your_scores = your_analysis.get('gap_analysis', {}).get('aspect_scores', {})
        comp_scores = competitor_analysis.get('gap_analysis', {}).get('aspect_scores', {})
        
        # 1. Radar Chart (Gap Map)
        if your_scores and comp_scores:
            plot_gap_radar_chart(
                your_scores,
                comp_scores,
                your_name,
                competitor_name
            )
            st.write("---")
        
        # 2. Comparison Table
        if your_scores and comp_scores:
            create_comparison_table(
                your_scores,
                comp_scores,
                your_name,
                competitor_name
            )
            st.write("---")
        
        # 3. Pain Points Bar Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {your_name}")
            your_gap = your_analysis.get('gap_analysis', {})
            if your_gap:
                plot_pain_points_bar_chart(your_gap, f"{your_name} - Pain Points", max_aspects=5)
        
        with col2:
            st.subheader(f"📊 {competitor_name}")
            comp_gap = competitor_analysis.get('gap_analysis', {})
            if comp_gap:
                plot_pain_points_bar_chart(comp_gap, f"{competitor_name} - Pain Points", max_aspects=5)
        
        st.write("---")
        
        # 4. Sentiment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            plot_sentiment_distribution(
                your_analysis.get('gap_analysis', {}),
                f"{your_name} - Sentiment"
            )
        
        with col2:
            plot_sentiment_distribution(
                competitor_analysis.get('gap_analysis', {}),
                f"{competitor_name} - Sentiment"
            )
        
        st.write("---")
        
        # 5. Word Clouds
        st.header("☁️ Complaint Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{your_name}")
            generate_complaint_word_cloud(your_reviews, title=f"{your_name} - Complaints")
        
        with col2:
            st.subheader(f"{competitor_name}")
            generate_complaint_word_cloud(competitor_reviews, title=f"{competitor_name} - Complaints")
        
        logger.info("Displayed comprehensive analysis")
        
    except Exception as e:
        logger.error(f"Error in comprehensive display: {e}", exc_info=True)
        st.error(f"Failed to display comprehensive analysis: {str(e)}")
