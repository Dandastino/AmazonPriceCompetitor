"""
Review Analyzer - Integrates review scraping with sentiment analysis and gap analysis.

This module provides high-level functions to:
1. Scrape reviews for competitors
2. Analyze sentiment and extract aspects
3. Perform gap analysis to identify competitor weaknesses
"""

import logging
from typing import List, Dict, Any, Optional

import streamlit as st

from src.core.oxylab_client import scrape_product_reviews
from .sentiment_analysis import (
    TextPreprocessor,
    LLMBasedABSA,
    GapAnalyzer,
    create_sentiment_analyzer
)
from src.utils import get_logger, get_timestamp, handle_exception

logger = get_logger(__name__)


class ReviewAnalyzer:
    """
    Complete review analysis pipeline combining scraping and NLP with LLM-based sentiment analysis.
    """
    
    def __init__(self, analysis_mode: str = "llm"):
        """
        Initialize review analyzer.
        
        Args:
            analysis_mode: Must be "llm" (uses OpenAI GPT-4o-mini)
        """
        self.analysis_mode = "llm"  # Always use LLM mode
        self.preprocessor = TextPreprocessor()
        self.gap_analyzer = GapAnalyzer()
        
        # Initialize LLM-based sentiment analyzer
        self.sentiment_analyzer = LLMBasedABSA()
        logger.info("ReviewAnalyzer initialized with LLM-based ABSA (GPT-4o-mini)")
    
    def analyze_competitor_product(
        self,
        asin: str,
        product_title: str,
        geo_location: str,
        domain: str,
        max_reviews: int = 50,
        product_category: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single competitor product's reviews.
        
        Args:
            asin: Product ASIN
            product_title: Product name
            geo_location: Geographic location
            domain: Amazon domain
            max_reviews: Maximum reviews to analyze
            product_category: Optional product category for better ABSA
            show_progress: Show progress in UI
            
        Returns:
            Complete analysis including reviews, sentiments, and gaps
        """
        try:
            if show_progress:
                st.write(f"### Analyzing: {product_title[:50]}...")
            
            # Step 1: Scrape reviews
            logger.info(f"Starting review analysis for {asin}")
            reviews = scrape_product_reviews(
                asin=asin,
                geo_location=geo_location,
                domain=domain,
                max_reviews=max_reviews,
                show_progress=show_progress
            )
            
            if not reviews:
                logger.warning(f"No reviews found for {asin}")
                return {
                    "asin": asin,
                    "product_title": product_title,
                    "total_reviews": 0,
                    "error": "No reviews found"
                }
            
            if show_progress:
                st.write(f"🤖 Analyzing sentiment for {len(reviews)} reviews...")
            
            # Step 2: Preprocess and analyze
            analyzed_reviews = []
            all_aspects = []
            
            for idx, review in enumerate(reviews):
                # Preprocess text
                review_text = review.get("text", "")
                cleaned_text = self.preprocessor.clean_text(review_text)
                
                if not cleaned_text:
                    continue
                
                # Extract aspects using LLM
                aspects = self.sentiment_analyzer.extract_aspects(
                    cleaned_text,
                    product_category
                )
                
                all_aspects.append(aspects)
                
                analyzed_reviews.append({
                    "review_id": review.get("id"),
                    "original_text": review_text,
                    "cleaned_text": cleaned_text,
                    "rating": review.get("rating"),
                    "date": review.get("date"),
                    "verified": review.get("verified"),
                    "aspects": aspects,
                    "aspect_count": len(aspects)
                })
                
                # Progress update
                if show_progress and (idx + 1) % 10 == 0:
                    st.write(f"Processed {idx + 1}/{len(reviews)} reviews...")
            
            # Step 3: Gap analysis
            if show_progress:
                st.write("📊 Performing gap analysis...")
            
            gap_analysis = self.gap_analyzer.analyze_aspects(all_aspects)
            
            # Compile results
            result = {
                "asin": asin,
                "product_title": product_title,
                "total_reviews_scraped": len(reviews),
                "total_reviews_analyzed": len(analyzed_reviews),
                "analysis_timestamp": get_timestamp(),
                "analysis_mode": self.analysis_mode,
                "reviews": analyzed_reviews,
                "gap_analysis": gap_analysis,
                "summary": self._generate_summary(gap_analysis)
            }
            
            logger.info(f"Completed analysis for {asin}: {len(analyzed_reviews)} reviews analyzed")
            return result
            
        except Exception as e:
            handle_exception(logger, e, f"Error analyzing product {asin}")
            return {
                "asin": asin,
                "product_title": product_title,
                "error": str(e)
            }
    
    def analyze_multiple_competitors(
        self,
        competitors: List[Dict[str, Any]],
        geo_location: str,
        domain: str,
        max_reviews_per_product: int = 50,
        product_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple competitor products and compare them.
        
        Args:
            competitors: List of competitor dicts with 'asin' and 'title'
            geo_location: Geographic location
            domain: Amazon domain
            max_reviews_per_product: Max reviews per product
            product_category: Optional product category
            
        Returns:
            Comparative gap analysis across all competitors
        """
        try:
            st.write(f"## 🔍 Analyzing {len(competitors)} Competitors")
            st.write("---")
            
            competitor_analyses = {}
            
            for idx, competitor in enumerate(competitors, 1):
                asin = competitor.get("asin")
                title = competitor.get("title", f"Product {asin}")
                
                st.write(f"### {idx}/{len(competitors)}: {title[:60]}")
                
                analysis = self.analyze_competitor_product(
                    asin=asin,
                    product_title=title,
                    geo_location=geo_location,
                    domain=domain,
                    max_reviews=max_reviews_per_product,
                    product_category=product_category,
                    show_progress=True
                )
                
                competitor_analyses[title] = analysis
                
                # Show quick summary
                if "gap_analysis" in analysis:
                    gap = analysis["gap_analysis"]
                    st.write(f"**Reviews Analyzed:** {gap.get('total_reviews_analyzed', 0)}")
                    st.write(f"**Negative Sentiment:** {gap.get('overall_negative_ratio', 0):.1%}")
                    
                    critical_gaps = gap.get("critical_gaps", [])
                    if critical_gaps:
                        st.write(f"**Critical Issues:** {', '.join([g['aspect'] for g in critical_gaps[:3]])}")
                
                st.write("---")
            
            # Comparative analysis
            st.write("## 📊 Comparative Gap Analysis")
            
            gap_analyses = {
                name: analysis.get("gap_analysis", {})
                for name, analysis in competitor_analyses.items()
                if "gap_analysis" in analysis
            }
            
            comparison = self.gap_analyzer.compare_competitors(gap_analyses)
            
            return {
                "total_competitors": len(competitors),
                "competitor_analyses": competitor_analyses,
                "comparative_analysis": comparison,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-competitor analysis: {e}", exc_info=True)
            st.error(f"Analysis error: {str(e)}")
            return {"error": str(e)}

    def _generate_summary(self, gap_analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of gap analysis."""
        total = gap_analysis.get("total_aspects", 0)
        negative_ratio = gap_analysis.get("overall_negative_ratio", 0)
        critical_gaps = gap_analysis.get("critical_gaps", [])
        
        if total == 0:
            return "No aspects analyzed."
        
        summary_parts = [
            f"Analyzed {total} aspect mentions.",
            f"{negative_ratio:.1%} negative sentiment overall."
        ]
        
        if critical_gaps:
            top_issues = [g['aspect'] for g in critical_gaps[:3]]
            summary_parts.append(f"Critical issues: {', '.join(top_issues)}.")
        else:
            summary_parts.append("No critical gaps identified.")
        
        return " ".join(summary_parts)


def display_gap_analysis_results(analysis_result: Dict[str, Any]):
    """
    Display gap analysis results in Streamlit UI.
    
    Args:
        analysis_result: Result from ReviewAnalyzer analysis
    """
    if "error" in analysis_result:
        st.error(f"Analysis Error: {analysis_result['error']}")
        return
    
    gap = analysis_result.get("gap_analysis", {})
    
    if not gap:
        st.warning("No gap analysis available")
        return
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Reviews",
            gap.get("total_reviews_analyzed", 0)
        )
    
    with col2:
        negative_ratio = gap.get("overall_negative_ratio", 0)
        st.metric(
            "Negative Sentiment",
            f"{negative_ratio:.1%}",
            delta=None,
            delta_color="inverse"
        )
    
    with col3:
        critical_count = len(gap.get("critical_gaps", []))
        st.metric(
            "Critical Issues",
            critical_count
        )
    
    # Sentiment breakdown
    st.write("### Sentiment Distribution")
    sentiment_breakdown = gap.get("sentiment_breakdown", {})
    
    if sentiment_breakdown:
        import pandas as pd
        
        df = pd.DataFrame([
            {"Sentiment": k.capitalize(), "Count": v}
            for k, v in sentiment_breakdown.items()
        ])
        st.bar_chart(df.set_index("Sentiment"))
    
    # Critical gaps
    critical_gaps = gap.get("critical_gaps", [])
    if critical_gaps:
        st.write("### 🚨 Critical Gaps (Opportunities)")
        st.write("These are the aspects where the competitor is failing:")
        
        for gap_item in critical_gaps[:5]:
            with st.expander(
                f"**{gap_item['aspect'].replace('_', ' ').title()}** - "
                f"{gap_item['negative_ratio']:.0%} negative "
                f"({gap_item['negative_count']} mentions)"
            ):
                st.write(f"**Total Mentions:** {gap_item['total_mentions']}")
                st.write(f"**Negative:** {gap_item['negative_count']}")
                st.write(f"**Positive:** {gap_item['positive_count']}")
                
                complaints = gap_item.get("common_complaints", [])
                if complaints:
                    st.write("**Common Complaints:**")
                    for complaint in complaints:
                        st.write(f"- {complaint}")
    else:
        st.success("No critical gaps found - this competitor has strong reviews overall.")
    
    # Top positive aspects
    top_positive = gap.get("top_positive_aspects", [])
    if top_positive:
        st.write("### ✅ Competitor Strengths")
        
        for aspect in top_positive[:3]:
            st.write(
                f"- **{aspect['aspect'].replace('_', ' ').title()}**: "
                f"{aspect['positive_count']} positive mentions"
            )


def display_comparative_analysis(comparison: Dict[str, Any]):
    """
    Display comparative analysis across competitors.
    
    Args:
        comparison: Result from GapAnalyzer.compare_competitors()
    """
    st.write("## 🏆 Competitor Rankings by Weakness")
    
    rankings = comparison.get("competitor_rankings", [])
    
    if not rankings:
        st.warning("No comparison data available")
        return
    
    import pandas as pd
    
    # Create ranking table
    rank_data = []
    for rank, item in enumerate(rankings, 1):
        rank_data.append({
            "Rank": rank,
            "Competitor": item["competitor"][:40],
            "Negative %": f"{item['negative_ratio']:.1%}",
            "Critical Issues": item["critical_gap_count"],
            "Top Complaint": item.get("top_complaint", "N/A")
        })
    
    df = pd.DataFrame(rank_data)
    st.dataframe(df, width='stretch')
    
    # Weakest competitor highlight
    weakest = comparison.get("weakest_competitor")
    if weakest:
        st.write("### 🎯 Weakest Competitor (Best Opportunity)")
        st.info(
            f"**{weakest['competitor']}** has {weakest['negative_ratio']:.1%} "
            f"negative sentiment with {weakest['critical_gap_count']} critical issues."
        )
        
        if weakest.get("critical_gaps"):
            st.write("**Their weak points:**")
            for aspect in weakest["critical_gaps"][:5]:
                st.write(f"- {aspect}")
    
    # Common weaknesses
    common = comparison.get("common_weaknesses", [])
    if common:
        st.write("### 🔍 Common Industry Weaknesses")
        st.write("These aspects are problematic across multiple competitors:")
        
        for aspect in common:
            st.write(f"- {aspect.replace('_', ' ').title()}")
