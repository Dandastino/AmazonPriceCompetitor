"""
Sentiment Analysis and Aspect-Based Sentiment Analysis (ABSA) Module

This module provides:
1. Text preprocessing using NLTK
2. LLM-based aspect extraction using OpenAI (GPT-4o-mini)
3. Gap analysis for competitor reviews
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from src.utils import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class TextPreprocessor:
    """Clean and preprocess review text using NLTK."""
    
    def __init__(self):
        """Initialize preprocessor with NLTK resources."""
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.punkt_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK resources: {e}")
            self.stop_words = set()
            self.lemmatizer = None
            self.punkt_available = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing noise and normalizing.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Use simple regex-based splitting if punkt is not available
        if not self.punkt_available:
            # Simple fallback: split by sentence-ending punctuation
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.debug(f"Failed to tokenize with NLTK (using fallback): {str(e)[:50]}")
            # Fallback to simple split
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Preprocessed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        if remove_stopwords and self.stop_words:
            # Tokenize
            words = word_tokenize(cleaned)
            
            # Remove stopwords
            words = [w for w in words if w not in self.stop_words]
            
            # Lemmatize if available
            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(w) for w in words]
            
            cleaned = ' '.join(words)
        
        return cleaned
    
    def extract_sentences_with_keywords(
        self, 
        text: str, 
        keywords: List[str]
    ) -> List[str]:
        """
        Extract sentences containing specific keywords.
        
        Args:
            text: Full review text
            keywords: List of keywords to search for
            
        Returns:
            List of sentences containing keywords
        """
        sentences = self.tokenize_sentences(text)
        matching_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                matching_sentences.append(sentence)
        
        return matching_sentences


class LLMBasedABSA:
    """
    Option B: LLM-based Aspect-Based Sentiment Analysis using OpenAI.
    
    Uses structured prompts to extract aspects and sentiments from reviews.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM-based ABSA.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        try:
            from openai import OpenAI
            import os
            
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized for ABSA")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def extract_aspects(
        self, 
        review_text: str,
        product_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract aspects and their sentiments from a review using LLM.
        
        Args:
            review_text: The review text to analyze
            product_category: Optional product category for context
            
        Returns:
            List of dicts with aspect, sentiment, and reason
            Example: [{"aspect": "battery", "sentiment": "negative", "reason": "short life"}]
        """
        if not self.client:
            logger.warning("OpenAI client not available for aspect extraction")
            return []
        
        # Build the prompt
        category_context = f" for a {product_category} product" if product_category else ""
        
        prompt = f"""Analyze this product review{category_context} and extract specific aspects mentioned along with their sentiment.

Review: "{review_text}"

Extract aspects as a JSON array where each item has:
- "aspect": the specific product feature/attribute mentioned (e.g., "battery", "packaging", "quality", "price")
- "sentiment": either "positive", "negative", or "neutral"
- "reason": a brief phrase explaining why (3-5 words)

Focus on tangible product attributes like: quality, price, packaging, durability, design, performance, size, material, shipping, value, etc.

Return ONLY valid JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing product reviews and extracting specific aspects with their sentiments. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*|\s*```', '', content)
            
            aspects = json.loads(content)
            
            # Validate structure
            if not isinstance(aspects, list):
                logger.warning("LLM returned non-list response")
                return []
            
            # Ensure all required fields
            validated_aspects = []
            for aspect in aspects:
                if isinstance(aspect, dict) and all(k in aspect for k in ['aspect', 'sentiment', 'reason']):
                    # Normalize sentiment
                    aspect['sentiment'] = aspect['sentiment'].lower()
                    if aspect['sentiment'] not in ['positive', 'negative', 'neutral']:
                        aspect['sentiment'] = 'neutral'
                    validated_aspects.append(aspect)
            
            return validated_aspects
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Aspect extraction failed: {e}")
            return []
    
    def analyze_reviews_batch(
        self, 
        reviews: List[str],
        product_category: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract aspects from multiple reviews.
        
        Args:
            reviews: List of review texts
            product_category: Optional product category
            
        Returns:
            List of aspect lists (one per review)
        """
        results = []
        for review in reviews:
            aspects = self.extract_aspects(review, product_category)
            results.append(aspects)
        
        return results


class GapAnalyzer:
    """
    Perform gap analysis on competitor reviews to identify weaknesses.
    Phase 3: Gap Analysis Logic with aspect scoring (0-1 scale).
    """
    
    # Common aspect categories for scoring
    COMMON_ASPECTS = [
        'price', 'quality', 'shipping', 'packaging', 'customer_service',
        'value', 'durability', 'design', 'performance', 'battery',
        'material', 'size', 'fit', 'comfort', 'delivery'
    ]
    
    def __init__(self):
        """Initialize gap analyzer."""
        self.preprocessor = TextPreprocessor()
    
    def calculate_aspect_scores(
        self,
        aspects_by_review: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Calculate sentiment scores (0 to 1) for each aspect category.
        
        Phase 3 Implementation:
        - 0.0 = 100% negative sentiment
        - 0.5 = neutral
        - 1.0 = 100% positive sentiment
        
        Args:
            aspects_by_review: List of aspect lists from each review
            
        Returns:
            Dict mapping aspect names to scores (0-1)
        """
        # Flatten all aspects
        all_aspects = [
            aspect 
            for review_aspects in aspects_by_review 
            for aspect in review_aspects
        ]
        
        if not all_aspects:
            return {}
        
        # Group by aspect
        aspect_groups = defaultdict(list)
        for aspect in all_aspects:
            aspect_name = aspect['aspect'].lower()
            aspect_groups[aspect_name].append(aspect)
        
        # Calculate scores for each aspect
        aspect_scores = {}
        for aspect_name, occurrences in aspect_groups.items():
            sentiment_counts = Counter(a['sentiment'] for a in occurrences)
            
            positive = sentiment_counts.get('positive', 0)
            negative = sentiment_counts.get('negative', 0)
            neutral = sentiment_counts.get('neutral', 0)
            total = len(occurrences)
            
            if total == 0:
                continue
            
            # Calculate score: weighted average
            # Positive = 1.0, Neutral = 0.5, Negative = 0.0
            score = (positive * 1.0 + neutral * 0.5 + negative * 0.0) / total
            aspect_scores[aspect_name] = round(score, 2)
        
        return aspect_scores
    
    def identify_gaps(
        self,
        your_product_scores: Dict[str, float],
        competitor_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Identify gaps between your product and competitor.
        
        Phase 3 Implementation:
        Compares lowest scores and identifies opportunities.
        
        Args:
            your_product_scores: Your product's aspect scores
            competitor_scores: Competitor's aspect scores
            
        Returns:
            Gap analysis with opportunities and threats
        """
        # Find common aspects
        common_aspects = set(your_product_scores.keys()) & set(competitor_scores.keys())
        
        opportunities = []  # Where competitor is weak and you're strong
        threats = []  # Where competitor is strong and you're weak
        shared_weaknesses = []  # Where both are weak
        shared_strengths = []  # Where both are strong
        
        for aspect in common_aspects:
            your_score = your_product_scores[aspect]
            competitor_score = competitor_scores[aspect]
            
            gap = your_score - competitor_score
            
            # Opportunity: You're significantly better
            if gap > 0.15:
                opportunities.append({
                    'aspect': aspect,
                    'your_score': your_score,
                    'competitor_score': competitor_score,
                    'gap': gap,
                    'insight': f"You excel at {aspect} (yours: {your_score:.1f}, theirs: {competitor_score:.1f})"
                })
            
            # Threat: They're significantly better
            elif gap < -0.15:
                threats.append({
                    'aspect': aspect,
                    'your_score': your_score,
                    'competitor_score': competitor_score,
                    'gap': gap,
                    'insight': f"Competitor stronger in {aspect} (yours: {your_score:.1f}, theirs: {competitor_score:.1f})"
                })
            
            # Shared weakness: Both low scores
            elif your_score < 0.5 and competitor_score < 0.5:
                shared_weaknesses.append({
                    'aspect': aspect,
                    'your_score': your_score,
                    'competitor_score': competitor_score,
                    'insight': f"Industry pain point: {aspect} (yours: {your_score:.1f}, theirs: {competitor_score:.1f})"
                })
            
            # Shared strength: Both high scores
            elif your_score > 0.7 and competitor_score > 0.7:
                shared_strengths.append({
                    'aspect': aspect,
                    'your_score': your_score,
                    'competitor_score': competitor_score,
                    'insight': f"Both excel at {aspect}"
                })
        
        # Sort by gap magnitude
        opportunities.sort(key=lambda x: x['gap'], reverse=True)
        threats.sort(key=lambda x: abs(x['gap']), reverse=True)
        
        # Find your lowest scores
        your_weakest = sorted(
            [(aspect, score) for aspect, score in your_product_scores.items()],
            key=lambda x: x[1]
        )[:5]
        
        # Find competitor's lowest scores
        competitor_weakest = sorted(
            [(aspect, score) for aspect, score in competitor_scores.items()],
            key=lambda x: x[1]
        )[:5]
        
        return {
            'opportunities': opportunities[:5],
            'threats': threats[:5],
            'shared_weaknesses': shared_weaknesses,
            'shared_strengths': shared_strengths,
            'your_weakest_aspects': [{'aspect': a, 'score': s} for a, s in your_weakest],
            'competitor_weakest_aspects': [{'aspect': a, 'score': s} for a, s in competitor_weakest],
            'summary': self._generate_gap_summary(opportunities, threats, shared_weaknesses)
        }
    
    def _generate_gap_summary(
        self,
        opportunities: List[Dict],
        threats: List[Dict],
        shared_weaknesses: List[Dict]
    ) -> str:
        """Generate human-readable gap summary."""
        parts = []
        
        if opportunities:
            top_opp = opportunities[0]
            parts.append(
                f"Your strength: {top_opp['aspect']} "
                f"(you: {top_opp['your_score']:.1f} vs competitor: {top_opp['competitor_score']:.1f})"
            )
        
        if threats:
            top_threat = threats[0]
            parts.append(
                f"Competitor advantage: {top_threat['aspect']} "
                f"(theirs: {top_threat['competitor_score']:.1f} vs yours: {top_threat['your_score']:.1f})"
            )
        
        if shared_weaknesses:
            weak_aspects = [w['aspect'] for w in shared_weaknesses[:2]]
            parts.append(f"Common pain points: {', '.join(weak_aspects)}")
        
        return ". ".join(parts) if parts else "No significant gaps identified."
    
    def analyze_aspects(
        self, 
        aspects_by_review: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Aggregate and analyze aspects from multiple reviews.
        
        Args:
            aspects_by_review: List of aspect lists from each review
            
        Returns:
            Gap analysis summary with top negative aspects and aspect scores
        """
        # Flatten all aspects
        all_aspects = [
            aspect 
            for review_aspects in aspects_by_review 
            for aspect in review_aspects
        ]
        
        if not all_aspects:
            return {
                "total_aspects": 0,
                "sentiment_breakdown": {},
                "top_negative_aspects": [],
                "top_positive_aspects": [],
                "critical_gaps": [],
                "aspect_scores": {}
            }
        
        # Count sentiments
        sentiment_counts = Counter(a['sentiment'] for a in all_aspects)
        
        # Group by aspect
        aspect_groups = defaultdict(list)
        for aspect in all_aspects:
            aspect_name = aspect['aspect'].lower()
            aspect_groups[aspect_name].append(aspect)
        
        # Analyze each aspect
        aspect_analysis = []
        for aspect_name, occurrences in aspect_groups.items():
            sentiment_breakdown = Counter(a['sentiment'] for a in occurrences)
            
            negative_count = sentiment_breakdown.get('negative', 0)
            positive_count = sentiment_breakdown.get('positive', 0)
            neutral_count = sentiment_breakdown.get('neutral', 0)
            total_count = len(occurrences)
            
            # Calculate negative ratio
            negative_ratio = negative_count / total_count if total_count > 0 else 0
            
            # Get common reasons
            reasons = [a.get('reason', '') for a in occurrences if a['sentiment'] == 'negative']
            
            aspect_analysis.append({
                "aspect": aspect_name,
                "total_mentions": total_count,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "neutral_count": neutral_count,
                "negative_ratio": negative_ratio,
                "common_complaints": list(set(reasons))[:3]
            })
        
        # Sort by negative ratio and count
        aspect_analysis.sort(
            key=lambda x: (x['negative_ratio'], x['negative_count']), 
            reverse=True
        )
        
        # Identify critical gaps (high negative ratio and high mention count)
        critical_gaps = [
            a for a in aspect_analysis 
            if a['negative_ratio'] > 0.6 and a['total_mentions'] >= 3
        ]
        
        # Calculate aspect scores (0-1 scale)
        aspect_scores = self.calculate_aspect_scores(aspects_by_review)
        
        return {
            "total_aspects": len(all_aspects),
            "total_reviews_analyzed": len(aspects_by_review),
            "sentiment_breakdown": {
                "positive": sentiment_counts.get('positive', 0),
                "negative": sentiment_counts.get('negative', 0),
                "neutral": sentiment_counts.get('neutral', 0)
            },
            "top_negative_aspects": aspect_analysis[:10],
            "top_positive_aspects": sorted(
                aspect_analysis, 
                key=lambda x: (x['positive_count'], -x['negative_ratio']), 
                reverse=True
            )[:5],
            "critical_gaps": critical_gaps,
            "overall_negative_ratio": sentiment_counts.get('negative', 0) / len(all_aspects) if all_aspects else 0,
            "aspect_scores": aspect_scores  # Phase 3: Aspect scores (0-1)
        }
    
    def compare_competitors(
        self, 
        competitor_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare gap analyses across multiple competitors.
        
        Args:
            competitor_analyses: Dict mapping competitor name to their gap analysis
            
        Returns:
            Comparative analysis showing weakest competitors
        """
        comparisons = []
        
        for competitor_name, analysis in competitor_analyses.items():
            negative_ratio = analysis.get('overall_negative_ratio', 0)
            critical_gaps = analysis.get('critical_gaps', [])
            
            comparisons.append({
                "competitor": competitor_name,
                "negative_ratio": negative_ratio,
                "critical_gap_count": len(critical_gaps),
                "critical_gaps": [g['aspect'] for g in critical_gaps],
                "top_complaint": critical_gaps[0]['aspect'] if critical_gaps else None
            })
        
        # Sort by negative ratio
        comparisons.sort(key=lambda x: x['negative_ratio'], reverse=True)
        
        return {
            "competitor_rankings": comparisons,
            "weakest_competitor": comparisons[0] if comparisons else None,
            "strongest_competitor": comparisons[-1] if comparisons else None,
            "common_weaknesses": self._find_common_weaknesses(competitor_analyses)
        }
    
    def _find_common_weaknesses(
        self, 
        competitor_analyses: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Find aspects that are negative across multiple competitors."""
        aspect_mentions = defaultdict(int)
        
        for analysis in competitor_analyses.values():
            critical_gaps = analysis.get('critical_gaps', [])
            for gap in critical_gaps:
                aspect_mentions[gap['aspect']] += 1
        
        # Return aspects mentioned negatively by multiple competitors
        common = [
            aspect 
            for aspect, count in aspect_mentions.items() 
            if count >= 2
        ]
        
        return sorted(common, key=lambda x: aspect_mentions[x], reverse=True)


# Factory function to create LLM-based sentiment analyzer
def create_sentiment_analyzer() -> LLMBasedABSA:
    """
    Factory function to create sentiment analyzer.
    
    Returns:
        LLM-based ABSA instance
    """
    return LLMBasedABSA()
