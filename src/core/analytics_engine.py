"""Analytics engine for scoring and analyzing products and competitors."""

import logging
import math

from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Handles all scoring and analytics logic."""

    @staticmethod
    def score_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank competitors based on multiple criteria."""
        if not competitors:
            return []
        
        scored_competitors = []
        
        prices = [float(c.get("price")) for c in competitors if c.get("price") is not None and float(c.get("price") or 0) > 0]
        avg_price = sum(prices) / len(prices) if prices else 1.0

        def get_reviews(c):
            rev = c.get("review_count") or c.get("reviews_count") or c.get("num_reviews")
            try:
                return float(rev) if rev is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        all_reviews = [get_reviews(c) for c in competitors]
        valid_reviews = [r for r in all_reviews if r > 0]
        max_reviews = max(valid_reviews) if valid_reviews else 1.0
        average_reviews = sum(valid_reviews[:10]) / max(len(valid_reviews[:10]), 1)
        base_rating_constant = max(average_reviews, 10.0)

        for original_index, competitor in enumerate(competitors):
            score = 0.0
            details = []
            
            reviews = get_reviews(competitor)
            
            try:
                rating = float(competitor.get("rating") or 0)
            except (ValueError, TypeError):
                rating = 0.0
                
            try:
                price = float(competitor.get("price") or 0)
            except (ValueError, TypeError):
                price = 0.0

            has_valid_reviews = reviews > 0
            has_valid_rating = rating > 0
            has_valid_price = price > 0

            # Zero Tolleranza: se manca un dato fondamentale, si becca 0 per evitare "fantasmi"
            if not (has_valid_reviews and has_valid_rating and has_valid_price):
                missing_critical = True
                if not has_valid_reviews:
                    details.append("Zero Tolleranza: Missing reviews (0 pts)")
                if not has_valid_rating:
                    details.append("Zero Tolleranza: Missing rating (0 pts)")
                if not has_valid_price:
                    details.append("Zero Tolleranza: Missing price (0 pts)")
            else:
                missing_critical = False
                
                # Calcolo del Premium Value Index
                pvi = price / avg_price
                effective_pvi = max(0.5, min(pvi, 2.0))
                
                # adjusted_constant: un prodotto costoso (alto PVI) avrà una costante più piccola,
                # rendendo le se sue 4.7 stelle molto "pesanti" e solide anche con meno recensioni.
                # Al contrario, un prodotto molto cheap avrà una costante grande, e per pesare
                # tanto avrà bisogno di valangate di recensioni per dimostrare vera qualità.
                adjusted_constant = base_rating_constant / effective_pvi
                
                # 1. Logarithmic reviews (40 points max)
                reviews_score = (math.log10(reviews + 1) / math.log10(max_reviews + 1)) * 40
                score += reviews_score
                details.append(f"Reviews: {int(reviews)} ({reviews_score:.0f} pts)")
                
                # 2. Rating weighted con adjusted_constant (35 points max)
                weighted_rating = (rating * reviews) / (reviews + adjusted_constant)
                rating_score = (weighted_rating / 5) * 35
                score += rating_score
                details.append(f"Weighted rating (Adj.C={adjusted_constant:.0f}): {weighted_rating:.2f}/5 ({rating_score:.0f} pts)")
                
                # 3. Premium Value Index Score (25 points max)
                # Al posto di premiare chi costa meno, premiamo proporzionalmente
                # un alto prezzo MANTENUTO con alto grado di rating (PVI "messo a frutto").
                pvi_normalized = effective_pvi / 2.0
                pvi_score = pvi_normalized * (weighted_rating / 5) * 25
                score += pvi_score
                currency = competitor.get("currency", "€")
                details.append(f"Premium Value Index: {pvi:.2f}x media ({pvi_score:.0f} pts)")
                        
            # Tie-breaker infinitesimale
            tie_breaker = (1 / (original_index + 1)) * 0.001
            score += tie_breaker

            scored_competitors.append({
                "competitor": competitor,
                "score": score,
                "details": details,
                "missing_critical": missing_critical,
                "tie_breaker": tie_breaker
            })
        
        # Sort by completeness first, then score descending
        scored_competitors.sort(key=lambda x: (x.get("missing_critical", False), -x["score"]))
        
        # Normalize scores so the best competitor always has maximum score (100)
        valid_competitors = [c for c in scored_competitors if not c.get("missing_critical", False)]
        if valid_competitors:
            max_score = valid_competitors[0]["score"]
            for competitor in scored_competitors:
                if competitor.get("missing_critical", False):
                    competitor["score"] = 0
                else:
                    competitor["score"] = (competitor["score"] / max_score) * 100
        else:
            for competitor in scored_competitors:
                competitor["score"] = 0
        
        return scored_competitors


    @staticmethod
    def _calculate_stock_score(stock: Any) -> float:
        """Calculate stock score based on availability."""
        if not isinstance(stock, str):
            stock = str(stock)
        
        stock_lower = stock.lower()
        
        if stock_lower in ["in stock", "available"]:
            return 15
        elif stock_lower in ["out of stock"]:
            return 3
        else:
            return 10

    @staticmethod
    def get_top_competitors(scored_competitors: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N competitors from scored list."""
        return scored_competitors[:limit]

    @staticmethod
    def calculate_competitor_stats(competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics for a list of competitors."""
        if not competitors:
            return {
                "total": 0,
                "avg_price": 0,
                "avg_rating": 0,
                "avg_reviews": 0,
                "in_stock_count": 0
            }
        
        total = len(competitors)
        
        # Average price
        prices = [c.get("price", 0) for c in competitors 
                 if isinstance(c.get("price"), (int, float)) and c.get("price") > 0]
        avg_price = sum(prices) / len(prices) if prices else 0
        
        # Average rating
        ratings = [c.get("rating", 0) for c in competitors 
                  if isinstance(c.get("rating"), (int, float))]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Average reviews
        reviews = [c.get("reviews_count") or c.get("num_reviews") or 0 for c in competitors]
        avg_reviews = sum(reviews) / len(reviews) if reviews else 0
        
        # In stock count
        in_stock = sum(1 for c in competitors 
                      if isinstance(c.get("stock"), str) and c.get("stock").lower() in ["in stock", "available"])
        
        return {
            "total": total,
            "avg_price": avg_price,
            "avg_rating": avg_rating,
            "avg_reviews": avg_reviews,
            "in_stock_count": in_stock,
            "in_stock_percentage": (in_stock / total * 100) if total > 0 else 0
        }
