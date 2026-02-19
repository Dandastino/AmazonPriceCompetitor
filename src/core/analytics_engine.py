"""Analytics engine for scoring and analyzing products and competitors."""

import logging

from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Handles all scoring and analytics logic."""

    @staticmethod
    def score_competitors(competitors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank competitors based on multiple criteria.
        
        Args:
            competitors: List of competitor dictionaries
            
        Returns:
            List of scored competitors with ranking details
        """
        if not competitors:
            return []
        
        scored_competitors = []
        
        for competitor in competitors:
            score = 0
            details = []
            missing_critical = False
            
            # 1. Reviews count (40 points max)
            reviews = competitor.get("reviews_count") or competitor.get("num_reviews") or 0
            has_reviews = isinstance(reviews, (int, float)) and reviews > 0
            if has_reviews:
                max_reviews = max([c.get("reviews_count") or c.get("num_reviews") or 0 for c in competitors]) or 1
                reviews_score = (reviews / max_reviews) * 40
                score += reviews_score
                details.append(f"👥 Bought: {int(reviews)} ({reviews_score:.0f} pts)")
            else:
                missing_critical = True
                details.append("⚠️ Missing reviews")
            
            # 2. Rating (35 points max)
            rating = competitor.get("rating", 0)
            if isinstance(rating, (int, float)):
                rating_score = (rating / 5) * 35
                score += rating_score
                details.append(f"⭐ Rating: {rating:.1f}/5 ({rating_score:.0f} pts)")
            
            # 3. Price (20 points max - lower price = higher score)
            price = competitor.get("price", 0)
            has_price = isinstance(price, (int, float)) and price > 0
            if has_price:
                max_price = max([c.get("price", 0) for c in competitors 
                               if isinstance(c.get("price"), (int, float)) and c.get("price") > 0] or [1])
                price_score = ((max_price - price) / max_price) * 20 if max_price > 0 else 0
                score += price_score
                currency = competitor.get("currency", "$")
                details.append(f"💰 Price: {currency}{price:.2f} ({price_score:.0f} pts)")
            else:
                missing_critical = True
                details.append("⚠️ Missing price")
            
            # 4. Stock availability (15 points max)
            stock = competitor.get("stock", "Unknown")
            stock_score = AnalyticsEngine._calculate_stock_score(stock)
            score += stock_score
            
            scored_competitors.append({
                "competitor": competitor,
                "score": score,
                "details": details,
                "stock_score": stock_score,
                "missing_critical": missing_critical
            })
        
        # Sort by completeness first, then score descending
        scored_competitors.sort(key=lambda x: (x.get("missing_critical", False), -x["score"]))
        return scored_competitors

    @staticmethod
    def score_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank products based on market potential.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of scored products with ranking details
        """
        if not products:
            return []
        
        scored_products = []
        
        for product in products:
            score = 0
            details = []
            
            # 1. Reviews count (40 points max)
            reviews = product.get("reviews_count") or product.get("num_reviews") or 0
            if isinstance(reviews, (int, float)) and reviews > 0:
                max_reviews = max([p.get("reviews_count") or p.get("num_reviews") or 0 for p in products]) or 1
                reviews_score = (reviews / max_reviews) * 40
                score += reviews_score
                details.append(f"👥 Bought: {int(reviews)} ({reviews_score:.0f} pts)")
            
            # 2. Rating (35 points max)
            rating = product.get("rating", 0)
            if isinstance(rating, (int, float)):
                rating_score = (rating / 5) * 35
                score += rating_score
                details.append(f"⭐ Rating: {rating:.1f}/5 ({rating_score:.0f} pts)")
            
            # 3. Price (20 points max)
            price = product.get("price", 0)
            if isinstance(price, (int, float)) and price > 0:
                max_price = max([p.get("price", 0) for p in products 
                               if isinstance(p.get("price"), (int, float)) and p.get("price") > 0] or [1])
                price_score = ((max_price - price) / max_price) * 20 if max_price > 0 else 0
                score += price_score
                currency = product.get("currency", "$")
                details.append(f"💰 Price: {currency}{price:.2f} ({price_score:.0f} pts)")
            
            # 4. Stock availability (15 points max)
            stock = product.get("stock", "Unknown")
            stock_score = AnalyticsEngine._calculate_stock_score(stock)
            score += stock_score
            
            scored_products.append({
                "product": product,
                "score": score,
                "details": details,
                "stock_score": stock_score
            })
        
        # Sort by score descending
        scored_products.sort(key=lambda x: x["score"], reverse=True)
        return scored_products

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
        """Get top N competitors from scored list.
        
        Args:
            scored_competitors: List of scored competitor dictionaries
            limit: Maximum number of competitors to return
            
        Returns:
            Top N competitors (already sorted)
        """
        return scored_competitors[:limit]

    @staticmethod
    def get_top_products(scored_products: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top N products from scored list.
        
        Args:
            scored_products: List of scored product dictionaries
            limit: Maximum number of products to return
            
        Returns:
            Top N products (already sorted)
        """
        return scored_products[:limit]

    @staticmethod
    def calculate_competitor_stats(competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics for a list of competitors.
        
        Args:
            competitors: List of competitor dictionaries
            
        Returns:
            Dictionary with aggregate stats
        """
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
