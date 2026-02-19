"""Building the Regression Model"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.core import MongoDB
from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PriceAnalytics:
    """Machine Learning model for predicting fair prices using competitor analysis."""

    def __init__(self, model_type: str = "random_forest", test_size: float = 0.2):
        """Initialize the price analytics engine."""
        self.db = MongoDB()
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.feature_names = ["price", "rating", "reviews_log", "in_stock"]
        self.is_trained = False
        self.training_metrics = {}

        if self.model_type not in ["linear", "random_forest"]:
            logger.warning(f"Invalid model type '{model_type}', defaulting to 'random_forest'")
            self.model_type = "random_forest"

    def extract_features(self, product: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numeric features from a product for ML model."""
        try:
            # Extract raw values with defaults
            price = float(product.get("price", 0) or 0)
            rating = float(product.get("rating", 0) or 0)
            reviews_count = int(product.get("reviews_count", 0) or 0)
            in_stock = 1 if product.get("in_stock", False) else 0
            
            if price <= 0:
                return None
                
            # Log-transform review count to handle wide range (0 to thousands)
            reviews_log = np.log1p(reviews_count)
            
            features = [
                price,
                rating,
                reviews_log,
                in_stock
            ]
            
            return features
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Error extracting features from product: {e}")
            return None

    def prepare_dataset(self, category_filter: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load competitors from database and prepare features & targets for training."""
        try:
            # Get all products from database
            all_products = self.db.get_all_products()
            
            if not all_products:
                logger.warning("No products in database for training")
                return None, None
            
            # Filter by category if specified
            if category_filter:
                products = [p for p in all_products 
                           if category_filter.lower() in str(p.get("categories", [])).lower()]
            else:
                products = all_products
            
            if len(products) < 3: 
                logger.warning(f"Insufficient products for training: {len(products)} < 3")
                return None, None
            
            X_list = [] # List of feature vectors
            y_list = [] # List of target prices
            
            for product in products:
                features = self.extract_features(product)
                if features is not None:
                    X_list.append(features)
                    y_list.append(float(product.get("price", 0)))
            
            if len(X_list) < 3:
                logger.warning(f"Insufficient valid products for training: {len(X_list)} < 3")
                return None, None
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            logger.info(f"Dataset prepared: {len(X)} products with features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}", exc_info=True)
            return None, None

    def train(self, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Train the regression model on competitor data."""
        try:
            # Prepare data
            X, y = self.prepare_dataset(category_filter)
            
            if X is None or y is None:
                logger.error("Failed to prepare dataset for training")
                return {"success": False, "error": "Insufficient data"}
            
            # Split into train and test sets (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            
            logger.info(f"Train/test split: {len(X_train)} training, {len(X_test)} test samples")
            
            # Scale features for better model performance
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            if self.model_type == "linear":
                self.model = LinearRegression()
            else:  # random_forest
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train_scaled, y_train)
            logger.info(f"Model trained successfully ({self.model_type})")
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store metrics
            self.training_metrics = {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "model_type": self.model_type
            }
            
            self.is_trained = True
            
            logger.info(f"Model evaluation - R²: {r2:.4f}, RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")
            
            return {
                "success": True,
                "metrics": self.training_metrics,
                "message": f"Model trained on {len(X_train)} products"
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def predict_fair_price(self, product: Dict[str, Any]) -> Optional[float]:
        """Predict the fair market price for a product based on its features."""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained yet. Call train() first.")
                return None
            
            # Extract features
            features = self.extract_features(product)
            if features is None:
                logger.warning(f"Could not extract features from product")
                return None
            
            # Scale and predict
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            predicted_price = float(self.model.predict(X_scaled)[0])
            
            # Ensure positive price
            predicted_price = max(predicted_price, 0.01)
            
            logger.debug(f"Predicted fair price: ${predicted_price:.2f}")
            return round(predicted_price, 2)
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}", exc_info=True)
            return None

    def predict_fair_price_with_overrides(
        self,
        product: Dict[str, Any],
        rating: Optional[float] = None,
        reviews_count: Optional[int] = None,
    ) -> Optional[float]:
        """Predict fair price with optional rating/reviews overrides for what-if analysis."""
        try:
            updated = dict(product)
            if rating is not None:
                updated["rating"] = rating
            if reviews_count is not None:
                updated["reviews_count"] = reviews_count
            return self.predict_fair_price(updated)
        except Exception as e:
            logger.error(f"Error predicting price with overrides: {e}", exc_info=True)
            return None

    def get_feature_importances(self) -> Optional[List[Dict[str, Any]]]:
        """Return feature importances when supported by the model."""
        try:
            if not self.is_trained or self.model is None:
                return None
            if not hasattr(self.model, "feature_importances_"):
                return None

            importances = list(self.model.feature_importances_)
            names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]

            results = []
            for name, value in zip(names, importances):
                results.append({
                    "feature": name,
                    "importance": float(value),
                    "importance_pct": round(float(value) * 100, 2)
                })

            results.sort(key=lambda item: item["importance"], reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error getting feature importances: {e}", exc_info=True)
            return None

    def get_price_recommendation(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a detailed price recommendation for a product."""
        try:
            current_price = float(product.get("price", 0) or 0)
            fair_price = self.predict_fair_price(product)
            
            if fair_price is None or current_price <= 0:
                return None
            
            difference = current_price - fair_price
            deviation_pct = (difference / fair_price) * 100

            # Business logic thresholds based on deviation from fair price
            if deviation_pct > 10:
                recommendation = "Overpriced - Risk of losing sales."
            elif deviation_pct < -10:
                recommendation = "Underpriced - Potential to increase margin."
            elif -5 <= deviation_pct <= 5:
                recommendation = "Optimally Priced - Competitive."
            elif deviation_pct > 0:
                recommendation = "Slightly Overpriced - Monitor."
            else:
                recommendation = "Slightly Underpriced - Monitor."
            
            return {
                "current_price": round(current_price, 2),
                "fair_price": fair_price,
                "difference": round(difference, 2),
                "percentage_diff": round(deviation_pct, 2),
                "deviation_pct": round(deviation_pct, 2),
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}", exc_info=True)
            return None

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get the training metrics from the last model training."""
        return self.training_metrics

    def is_model_trained(self) -> bool:
        """Check if model has been trained."""
        return self.is_trained
