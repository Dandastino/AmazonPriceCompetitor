import numpy as np
import pandas as pd
import logging

from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.utils import get_logger, safe_float, safe_int

logger = get_logger(__name__)


class FeatureExtractor:
    """Prepares features from already-normalized product data (DRY principle)."""

    def __init__(self, use_frequency_encoding: bool = True):
        """Initialize feature extractor."""
        self.use_frequency_encoding = use_frequency_encoding
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.brand_frequency_map: Dict[str, float] = {}
        self.feature_columns: List[str] = []
        
    def extract_features_from_db(self, db_products: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features from TinyDB product records (already normalized by oxylab_client)."""
        data = []
        
        for product in db_products:
            try:
                features = self._prepare_product_features(product)
                data.append(features)
            except Exception as e:
                logger.warning(f"Error processing product {product.get('asin', 'unknown')}: {e}")
                continue
        
        if not data:
            raise ValueError("No valid products found for feature extraction")
        
        df = pd.DataFrame(data)
        
        # Separate target variable (price) from features
        if 'price' not in df.columns:
            raise ValueError("Price column not found in extracted features")
        
        X = df.drop(columns=['price', 'asin'])
        y = df['price']
        
        # Store feature column names for later use
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def _prepare_product_features(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare features from already-normalized product (from oxylab_client)."""
        return {
            'asin': product.get('asin', ''),
            'rating': self._get_rating(product),
            'review_count': self._get_review_count(product),
            'category_rank': self._get_category_rank(product),
            'is_prime': self._get_is_prime(product),
            'brand': product.get('brand', 'Unknown'),
            'price': self._get_price(product),
        }
    
    def _get_rating(self, product: Dict[str, Any]) -> float:
        """Get rating from already-extracted product data."""
        rating = safe_float(product.get('rating'), default=0.0)
        # Validate range (0-5 or 0-10)
        return max(0.0, min(rating, 10.0))
    
    def _get_review_count(self, product: Dict[str, Any]) -> int:
        """Get review count from already-extracted product data."""
        review_count = safe_int(product.get('review_count'), default=0)
        return max(0, review_count)
    
    def _get_category_rank(self, product: Dict[str, Any]) -> int:
        """Get category rank from already-extracted product data."""
        rank = safe_int(product.get('category_rank'), default=0)
        return max(0, rank)
    
    def _get_is_prime(self, product: Dict[str, Any]) -> int:
        """Determine Prime eligibility from seller info (already extracted)."""
        seller = product.get('seller_name', '').lower()
        # amazon.com typically means Prime eligible
        return 1 if seller == 'amazon.com' else 0
    
    def _get_price(self, product: Dict[str, Any]) -> float:
        """Get price from already-extracted product data."""
        price = safe_float(product.get('price'), default=np.nan)
        # Reject invalid prices (0 or negative means unavailable)
        if price <= 0:
            return np.nan
        return price
    
    def encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and handle missing values."""
        X_processed = X.copy()
        
        # Handle missing numerical values (impute with mean if data is sparse)
        numerical_cols = ['rating', 'review_count', 'category_rank']
        imputer = SimpleImputer(strategy='mean')
        X_processed[numerical_cols] = imputer.fit_transform(X_processed[numerical_cols])
        
        # Encode brand (categorical)
        if 'brand' in X_processed.columns:
            if self.use_frequency_encoding:
                X_processed['brand'] = self._frequency_encode_brand(X_processed['brand'])
            else:
                X_processed['brand'] = self.label_encoder.fit_transform(X_processed['brand'])
        
        # is_prime is already binary (0 or 1)
        
        return X_processed
    
    def _frequency_encode_brand(self, brands: pd.Series) -> pd.Series:
        """Frequency encode brand: higher frequency = higher value."""
        if not self.brand_frequency_map:
            # Build frequency map from data
            brand_counts = brands.value_counts(normalize=True)
            self.brand_frequency_map = brand_counts.to_dict()
        
        return brands.map(self.brand_frequency_map)
    
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features using StandardScaler for ML model compatibility."""
        return self.scaler.fit_transform(X)
    
    def process_pipeline(
        self, 
        db_products: List[Dict[str, Any]], 
        remove_missing_targets: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Complete feature processing pipeline (no data re-extraction)."""
        logger.info(f"Processing {len(db_products)} products (no re-extraction)...")
        
        # Prepare features from already-normalized data
        X, y = self.extract_features_from_db(db_products)
        logger.info(f"Prepared features for {len(X)} products")
        
        # Remove rows with missing target (price)
        if remove_missing_targets:
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            logger.info(f"Retained {len(X)} products with valid prices")
        
        # Encode features
        X_encoded = self.encode_features(X)
        logger.info("Encoded categorical features (brand, is_prime)")
        
        # Scale features
        X_scaled = self.scale_features(X_encoded)
        logger.info("Scaled numerical features with StandardScaler")
        
        # Remove any NaN from y
        y_clean = y.values[~np.isnan(y.values)]
        
        return X_scaled[:len(y_clean)], y_clean


def prepare_data_for_model(
    db_instance,
    use_frequency_encoding: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to prepare data from TinyDB instance."""
    try:
        # Get products from database (already extracted by oxylab_client)
        products = db_instance.products.all()
        
        # Initialize extractor and process
        extractor = FeatureExtractor(use_frequency_encoding=use_frequency_encoding)
        X_scaled, y = extractor.process_pipeline(
            [p.dict() for p in products],
            remove_missing_targets=True
        )
        
        logger.info(f"Data preparation complete. Features: {X_scaled.shape}, Targets: {y.shape}")
        return X_scaled, y
        
    except Exception as e:
        logger.error(f"Error in data preparation: {e}", exc_info=True)
        raise
