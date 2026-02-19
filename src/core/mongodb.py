"""
MongoDB Database Wrapper for Amazon Price Competitor Analysis.

This module provides a clean separation between:
1. Products Collection: Only the user's main products
2. Competitors Collection: Competitors for each main product

Key design principles:
- Main products don't have 'parent_asin' field
- Competitors always have 'parent_asin' linking to main product
- Can only find competitors for main products (not recursive)
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from dotenv import load_dotenv

from src.utils.common_utils import (
    get_logger,
    validate_not_empty,
    validate_string,
    add_timestamps,
    handle_exception
)

load_dotenv()

logger = get_logger(__name__)


class MongoDB:
    """
    MongoDB wrapper with separate collections for products and competitors.
    
    Collections:
    - products: User's main products (what they're selling/tracking)
    - competitors: Competitor products (linked to main products via parent_asin)
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "amazon_competitor_db"
    ):
        """
        Initialize MongoDB connection.

        Args:
            connection_string: MongoDB connection string (defaults to env var)
            database_name: Name of the database
        """
        try:
            # Get connection string from env or parameter
            self.connection_string = connection_string or os.getenv(
                "MONGODB_URI",
                "mongodb://localhost:27017/"
            )
            
            # Connect to MongoDB
            self.client = MongoClient(self.connection_string)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB successfully")
            
            # Get database
            self.db: MongoDatabase = self.client[database_name]
            
            # Get collections
            self.products: Collection = self.db["products"]
            self.competitors: Collection = self.db["competitors"]
            
            # Create indexes for performance
            self._create_indexes()
            
            logger.info(f"MongoDB initialized: {database_name}")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(
                "Cannot connect to MongoDB. Ensure MongoDB is running "
                "(use 'docker-compose up -d' if using Docker)"
            )
        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            raise

    def _create_indexes(self) -> None:
        """Create database indexes for better query performance."""
        try:
            # Products collection indexes
            self.products.create_index([("asin", ASCENDING)], unique=True)
            self.products.create_index([("created_at", DESCENDING)])
            self.products.create_index([("title", ASCENDING)])
            
            # Competitors collection indexes
            self.competitors.create_index([("asin", ASCENDING)])
            self.competitors.create_index([("parent_asin", ASCENDING)])
            self.competitors.create_index(
                [("parent_asin", ASCENDING), ("asin", ASCENDING)],
                unique=True  # Prevent duplicate competitor for same parent
            )
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    # ========================================================================
    # MAIN PRODUCTS METHODS (User's Products)
    # ========================================================================

    def add_product(self, product_data: Dict[str, Any]) -> str:
        """
        Add a new main product (user's product).

        Args:
            product_data: Product information dict

        Returns:
            Product ASIN

        Raises:
            ValueError: Invalid product data
            DuplicateKeyError: Product already exists
        """
        try:
            # Validate product data
            validate_not_empty(product_data, "product_data", dict)
            validate_not_empty(product_data.get("asin"), "ASIN", str)

            # Ensure this is marked as a main product
            product_data["is_main_product"] = True
            
            # Remove parent_asin if somehow present (main products don't have parents)
            product_data.pop("parent_asin", None)

            # Add timestamps
            product_data = add_timestamps(product_data, created=True, updated=True)

            # Insert into products collection
            result = self.products.insert_one(product_data)
            asin = product_data["asin"]
            
            logger.info(f"Main product added: ASIN={asin}, ID={result.inserted_id}")
            return asin

        except DuplicateKeyError:
            asin = product_data.get("asin", "unknown")
            error_msg = f"Product {asin} already exists"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        except ValueError as e:
            logger.error(f"Validation error adding product: {e}")
            raise
        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            raise

    def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Get a main product by ASIN.

        Args:
            asin: Product ASIN

        Returns:
            Product dict or None if not found
        """
        try:
            validate_string(asin, "ASIN", allow_empty=False)
            
            product = self.products.find_one({"asin": asin})
            
            if product:
                # Remove MongoDB _id for cleaner output
                product.pop("_id", None)
                logger.debug(f"Found main product: {asin}")
            else:
                logger.debug(f"Main product not found: {asin}")
            
            return product

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return None

    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get all main products (user's products only, no competitors).

        Returns:
            List of main product dicts
        """
        try:
            products = list(self.products.find(
                {"is_main_product": True}
            ).sort("created_at", DESCENDING))
            
            # Remove MongoDB _id fields
            for product in products:
                product.pop("_id", None)
            
            logger.debug(f"Retrieved {len(products)} main products")
            return products

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return []

    def update_product(self, asin: str, updates: Dict[str, Any]) -> bool:
        """
        Update a main product.

        Args:
            asin: Product ASIN
            updates: Dict of fields to update

        Returns:
            True if updated, False if not found
        """
        try:
            validate_string(asin, "ASIN", allow_empty=False)
            validate_not_empty(updates, "updates", dict)

            # Add updated timestamp
            updates = add_timestamps(updates, created=False, updated=True)
            
            # Don't allow changing is_main_product or adding parent_asin
            updates["is_main_product"] = True
            updates.pop("parent_asin", None)

            result = self.products.update_one(
                {"asin": asin},
                {"$set": updates}
            )

            if result.modified_count > 0:
                logger.info(f"Main product updated: {asin}")
                return True
            else:
                logger.warning(f"Main product not found for update: {asin}")
                return False

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return False

    def delete_product(self, asin: str, delete_competitors: bool = True) -> bool:
        """
        Delete a main product.

        Args:
            asin: Product ASIN
            delete_competitors: Also delete associated competitors

        Returns:
            True if deleted, False if not found
        """
        try:
            validate_string(asin, "ASIN", allow_empty=False)

            # Delete the main product
            result = self.products.delete_one({"asin": asin})

            if result.deleted_count > 0:
                logger.info(f"Main product deleted: {asin}")
                
                # Also delete competitors if requested
                if delete_competitors:
                    comp_result = self.competitors.delete_many({"parent_asin": asin})
                    logger.info(f"Deleted {comp_result.deleted_count} competitors for {asin}")
                
                return True
            else:
                logger.warning(f"Main product not found for deletion: {asin}")
                return False

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return False

    def delete_all_products(self) -> bool:
        """
        Delete all products and all competitors from the database.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all competitors first
            comp_result = self.competitors.delete_many({})
            logger.info(f"Deleted {comp_result.deleted_count} competitors")
            
            # Delete all products
            prod_result = self.products.delete_many({})
            logger.info(f"Deleted {prod_result.deleted_count} products")
            
            return True

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return False

    def product_exists(self, asin: str) -> bool:
        """Check if a main product exists by ASIN."""
        try:
            return self.products.count_documents({"asin": asin}, limit=1) > 0
        except Exception as e:
            logger.error(f"Error checking product existence: {e}")
            return False

    def count_products(self) -> int:
        """Get total count of main products."""
        try:
            count = self.products.count_documents({"is_main_product": True})
            logger.debug(f"Total main products: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting products: {e}")
            return 0

    # ========================================================================
    # COMPETITORS METHODS (Competitor Products)
    # ========================================================================

    def add_competitor(
        self,
        competitor_data: Dict[str, Any],
        parent_asin: str
    ) -> str:
        """
        Add a competitor product for a main product.

        Args:
            competitor_data: Competitor information dict
            parent_asin: ASIN of the parent main product

        Returns:
            Competitor ASIN

        Raises:
            ValueError: Invalid data or parent doesn't exist
        """
        try:
            # Validate inputs
            validate_not_empty(competitor_data, "competitor_data", dict)
            validate_string(parent_asin, "parent_asin", allow_empty=False)
            validate_not_empty(competitor_data.get("asin"), "competitor ASIN", str)

            # Verify parent product exists
            if not self.product_exists(parent_asin):
                raise ValueError(f"Parent product {parent_asin} does not exist")

            # Set parent_asin and mark as competitor
            competitor_data["parent_asin"] = parent_asin
            competitor_data["is_main_product"] = False

            # Add timestamps
            competitor_data = add_timestamps(competitor_data, created=True, updated=True)

            # Insert into competitors collection
            result = self.competitors.insert_one(competitor_data)
            asin = competitor_data["asin"]
            
            logger.info(f"Competitor added: ASIN={asin}, Parent={parent_asin}, ID={result.inserted_id}")
            return asin

        except DuplicateKeyError:
            asin = competitor_data.get("asin", "unknown")
            error_msg = f"Competitor {asin} already exists for parent {parent_asin}"
            logger.warning(error_msg)
            # Don't raise error, just log - duplicate competitors are not critical
            return asin
        except ValueError as e:
            logger.error(f"Validation error adding competitor: {e}")
            raise
        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            raise

    def get_competitors(self, parent_asin: str) -> List[Dict[str, Any]]:
        """
        Get all competitors for a main product.

        Args:
            parent_asin: ASIN of the parent main product

        Returns:
            List of competitor dicts
        """
        try:
            validate_string(parent_asin, "parent_asin", allow_empty=False)

            normalized_asin = parent_asin.strip()
            asin_candidates = list({
                normalized_asin,
                normalized_asin.upper(),
                normalized_asin.lower(),
            })

            competitors = list(self.competitors.find(
                {"parent_asin": {"$in": asin_candidates}}
            ).sort("created_at", DESCENDING))

            if not competitors:
                competitors = list(self.products.find(
                    {"parent_asin": {"$in": asin_candidates}, "is_main_product": {"$ne": True}}
                ).sort("created_at", DESCENDING))

            # Remove MongoDB _id fields
            for comp in competitors:
                comp.pop("_id", None)

            logger.debug(f"Retrieved {len(competitors)} competitors for {parent_asin}")
            return competitors

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return []

    def update_competitor(self, competitor_asin: str, parent_asin: str, updates: Dict[str, Any]) -> bool:
        """
        Update a competitor product.

        Args:
            competitor_asin: ASIN of the competitor
            parent_asin: ASIN of the parent product
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        try:
            validate_string(competitor_asin, "competitor_asin", allow_empty=False)
            validate_string(parent_asin, "parent_asin", allow_empty=False)
            validate_not_empty(updates, "updates", dict)

            # Add timestamp for updated_at
            updates = add_timestamps(updates, created=False, updated=True)

            # Update in competitors collection
            result = self.competitors.update_one(
                {"asin": competitor_asin, "parent_asin": parent_asin},
                {"$set": updates}
            )

            if result.matched_count > 0:
                logger.info(f"Competitor updated: {competitor_asin}")
                return True
            else:
                logger.warning(f"Competitor not found for update: {competitor_asin}")
                return False

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return False

    def delete_competitors(self, parent_asin: str) -> int:
        """
        Delete all competitors for a main product.

        Args:
            parent_asin: ASIN of the parent main product

        Returns:
            Number of competitors deleted
        """
        try:
            validate_string(parent_asin, "parent_asin", allow_empty=False)

            result = self.competitors.delete_many({"parent_asin": parent_asin})
            count = result.deleted_count
            
            logger.info(f"Deleted {count} competitors for {parent_asin}")
            return count

        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return 0

    def count_competitors(self, parent_asin: Optional[str] = None) -> int:
        """
        Get count of competitors.

        Args:
            parent_asin: If provided, count for specific product; otherwise total

        Returns:
            Number of competitors
        """
        try:
            if parent_asin:
                count = self.competitors.count_documents({"parent_asin": parent_asin})
            else:
                count = self.competitors.count_documents({})
            
            logger.debug(f"Total competitors: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting competitors: {e}")
            return 0

    def get_competitor_parent_asins(self, limit: int = 50) -> List[str]:
        """
        Get distinct parent ASINs from competitors collection.

        Args:
            limit: Maximum number of parent ASINs to return

        Returns:
            List of parent ASINs
        """
        try:
            parents = self.competitors.distinct("parent_asin")
            parents = [p for p in parents if isinstance(p, str) and p.strip()]
            parents = sorted(set(parents))
            return parents[:max(limit, 0)]
        except Exception as e:
            logger.error(f"Error getting competitor parent ASINs: {e}")
            return []

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def clear_all_data(self) -> bool:
        """
        Clear all data from both collections (use with caution!).

        Returns:
            True if successful
        """
        try:
            self.products.delete_many({})
            self.competitors.delete_many({})
            logger.warning("All data cleared from database")
            return True
        except Exception as e:
            handle_exception(logger, e, show_ui=False)
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with database stats
        """
        try:
            stats = {
                "total_products": self.count_products(),
                "total_competitors": self.count_competitors(),
                "database_name": self.db.name,
                "collections": self.db.list_collection_names()
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def close(self) -> None:
        """Close MongoDB connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"MongoDB(database={self.db.name}, products={self.count_products()}, competitors={self.count_competitors()})"
