import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)


class Database:
    """TinyDB wrapper for product data management with error handling."""

    def __init__(self, db_path: str = "data.json"):
        """
        Initialize database connection.

        Args:
            db_path: Path to JSON database file
        """
        try:
            import os

            dirname = os.path.dirname(db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            self.db_path = db_path
            self.db = TinyDB(db_path)
            self.products = self.db.table("products")

            logger.info(f"Database initialized: {db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            raise

    def insert_product(self, product_data: Dict[str, Any]) -> int:
        """
        Insert a new product into database.

        Args:
            product_data: Product information dict

        Returns:
            Document ID of inserted product

        Raises:
            ValueError: Invalid product data
            Exception: Database error
        """
        try:
            if not product_data or not isinstance(product_data, dict):
                raise ValueError(f"Invalid product data: {product_data}")

            if not product_data.get("asin"):
                raise ValueError("Product must have an ASIN")

            # Add metadata
            product_data["created_at"] = datetime.now().isoformat()
            product_data["updated_at"] = datetime.now().isoformat()

            doc_id = self.products.insert(product_data)
            logger.info(f"Product inserted: ASIN={product_data.get('asin')}, ID={doc_id}")
            return doc_id

        except ValueError as e:
            logger.error(f"Validation error inserting product: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error inserting product: {e}", exc_info=True)
            raise

    def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Get a single product by ASIN.

        Args:
            asin: Product ASIN

        Returns:
            Product dict or None if not found

        Raises:
            ValueError: Invalid ASIN
        """
        try:
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            query = Query()
            product = self.products.get(query.asin == asin)

            if product:
                logger.debug(f"Found product: {asin}")
            else:
                logger.debug(f"Product not found: {asin}")

            return product

        except ValueError as e:
            logger.error(f"Validation error fetching product: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error fetching product {asin}: {e}", exc_info=True)
            raise

    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get all products from database.

        Returns:
            List of all product dicts
        """
        try:
            products = self.products.all()
            logger.debug(f"Retrieved {len(products)} products")
            return products

        except Exception as e:
            logger.error(f"Database error fetching all products: {e}", exc_info=True)
            raise

    def search_products(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search products by criteria (AND logic).

        Args:
            search_criteria: Dict of field:value pairs to match

        Returns:
            List of matching product dicts

        Raises:
            ValueError: Invalid search criteria
        """
        try:
            if not search_criteria or not isinstance(search_criteria, dict):
                logger.warning("Empty search criteria provided")
                return []

            query = Query()
            combined_query = None

            for key, value in search_criteria.items():
                if key not in ["asin", "parent_asin", "title", "brand", "amazon_domain"]:
                    logger.warning(f"Searching on uncommon field: {key}")

                if combined_query is None:
                    combined_query = query[key] == value
                else:
                    combined_query &= query[key] == value

            if combined_query is None:
                logger.warning("No valid search criteria found")
                return []

            results = self.products.search(combined_query)
            logger.debug(f"Search for {search_criteria} returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Database error searching products: {e}", exc_info=True)
            raise

    def update_product(self, asin: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing product.

        Args:
            asin: Product ASIN
            updates: Dict of fields to update

        Returns:
            True if updated, False if not found

        Raises:
            ValueError: Invalid input
        """
        try:
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            if not updates or not isinstance(updates, dict):
                raise ValueError(f"Invalid updates: {updates}")

            updates["updated_at"] = datetime.now().isoformat()

            query = Query()
            doc_ids = self.products.update(updates, query.asin == asin)

            if doc_ids:
                logger.info(f"Product updated: {asin}")
                return True
            else:
                logger.warning(f"Product not found for update: {asin}")
                return False

        except ValueError as e:
            logger.error(f"Validation error updating product: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error updating product {asin}: {e}", exc_info=True)
            raise

    def delete_product(self, asin: str) -> bool:
        """
        Delete a product by ASIN.

        Args:
            asin: Product ASIN

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: Invalid ASIN
        """
        try:
            if not asin or not isinstance(asin, str):
                raise ValueError(f"Invalid ASIN: {asin}")

            query = Query()
            doc_ids = self.products.remove(query.asin == asin)

            if doc_ids:
                logger.info(f"Product deleted: {asin}")
                return True
            else:
                logger.warning(f"Product not found for deletion: {asin}")
                return False

        except ValueError as e:
            logger.error(f"Validation error deleting product: {e}")
            raise
        except Exception as e:
            logger.error(f"Database error deleting product {asin}: {e}", exc_info=True)
            raise

    def delete_all_products(self) -> bool:
        """
        Delete all products from database.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.products.truncate()
            logger.info("All products deleted from database")
            return True
        except Exception as e:
            logger.error(f"Database error deleting all products: {e}", exc_info=True)
            return False

    def product_exists(self, asin: str) -> bool:
        """Check if a product exists by ASIN."""
        try:
            return self.get_product(asin) is not None
        except Exception as e:
            logger.error(f"Error checking product existence: {e}")
            return False

    def get_competitors(self, parent_asin: str) -> List[Dict[str, Any]]:
        """
        Get all competitors for a parent product.

        Args:
            parent_asin: Parent product ASIN

        Returns:
            List of competitor products
        """
        return self.search_products({"parent_asin": parent_asin})

    def count_products(self) -> int:
        """Get total product count."""
        try:
            count = len(self.products)
            logger.debug(f"Total products: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting products: {e}")
            return 0

    def close(self) -> None:
        """Close database connection."""
        try:
            if self.db:
                self.db.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()