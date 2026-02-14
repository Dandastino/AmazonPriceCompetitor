import logging
from typing import List, Dict, Any, Optional

import streamlit as st

from src.db import Database
from src.oxylab_client import (
    scrape_multiple_products,
    scrape_product_details,
    search_competitors,
)

logger = logging.getLogger(__name__)


class ProductService:
    """Service for managing product scraping and storage."""

    def __init__(self):
        """Initialize service with database connection."""
        self.db = Database()

    def scrape_and_store_product(
        self, asin: str, geo_location: str, domain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape product details and store in database.

        Args:
            asin: Product ASIN
            geo_location: Geographic location
            domain: Amazon domain

        Returns:
            Product data dict or None on failure
        """
        try:
            if not asin or not isinstance(asin, str):
                error_msg = f"❌ Invalid ASIN format: '{asin}'. Please enter a valid Amazon ASIN (e.g., B0CX23VSAS)"
                st.error(error_msg)
                logger.error(error_msg)
                return None

            if not asin.strip():
                error_msg = "❌ ASIN cannot be empty. Please enter a product ASIN."
                st.error(error_msg)
                logger.error(error_msg)
                return None

            st.info(f"🔄 Scraping product: {asin} from amazon.{domain}")
            
            try:
                data = scrape_product_details(asin, geo_location, domain)
            except Exception as scrape_error:
                error_msg = f"❌ Failed to scrape product - {str(scrape_error)}"
                st.error(error_msg)
                logger.error(f"Scraping failed for {asin}: {scrape_error}")
                return None

            # Validate that we actually got product data
            if not data:
                error_msg = f"❌ No product data returned. ASIN '{asin}' may not exist on amazon.{domain}"
                st.error(error_msg)
                logger.error(f"No data returned for ASIN {asin}")
                return None

            # Check for critical fields
            title = data.get("title", "").strip()
            product_asin = data.get("asin", "").strip()
            
            if not title or not product_asin:
                error_msg = (
                    f"❌ Product not found. The ASIN '{asin}' does not appear to exist on amazon.{domain}\n\n"
                    "Possible reasons:\n"
                    "• The ASIN is incorrect or invalid\n"
                    "• The product has been removed from Amazon\n"
                    "• The product is not available in the selected region\n"
                    f"• Check that amazon.{domain} is the correct marketplace"
                )
                st.error(error_msg)
                logger.warning(f"Product not found for ASIN {asin}: missing title or ASIN in response")
                return None

            # Store product
            self.db.insert_product(data)
            success_msg = f"✅ Product scraped successfully!\n\n**Title:** {title}\n**ASIN:** {product_asin}"
            st.success(success_msg)
            logger.info(f"Product {asin} stored: {title}")
            return data

        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}\n\nPlease check your ASIN and try again."
            st.error(error_msg)
            logger.error(f"Unexpected error scraping {asin}: {e}", exc_info=True)
            return None

    def fetch_and_store_competitors(
        self,
        parent_asin: str,
        domain: str,
        geo_location: str,
        pages: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Fetch competitors for a product and store them.

        Args:
            parent_asin: Parent product ASIN
            domain: Amazon domain
            geo_location: Geographic location
            pages: Number of search pages to query

        Returns:
            List of stored competitor products
        """
        try:
            # Validate input
            if not parent_asin or not isinstance(parent_asin, str):
                st.error("Invalid parent ASIN")
                return []

            if pages < 1 or pages > 5:
                st.warning(f"Pages {pages} out of range, using 2")
                pages = 2

            st.write(f"🔍 Fetching parent product: {parent_asin}")

            # Fetch parent product
            parent = self.db.get_product(parent_asin)
            if not parent:
                st.error(f"Parent product {parent_asin} not found in database")
                logger.warning(f"Parent product {parent_asin} not found")
                return []

            # Get search parameters from parent
            search_domain = parent.get("amazon_domain", domain)
            search_geo = parent.get("amazon_geo_location", geo_location)

            st.write(f"🌍 Searching domain: {search_domain} | Region: {search_geo}")

            # Extract and clean categories
            search_categories = self._extract_categories(parent)
            if not search_categories:
                st.warning("No categories found for competitor search")
                logger.warning(f"No categories found for {parent_asin}")
                return []

            st.write(f"📂 Searching in categories: {', '.join(search_categories[:3])}")

            # Search competitors across categories
            all_results = self._search_across_categories(
                parent, search_categories, search_domain, search_geo, pages
            )

            if not all_results:
                st.warning("No competitors found")
                logger.warning(f"No competitors found for {parent_asin}")
                return []

            # Extract unique competitor ASINs
            competitor_asins = self._extract_unique_competitors(all_results, parent_asin)
            if not competitor_asins:
                st.warning("No valid competitor ASINs extracted")
                return []

            st.write(f"🎯 Found {len(competitor_asins)} unique competitors")

            # Scrape and store competitors
            stored_competitors = self._scrape_and_store_competitors(
                competitor_asins, parent_asin, search_geo, search_domain
            )

            # Display results
            self._display_competitor_results(stored_competitors)

            return stored_competitors

        except Exception as e:
            st.error(f"Error fetching competitors: {str(e)}")
            logger.error(f"Error fetching competitors for {parent_asin}: {e}", exc_info=True)
            return []

    def _extract_categories(self, parent: Dict[str, Any]) -> List[str]:
        """Extract and clean categories from parent product."""
        categories = []

        if parent.get("categories") and isinstance(parent["categories"], list):
            categories.extend(str(cat).strip() for cat in parent["categories"] if cat)

        if parent.get("category_path") and isinstance(parent["category_path"], list):
            categories.extend(str(cat).strip() for cat in parent["category_path"] if cat)

        # Remove duplicates and empty strings
        unique_categories = list(set(cat for cat in categories if cat and len(cat) > 0))
        return unique_categories[:5]  # Limit to top 5

    def _search_across_categories(
        self,
        parent: Dict[str, Any],
        categories: List[str],
        domain: str,
        geo_location: str,
        pages: int,
    ) -> List[Dict[str, Any]]:
        """Search competitors across all categories."""
        all_results = []
        progress_bar = st.progress(0)

        for idx, category in enumerate(categories[:3]):  # Limit to 3 categories
            try:
                st.write(f"  Searching category: {category}")
                search_result = search_competitors(
                    query_title=parent.get("title", ""),
                    domain=domain,
                    categories=[category],
                    pages=pages,
                    geo_location=geo_location,
                )
                all_results.extend(search_result or [])
                progress_bar.progress((idx + 1) / 3)
            except Exception as e:
                logger.error(f"Error searching category {category}: {e}")
                st.warning(f"Error searching category {category}: {str(e)}")

        return all_results

    def _extract_unique_competitors(
        self, all_results: List[Dict[str, Any]], parent_asin: str
    ) -> List[str]:
        """Extract unique valid competitor ASINs."""
        competitors = set()

        for result in all_results:
            asin = result.get("asin")
            title = result.get("title")

            # Validate competitor
            if (
                asin
                and isinstance(asin, str)
                and asin != parent_asin
                and title
                and isinstance(title, str)
            ):
                competitors.add(asin)

        return list(competitors)

    def _scrape_and_store_competitors(
        self, competitor_asins: List[str], parent_asin: str, geo_location: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Scrape and store competitor products."""
        stored = []
        progress_bar = st.progress(0)

        for idx, asin in enumerate(competitor_asins):
            try:
                details = scrape_product_details(asin, geo_location, domain)
                if details:
                    details["parent_asin"] = parent_asin
                    self.db.insert_product(details)
                    stored.append(details)
                    logger.info(f"Stored competitor {asin}")
            except Exception as e:
                logger.error(f"Error scraping competitor {asin}: {e}")
                st.warning(f"Failed to scrape competitor {asin}")

            progress_bar.progress(min((idx + 1) / len(competitor_asins), 1.0))

        return stored

    def _display_competitor_results(self, competitors: List[Dict[str, Any]]) -> None:
        """Display competitor results in Streamlit UI."""
        st.markdown("---")
        st.subheader(f"✅ Stored {len(competitors)} Competitors")

        for competitor in competitors:
            try:
                title = competitor.get("title", "Unknown Product")
                asin = competitor.get("asin", "N/A")
                price = competitor.get("price")
                currency = competitor.get("currency", "$")

                # Format price
                if isinstance(price, (int, float)) and price > 0:
                    price_str = f"{currency} {price:.2f}"
                else:
                    price_str = "N/A"

                rating = competitor.get("rating")
                rating_str = f"⭐ {rating:.1f}" if isinstance(rating, (int, float)) else "N/A"

                stock = competitor.get("stock", "Unknown")

                st.write(
                    f"• **{title}** [ASIN: {asin}]  \n"
                    f"  Price: {price_str} | Rating: {rating_str} | Stock: {stock}"
                )
            except Exception as e:
                logger.error(f"Error displaying competitor: {e}")
                st.warning("Error displaying competitor details")

        st.markdown("---")


# Backward-compatible function interfaces
def scrape_and_store_product(asin: str, geo_location: str, domain: str) -> Optional[Dict[str, Any]]:
    """Scrape and store product (uses ProductService)."""
    service = ProductService()
    return service.scrape_and_store_product(asin, geo_location, domain)


def fetch_and_store_competitors(
    parent_asin: str, domain: str, geo_location: str, pages: int = 2
) -> List[Dict[str, Any]]:
    """Fetch and store competitors (uses ProductService)."""
    service = ProductService()
    return service.fetch_and_store_competitors(parent_asin, domain, geo_location, pages)