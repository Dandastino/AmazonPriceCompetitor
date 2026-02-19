"""
Common utilities for DRY principles across the project.

This module provides:
1. Error handling helpers
2. Validation functions
3. Streamlit UI helpers
4. Progress bar context manager
"""

import logging
from typing import Any, Callable, Optional, TypeVar, Type, List
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

import streamlit as st


T = TypeVar('T')


# ============================================================================
# Error Handling Helpers
# ============================================================================

def handle_exception(
    logger: logging.Logger,
    error: Exception,
    user_message: Optional[str] = None,
    show_ui: bool = True,
    log_traceback: bool = True
) -> None:
    """
    Centralized exception handling with logging and UI feedback.
    
    Args:
        logger: Logger instance
        error: Exception to handle
        user_message: Custom message for user (defaults to error message)
        show_ui: Whether to show error in Streamlit UI
        log_traceback: Whether to log full traceback
    """
    # Log the error
    error_msg = str(error)
    if log_traceback:
        logger.error(f"Error: {error_msg}", exc_info=True)
    else:
        logger.error(f"Error: {error_msg}")
    
    # Show in UI if requested
    if show_ui:
        display_msg = user_message if user_message else f"Error: {error_msg}"
        st.error(display_msg)


def safe_execute(
    logger: logging.Logger,
    default_return: Any = None,
    user_message: Optional[str] = None,
    show_ui: bool = True
):
    """
    Decorator for safe function execution with centralized error handling.
    
    Usage:
        @safe_execute(logger, default_return=[], user_message="Failed to load data")
        def my_function():
            # your code here
    
    Args:
        logger: Logger instance
        default_return: Value to return on exception
        user_message: Custom error message for user
        show_ui: Whether to show error in UI
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_exception(logger, e, user_message, show_ui)
                return default_return
        return wrapper
    return decorator


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_not_empty(
    value: Any,
    var_name: str,
    expected_type: Optional[Type] = None
) -> bool:
    """
    Validate that a value is not None/empty and optionally check type.
    
    Args:
        value: Value to validate
        var_name: Variable name for error message
        expected_type: Expected type (e.g., str, list, dict)
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not value:
        raise ValueError(f"{var_name} cannot be empty or None")
    
    if expected_type and not isinstance(value, expected_type):
        raise ValueError(
            f"{var_name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    return True


def validate_string(value: Any, var_name: str, allow_empty: bool = False) -> bool:
    """
    Validate that a value is a valid string.
    
    Args:
        value: Value to validate
        var_name: Variable name for error message
        allow_empty: Whether empty strings are allowed
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, str):
        raise ValueError(f"{var_name} must be a string, got {type(value).__name__}")
    
    if not allow_empty and not value.strip():
        raise ValueError(f"{var_name} cannot be empty")
    
    return True


def validate_asin(asin: Any) -> bool:
    """
    Validate Amazon ASIN format.
    
    Args:
        asin: ASIN to validate
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If invalid ASIN
    """
    validate_string(asin, "ASIN", allow_empty=False)
    
    if len(asin.strip()) != 10:
        raise ValueError(f"ASIN must be 10 characters, got {len(asin.strip())}")
    
    return True


def validate_range(
    value: Any,
    var_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Validate that a numeric value is within a range.
    
    Args:
        value: Value to validate
        var_name: Variable name for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{var_name} must be numeric, got {type(value).__name__}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{var_name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{var_name} must be <= {max_val}, got {value}")
    
    return True


def validate_list(
    value: Any,
    var_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    item_type: Optional[Type] = None
) -> bool:
    """
    Validate a list with optional length and item type checks.
    
    Args:
        value: Value to validate
        var_name: Variable name for error message
        min_length: Minimum list length
        max_length: Maximum list length
        item_type: Expected type of list items
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, list):
        raise ValueError(f"{var_name} must be a list, got {type(value).__name__}")
    
    if min_length is not None and len(value) < min_length:
        raise ValueError(f"{var_name} must have at least {min_length} items, got {len(value)}")
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{var_name} must have at most {max_length} items, got {len(value)}")
    
    if item_type:
        for idx, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValueError(
                    f"{var_name}[{idx}] must be {item_type.__name__}, "
                    f"got {type(item).__name__}"
                )
    
    return True


# ============================================================================
# Streamlit UI Helpers
# ============================================================================

class UINotifier:
    """Centralized Streamlit UI notification helper."""
    
    @staticmethod
    def success(message: str, icon: str = "✅") -> None:
        """Show success message."""
        st.success(f"{icon} {message}")
    
    @staticmethod
    def error(message: str, icon: str = "❌") -> None:
        """Show error message."""
        st.error(f"{icon} {message}")
    
    @staticmethod
    def warning(message: str, icon: str = "⚠️") -> None:
        """Show warning message."""
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def info(message: str, icon: str = "ℹ️") -> None:
        """Show info message."""
        st.info(f"{icon} {message}")
    
    @staticmethod
    def progress_update(message: str, icon: str = "🔄") -> None:
        """Show progress update."""
        st.write(f"{icon} {message}")


@contextmanager
def progress_tracker(total: int, desc: str = "Processing"):
    """
    Context manager for Streamlit progress tracking.
    
    Usage:
        with progress_tracker(100, "Scraping products") as progress:
            for i in range(100):
                # do work
                progress.update(i + 1, f"Processing item {i+1}")
    
    Args:
        total: Total number of items
        desc: Description of the operation
    """
    class ProgressTracker:
        def __init__(self, total: int, desc: str):
            self.total = total
            self.desc = desc
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
        
        def update(self, current: int, message: Optional[str] = None) -> None:
            """Update progress."""
            progress = min(current / self.total, 1.0)
            self.progress_bar.progress(progress)
            if message:
                self.status_text.write(f"{self.desc}: {message}")
        
        def cleanup(self) -> None:
            """Clean up progress widgets."""
            self.progress_bar.empty()
            self.status_text.empty()
    
    tracker = ProgressTracker(total, desc)
    try:
        yield tracker
    finally:
        tracker.cleanup()


# ============================================================================
# Timestamp Helpers
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def add_timestamps(data: dict, created: bool = True, updated: bool = True) -> dict:
    """
    Add timestamp fields to a data dictionary.
    
    Args:
        data: Dictionary to add timestamps to
        created: Whether to add 'created_at' timestamp
        updated: Whether to add 'updated_at' timestamp
    
    Returns:
        Dictionary with timestamps added
    """
    result = data.copy()
    timestamp = get_timestamp()
    
    if created and 'created_at' not in result:
        result['created_at'] = timestamp
    
    if updated:
        result['updated_at'] = timestamp
    
    return result


# ============================================================================
# Data Helpers
# ============================================================================

def safe_get(
    data: dict,
    key: str,
    default: Any = None,
    expected_type: Optional[Type] = None
) -> Any:
    """
    Safely get value from dictionary with type checking.
    
    Args:
        data: Dictionary to get value from
        key: Key to retrieve
        default: Default value if key not found or type mismatch
        expected_type: Expected type of value
    
    Returns:
        Value from dictionary or default
    """
    value = data.get(key, default)
    
    if expected_type and value is not None and not isinstance(value, expected_type):
        return default
    
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Int value or default
    """
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def remove_duplicates_by_key(items: List[dict], key: str) -> List[dict]:
    """
    Remove duplicate dictionaries based on a key.
    
    Args:
        items: List of dictionaries
        key: Key to use for deduplication
    
    Returns:
        List with duplicates removed (preserves order)
    """
    seen = set()
    result = []
    
    for item in items:
        value = item.get(key)
        if value and value not in seen:
            seen.add(value)
            result.append(item)
    
    return result


# ============================================================================
# Logger Helper
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# Formatting Helpers (merged from utils.py)
# ============================================================================

def format_price(
    price: Any,
    currency: str = "$",
    decimals: int = 2,
    fallback: str = "N/A",
    min_value: Optional[float] = None,
) -> str:
    """Format a price with currency and fixed decimals."""
    if not isinstance(price, (int, float)):
        return fallback
    if min_value is not None and price < min_value:
        return fallback
    return f"{currency} {price:.{decimals}f}"


def format_rating(
    rating: Any,
    decimals: int = 1,
    fallback: str = "N/A",
    style: str = "star_suffix",
) -> str:
    """Format a rating with a consistent style."""
    if not isinstance(rating, (int, float)):
        return fallback

    if style == "star_prefix":
        return f"⭐ {rating:.{decimals}f}"
    if style == "fraction":
        return f"{rating:.{decimals}f}/5.0"
    return f"{rating:.{decimals}f}⭐"


def extract_category_names(
    categories: List[Any],
    split_on_ampersand: bool = False,
    max_items: Optional[int] = None,
) -> List[str]:
    """Extract category names from known shapes (ladder, dict, str)."""
    names: List[str] = []

    for cat in categories or []:
        if isinstance(cat, dict):
            ladder = cat.get("ladder")
            if isinstance(ladder, list):
                for item in ladder:
                    name = item.get("name") if isinstance(item, dict) else None
                    if name:
                        names.append(str(name))
            elif "name" in cat:
                names.append(str(cat.get("name")))
        elif isinstance(cat, str):
            names.append(cat)

    if split_on_ampersand:
        split_names: List[str] = []
        for name in names:
            parts = [part.strip() for part in name.split("&")]
            split_names.extend([part for part in parts if part])
        names = split_names

    names = [name.strip() for name in names if name and str(name).strip()]
    unique = list(dict.fromkeys(names))

    if max_items is not None:
        return unique[:max_items]
    return unique


def extract_product_categories(product: dict, max_items: int = 5) -> List[str]:
    """Extract product categories with category_path and ladder support."""
    names = extract_category_names(
        product.get("categories", []),
        split_on_ampersand=True,
    )

    category_path = product.get("category_path")
    if isinstance(category_path, list):
        names.extend([str(cat).strip() for cat in category_path if cat])

    unique = list(dict.fromkeys([name for name in names if name]))
    return unique[:max_items]
