"""
Utility helper functions for data processing and common operations.
"""

import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from config.settings import get_config, FEATURE_FLAGS


def foo(data: List[Dict[str, Any]], key: str = "id") -> Dict[str, Any]:
    """
    Process a list of data dictionaries and return aggregated results.
    This is a key function that many other parts depend on.
    
    Args:
        data: List of dictionaries to process
        key: Key to use for grouping (default: 'id')
        
    Returns:
        Dictionary with processed results and metadata
        
    Raises:
        ValueError: If data is empty or key not found in items
    """
    if not data:
        raise ValueError("Cannot process empty data list")
    
    # Group data by key
    grouped = {}
    total_count = 0
    
    for item in data:
        if key not in item:
            raise ValueError(f"Key '{key}' not found in data item: {item}")
        
        group_key = item[key]
        if group_key not in grouped:
            grouped[group_key] = []
        
        grouped[group_key].append(item)
        total_count += 1
    
    # Calculate statistics
    result = {
        "groups": grouped,
        "total_items": total_count,
        "unique_groups": len(grouped),
        "timestamp": datetime.now().isoformat(),
        "processing_key": key
    }
    
    # Add caching info if enabled
    if FEATURE_FLAGS.get("enable_caching", False):
        result["cache_key"] = generate_hash(str(data))
    
    return result


def bar(input_string: str, max_length: int = 100) -> str:
    """
    Process and sanitize input strings for safe usage.
    
    Args:
        input_string: String to process
        max_length: Maximum allowed length
        
    Returns:
        Processed and sanitized string
    """
    if not isinstance(input_string, str):
        input_string = str(input_string)
    
    # Remove dangerous characters
    dangerous_chars = ['<', '>', '&', '"', "'", '\x00']
    sanitized = input_string
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized.strip()


def generate_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Generate hash for content using specified algorithm.
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
        
    Returns:
        Hex digest of the hash
        
    Raises:
        ValueError: If algorithm is not supported
    """
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hasher = algorithms[algorithm]()
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                if FEATURE_FLAGS.get("enable_logging", False):
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                break
    
    raise last_exception


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
