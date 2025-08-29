"""
Configuration settings for the application.
Contains database configuration, API keys, and feature flags.
"""

import os
from typing import Dict, Any, Optional


# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "name": os.getenv("DB_NAME", "myapp"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# API configuration
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "https://api.example.com"),
    "timeout": int(os.getenv("API_TIMEOUT", "30")),
    "retries": int(os.getenv("API_RETRIES", "3")),
    "api_key": os.getenv("API_KEY"),
}

# Feature flags
FEATURE_FLAGS = {
    "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
    "enable_logging": os.getenv("ENABLE_LOGGING", "true").lower() == "true",
    "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
    "experimental_features": os.getenv("EXPERIMENTAL_FEATURES", "false").lower() == "true",
}

# Critical configuration that the system depends on
CONFIG_X = os.getenv("CONFIG_X")  # This is critical - system breaks without it!


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by key.
    
    Args:
        key: Configuration key in dot notation (e.g., 'database.host')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config_map = {
        "database": DATABASE_CONFIG,
        "api": API_CONFIG,
        "features": FEATURE_FLAGS,
    }
    
    parts = key.split(".")
    if len(parts) != 2:
        return default
    
    section, setting = parts
    return config_map.get(section, {}).get(setting, default)


def validate_config() -> Dict[str, str]:
    """
    Validate critical configuration settings.
    
    Returns:
        Dictionary of validation errors (empty if all valid)
    """
    errors = {}
    
    # Check critical CONFIG_X
    if not CONFIG_X:
        errors["CONFIG_X"] = "CONFIG_X environment variable is required but not set"
    
    # Check database config
    if not DATABASE_CONFIG["password"]:
        errors["database.password"] = "Database password not configured"
    
    # Check API key
    if not API_CONFIG["api_key"]:
        errors["api.api_key"] = "API key not configured"
    
    return errors


def get_database_url() -> str:
    """
    Construct database URL from configuration.
    
    Returns:
        Complete database connection URL
    """
    cfg = DATABASE_CONFIG
    return f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['name']}"
