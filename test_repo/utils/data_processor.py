"""
Advanced data processing utilities that depend on helpers and config.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from utils.helpers import foo, bar, generate_hash, load_json_file, save_json_file
from config.settings import get_config, CONFIG_X, validate_config


class DataProcessor:
    """
    Main data processor that orchestrates various data operations.
    This class is critical to the application's functionality.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize data processor with optional config override.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = config_override or {}
        self.cache = {}
        self.processing_stats = {
            "total_processed": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        # Validate critical configuration
        config_errors = validate_config()
        if config_errors:
            raise RuntimeError(f"Configuration validation failed: {config_errors}")
    
    def process_batch(self, data_batch: List[Dict[str, Any]], 
                     processing_key: str = "id") -> Dict[str, Any]:
        """
        Process a batch of data using the foo function with additional validation.
        
        Args:
            data_batch: Batch of data dictionaries to process
            processing_key: Key to use for processing
            
        Returns:
            Processed batch results with additional metadata
            
        Raises:
            ValueError: If CONFIG_X is missing or data is invalid
        """
        # Critical check - this will break if CONFIG_X is missing!
        if not CONFIG_X:
            raise ValueError("CONFIG_X is required for batch processing but not configured")
        
        try:
            # Use the foo function for core processing
            base_result = foo(data_batch, processing_key)
            
            # Add our own metadata
            enhanced_result = {
                **base_result,
                "config_x_value": CONFIG_X,
                "processor_id": generate_hash(f"processor_{datetime.now().isoformat()}"),
                "batch_size": len(data_batch),
                "config_source": "DataProcessor"
            }
            
            # Update stats
            self.processing_stats["total_processed"] += len(data_batch)
            
            return enhanced_result
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            raise ValueError(f"Batch processing failed: {e}")
    
    def sanitize_and_validate(self, raw_data: List[str]) -> List[str]:
        """
        Sanitize and validate raw string data using bar function.
        
        Args:
            raw_data: List of raw strings to process
            
        Returns:
            List of sanitized and validated strings
        """
        max_length = get_config("processing.max_string_length", 200)
        
        cleaned_data = []
        for item in raw_data:
            try:
                # Use bar function for sanitization
                cleaned = bar(item, max_length)
                if cleaned:  # Only keep non-empty results
                    cleaned_data.append(cleaned)
            except Exception as e:
                print(f"Error processing item '{item}': {e}")
                continue
        
        return cleaned_data
    
    def process_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """
        Process a JSON file and save results.
        
        Args:
            file_path: Path to input JSON file
            output_path: Optional output path (auto-generated if not provided)
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Load data
            data = load_json_file(file_path)
            
            # Convert to list format if needed
            if isinstance(data, dict):
                data_list = [data]
            elif isinstance(data, list):
                data_list = data
            else:
                raise ValueError(f"Unsupported data format: {type(data)}")
            
            # Process the data
            result = self.process_batch(data_list)
            
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = f"{base_name}_processed.json"
            
            # Save results
            success = save_json_file(result, output_path)
            
            if success:
                print(f"Successfully processed {file_path} -> {output_path}")
            
            return success
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            self.processing_stats["errors"] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and system health info.
        
        Returns:
            Dictionary with stats and health information
        """
        current_time = datetime.now()
        uptime = current_time - self.processing_stats["start_time"]
        
        return {
            **self.processing_stats,
            "uptime_seconds": uptime.total_seconds(),
            "current_time": current_time.isoformat(),
            "config_x_present": bool(CONFIG_X),
            "cache_size": len(self.cache),
            "config_errors": validate_config()
        }
    
    def health_check(self) -> Tuple[bool, List[str]]:
        """
        Perform system health check.
        
        Returns:
            Tuple of (is_healthy, list_of_issues)
        """
        issues = []
        
        # Check critical configuration
        if not CONFIG_X:
            issues.append("CONFIG_X is missing - system will fail to process batches")
        
        # Check other config issues
        config_errors = validate_config()
        if config_errors:
            issues.extend([f"Config error: {k} - {v}" for k, v in config_errors.items()])
        
        # Check processing stats
        if self.processing_stats["errors"] > 0:
            error_rate = self.processing_stats["errors"] / max(self.processing_stats["total_processed"], 1)
            if error_rate > 0.1:  # More than 10% error rate
                issues.append(f"High error rate: {error_rate:.1%}")
        
        is_healthy = len(issues) == 0
        return is_healthy, issues


def create_sample_data(count: int = 5) -> List[Dict[str, Any]]:
    """
    Create sample data for testing the processor.
    
    Args:
        count: Number of sample items to create
        
    Returns:
        List of sample data dictionaries
    """
    sample_data = []
    
    for i in range(count):
        item = {
            "id": f"item_{i}",
            "name": f"Sample Item {i}",
            "value": i * 10,
            "category": "test" if i % 2 == 0 else "production",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processed": False,
                "priority": "high" if i < 2 else "normal"
            }
        }
        sample_data.append(item)
    
    return sample_data


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Create and process sample data
    sample = create_sample_data()
    result = processor.process_batch(sample)
    
    print("Processing complete!")
    print(f"Processed {result['total_items']} items into {result['unique_groups']} groups")
    print(f"Stats: {processor.get_stats()}")
