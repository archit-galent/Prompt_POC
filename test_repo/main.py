"""
Main application entry point that orchestrates the data processing workflow.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from utils.data_processor import DataProcessor, create_sample_data
from utils.helpers import foo, format_duration
from config.settings import validate_config, CONFIG_X


def main():
    """
    Main application function that sets up and runs data processing.
    """
    parser = argparse.ArgumentParser(description="Data Processing Application")
    parser.add_argument("--input", help="Input JSON file to process")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--sample", action="store_true", help="Generate and process sample data")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--count", type=int, default=10, help="Number of sample items to generate")
    
    args = parser.parse_args()
    
    # First, validate configuration
    print("Validating configuration...")
    config_errors = validate_config()
    
    if config_errors:
        print("❌ Configuration validation failed:")
        for key, error in config_errors.items():
            print(f"  - {key}: {error}")
        
        if not CONFIG_X:
            print("\n💡 Hint: Set CONFIG_X environment variable to fix the main issue:")
            print("   export CONFIG_X='some-important-value'")
        
        if not args.health:  # Only exit if not doing health check
            sys.exit(1)
    else:
        print("✅ Configuration validation passed!")
    
    try:
        # Initialize processor
        processor = DataProcessor()
        
        # Handle different modes
        if args.health:
            print("\n" + "="*50)
            print("SYSTEM HEALTH CHECK")
            print("="*50)
            
            is_healthy, issues = processor.health_check()
            
            if is_healthy:
                print("✅ System is healthy!")
            else:
                print("❌ System has issues:")
                for issue in issues:
                    print(f"  - {issue}")
            
            stats = processor.get_stats()
            print(f"\nStats:")
            print(f"  - Uptime: {format_duration(stats['uptime_seconds'])}")
            print(f"  - Total processed: {stats['total_processed']}")
            print(f"  - Errors: {stats['errors']}")
            print(f"  - CONFIG_X present: {stats['config_x_present']}")
            
            return
        
        elif args.sample:
            print(f"\nGenerating {args.count} sample items...")
            sample_data = create_sample_data(args.count)
            
            print("Processing sample data...")
            result = foo(sample_data, "category")
            
            print(f"✅ Sample processing complete!")
            print(f"  - Total items: {result['total_items']}")
            print(f"  - Unique groups: {result['unique_groups']}")
            print(f"  - Processing key: {result['processing_key']}")
            
            # Also test the processor
            processor_result = processor.process_batch(sample_data, "category")
            print(f"  - Processor ID: {processor_result['processor_id']}")
            print(f"  - CONFIG_X value: {processor_result['config_x_value']}")
            
        elif args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"❌ Input file not found: {input_path}")
                sys.exit(1)
            
            print(f"Processing file: {input_path}")
            success = processor.process_file(str(input_path), args.output)
            
            if success:
                print("✅ File processing complete!")
            else:
                print("❌ File processing failed!")
                sys.exit(1)
        
        else:
            print("No operation specified. Use --help for options.")
            print("\nQuick examples:")
            print("  python main.py --sample --count 5")
            print("  python main.py --health")
            print("  python main.py --input data.json --output processed.json")
    
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
