#!/usr/bin/env python3
"""
FractalGenesis AI Management Utility

Easy-to-use command line interface for managing AI selectors,
training data, and preference learning models.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ai.preference_learner import (
    analyze_user_data, manage_selection_data, list_selectors,
    export_selector, import_selector, list_ai_selectors, AISelector
)


def show_help():
    """Display help information."""
    print("""
FractalGenesis AI Management Utility
=====================================

TRAINING:
  manage_ai train              Train new AI model from user selection data
  
DATA MANAGEMENT:
  manage_ai data list          List all selection sessions
  manage_ai data stats         Show selection data statistics  
  manage_ai data clean         Remove invalid session files
  manage_ai data export        Export all data to consolidated file

AI SELECTORS:
  manage_ai selectors          List all available AI selectors
  manage_ai export <name>      Export AI selector for sharing
  manage_ai import <file>      Import shared AI selector
  manage_ai test <name>        Test AI selector (requires test data)

EXAMPLES:
  # Train a new AI from your selections
  manage_ai train
  
  # Check how much data you have
  manage_ai data stats
  
  # List available AI selectors  
  manage_ai selectors
  
  # Export your AI for sharing
  manage_ai export "My Preferences 2025-01-10"
  
  # Import someone else's AI
  manage_ai import shared_selector.pkl
""")


def test_selector(selector_name: str):
    """Test an AI selector with sample data."""
    selectors = list_ai_selectors()
    matching = [s for s in selectors if s.get('name', '').lower() == selector_name.lower()]
    
    if not matching:
        print(f"No selector found with name '{selector_name}'")
        return
    
    try:
        # Load the AI selector
        selector_path = matching[0]['filepath']
        ai_selector = AISelector(selector_path)
        
        print(f"Testing AI selector: {ai_selector.metadata['name']}")
        print(f"Created: {ai_selector.metadata.get('created_at', 'Unknown')}")
        print(f"Features: {ai_selector.metadata.get('feature_count', 'Unknown')}")
        
        # Test with dummy fractal data (would normally come from fractal evolution)
        test_fractals = [
            {"scale": 2.5, "rotate": 45, "brightness": 1.2, "xforms": []},
            {"scale": 1.8, "rotate": 90, "brightness": 0.8, "xforms": []},
            {"scale": 3.2, "rotate": 180, "brightness": 1.5, "xforms": []}
        ]
        
        selection = ai_selector.select_fractal(test_fractals)
        print(f"\nAI selected fractal #{selection} from 3 test candidates")
        print("AI selector is working correctly!")
        
    except Exception as e:
        print(f"Error testing selector: {e}")


def main():
    """Main command-line interface."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "help" or command == "--help" or command == "-h":
            show_help()
        
        elif command == "train":
            print("Training AI model from user selection data...\n")
            analyze_user_data()
        
        elif command == "data":
            if len(sys.argv) >= 3:
                subcommand = sys.argv[2].lower()
                if subcommand in ["list", "stats", "clean", "export"]:
                    manage_selection_data(subcommand)
                else:
                    print(f"Unknown data command: {subcommand}")
                    print("Available: list, stats, clean, export")
            else:
                manage_selection_data("stats")
        
        elif command == "selectors":
            list_selectors()
        
        elif command == "export":
            if len(sys.argv) >= 3:
                selector_name = sys.argv[2]
                export_selector(selector_name)
            else:
                print("Please specify selector name to export")
                print("Use 'manage_ai selectors' to see available selectors")
        
        elif command == "import":
            if len(sys.argv) >= 3:
                file_path = sys.argv[2]
                # Optional new name
                new_name = sys.argv[3] if len(sys.argv) >= 4 else None
                import_selector(file_path, new_name)
            else:
                print("Please specify file path to import")
        
        elif command == "test":
            if len(sys.argv) >= 3:
                selector_name = sys.argv[2]
                test_selector(selector_name)
            else:
                print("Please specify selector name to test")
                print("Use 'manage_ai selectors' to see available selectors")
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'manage_ai help' for usage information")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
