#!/usr/bin/env python3
"""
Daily runner script for Altseason Radar
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from altseason.runner import AltseasonRunner

def main():
    runner = AltseasonRunner()
    success = runner.run_daily_analysis()
    
    if not success:
        print("Daily analysis failed!")
        sys.exit(1)
    
    print("Daily analysis completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()