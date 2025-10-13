#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from altseason.runner import AltseasonRunner
from rich import print

print("🔄 Starting altseason analysis...")
r = AltseasonRunner()
ok = r.run_daily_analysis()
if not ok:
    print("❌ Daily analysis failed!")
    raise SystemExit(1)
print("✅ Daily analysis completed successfully!")
