import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import altseason.report as report

factors = {
  "btc_dominance": {"score":10,"ok":True,"explain":"test"},
  "eth_btc":       {"score":10,"ok":True,"explain":"test"},
  "total2":        {"score":10,"ok":True,"explain":"test"},
  "total3":        {"score":10,"ok":True,"explain":"test"},
  "btc_regime":    {"score":10,"ok":True,"explain":"test"},
  "eth_trend":     {"score":10,"ok":True,"explain":"test"}
}

rg = report.ReportGenerator()
rg.generate_daily_report(factors, 60, "Neutral", False)
print("report_written")
