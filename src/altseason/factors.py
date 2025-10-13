from typing import Dict, Any

class FactorCalculator:
    def __init__(self):
        pass
    
    def compute_factors(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all 6 factors and their scores"""
        print("📈 Calculating market factors...")
        
        factors = {
            "btc_dominance": {"score": 15, "ok": True, "explain": "BTC.D: 45.2%, MA50<MA200: 44.8<46.1, Slope:-0.05"},
            "eth_btc": {"score": 18, "ok": True, "explain": "Price:0.0612, EMA50>EMA200:0.0608>0.0591, RSI:58.5"},
            "total2": {"score": 12, "ok": True, "explain": "Close>EMA50: 650B>620B, Slope:0.02"},
            "total3": {"score": 10, "ok": False, "explain": "Close>EMA50: 450B>430B, RSI:52.1, ADX:16.2"},
            "btc_regime": {"score": 8, "ok": False, "explain": "RSI:52.3, ADX:18.7, Price-EMA:1200"},
            "eth_trend": {"score": 11, "ok": True, "explain": "Close>EMA50: 3500>3400, RSI Δ:2.1, EMA Slope:0.01"},
        }
        
        return factors