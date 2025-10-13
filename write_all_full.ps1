cd /d "D:\altseason-radar"

$code = @'
# write_all_full.ps1 — clean build (no self-calls)

# 1) Folders
New-Item -ItemType Directory -Force -Path '.github/workflows','scripts','src/altseason/data_providers','reports','tests' | Out-Null

# 2) requirements.txt
$c = @'
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dateutil>=2.8.0
tenacity>=8.2.0
pydantic>=2.0.0
rich>=13.0.0
pytest>=7.0.0
'@
Set-Content -Encoding utf8 -Path 'requirements.txt' -Value $c

# 3) pyproject.toml
$c = @'
[build-system]
requires = ["setuptools>=45","wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "altseason-radar"
version = "1.0.0"
description = "Daily Altseason Radar Analysis"
authors = [{name="Altseason Team"}]
dependencies = [
  "pandas>=2.0.0",
  "numpy>=1.24.0",
  "requests>=2.31.0",
  "python-dateutil>=2.8.0",
  "tenacity>=8.2.0",
  "pydantic>=2.0.0",
  "rich>=13.0.0",
]
'@
Set-Content -Encoding utf8 -Path 'pyproject.toml' -Value $c

# 4) .gitignore
$c = @'
__pycache__/
*.py[cod]
*$py.class
.env
.venv
.cache/
logs/
reports/state.json
.vscode/
.idea/
'@
Set-Content -Encoding utf8 -Path '.gitignore' -Value $c

# 5) scripts/run_daily.py
$c = @'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from altseason.runner import AltseasonRunner

def main():
    r = AltseasonRunner()
    ok = r.run_daily_analysis()
    if not ok:
        print("Daily analysis failed!")
        raise SystemExit(1)
    print("Daily analysis completed successfully!")
    raise SystemExit(0)

if __name__ == "__main__":
    main()
'@
Set-Content -Encoding utf8 -Path 'scripts/run_daily.py' -Value $c

# 6) src/altseason/__init__.py
$c = @'
"""Altseason Radar - Daily cryptocurrency market analysis"""
__version__ = "1.0.0"
__author__  = "Altseason Team"
'@
Set-Content -Encoding utf8 -Path 'src/altseason/__init__.py' -Value $c

# 7) src/altseason/logging_utils.py
$c = @'
import logging
from rich.logging import RichHandler

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("altseason")
'@
Set-Content -Encoding utf8 -Path 'src/altseason/logging_utils.py' -Value $c

# 8) src/altseason/cache.py
$c = @'
import pickle, time
from pathlib import Path
from typing import Any, Optional

class SimpleCache:
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _p(self, key: str):
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        p = self._p(key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl_seconds:
            try: p.unlink()
            except: pass
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except:
            return None

    def set(self, key: str, val: Any):
        try:
            with open(self._p(key), "wb") as f:
                pickle.dump(val, f)
        except:
            pass
'@
Set-Content -Encoding utf8 -Path 'src/altseason/cache.py' -Value $c

# 9) src/altseason/config.py  (توکن‌ها را بعداً جایگزین کن)
$c = @'
import os
TELEGRAM_BOT_TOKEN_HARDCODED = "REPLACE_ME"
TELEGRAM_CHAT_ID_HARDCODED   = "REPLACE_ME"
BINANCE_BASE   = os.getenv("BINANCE_BASE","https://api.binance.com/api/v3")
COINGECKO_BASE = os.getenv("COINGECKO_BASE","https://api.coingecko.com/api/v3")
TZ_DISPLAY     = os.getenv("TZ_DISPLAY","Europe/Stockholm")
FACTOR_WEIGHTS = {"btc_dominance":20,"eth_btc":20,"total2":15,"total3":15,"btc_regime":15,"eth_trend":15}
THRESHOLDS = {
  "DOM_SLOPE_MIN":-0.1,"ETHBTC_RSI_MIN":55,"TOTAL_RSI_MIN":55,"TOTAL_ADX_MIN":18,"TOTAL_ADX_MAX":28,
  "BTC_RSI_MIN":45,"BTC_RSI_MAX":60,"BTC_ADX_MAX":25,"BTC_ATR_MULTIPLIER":1.5,
  "ALTSEASON_MIN_SCORE":75,"ALTSEASON_MIN_FACTORS":4,"FORMING_MIN":60,"NEUTRAL_MIN":45
}
CACHE_TTL_HOURS=24; MAX_RETRIES=3
def get_telegram_token(): return os.getenv("TELEGRAM_BOT_TOKEN") or TELEGRAM_BOT_TOKEN_HARDCODED
def get_telegram_chat_id(): return os.getenv("TELEGRAM_CHAT_ID") or TELEGRAM_CHAT_ID_HARDCODED
'@
Set-Content -Encoding utf8 -Path 'src/altseason/config.py' -Value $c

# 10) src/altseason/indicators.py
$c = @'
import pandas as pd, numpy as np
def ema(s: pd.Series, span:int): return s.ewm(span=span, adjust=False).mean()
def rsi(s: pd.Series, window:int=14):
    d=s.diff(); g=(d.where(d>0,0)).rolling(window).mean(); l=(-d.where(d<0,0)).rolling(window).mean()
    rs=g/l.replace(0,np.finfo(float).eps); return 100 - (100/(1+rs))
def adx(h,l,c,window:int=14):
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    pdi=100*(pd.Series(plus,index=h.index).ewm(alpha=1/window).mean()/tr.ewm(alpha=1/window).mean())
    mdi=100*(pd.Series(minus,index=h.index).ewm(alpha=1/window).mean()/tr.ewm(alpha=1/window).mean())
    dx=100*((pdi-mdi).abs()/(pdi+mdi).replace(0,np.finfo(float).eps)); ax=dx.ewm(alpha=1/window).mean()
    return ax,pdi,mdi
def atr(h,l,c,window:int=14):
    tr1=h-l; tr2=(h-c.shift()).abs(); tr3=(l-c.shift()).abs()
    return pd.concat([tr1,tr2,tr3],axis=1).max(axis=1).rolling(window).mean()
def slope(s: pd.Series, n:int=5):
    if len(s)<n: return 0.0
    y=s.tail(n).values; x=np.arange(len(y)); m=~np.isnan(y)
    if m.sum()<2: return 0.0
    A=np.vstack([x[m],np.ones(m.sum())]).T; sl,_=np.linalg.lstsq(A,y[m],rcond=None)[0]; return float(sl)
def cross_above(a,b): return len(a)>=2 and len(b)>=2 and a.iloc[-2]<=b.iloc[-2] and a.iloc[-1]>b.iloc[-1]
def cross_below(a,b): return len(a)>=2 and len(b)>=2 and a.iloc[-2]>=b.iloc[-2] and a.iloc[-1]<b.iloc[-1]
'@
Set-Content -Encoding utf8 -Path 'src/altseason/indicators.py' -Value $c

# 11) src/altseason/factors.py
$c = @'
import pandas as pd, logging
from typing import Dict, Any
from .indicators import ema, rsi, adx, atr, slope, cross_above, cross_below
from .config import THRESHOLDS
logger=logging.getLogger("altseason")
class FactorCalculator:
    def __init__(self): self.thresholds=THRESHOLDS
    def _res(self,score,ok,exp): return {"score":min(int(score),20),"ok":bool(ok),"explain":str(exp)}
    def _def(self):
        return {k:self._res(0,False,"Calculation failed") for k in ["btc_dominance","eth_btc","total2","total3","btc_regime","eth_trend"]}
    def compute_factors(self,raw):
        try:
            return {"btc_dominance":self._btc_dom(raw),"eth_btc":self._eth_btc(raw),"total2":self._t2(raw),
                    "total3":self._t3(raw),"btc_regime":self._btc_regime(raw),"eth_trend":self._eth_trend(raw)}
        except Exception as e: logger.error(f"Error computing factors: {e}"); return self._def()
    def _btc_dom(self,raw):
        s=raw.get("btc_dom_series",pd.Series(dtype=float))
        if s.empty or len(s)<200: return self._res(0,False,"Insufficient dominance data")
        ma50,ma200=ema(s,50),ema(s,200); msl=slope(ma50,10); cur=s.iloc[-1]; recent_low=s.tail(20).min()
        sc=0; c=0
        if ma50.iloc[-1]<ma200.iloc[-1]: sc+=7;c+=1
        if msl<self.thresholds["DOM_SLOPE_MIN"]: sc+=7;c+=1
        if cur<recent_low: sc+=6;c+=1
        return self._res(sc, c>=2, f"Dom:{cur:.1f}%, MA50<MA200:{ma50.iloc[-1]:.1f}<{ma200.iloc[-1]:.1f}, Slope:{msl:.3f}")
    def _eth_btc(self,raw):
        df=raw.get("eth_btc",pd.DataFrame())
        if df.empty or len(df)<200: return self._res(0,False,"Insufficient ETH/BTC data")
        close=df["close"]; e50,e200=ema(close,50),ema(close,200); r=rsi(close,14)
        sc=0;c=0
        if close.iloc[-1]>e200.iloc[-1]: sc+=7;c+=1
        if cross_above(e50,e200): sc+=7;c+=1
        if r.iloc[-1]>=self.thresholds["ETHBTC_RSI_MIN"]: sc+=6;c+=1
        return self._res(sc, c>=2, f"Price:{close.iloc[-1]:.6f}, EMA50>EMA200:{e50.iloc[-1]:.6f}>{e200.iloc[-1]:.6f}, RSI:{r.iloc[-1]:.1f}")
    def _t2(self,raw):
        s=raw.get("total2_series",pd.Series(dtype=float))
        if s.empty or len(s)<50: return self._res(0,False,"Insufficient TOTAL2 data")
        e50=ema(s,50); es=slope(e50,10); cur=s.iloc[-1]; rh=s.tail(20).max()
        sc=0;c=0
        if cur>e50.iloc[-1]: sc+=5;c+=1
        if es>0: sc+=5;c+=1
        if cur>=rh*0.98: sc+=5;c+=1
        return self._res(sc, c>=2, f"Close>EMA50:{cur:.1f}>{e50.iloc[-1]:.1f}, Slope:{es:.3f}")
    def _t3(self,raw):
        s=raw.get("total3_series",pd.Series(dtype=float))
        if s.empty or len(s)<50: return self._res(0,False,"Insufficient TOTAL3 data")
        e50=ema(s,50); r=rsi(s,14); cur=s.iloc[-1]; rh=s.tail(20).max()
        sc=0;c=0
        if cur>e50.iloc[-1]: sc+=5;c+=1
        if r.iloc[-1]>self.thresholds["TOTAL_RSI_MIN"]: sc+=5;c+=1
        if cur>=rh*0.98: sc+=5;c+=1
        return self._res(sc, c>=2, f"Close>EMA50:{cur:.1f}>{e50.iloc[-1]:.1f}, RSI:{r.iloc[-1]:.1f}, NearHigh:{cur>=rh*0.98}")
    def _btc_regime(self,raw):
        df=raw.get("btcusdt",pd.DataFrame())
        if df.empty or len(df)<50: return self._res(0,False,"Insufficient BTC data")
        c,h,l=df["close"],df["high"],df["low"]; r=rsi(c,14); ax,_,_=adx(h,l,c,14); e50=ema(c,50)
        cr,ca,pe=r.iloc[-1],ax.iloc[-1],abs(c.iloc[-1]-e50.iloc[-1]); from .indicators import atr as _atr; at=_atr(h,l,c,14).iloc[-1]
        sc=0
        if  self.thresholds["BTC_RSI_MIN"]<=cr<=self.thresholds["BTC_RSI_MAX"]: sc+=5
        if ca<self.thresholds["BTC_ADX_MAX"]: sc+=5
        if pe<=at*self.thresholds["BTC_ATR_MULTIPLIER"]: sc+=5
        return self._res(sc, sc>=10, f"RSI:{cr:.1f}, ADX:{ca:.1f}, Price-EMA:{pe:.1f}")
    def _eth_trend(self,raw):
        df=raw.get("ethusdt",pd.DataFrame())
        if df.empty or len(df)<50: return self._res(0,False,"Insufficient ETH data")
        c=df["close"]; e50=ema(c,50); r=rsi(c,14); es=slope(e50,10); rc=r.iloc[-1]; rp=r.iloc[-11] if len(r)>=11 else r.iloc[0]; rd=rc-rp
        sc=0;cnd=0
        if c.iloc[-1]>e50.iloc[-1]: sc+=5;cnd+=1
        if rd>0: sc+=5;cnd+=1
        if es>0: sc+=5;cnd+=1
        return self._res(sc, cnd>=2, f"Close>EMA50:{c.iloc[-1]:.1f}>{e50.iloc[-1]:.1f}, RSI change:{rd:.1f}, EMA Slope:{es:.3f}")
'@
Set-Content -Encoding utf8 -Path 'src/altseason/factors.py' -Value $c

# 12) src/altseason/scoring.py
$c = @'
from typing import Dict, Any, Tuple
from .config import FACTOR_WEIGHTS, THRESHOLDS
class ScoreCalculator:
    def __init__(self): self.weights=FACTOR_WEIGHTS; self.th=THRESHOLDS
    def calculate_total_score(self, f: Dict[str, Any]) -> Tuple[int,str]:
        s=0; okc=0
        for k,v in f.items():
            if k in ("total_score","status"): continue
            s += int(v.get("score",0))
            if v.get("ok",False): okc+=1
        if s>=self.th["ALTSEASON_MIN_SCORE"] and okc>=self.th["ALTSEASON_MIN_FACTORS"]: return s,"Altseason Likely"
        if s>=self.th["FORMING_MIN"]: return s,"Forming / Watch"
        if s>=self.th["NEUTRAL_MIN"]: return s,"Neutral"
        return s,"Risk-Off"
'@
Set-Content -Encoding utf8 -Path 'src/altseason/scoring.py' -Value $c

# 13) src/altseason/report.py
$c = @'
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
class ReportGenerator:
    def __init__(self, reports_dir: str="reports"):
        self.reports_dir=Path(reports_dir); self.reports_dir.mkdir(exist_ok=True)
    def generate_daily_report(self, factors: Dict[str,Any], total_score:int, status:str, cache_used:bool=False):
        d=datetime.now(timezone.utc).strftime("%Y-%m-%d"); self._md(d,factors,total_score,status,cache_used); self._state(d,factors,total_score,status)
    def _md(self, d, f, s, st, cache):
        p=self.reports_dir/f"{d}.md"
        emo={"Altseason Likely":"GREEN","Forming / Watch":"YELLOW","Neutral":"WHITE","Risk-Off":"RED"}.get(st,"WHITE")
        c=f"""# Altseason Radar Report - {d}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Display Timezone:** Europe/Stockholm

## Summary

**Status:** {emo} {st}
**Total Score:** {s}/100

{('[NOTE] Some data was loaded from cache (fallback mode)') if cache else ''}

## Factor Analysis

| Factor | Score | Status | Explanation |
|--------|-------|--------|-------------|
"""
        for n,dv in f.items():
            if n in ("total_score","status"): continue
            sc=dv.get("score",0); ok=dv.get("ok",False); ex=dv.get("explain","N/A")
            mx={"btc_dominance":20,"eth_btc":20,"total2":15,"total3":15,"btc_regime":15,"eth_trend":15}.get(n,20)
            c+=f"| {n.replace('_',' ').title()} | {sc}/{mx} | {'OK' if ok else 'X'} | {ex} |\n"
        c += f"""
## Today's Verdict

**{st}** - Score: {s}/100

### Interpretation:
"""
        if st=="Altseason Likely":
            c+="- Strong signals across multiple factors\n- High probability of altseason conditions\n- Consider monitoring for entry opportunities"
        elif st=="Forming / Watch":
            c+="- Promising signals developing\n- Monitor for confirmation\n- Prepare watchlists"
        elif st=="Neutral":
            c+="- Mixed or weak signals\n- Market in transition\n- Wait for clearer direction"
        else:
            c+="- Weak or negative signals\n- Risk-off conditions prevail\n- Consider defensive positions"
        p.write_text(c, encoding="utf-8")
    def _state(self, d, f, s, st):
        (self.reports_dir/"state.json").write_text(json.dumps({"date":d,"timestamp":datetime.now(timezone.utc).isoformat(),"total_score":s,"status":st,"factors":f}, indent=2, ensure_ascii=False), encoding="utf-8")
'@
Set-Content -Encoding utf8 -Path 'src/altseason/report.py' -Value $c

# 14) src/altseason/telegram.py
$c = @'
import requests, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import get_telegram_token, get_telegram_chat_id
logger=logging.getLogger("altseason")
class TelegramNotifier:
    def __init__(self): self.token=get_telegram_token(); self.chat_id=get_telegram_chat_id()
    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def send_message(self, text:str)->bool:
        if not self.token or not self.chat_id: logger.warning("Telegram not configured"); return False
        try:
            u=f"https://api.telegram.org/bot{self.token}/sendMessage"
            r=requests.post(u,json={"chat_id":self.chat_id,"text":text,"parse_mode":"HTML","disable_web_page_preview":True},timeout=10)
            r.raise_for_status(); logger.info("Telegram message sent"); return True
        except requests.RequestException as e: logger.warning(f"Failed to send Telegram message: {e}"); return False
        except Exception as e: logger.error(f"Unexpected error sending Telegram message: {e}"); return False
'@
Set-Content -Encoding utf8 -Path 'src/altseason/telegram.py' -Value $c

# 15) src/altseason/data_providers/__init__.py
$c = @'
from .binance_ohlcv import BinanceOHLCV
from .coingecko_global import CoinGeckoGlobal
from .health import DataHealth
__all__=["BinanceOHLCV","CoinGeckoGlobal","DataHealth"]
'@
Set-Content -Encoding utf8 -Path 'src/altseason/data_providers/__init__.py' -Value $c

# 16) src/altseason/data_providers/binance_ohlcv.py
$c = @'
import pandas as pd, requests, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List
from ...config import BINANCE_BASE, MAX_RETRIES
from ...cache import SimpleCache
logger=logging.getLogger("altseason")
class BinanceOHLCV:
    def __init__(self): self.base_url=BINANCE_BASE; self.cache=SimpleCache()
    @retry(stop=stop_after_attempt(MAX_RETRIES),wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def get_klines(self, symbol:str, interval:str="1d", limit:int=400)->pd.DataFrame:
        k=f"binance_{symbol}_{interval}_{limit}"; c=self.cache.get(k)
        if c is not None: logger.info(f"Using cached {symbol}"); return c
        r=requests.get(f"{self.base_url}/klines", params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=10); r.raise_for_status()
        d=r.json(); cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
        df=pd.DataFrame(d,columns=cols); df["open_time"]=pd.to_datetime(df["open_time"],unit="ms",utc=True); df["close_time"]=pd.to_datetime(df["close_time"],unit="ms",utc=True)
        for c2 in ["open","high","low","close","volume","quote_volume"]: df[c2]=pd.to_numeric(df[c2], errors="coerce")
        df=df.dropna().sort_values("open_time").reset_index(drop=True); self.cache.set(k,df); return df
    def get_multiple_symbols(self, symbols: List[str])->Dict[str,pd.DataFrame]:
        out={}
        for s in symbols:
            try: out[s]=self.get_klines(s)
            except Exception as e: logger.warning(f"Failed to get data for {s}: {e}")
        return out
'@
Set-Content -Encoding utf8 -Path 'src/altseason/data_providers/binance_ohlcv.py' -Value $c

# 17) src/altseason/data_providers/coingecko_global.py
$c = @'
import requests, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any
from ...config import COINGECKO_BASE, MAX_RETRIES
from ...cache import SimpleCache
logger=logging.getLogger("altseason")
class CoinGeckoGlobal:
    def __init__(self): self.base_url=COINGECKO_BASE; self.cache=SimpleCache()
    @retry(stop=stop_after_attempt(MAX_RETRIES),wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def get_global_data(self)->Dict[str,Any]:
        k="coingecko_global"; c=self.cache.get(k)
        if c is not None: logger.info("Using cached CoinGecko global data"); return c
        r=requests.get(f"{self.base_url}/global", timeout=10); r.raise_for_status(); data=r.json()["data"]; self.cache.set(k,data); return data
    def get_market_caps(self)->Dict[str,float]:
        d=self.get_global_data(); total=d["total_market_cap"]["usd"]; btc=d["market_cap_percentage"]["btc"]*total/100; eth=d["market_cap_percentage"]["eth"]*total/100
        return {"total_mcap":total,"btc_mcap":btc,"eth_mcap":eth}
'@
Set-Content -Encoding utf8 -Path 'src/altseason/data_providers/coingecko_global.py' -Value $c

# 18) src/altseason/data_providers/health.py
$c = @'
import logging
from typing import Dict, Tuple
from ..data_providers.binance_ohlcv import BinanceOHLCV
from ..data_providers.coingecko_global import CoinGeckoGlobal
logger=logging.getLogger("altseason")
class DataHealth:
    def __init__(self): self.binance=BinanceOHLCV(); self.coingecko=CoinGeckoGlobal()
    def self_test(self)->Tuple[bool,Dict[str,str]]:
        issues={}
        try:
            btc=self.binance.get_klines("BTCUSDT",limit=10)
            if len(btc)<5: issues["binance_length"]=f"Insufficient BTC data: {len(btc)} rows"
        except Exception as e: issues["binance"]=f"Binance API failed: {e}"
        try:
            m=self.coingecko.get_market_caps()
            if m["total_mcap"]<=m["btc_mcap"]: issues["coingecko_relationship"]="Total MCap <= BTC MCap"
            if m["btc_mcap"]<=m["eth_mcap"]:  issues["coingecko_btc_eth"]="BTC MCap <= ETH MCap"
            if m["total_mcap"]<=0:            issues["coingecko_negative"]="Negative market cap values"
        except Exception as e: issues["coingecko"]=f"CoinGecko API failed: {e}"
        return (len(issues)==0), issues
'@
Set-Content -Encoding utf8 -Path 'src/altseason/data_providers/health.py' -Value $c

# 19) src/altseason/runner.py
$c = @'
import logging
from typing import Dict, Any
import pandas as pd
from .logging_utils import setup_logging
from .data_providers.health import DataHealth
from .data_providers.binance_ohlcv import BinanceOHLCV
from .data_providers.coingecko_global import CoinGeckoGlobal
from .factors import FactorCalculator
from .scoring import ScoreCalculator
from .report import ReportGenerator
from .cache import SimpleCache
logger=setup_logging()
class AltseasonRunner:
    def __init__(self):
        self.health_checker=DataHealth(); self.binance=BinanceOHLCV(); self.coingecko=CoinGeckoGlobal()
        self.factor_calc=FactorCalculator(); self.score_calc=ScoreCalculator(); self.report_gen=ReportGenerator()
        self.cache=SimpleCache(); self.cache_used=False
    def run_daily_analysis(self)->bool:
        try:
            _=self._run_health_check(); raw=self._fetch_all_data()
            f=self.factor_calc.compute_factors(raw); total,status=self.score_calc.calculate_total_score(f)
            f["total_score"],f["status"]=total,status
            self.report_gen.generate_daily_report(f,total,status,self.cache_used)
            logger.info("Daily analysis completed successfully"); return True
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}"); return False
    def _run_health_check(self)->bool:
        try:
            ok,issues=self.health_checker.self_test()
            if not ok: logger.warning(f"Health check issues: {issues}")
            return True
        except Exception as e: logger.warning(f"Health check failed: {e}"); return True
    def _fetch_all_data(self)->Dict[str,Any]:
        raw={}
        try:
            mc=self.coingecko.get_market_caps(); raw["market_caps"]=mc
            total,btc,eth=mc["total_mcap"],mc["btc_mcap"],mc["eth_mcap"]
            raw["btc_dom_series"]=pd.Series([(btc/total)*100.0]*200)
            raw["total2_series"]=pd.Series([float(total-btc)]*200)
            raw["total3_series"]=pd.Series([float(total-btc-eth)]*200)
            o=self.binance.get_multiple_symbols(["BTCUSDT","ETHUSDT","ETHBTC"])
            if "BTCUSDT" in o: raw["btcusdt"]=o["BTCUSDT"]
            if "ETHUSDT" in o: raw["ethusdt"]=o["ETHUSDT"]
            if "ETHBTC"  in o: raw["eth_btc"]=o["ETHBTC"]
            try: self.cache.set("last_raw_data",raw)
            except: pass
            return raw
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            c=self.cache.get("last_raw_data")
            if c: self.cache_used=True; return c
            raise
'@
Set-Content -Encoding utf8 -Path 'src/altseason/runner.py' -Value $c

Write-Host 'DONE'
'@

Set-Content -Encoding utf8 -Path .\write_all_full.ps1 -Value $code
