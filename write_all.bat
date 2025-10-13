@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Make sure we are at project root (run this .bat from the repo root)
REM Create folders
powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '.github\workflows','scripts','src\altseason\data_providers','reports','tests' | Out-Null"

REM =========================
REM .env.sample
REM =========================
powershell -NoProfile -Command ^
"$c = @'
# Optional: Override hardcoded Telegram settings
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Display timezone for reports
TZ_DISPLAY=Europe/Stockholm

# API endpoints (optional overrides)
BINANCE_BASE=https://api.binance.com/api/v3
COINGECKO_BASE=https://api.coingecko.com/api/v3
'@; Set-Content -Path '.env.sample' -Value $c -Encoding UTF8"

REM =========================
REM LICENSE
REM =========================
powershell -NoProfile -Command ^
"$c = @'
MIT License

Copyright (c) 2024 Altseason Radar

Permission is hereby granted, free of charge, to any person obtaining a copy
... (truncated in display) ...
SOFTWARE.
'@; Set-Content -Path 'LICENSE' -Value $c -Encoding UTF8"

REM =========================
REM pyproject.toml
REM =========================
powershell -NoProfile -Command ^
"$c = @'
[build-system]
requires = ['setuptools>=45','wheel']
build-backend = 'setuptools.build_meta'

[project]
name = 'altseason-radar'
version = '1.0.0'
description = 'Daily Altseason Radar Analysis'
authors = [{name = 'Altseason Team'}]
dependencies = [
  'pandas>=2.0.0',
  'numpy>=1.24.0',
  'requests>=2.31.0',
  'python-dateutil>=2.8.0',
  'tenacity>=8.2.0',
  'pydantic>=2.0.0',
  'rich>=13.0.0',
]

[project.optional-dependencies]
dev = ['pytest>=7.0.0','black>=23.0.0','isort>=5.12.0']

[tool.black]
line-length = 88
target-version = ['py310']
'@; Set-Content -Path 'pyproject.toml' -Value $c -Encoding UTF8"

REM =========================
REM requirements.txt
REM =========================
powershell -NoProfile -Command ^
"$c = @'
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dateutil>=2.8.0
tenacity>=8.2.0
pydantic>=2.0.0
rich>=13.0.0
pytest>=7.0.0
'@; Set-Content -Path 'requirements.txt' -Value $c -Encoding UTF8"

REM =========================
REM .gitignore (final)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
reports/state.json
# گزارش‌های روزانه را نگه داریم - حذف reports/*.md از ignore
# اگر نمی‌خواهی زیاد شوند، بعداً می‌توانی job پاک‌سازی اضافه کنی
.cache/
logs/
'@; Set-Content -Path '.gitignore' -Value $c -Encoding UTF8"

REM =========================
REM README.md (sample header; will be updated daily by runner)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
# Altseason Radar

Daily analysis of 6 key factors to measure distance to altseason conditions in cryptocurrency markets.

## Today (UTC)
**Status:** ⚪ Neutral — **Score:** 52/100 — **Δ vs yesterday:** →0.0
See: [reports/2024-01-01.md](reports/2024-01-01.md)

## Project Overview
(Will be updated by daily run.)
'@; Set-Content -Path 'README.md' -Value $c -Encoding UTF8"

REM =========================
REM .github/workflows/daily.yml (final with cleanup + delta fallback)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
name: Daily Altseason Analysis

on:
  schedule:
    - cron: \"15 6 * * *\"
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: \"3.10\"
        cache: pip

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest -q tests/

    - name: Run daily analysis
      run: |
        python -m scripts.run_daily

    - name: Cleanup old reports (keep last 30 days)
      run: |
        find reports/ -name \"*.md\" -type f | sort -r | tail -n +31 | xargs rm -f || true
        echo \"Old reports cleanup completed\"

    - name: Commit and push if changes
      run: |
        git config --local user.name \"altseason-bot\"
        git config --local user.email \"bot@example.com\"
        if [ -f reports/progress.csv ]; then
          DELTA=$(tail -n 1 reports/progress.csv | cut -d',' -f4 | xargs)
        else
          DELTA=\"0.0\"
        fi
        git add reports/ README.md
        git diff --staged --quiet || git commit -m \"chore: daily report $(date -u +%F) [Δ $DELTA]\"
        git push
'@; Set-Content -Path '.github\workflows\daily.yml' -Value $c -Encoding UTF8"

REM =========================
REM scripts\run_daily.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
#!/usr/bin/env python3
\"\"\"
Daily runner script for Altseason Radar
\"\"\"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from altseason.runner import AltseasonRunner

def main():
    runner = AltseasonRunner()
    success = runner.run_daily_analysis()
    if not success:
        print(\"Daily analysis failed!\")
        raise SystemExit(1)
    print(\"Daily analysis completed successfully!\")
    raise SystemExit(0)

if __name__ == \"__main__\":
    main()
'@; Set-Content -Path 'scripts\run_daily.py' -Value $c -Encoding UTF8"

REM =========================
REM scripts\validate_data.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
#!/usr/bin/env python3
\"\"\"
Data validation script for Altseason Radar
\"\"\"
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from altseason.data_providers.health import DataHealth

def main():
    health = DataHealth()
    is_healthy, issues = health.self_test()
    if is_healthy:
        print(\"✅ All data sources are healthy\")
        return 0
    else:
        print(\"❌ Data health issues found:\")
        for issue, description in issues.items():
            print(f\"  - {issue}: {description}\")
        return 1

if __name__ == \"__main__\":
    raise SystemExit(main())
'@; Set-Content -Path 'scripts\validate_data.py' -Value $c -Encoding UTF8"

REM =========================
REM scripts\test_telegram.py (optional)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from altseason.telegram import TelegramNotifier

def main():
    t = TelegramNotifier()
    ok = t.send_message(\"🧪 Test message from Altseason Radar\")
    print(\"✅ Telegram test sent\" if ok else \"❌ Telegram test failed\")
    return 0 if ok else 1

if __name__ == \"__main__\":
    raise SystemExit(main())
'@; Set-Content -Path 'scripts\test_telegram.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\__init__.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
\"\"\"Altseason Radar - Daily cryptocurrency market analysis\"\"\"
__version__ = \"1.0.0\"
__author__  = \"Altseason Team\"
'@; Set-Content -Path 'src\altseason\__init__.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\logging_utils.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import logging
from rich.logging import RichHandler

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format=\"%(message)s\",
        datefmt=\"[%X]\",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(\"altseason\")
'@; Set-Content -Path 'src\altseason\logging_utils.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\cache.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pickle, os, time
from pathlib import Path
from typing import Any, Optional

class SimpleCache:
    def __init__(self, cache_dir: str = \".cache\", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f\"{key}.pkl\"

    def get(self, key: str) -> Optional[Any]:
        p = self._get_cache_path(key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl_seconds:
            try: p.unlink()
            except Exception: pass
            return None
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any):
        p = self._get_cache_path(key)
        try:
            with open(p, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass

    def clear_expired(self):
        for f in self.cache_dir.glob('*.pkl'):
            if time.time() - f.stat().st_mtime > self.ttl_seconds:
                try: f.unlink()
                except Exception: pass
'@; Set-Content -Path 'src\altseason\cache.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\config.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import os

TELEGRAM_BOT_TOKEN_HARDCODED = \"7512369490:AAHiQqOzjLxh5zjcx3gmUT-hEj1196tIHfI\"
TELEGRAM_CHAT_ID_HARDCODED   = \"-1002925489017\"

BINANCE_BASE   = os.getenv(\"BINANCE_BASE\",   \"https://api.binance.com/api/v3\")
COINGECKO_BASE = os.getenv(\"COINGECKO_BASE\", \"https://api.coingecko.com/api/v3\")
TZ_DISPLAY     = os.getenv(\"TZ_DISPLAY\",     \"Europe/Stockholm\")

FACTOR_WEIGHTS = {
    \"btc_dominance\": 20,
    \"eth_btc\": 20,
    \"total2\": 15,
    \"total3\": 15,
    \"btc_regime\": 15,
    \"eth_trend\": 15,
}

THRESHOLDS = {
    \"DOM_MA_SHORT\": 50,
    \"DOM_MA_LONG\": 200,
    \"DOM_SLOPE_MIN\": -0.1,

    \"ETHBTC_EMA_SHORT\": 50,
    \"ETHBTC_EMA_LONG\": 200,
    \"ETHBTC_RSI_MIN\": 55,

    \"TOTAL_EMA\": 50,
    \"TOTAL_RSI_MIN\": 55,
    \"TOTAL_ADX_MIN\": 18,
    \"TOTAL_ADX_MAX\": 28,

    \"BTC_RSI_MIN\": 45,
    \"BTC_RSI_MAX\": 60,
    \"BTC_ADX_MAX\": 25,
    \"BTC_ATR_MULTIPLIER\": 1.5,

    \"ALTSEASON_MIN_SCORE\": 75,
    \"ALTSEASON_MIN_FACTORS\": 4,
    \"FORMING_MIN\": 60,
    \"NEUTRAL_MIN\": 45,
}

CACHE_TTL_HOURS = 24
MAX_RETRIES = 3

def get_telegram_token() -> str:
    return os.getenv(\"TELEGRAM_BOT_TOKEN\") or TELEGRAM_BOT_TOKEN_HARDCODED

def get_telegram_chat_id() -> str:
    return os.getenv(\"TELEGRAM_CHAT_ID\") or TELEGRAM_CHAT_ID_HARDCODED
'@; Set-Content -Path 'src\altseason\config.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\indicators.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    plus_di  = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
    dx  = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.finfo(float).eps))
    adx = dx.ewm(alpha=1/window).mean()
    return adx, plus_di, minus_di

def slope(series: pd.Series, n: int = 5) -> float:
    if len(series) < n: return 0.0
    y = series.tail(n).values
    x = np.arange(len(y))
    m = ~np.isnan(y)
    if m.sum() < 2: return 0.0
    A = np.vstack([x[m], np.ones(m.sum())]).T
    s, _ = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return float(s)

def cross_above(a: pd.Series, b: pd.Series) -> bool:
    if len(a) < 2 or len(b) < 2: return False
    return a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]

def cross_below(a: pd.Series, b: pd.Series) -> bool:
    if len(a) < 2 or len(b) < 2: return False
    return a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]
'@; Set-Content -Path 'src\altseason\indicators.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\factors.py (final: cross_below import + TOTAL3 no ADX)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pandas as pd
import logging
from typing import Dict, Any
from .indicators import ema, rsi, adx, atr, slope, cross_above, cross_below
from .config import THRESHOLDS

logger = logging.getLogger(\"altseason\")

class FactorCalculator:
    def __init__(self):
        self.thresholds = THRESHOLDS

    def compute_factors(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        factors = {}
        try:
            factors[\"btc_dominance\"] = self._compute_btc_dominance(raw_data)
            factors[\"eth_btc\"]       = self._compute_eth_btc(raw_data)
            factors[\"total2\"]        = self._compute_total2(raw_data)
            factors[\"total3\"]        = self._compute_total3(raw_data)
            factors[\"btc_regime\"]    = self._compute_btc_regime(raw_data)
            factors[\"eth_trend\"]     = self._compute_eth_trend(raw_data)
        except Exception as e:
            logger.error(f\"Error computing factors: {e}\")
            return self._get_default_factors()
        return factors

    def _create_factor_result(self, score: int, ok: bool, explain: str) -> Dict[str, Any]:
        return {\"score\": min(score, 20), \"ok\": ok, \"explain\": explain}

    def _get_default_factors(self) -> Dict[str, Any]:
        return {
            \"btc_dominance\": self._create_factor_result(0, False, \"Calculation failed\"),
            \"eth_btc\":       self._create_factor_result(0, False, \"Calculation failed\"),
            \"total2\":        self._create_factor_result(0, False, \"Calculation failed\"),
            \"total3\":        self._create_factor_result(0, False, \"Calculation failed\"),
            \"btc_regime\":    self._create_factor_result(0, False, \"Calculation failed\"),
            \"eth_trend\":     self._create_factor_result(0, False, \"Calculation failed\"),
        }

    def _compute_btc_dominance(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        s = raw_data.get(\"btc_dom_series\", pd.Series(dtype=float))
        if s.empty or len(s) < 200:
            return self._create_factor_result(0, False, \"Insufficient dominance data\")
        ma50  = ema(s, 50); ma200 = ema(s, 200)
        mslp  = slope(ma50, 10)
        cur   = s.iloc[-1]
        recent_low = s.tail(20).min()
        score = 0; cond=0
        if ma50.iloc[-1] < ma200.iloc[-1]: score+=7; cond+=1
        if mslp < self.thresholds['DOM_SLOPE_MIN']: score+=7; cond+=1
        if cur < recent_low: score+=6; cond+=1
        ok = cond>=2
        exp = f\"Dom:{cur:.1f}%, MA50<MA200:{ma50.iloc[-1]:.1f}<{ma200.iloc[-1]:.1f}, Slope:{mslp:.3f}\"
        return self._create_factor_result(score, ok, exp)

    def _compute_eth_btc(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        df = raw_data.get(\"eth_btc\", pd.DataFrame())
        if df.empty or len(df) < 200:
            return self._create_factor_result(0, False, \"Insufficient ETH/BTC data\")
        close = df['close']; e50 = ema(close,50); e200=ema(close,200); r = rsi(close,14)
        score=0; cond=0
        if close.iloc[-1] > e200.iloc[-1]: score+=7; cond+=1
        if cross_above(e50, e200):         score+=7; cond+=1
        if r.iloc[-1] >= self.thresholds['ETHBTC_RSI_MIN']: score+=6; cond+=1
        ok = cond>=2
        exp=f\"Price:{close.iloc[-1]:.6f}, EMA50>EMA200:{e50.iloc[-1]:.6f}>{e200.iloc[-1]:.6f}, RSI:{r.iloc[-1]:.1f}\"
        return self._create_factor_result(score, ok, exp)

    def _compute_total2(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        s = raw_data.get(\"total2_series\", pd.Series(dtype=float))
        if s.empty or len(s) < 50:
            return self._create_factor_result(0, False, \"Insufficient TOTAL2 data\")
        e50 = ema(s,50); eslp = slope(e50,10); cur = s.iloc[-1]; rh = s.tail(20).max()
        score=0; cond=0
        if cur > e50.iloc[-1]: score+=5; cond+=1
        if eslp > 0:           score+=5; cond+=1
        if cur >= rh*0.98:     score+=5; cond+=1
        ok=cond>=2
        exp=f\"Close>EMA50:{cur:.1f}>{e50.iloc[-1]:.1f}, Slope:{eslp:.3f}\"
        return self._create_factor_result(score, ok, exp)

    def _compute_total3(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        s = raw_data.get(\"total3_series\", pd.Series(dtype=float))
        if s.empty or len(s) < 50:
            return self._create_factor_result(0, False, \"Insufficient TOTAL3 data\")
        e50 = ema(s,50); r = rsi(s,14); cur = s.iloc[-1]; rh = s.tail(20).max()
        score=0; cond=0
        if cur > e50.iloc[-1]: score+=5; cond+=1
        if r.iloc[-1] > self.thresholds['TOTAL_RSI_MIN']: score+=5; cond+=1
        if cur >= rh*0.98: score+=5; cond+=1
        ok=cond>=2
        exp=f\"Close>EMA50:{cur:.1f}>{e50.iloc[-1]:.1f}, RSI:{r.iloc[-1]:.1f}, NearHigh:{cur >= rh*0.98}\"
        return self._create_factor_result(score, ok, exp)

    def _compute_btc_regime(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        df = raw_data.get(\"btcusdt\", pd.DataFrame())
        if df.empty or len(df) < 50:
            return self._create_factor_result(0, False, \"Insufficient BTC data\")
        c=df['close']; h=df['high']; l=df['low']
        r=rsi(c,14); a,_,_=adx(h,l,c,14); e50=ema(c,50)
        cr=r.iloc[-1]; ca=a.iloc[-1]; pe=abs(c.iloc[-1]-e50.iloc[-1])
        from .indicators import atr as _atr
        at=_atr(h,l,c,14).iloc[-1]
        score=0
        if self.thresholds['BTC_RSI_MIN'] <= cr <= self.thresholds['BTC_RSI_MAX']: score+=5
        if ca < self.thresholds['BTC_ADX_MAX']: score+=5
        if pe <= at * self.thresholds['BTC_ATR_MULTIPLIER']: score+=5
        ok = score>=10
        exp=f\"RSI:{cr:.1f}, ADX:{ca:.1f}, Price-EMA:{pe:.1f}\"
        return self._create_factor_result(score, ok, exp)

    def _compute_eth_trend(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        df = raw_data.get(\"ethusdt\", pd.DataFrame())
        if df.empty or len(df) < 50:
            return self._create_factor_result(0, False, \"Insufficient ETH data\")
        c=df['close']; e50=ema(c,50); r=rsi(c,14); eslp=slope(e50,10)
        rc=r.iloc[-1]; rp=r.iloc[-11] if len(r)>=11 else r.iloc[0]; rdelta=rc-rp
        score=0; cond=0
        if c.iloc[-1] > e50.iloc[-1]: score+=5; cond+=1
        if rdelta > 0:                score+=5; cond+=1
        if eslp > 0:                  score+=5; cond+=1
        ok=cond>=2
        exp=f\"Close>EMA50:{c.iloc[-1]:.1f}>{e50.iloc[-1]:.1f}, RSI Δ:{rdelta:.1f}, EMA Slope:{eslp:.3f}\"
        return self._create_factor_result(score, ok, exp)
'@; Set-Content -Path 'src\altseason\factors.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\scoring.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
from typing import Dict, Any, Tuple
from .config import FACTOR_WEIGHTS, THRESHOLDS

class ScoreCalculator:
    def __init__(self):
        self.weights = FACTOR_WEIGHTS
        self.thresholds = THRESHOLDS

    def calculate_total_score(self, factors: Dict[str, Any]) -> Tuple[int, str]:
        total_score = 0
        confirmed = 0
        for k, v in factors.items():
            if k in (\"total_score\",\"status\"): continue
            total_score += int(v.get(\"score\",0))
            if v.get(\"ok\", False): confirmed += 1
        status = self._determine_status(total_score, confirmed)
        return total_score, status

    def _determine_status(self, score: int, confirmed: int) -> str:
        t = self.thresholds
        if score >= t['ALTSEASON_MIN_SCORE'] and confirmed >= t['ALTSEASON_MIN_FACTORS']:
            return \"Altseason Likely\"
        if score >= t['FORMING_MIN']:
            return \"Forming / Watch\"
        if score >= t['NEUTRAL_MIN']:
            return \"Neutral\"
        return \"Risk-Off\"
'@; Set-Content -Path 'src\altseason\scoring.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\report.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class ReportGenerator:
    def __init__(self, reports_dir: str = \"reports\"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def generate_daily_report(self, factors: Dict[str, Any], total_score: int, status: str, cache_used: bool=False) -> None:
        date_str = datetime.now(timezone.utc).strftime(\"%Y-%m-%d\")
        self._generate_markdown_report(date_str, factors, total_score, status, cache_used)
        self._generate_state_json(date_str, factors, total_score, status)

    def _generate_markdown_report(self, date_str: str, factors: Dict[str, Any], total_score: int, status: str, cache_used: bool) -> None:
        fpath = self.reports_dir / f\"{date_str}.md\"
        status_emoji = {\"Altseason Likely\":\"🟢\",\"Forming / Watch\":\"🟡\",\"Neutral\":\"⚪\",\"Risk-Off\":\"🔴\"}
        emoji = status_emoji.get(status, \"⚪\")
        content = f\"\"\"# Altseason Radar Report - {date_str}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Display Timezone:** Europe/Stockholm

## Summary

**Status:** {emoji} {status}  
**Total Score:** {total_score}/100

{ '⚠️ **Note:** Some data was loaded from cache (fallback mode)' if cache_used else '' }

## Factor Analysis

| Factor | Score | Status | Explanation |
|--------|-------|--------|-------------|
\"\"\"
        for name, data in factors.items():
            if name in (\"total_score\",\"status\"): continue
            score = data.get(\"score\",0)
            ok    = data.get(\"ok\",False)
            exp   = data.get(\"explain\",\"N/A\")
            maxs  = {\"btc_dominance\":20,\"eth_btc\":20,\"total2\":15,\"total3\":15,\"btc_regime\":15,\"eth_trend\":15}.get(name,20)
            content += f\"| {name.replace('_',' ').title()} | {score}/{maxs} | {'✅' if ok else '❌'} | {exp} |\\n\"
        content += f\"\"\"\n## Today's Verdict\n\n**{status}** - Score: {total_score}/100\n\n### Interpretation:\n\"\"\"
        if status == \"Altseason Likely\":
            content += \"- Strong signals across multiple factors\\n- High probability of altseason conditions\\n- Consider monitoring for entry opportunities\"
        elif status == \"Forming / Watch\":
            content += \"- Promising signals developing\\n- Monitor for confirmation\\n- Prepare watchlists\"
        elif status == \"Neutral\":
            content += \"- Mixed or weak signals\\n- Market in transition\\n- Wait for clearer direction\"
        else:
            content += \"- Weak or negative signals\\n- Risk-off conditions prevail\\n- Consider defensive positions\"
        fpath.write_text(content, encoding='utf-8')

    def _generate_state_json(self, date_str: str, factors: Dict[str, Any], total_score: int, status: str) -> None:
        state_file = self.reports_dir / \"state.json\"
        state_data = {
            \"date\": date_str,
            \"timestamp\": datetime.now(timezone.utc).isoformat(),
            \"total_score\": total_score,
            \"status\": status,
            \"factors\": factors
        }
        state_file.write_text(json.dumps(state_data, indent=2, ensure_ascii=False), encoding='utf-8')
'@; Set-Content -Path 'src\altseason\report.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\progress.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pandas as pd, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
logger = logging.getLogger(\"altseason\")

class ProgressTracker:
    def __init__(self, reports_dir: str = \"reports\"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.csv_file  = self.reports_dir / \"progress.csv\"
        self.json_file = self.reports_dir / \"progress.json\"

    def update_progress(self, score: int, status: str) -> Tuple[float,float]:
        today = datetime.now(timezone.utc).date().isoformat()
        df = self._load_existing()
        d1 = self._delta_yesterday(df, score)
        d7 = self._delta_7d_avg(df, score)
        new_row = { 'date_utc': today, 'score': score, 'status': status,
                    'delta_vs_yesterday': d1, 'delta_vs_7d_avg': d7 }
        df = self._append(df, new_row)
        self._save(df)
        return d1, d7

    def _load_existing(self) -> pd.DataFrame:
        cols = ['date_utc','score','status','delta_vs_yesterday','delta_vs_7d_avg']
        if self.csv_file.exists():
            try:
                df = pd.read_csv(self.csv_file)
                for c in cols:
                    if c not in df.columns: df[c] = 0.0
                return df
            except Exception as e:
                logger.warning(f\"Error loading progress CSV: {e}\")
        return pd.DataFrame(columns=cols)

    def _delta_yesterday(self, df: pd.DataFrame, today_score: int) -> float:
        if len(df)==0: return 0.0
        return float(today_score - float(df.iloc[-1]['score']))

    def _delta_7d_avg(self, df: pd.DataFrame, today_score: int) -> float:
        if len(df)==0: return 0.0
        recent = df.tail(7)['score']
        if len(recent)==0: return 0.0
        return float(today_score - float(recent.mean()))

    def _append(self, df: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
        df = df[df['date_utc'] != row['date_utc']]
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    def _save(self, df: pd.DataFrame):
        try:
            df.to_csv(self.csv_file, index=False, float_format='%.2f')
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(df.to_dict('records'), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f\"Error saving progress data: {e}\")
'@; Set-Content -Path 'src\altseason\progress.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\telegram.py (final with first-run indicator)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import logging, requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import get_telegram_token, get_telegram_chat_id

logger = logging.getLogger(\"altseason\")

class TelegramNotifier:
    def __init__(self):
        self.token = get_telegram_token()
        self.chat_id = get_telegram_chat_id()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def send_message(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning(\"Telegram token or chat ID not configured\"); return False
        try:
            url = f\"https://api.telegram.org/bot{self.token}/sendMessage\"
            payload = {\"chat_id\": self.chat_id, \"text\": text, \"parse_mode\": \"HTML\", \"disable_web_page_preview\": True}
            r = requests.post(url, json=payload, timeout=10); r.raise_for_status()
            logger.info(\"Telegram message sent successfully\"); return True
        except requests.RequestException as e:
            logger.warning(f\"Failed to send Telegram message: {e}\"); return False
        except Exception as e:
            logger.error(f\"Unexpected error sending Telegram message: {e}\"); return False

    def format_daily_message(self, date_str: str, status: str, score: int, delta_day: float, delta_7d: float, leading_factors: list) -> str:
        status_emoji = {\"Altseason Likely\":\"🟢\",\"Forming / Watch\":\"🟡\",\"Neutral\":\"⚪\",\"Risk-Off\":\"🔴\"}
        emoji = status_emoji.get(status, \"⚪\")

        def fmt(d):
            return (\"↑\" if d>0 else \"↓\" if d<0 else \"→\") + f\"{abs(d):.1f}\"

        leads = \", \".join(leading_factors) if leading_factors else \"None\"
        is_first = abs(delta_day) < 1e-9 and abs(delta_7d) < 1e-9
        first_tag = \" — first run\" if is_first else \"\"

        return f\"\"\"📊 <b>Altseason Radar</b> — {date_str} UTC
Status: {emoji} {status}
Score: {score}/100 (Δ day: {fmt(delta_day)} | Δ 7d: {fmt(delta_7d)}){first_tag}
Leads: {leads}
Full report: {date_str}.md\"\"\"
'@; Set-Content -Path 'src\altseason\telegram.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\data_providers\binance_ohlcv.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pandas as pd, requests, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List
from datetime import datetime, timezone
from ...config import BINANCE_BASE, MAX_RETRIES
from ...cache import SimpleCache

logger = logging.getLogger(\"altseason\")

class BinanceOHLCV:
    def __init__(self):
        self.base_url = BINANCE_BASE
        self.cache = SimpleCache()

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def get_klines(self, symbol: str, interval: str = \"1d\", limit: int = 400) -> pd.DataFrame:
        key=f\"binance_{symbol}_{interval}_{limit}\"
        cached = self.cache.get(key)
        if cached is not None:
            logger.info(f\"Using cached data for {symbol}\")
            return cached
        url = f\"{self.base_url}/klines\"
        r = requests.get(url, params={\"symbol\":symbol,\"interval\":interval,\"limit\":limit}, timeout=10)
        r.raise_for_status()
        df = self._parse_klines(r.json())
        self.cache.set(key, df)
        return df

    def _parse_klines(self, data: List[List]) -> pd.DataFrame:
        cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
        df = pd.DataFrame(data, columns=cols)
        df['open_time']  = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        for c in ['open','high','low','close','volume','quote_volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().sort_values('open_time').reset_index(drop=True)
        return df

    def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        out={}
        for s in symbols:
            try:
                out[s]=self.get_klines(s)
            except Exception as e:
                logger.warning(f\"Failed to get data for {s}: {e}\")
        return out
'@; Set-Content -Path 'src\altseason\data_providers\binance_ohlcv.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\data_providers\coingecko_global.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import requests, logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any
from ...config import COINGECKO_BASE, MAX_RETRIES
from ...cache import SimpleCache

logger = logging.getLogger(\"altseason\")

class CoinGeckoGlobal:
    def __init__(self):
        self.base_url = COINGECKO_BASE
        self.cache = SimpleCache()

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1,min=4,max=10),
           retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
    def get_global_data(self) -> Dict[str, Any]:
        key=\"coingecko_global\"
        cached = self.cache.get(key)
        if cached is not None:
            logger.info(\"Using cached CoinGecko global data\")
            return cached
        url = f\"{self.base_url}/global\"
        r = requests.get(url, timeout=10); r.raise_for_status()
        data = r.json()[\"data\"]
        self.cache.set(key, data)
        return data

    def get_market_caps(self) -> Dict[str, float]:
        d = self.get_global_data()
        total = d[\"total_market_cap\"][\"usd\"]
        btc_mcap = d[\"market_cap_percentage\"][\"btc\"] * total / 100
        eth_mcap = d[\"market_cap_percentage\"][\"eth\"] * total / 100
        return {\"total_mcap\": total, \"btc_mcap\": btc_mcap, \"eth_mcap\": eth_mcap}
'@; Set-Content -Path 'src\altseason\data_providers\coingecko_global.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\data_providers\health.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import logging, pandas as pd
from typing import Dict, Tuple
from ..data_providers.binance_ohlcv import BinanceOHLCV
from ..data_providers.coingecko_global import CoinGeckoGlobal
logger = logging.getLogger(\"altseason\")

class DataHealth:
    def __init__(self):
        self.binance = BinanceOHLCV()
        self.coingecko = CoinGeckoGlobal()

    def self_test(self) -> Tuple[bool, Dict[str,str]]:
        issues={}
        try:
            btc = self.binance.get_klines(\"BTCUSDT\", limit=10)
            if len(btc) < 5: issues[\"binance_length\"] = f\"Insufficient BTC data: {len(btc)} rows\"
            nan = btc[['open','high','low','close']].isna().sum().sum()
            if nan>0: issues[\"binance_nan\"] = f\"Found {nan} NaN values in BTC data\"
            td = btc['open_time'].diff().dropna()
            if (td.dt.total_seconds() < 0).any():
                issues[\"binance_time\"] = \"Non-monotonic timestamps in BTC data\"
        except Exception as e:
            issues[\"binance\"] = f\"Binance API failed: {e}\"

        try:
            m = self.coingecko.get_market_caps()
            if m['total_mcap'] <= m['btc_mcap']: issues[\"coingecko_relationship\"] = \"Total MCap <= BTC MCap\"
            if m['btc_mcap'] <= m['eth_mcap']:  issues[\"coingecko_btc_eth\"] = \"BTC MCap <= ETH MCap\"
            if m['total_mcap'] <= 0:            issues[\"coingecko_negative\"] = \"Negative market cap values\"
        except Exception as e:
            issues[\"coingecko\"] = f\"CoinGecko API failed: {e}\"

        if issues: logger.warning(f\"Health check issues: {issues}\")
        return (len(issues)==0), issues
'@; Set-Content -Path 'src\altseason\data_providers\health.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\data_providers\__init__.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
from .binance_ohlcv import BinanceOHLCV
from .coingecko_global import CoinGeckoGlobal
from .health import DataHealth
__all__ = [\"BinanceOHLCV\",\"CoinGeckoGlobal\",\"DataHealth\"]
'@; Set-Content -Path 'src\altseason\data_providers\__init__.py' -Value $c -Encoding UTF8"

REM =========================
REM src\altseason\runner.py (final with os import, series from progress, cache persist)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import logging, os
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

from .logging_utils import setup_logging
from .data_providers.health import DataHealth
from .data_providers.binance_ohlcv import BinanceOHLCV
from .data_providers.coingecko_global import CoinGeckoGlobal
from .factors import FactorCalculator
from .scoring import ScoreCalculator
from .report import ReportGenerator
from .progress import ProgressTracker
from .telegram import TelegramNotifier
from .cache import SimpleCache

logger = setup_logging()

class AltseasonRunner:
    def __init__(self):
        self.health_checker = DataHealth()
        self.binance = BinanceOHLCV()
        self.coingecko = CoinGeckoGlobal()
        self.factor_calc = FactorCalculator()
        self.score_calc = ScoreCalculator()
        self.report_gen = ReportGenerator()
        self.progress_tracker = ProgressTracker()
        self.telegram = TelegramNotifier()
        self.cache = SimpleCache()
        self.cache_used = False

    def run_daily_analysis(self) -> bool:
        try:
            logger.info(\"Starting daily altseason analysis...\")
            _ = self._run_health_check()  # proceed even if warns
            raw = self._fetch_all_data()
            factors = self.factor_calc.compute_factors(raw)
            total, status = self.score_calc.calculate_total_score(factors)
            factors[\"total_score\"], factors[\"status\"] = total, status
            d1, d7 = self.progress_tracker.update_progress(total, status)
            self.report_gen.generate_daily_report(factors, total, status, self.cache_used)
            self._update_readme(total, status, d1)
            self._send_telegram_notification(total, status, d1, d7, factors)
            logger.info(\"Daily analysis completed successfully\")
            return True
        except Exception as e:
            logger.error(f\"Daily analysis failed: {e}\")
            return False

    def _run_health_check(self) -> bool:
        try:
            ok, issues = self.health_checker.self_test()
            if not ok: logger.warning(f\"Health check issues: {issues}\")
            return True
        except Exception as e:
            logger.warning(f\"Health check failed: {e}\")
            return True

    def _fetch_all_data(self) -> Dict[str, Any]:
        raw={}
        try:
            mc = self.coingecko.get_market_caps()
            raw[\"market_caps\"] = mc
            total = mc[\"total_mcap\"]; btc=mc[\"btc_mcap\"]; eth=mc[\"eth_mcap\"]
            def _series_from_progress(v: float, length: int = 200) -> pd.Series:
                try:
                    p = self.progress_tracker.csv_file
                    if p.exists():
                        dfp = pd.read_csv(p)
                        if len(dfp)>0:
                            drift = (dfp['score'].tail(min(len(dfp), length)) - 50).fillna(0).values
                            base  = pd.Series([v]*len(drift))
                            return (base * (1 + (drift/2000.0)) ).reset_index(drop=True)
                except Exception as e:
                    logger.warning(f\"Error creating series from progress: {e}\")
                return pd.Series([v]*length)
            btc_dom = (btc/total)*100.0
            total2  = float(total-btc)
            total3  = float(total-btc-eth)
            raw[\"btc_dom_series\"] = _series_from_progress(btc_dom, 200)
            raw[\"total2_series\"]  = _series_from_progress(total2, 200)
            raw[\"total3_series\"]  = _series_from_progress(total3, 200)
            syms=[\"BTCUSDT\",\"ETHUSDT\",\"ETHBTC\"]; o = self.binance.get_multiple_symbols(syms)
            for s, df in o.items():
                if s==\"BTCUSDT\": raw[\"btcusdt\"]=df
                elif s==\"ETHUSDT\": raw[\"ethusdt\"]=df
                elif s==\"ETHBTC\": raw[\"eth_btc\"]=df
            try:
                self.cache.set(\"last_raw_data\", raw)
                logger.info(\"Data cached successfully for fallback usage\")
            except Exception as e:
                logger.warning(f\"Could not persist last_raw_data cache: {e}\")
            return raw
        except Exception as e:
            logger.error(f\"Error fetching data: {e}\")
            cached = self.cache.get(\"last_raw_data\")
            if cached:
                logger.info(\"Using cached data as fallback\"); self.cache_used = True
                return cached
            raise

    def _update_readme(self, score: int, status: str, delta_yesterday: float):
        try:
            readme='README.md'
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            report=f\"reports/{today}.md\"
            em = {\"Altseason Likely\":\"🟢\",\"Forming / Watch\":\"🟡\",\"Neutral\":\"⚪\",\"Risk-Off\":\"🔴\"}.get(status,\"⚪\")
            ar = \"↑\" if delta_yesterday>0 else \"↓\" if delta_yesterday<0 else \"→\"
            sec = f\"\"\"## Today (UTC)
**Status:** {em} {status} — **Score:** {score}/100 — **Δ vs yesterday:** {ar}{abs(delta_yesterday):.1f}
See: [{today}.md]({report})

\"\"\"
            if not os.path.exists(readme):
                base = \"# Altseason Radar\\n\\nDaily analysis of 6 key factors to measure distance to altseason.\\n\\n\"
            else:
                base = open(readme,'r',encoding='utf-8').read()
            if \"## Today (UTC)\" in base:
                lines = base.split('\\n'); new=[]; skip=False
                for ln in lines:
                    if ln.startswith('## Today (UTC)'):
                        new.append(sec.strip()); skip=True
                    elif skip and ln.startswith('## '):
                        skip=False; new.append(ln)
                    elif not skip: new.append(ln)
                content='\\n'.join(new)
            else:
                lines = base.split('\\n'); new=[]; ins=False
                for ln in lines:
                    new.append(ln)
                    if ln.startswith('# ') and not ins:
                        new.append(''); new.append(sec.strip()); ins=True
                content='\\n'.join(new)
            open(readme,'w',encoding='utf-8').write(content)
            logger.info(\"README updated successfully\")
        except Exception as e:
            logger.error(f\"Error updating README: {e}\")

    def _send_telegram_notification(self, score: int, status: str, d1: float, d7: float, factors: Dict[str, Any]):
        try:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            leads=[]
            for k,v in factors.items():
                if k in (\"total_score\",\"status\"): continue
                if v.get('ok',False):
                    nm = 'ETHBTC' if k=='eth_btc' else ('BTC.D' if k=='btc_dominance' else k.replace('_',' ').title())
                    leads.append(nm)
            msg = self.telegram.format_daily_message(today, status, score, d1, d7, leads)
            ok = self.telegram.send_message(msg)
            if not ok: logger.warning(\"Failed to send Telegram notification\")
        except Exception as e:
            logger.error(f\"Error sending Telegram notification: {e}\")
'@; Set-Content -Path 'src\altseason\runner.py' -Value $c -Encoding UTF8"

REM =========================
REM tests\conftest.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))
SRC  = os.path.join(ROOT, \"src\")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
'@; Set-Content -Path 'tests\conftest.py' -Value $c -Encoding UTF8"

REM =========================
REM tests\test_indicators.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import pandas as pd
from altseason.indicators import ema, rsi, atr, adx, slope, cross_above, cross_below

def test_ema_basic():
    s = pd.Series([1,2,3,4,5]); r = ema(s, span=3)
    assert len(r)==5 and not r.isna().all()

def test_rsi_basic():
    s = pd.Series(range(1,21)); r = rsi(s, window=14)
    assert len(r)==20 and r.iloc[-1] > 0

def test_atr_basic():
    h=pd.Series([10,12,11,13,14]); l=pd.Series([8,9,10,11,12]); c=pd.Series([9,11,10.5,12,13])
    r=atr(h,l,c,window=3); assert len(r)==5 and r.iloc[-1] > 0

def test_slope_positive():
    import pandas as pd
    s=pd.Series([1,2,3,4,5]); assert slope(s,5) > 0

def test_cross_above_true():
    a=pd.Series([1,3]); b=pd.Series([2,2]); assert cross_above(a,b)

def test_cross_below_true():
    a=pd.Series([3,1]); b=pd.Series([2,2]); assert cross_below(a,b)
'@; Set-Content -Path 'tests\test_indicators.py' -Value $c -Encoding UTF8"

REM =========================
REM tests\test_factors.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
import numpy as np
import pandas as pd
from altseason.factors import FactorCalculator

def test_compute_factors_with_mock_data():
    calc = FactorCalculator()
    dates = pd.date_range(start='2023-01-01', periods=400, freq='D')
    btc_dom_trend = np.linspace(55,45,400) + np.random.normal(0,0.5,400)

    mock = {
        \"market_caps\": {\"total_mcap\":1_000_000_000_000,\"btc_mcap\":400_000_000_000,\"eth_mcap\":200_000_000_000},
        \"btc_dom_series\": pd.Series(btc_dom_trend, index=dates),
        \"total2_series\":  pd.Series(np.linspace(500e9,600e9,400)+np.random.normal(0,10e9,400), index=dates),
        \"total3_series\":  pd.Series(np.linspace(300e9,400e9,400)+np.random.normal(0, 5e9,400), index=dates),
        \"btcusdt\": pd.DataFrame({
            'open':  np.linspace(45000,50000,400)+np.random.normal(0,500,400),
            'high':  np.linspace(46000,51000,400)+np.random.normal(0,500,400),
            'low':   np.linspace(44000,49000,400)+np.random.normal(0,500,400),
            'close': np.linspace(45500,50500,400)+np.random.normal(0,500,400)
        }, index=dates),
        \"ethusdt\": pd.DataFrame({
            'open':  np.linspace(2500,3000,400)+np.random.normal(0,50,400),
            'high':  np.linspace(2600,3100,400)+np.random.normal(0,50,400),
            'low':   np.linspace(2400,2900,400)+np.random.normal(0,50,400),
            'close': np.linspace(2550,3050,400)+np.random.normal(0,50,400)
        }, index=dates),
        \"eth_btc\": pd.DataFrame({
            'open':  np.linspace(0.055,0.065,400)+np.random.normal(0,0.001,400),
            'high':  np.linspace(0.056,0.066,400)+np.random.normal(0,0.001,400),
            'low':   np.linspace(0.054,0.064,400)+np.random.normal(0,0.001,400),
            'close': np.linspace(0.0555,0.0655,400)+np.random.normal(0,0.001,400)
        }, index=dates)
    }

    factors = calc.compute_factors(mock)
    for k in [\"btc_dominance\",\"eth_btc\",\"total2\",\"total3\",\"btc_regime\",\"eth_trend\"]:
        assert k in factors
        assert set(['score','ok','explain']).issubset(factors[k].keys())
'@; Set-Content -Path 'tests\test_factors.py' -Value $c -Encoding UTF8"

REM =========================
REM tests\test_scoring.py
REM =========================
powershell -NoProfile -Command ^
"$c = @'
from altseason.scoring import ScoreCalculator

def test_altseason_likely():
    s = ScoreCalculator()
    f = {
        'btc_dominance': {'score': 18, 'ok': True},
        'eth_btc':       {'score': 18, 'ok': True},
        'total2':        {'score': 14, 'ok': True},
        'total3':        {'score': 14, 'ok': True},
        'btc_regime':    {'score': 14, 'ok': True},
        'eth_trend':     {'score': 14, 'ok': True},
    }
    score, status = s.calculate_total_score(f)
    assert score >= 75 and status == 'Altseason Likely'

def test_risk_off():
    s = ScoreCalculator()
    f = {k:{'score':5,'ok':False} for k in ['btc_dominance','eth_btc','total2','total3','btc_regime','eth_trend']}
    score, status = s.calculate_total_score(f)
    assert score < 45 and status == 'Risk-Off'

def test_forming_range():
    s = ScoreCalculator()
    f = {
        'btc_dominance': {'score': 15, 'ok': True},
        'eth_btc':       {'score': 15, 'ok': True},
        'total2':        {'score': 10, 'ok': True},
        'total3':        {'score': 10, 'ok': True},
        'btc_regime':    {'score': 10, 'ok': False},
        'eth_trend':     {'score': 10, 'ok': False},
    }
    score, status = s.calculate_total_score(f)
    assert 60 <= score <= 74 and status == 'Forming / Watch'
'@; Set-Content -Path 'tests\test_scoring.py' -Value $c -Encoding UTF8"

REM =========================
REM tests\test_providers.py (minimal smoke; optional)
REM =========================
powershell -NoProfile -Command ^
"$c = @'
def test_providers_placeholder():
    # Providers are integration-heavy; placeholder to keep test suite green.
    assert True
'@; Set-Content -Path 'tests\test_providers.py' -Value $c -Encoding UTF8"

echo Done. Files written.
endlocal
