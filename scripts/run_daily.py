#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altseason Radar — daily runner with Telegram notify and robust logging.

Usage:
  python -m scripts.run_daily
  python -m scripts.run_daily --no-telegram
  python -m scripts.run_daily --state ./reports/state.json --reports ./reports
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from dateutil import tz

# Allow "python -m scripts.run_daily"
CURR = Path(__file__).resolve()
ROOT = CURR.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from altseason.runner import AltseasonRunner  # noqa: E402
from altseason.config import (                 # noqa: E402
    get_telegram_token,
    get_telegram_chat_id,
    TZ_DISPLAY,
)

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    # Minimal fallback if 'rich' is unavailable
    class _DummyConsole:
        def print(self, *a, **k):  # type: ignore
            print(*a)
        def rule(self, *a, **k):   # type: ignore
            print("-" * 60)
    Console = _DummyConsole  # type: ignore
    class _DummyTable:
        def __init__(self, *_, **__): pass
        def add_column(self, *_, **__): pass
        def add_row(self, *a, **k): pass
    Table = _DummyTable  # type: ignore

import requests  # After requirements install

console: Console = Console()  # type: ignore


# ----------------------------- small time helpers -----------------------------

def iso_utc_now() -> str:
    """Return ISO-8601 in UTC with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fmt_local(dt_iso: str, tz_name: str) -> str:
    """Format ISO datetime string to display timezone for human-friendly message."""
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        to_zone = tz.gettz(tz_name or "UTC")
        return dt.astimezone(to_zone).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt_iso


# ----------------------------- files & state ----------------------------------

def ensure_state_file(path: Path) -> None:
    """Create a minimal state.json if it doesn't exist (prevents hard failure)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    minimal = {
        "total_score": None,
        "status": "",
        "forming": False,
        "as_of": iso_utc_now(),
        "factors": {},
    }
    path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")


def read_state(state_path: Path) -> Dict[str, Any]:
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------- fallback: parse latest md ----------------------

MD_SCORE_RE = re.compile(r"(?i)Total\s*Score\s*:\s*(\d+)\s*/\s*100")
MD_STATUS_RE = re.compile(r"(?i)Status\s*:\s*([^\n]+)")

def parse_latest_md(reports_dir: Path) -> Dict[str, Any]:
    """Parse the most recent Markdown report to extract score/status as a fallback."""
    md_files: List[Path] = sorted(
        reports_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not md_files:
        return {}

    text = md_files[0].read_text(encoding="utf-8", errors="ignore")
    score: Optional[int] = None
    status: Optional[str] = None

    m = MD_SCORE_RE.search(text)
    if m:
        try:
            score = int(m.group(1))
        except Exception:
            pass

    m = MD_STATUS_RE.search(text)
    if m:
        status = m.group(1).strip()

    return {
        "total_score": score,
        "status": status,
        "forming": bool(status and ("Form" in status or "Watch" in status)),
        "as_of": iso_utc_now(),
        "factors": {},
    }


def load_summary(state_path: Path, reports_dir: Path) -> Dict[str, Any]:
    """
    Try state.json → fallback to newest Markdown (recursive).
    Return dict with: total_score(int), status(str), forming(bool), as_of(str ISO).
    """
    def _has_values(d: Dict[str, Any]) -> bool:
        try:
            sc = d.get("total_score", None)
            st = str(d.get("status", "")).strip()
            return isinstance(sc, (int, float)) and st != ""
        except Exception:
            return False

    # 1) state.json
    data: Dict[str, Any] = {}
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[yellow]Failed to read state.json: {e}[/yellow]")

    if _has_values(data):
        data.setdefault("as_of", iso_utc_now())
        data.setdefault("forming", False)
        return data

    # 2) newest *.md (recursive)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(reports_dir.rglob("*.md"))
    if not md_files:
        console.print("[yellow]No Markdown reports found under reports/**.md[/yellow]")
    else:
        latest_md = md_files[-1]
        try:
            text = latest_md.read_text(encoding="utf-8")
            score = None
            status = None

            # score patterns
            for pat in [
                r"\bTotal\s*Score\s*:\s*(\d{1,3})\s*/\s*100",
                r"\bScore\s*:\s*(\d{1,3})\s*/\s*100",
                r"\bFinal\s*Verdict.*?(\d{1,3})\s*/\s*100",
            ]:
                m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    score = int(m.group(1))
                    break

            # status patterns (strip emojis/pipes/dashes)
            for pat in [
                r"\bStatus\s*:\s*([^\n\r]+)",
                r"\bFinal\s*Verdict\s*:\s*([^\n\r]+)",
            ]:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    clean = (
                        raw.replace("🟢", "")
                           .replace("🟡", "")
                           .replace("🟠", "")
                           .replace("🔴", "")
                           .replace("⚪️", "")
                           .replace("⚪", "")
                           .strip(" -:|")
                           .strip()
                    )
                    status = clean.split("—")[0].strip()
                    break

            if score is not None:
                data["total_score"] = score
            if status:
                data["status"] = status

            st_low = str(data.get("status", "")).lower()
            if "form" in st_low or "watch" in st_low:
                data["forming"] = True
            elif "neutral" in st_low:
                data["forming"] = False

            if "as_of" not in data or not data["as_of"]:
                data["as_of"] = iso_utc_now()

        except Exception as e:
            console.print(f"[yellow]Failed to parse {latest_md.name}: {e}[/yellow]")

    # 3) final guards
    data.setdefault("total_score", None)
    data.setdefault("status", "")
    data.setdefault("forming", False)
    data.setdefault("as_of", iso_utc_now())

    if not _has_values(data):
        console.print(
            "[yellow]Summary still incomplete (score/status missing). "
            "Ensure state.json is written by the runner OR the Markdown includes 'Score:' and 'Status:' lines.[/yellow]"
        )
    return data


# ----------------------------- message & telegram -----------------------------

FACTOR_TITLES = {
    "btc_dominance": "BTC Dominance",
    "eth_btc": "ETH/BTC",
    "total2": "TOTAL2 (Alts ex-BTC)",
    "total3": "TOTAL3 (Smaller Alts)",
    "btc_regime": "BTC Regime",
    "eth_trend": "ETH Trend",
}

DISPLAY_ORDER = [
    "btc_dominance",
    "eth_btc",
    "total2",
    "total3",
    "btc_regime",
    "eth_trend",
]

def _status_emoji(status: str, forming: bool) -> str:
    s = (status or "").lower()
    if "altseason likely" in s:
        return "🟢"
    if "forming" in s or forming:
        return "🟡"
    if "neutral" in s:
        return "⚪️"
    if "risk-off" in s or "risk off" in s:
        return "🔴"
    return "❔"


def build_message(state: Dict[str, Any]) -> str:
    score = state.get("total_score")
    status = state.get("status") or "Unknown"
    forming = bool(state.get("forming"))
    dt_raw = state.get("as_of") or state.get("date") or iso_utc_now()
    when = fmt_local(dt_raw, TZ_DISPLAY)

    # single, correct emoji
    emoji = _status_emoji(status, forming)

    parts = [
        "📡 *Altseason Radar — Daily*",
        f"📊 *Score:* `{score}/100`" if score is not None else "📊 *Score:* `N/A`",
        f"🎯 *Status:* {status} {emoji}",
        f"🕒 *As of:* {when}",
    ]

    # Factors (nice titles, fixed order, cap to 6)
    facs: Dict[str, Any] = state.get("factors") or {}
    if facs:
        lines = []
        ordered_keys = [k for k in DISPLAY_ORDER if k in facs] + [k for k in facs.keys() if k not in DISPLAY_ORDER]
        for k in ordered_keys[:6]:
            v = facs.get(k, {})
            sc = v.get("score", "—")
            ok = "✅" if v.get("ok") else "❌"
            title = FACTOR_TITLES.get(k, k.replace("_", " ").title())
            lines.append(f"• {title}: {sc} {ok}")
        if lines:
            parts.append("—\n*Factors*\n" + "\n".join(lines))

    return "\n".join(parts)


def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    """Send a Markdown message to Telegram using env or config getters."""
    token = os.getenv("TELEGRAM_BOT_TOKEN") or get_telegram_token()
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or get_telegram_chat_id()

    if not token or not chat_id:
        console.print("[yellow]Telegram not configured (missing token/chat_id). Skipping.[/yellow]")
        return None

    api = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(api, json=payload, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram API error: {r.status_code} {r.text}")
    return r.json()


# ----------------------------- runner ----------------------------------------

def run_analysis() -> bool:
    console.rule("[bold cyan]Altseason Radar — Daily Run")
    console.print("🔄 Starting altseason analysis...")
    runner = AltseasonRunner()
    ok = runner.run_daily_analysis()
    if ok:
        console.print("✅ Analysis completed successfully!")
    else:
        console.print("[red]❌ Analysis failed[/red]")
    return ok


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run daily analysis and optionally notify Telegram.")
    p.add_argument(
        "--no-telegram",
        action="store_true",
        help="Do not send Telegram notification even if analysis succeeds.",
    )
    p.add_argument(
        "--state",
        type=str,
        default=str(ROOT / "reports" / "state.json"),
        help="Path to state.json generated by analysis.",
    )
    p.add_argument(
        "--reports",
        type=str,
        default=str(ROOT / "reports"),
        help="Reports directory (for Markdown fallback parsing).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = make_parser().parse_args(argv)

    state_path = Path(args.state)
    reports_dir = Path(args.reports)

    try:
        # Make sure state file exists so the next steps never hard-fail
        ensure_state_file(state_path)

        ok = run_analysis()
        if not ok:
            return 1

        # Load summary with Markdown fallback when state.json is empty/partial
        state = load_summary(state_path, reports_dir)
        msg = build_message(state)

        # Console table
        try:
            table = Table(title="Daily Summary")
            table.add_column("Field", style="bold")
            table.add_column("Value")
            table.add_row("Score", f"{state.get('total_score', 'N/A')}")
            table.add_row("Status", state.get("status", "Unknown") or "Unknown")
            table.add_row("Forming", "Yes" if state.get("forming") else "No")
            when = fmt_local(state.get("as_of") or "", TZ_DISPLAY)
            table.add_row("As of", when or "-")
            console.print(table)
        except Exception:
            console.print(msg)

        if args.no_telegram:
            console.print("[yellow]Skipping Telegram by flag.[/yellow]")
            return 0

        console.print("📨 Sending Telegram notification…")
        _res = send_telegram(msg)
        console.print("✅ Telegram notification sent.")

        console.print("[green]Daily analysis completed successfully![/green]")
        return 0

    except requests.RequestException as e:
        console.print(f"[red]Network/Telegram error:[/red] {e}")
        return 3
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        return 9


if __name__ == "__main__":
    sys.exit(main())
