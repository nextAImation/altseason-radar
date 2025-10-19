#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altseason Radar — daily runner with Telegram notify, delta-aware messaging, and robust logging.

Usage:
  python -m scripts.run_daily
  python -m scripts.run_daily --no-telegram
  python -m scripts.run_daily --state ./reports/state.json --reports ./reports
  python -m scripts.run_daily --only-on-change --min-delta 2
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
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
    class _DummyConsole:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("-" * 60)
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def fmt_local(dt_iso: str, tz_name: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        to_zone = tz.gettz(tz_name or "UTC")
        return dt.astimezone(to_zone).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt_iso

# ----------------------------- files & state ----------------------------------

def ensure_state_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    minimal = {"total_score": None, "status": "", "forming": False, "as_of": iso_utc_now(), "factors": {}}
    path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")

def read_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ----------------------------- markdown parsing -------------------------------

MD_SCORE_RE = re.compile(r"(?i)\*\*Score:\*\*\s*(\d{1,3})\s*/\s*100")
MD_STATUS_RE = re.compile(r"(?i)\*\*Status:\*\*\s*([^\n\r]+)")
MD_OKCNT_RE  = re.compile(r"(?i)\*\*OK\s*Factors:\*\*\s*(\d+)")
MD_CONF_RE   = re.compile(r"(?i)\*\*Confidence:\*\*\s*([0-9]*\.?[0-9]+)")

def _read_prev_md_summaries(reports_dir: Path, n: int = 5) -> List[Dict[str, Any]]:
    mds = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for p in mds[:n]:
        text = p.read_text(encoding="utf-8", errors="ignore")
        score = None
        status = None
        okc = None
        conf = None
        m = MD_SCORE_RE.search(text);  score  = int(m.group(1)) if m else None
        m = MD_STATUS_RE.search(text); status = m.group(1).strip() if m else None
        m = MD_OKCNT_RE.search(text);  okc    = int(m.group(1)) if m else None
        m = MD_CONF_RE.search(text);   conf   = float(m.group(1)) if m else None
        out.append({"path": p, "score": score, "status": status, "ok_count": okc, "confidence": conf})
    return out

# ----------------------------- message & telegram -----------------------------

FACTOR_TITLES = {
    "btc_dominance": "BTC Dominance",
    "eth_btc": "ETH/BTC",
    "total2": "TOTAL2 (Alts ex-BTC)",
    "total3": "TOTAL3 (Smaller Alts)",
    "btc_regime": "BTC Regime",
    "eth_trend": "ETH Trend",
}

DISPLAY_ORDER = ["btc_dominance","eth_btc","total2","total3","btc_regime","eth_trend"]

def _status_emoji(status: str, forming: bool) -> str:
    s = (status or "").lower()
    if "altseason likely" in s: return "🟢"
    if "forming" in s or forming: return "🟡"
    if "neutral" in s: return "⚪️"
    if "risk-off" in s or "risk off" in s: return "🔴"
    return "❔"

def _factor_flip_lines(curr: Dict[str, Any], prev: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    cfac = curr.get("factors") or {}
    pfac = prev.get("factors") or {}
    for k in DISPLAY_ORDER:
        if k in cfac and k in pfac:
            c_ok = bool(cfac[k].get("ok"))
            p_ok = bool(pfac[k].get("ok"))
            if c_ok != p_ok:
                title = FACTOR_TITLES.get(k, k)
                arrow = "✅" if c_ok else "❌"
                lines.append(f"• {title}: flip → {arrow}")
    return lines

def build_message(state: Dict[str, Any]) -> str:
    score   = state.get("total_score")
    status  = state.get("status") or "Unknown"
    forming = bool(state.get("forming"))
    when    = fmt_local(state.get("as_of") or iso_utc_now(), TZ_DISPLAY)
    emoji   = _status_emoji(status, forming)

    parts = [
        "📡 *Altseason Radar — Daily*",
        f"📊 *Score:* `{score}/100`" if score is not None else "📊 *Score:* `N/A`",
        f"🎯 *Status:* {status} {emoji}",
        f"🕒 *As of:* {when}",
    ]

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

    # footer diagnostics if available
    okc = state.get("ok_count")
    conf = state.get("confidence")
    diag = []
    if okc is not None:  diag.append(f"**OK Factors:** {okc}")
    if conf is not None: diag.append(f"**Confidence:** {conf:.3f}")
    if diag:
        parts.append("—\n" + " | ".join(diag))

    return "\n".join(parts)

def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or get_telegram_token()
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or get_telegram_chat_id()
    if not token or not chat_id:
        console.print("[yellow]Telegram not configured (missing token/chat_id). Skipping.[/yellow]")
        return None
    api = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    r = requests.post(api, json=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram API error: {r.status_code} {r.text}")
    return r.json()

# ----------------------------- delta / gating --------------------------------

def _delta_block(curr: Dict[str, Any], prev_md_list: List[Dict[str, Any]], prev_state: Optional[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Returns (delta_text, score_diff). Uses last .md as 'yesterday'; falls back to prev_state if needed.
    """
    if prev_md_list:
        prev = prev_md_list[0]
        p_score  = prev.get("score")
        p_status = prev.get("status")
        p_okc    = prev.get("ok_count")
        p_conf   = prev.get("confidence")
    elif prev_state:
        p_score  = prev_state.get("total_score")
        p_status = prev_state.get("status")
        p_okc    = prev_state.get("ok_count")
        p_conf   = prev_state.get("confidence")
    else:
        return ("", 0)

    c_score = curr.get("total_score")
    c_status= curr.get("status")
    c_okc   = curr.get("ok_count")
    c_conf  = curr.get("confidence")

    lines: List[str] = []
    score_diff = 0
    if isinstance(c_score, int) and isinstance(p_score, int):
        score_diff = c_score - p_score
        arrow = "▲" if score_diff > 0 else "▼" if score_diff < 0 else "➖"
        lines.append(f"📈 *Change vs yesterday:* {arrow} {score_diff:+d} points")

    if p_status and c_status and p_status.strip() != c_status.strip():
        lines.append(f"🔄 *Status changed:* `{p_status}` → `{c_status}`")

    if isinstance(c_okc, int) and isinstance(p_okc, int) and c_okc != p_okc:
        arrow = "▲" if (c_okc - p_okc) > 0 else "▼"
        lines.append(f"🧩 OK factors: {arrow} {c_okc - p_okc:+d} → `{c_okc}`")

    if isinstance(c_conf, float) and isinstance(p_conf, float) and abs(c_conf - p_conf) >= 0.01:
        arrow = "▲" if (c_conf - p_conf) > 0 else "▼"
        lines.append(f"🔐 Confidence: {arrow} {c_conf - p_conf:+.3f} → `{c_conf:.3f}`")

    # factor flips
    if prev_state:
        flips = _factor_flip_lines(curr, prev_state)
        if flips:
            lines.append("*Factor flips:*")
            lines.extend(flips)

    return ("\n".join(lines), score_diff)

def _status_streak(reports_dir: Path, current_status: str, max_days: int = 30) -> int:
    """Count consecutive days with the same status in reverse-chronological .md files."""
    mds = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    streak = 0
    for p in mds[:max_days]:
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = MD_STATUS_RE.search(text)
        st = m.group(1).strip() if m else None
        if st == current_status:
            streak += 1
        else:
            break
    return streak

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
    p.add_argument("--no-telegram", action="store_true", help="Do not send Telegram notification.")
    p.add_argument("--state", type=str, default=str(ROOT / "reports" / "state.json"),
                   help="Path to state.json generated by analysis.")
    p.add_argument("--reports", type=str, default=str(ROOT / "reports"),
                   help="Reports directory (for Markdown fallback parsing).")
    p.add_argument("--only-on-change", action="store_true",
                   help="Send Telegram only if change is material (score delta, status flip, or factor flips).")
    p.add_argument("--min-delta", type=int, default=int(os.getenv("ALT_MIN_DELTA", "2")),
                   help="Minimum absolute score delta to qualify as material change.")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    state_path  = Path(args.state)
    reports_dir = Path(args.reports)

    try:
        ensure_state_file(state_path)

        ok = run_analysis()
        if not ok:
            return 1

        # Load fresh state (produced by runner)
        current = read_state(state_path)

        # Previous references for delta
        prev_mds   = _read_prev_md_summaries(reports_dir, n=2)  # [today, yesterday] after run
        prev_state = {}
        if len(prev_mds) >= 2:
            # try to read yesterday's state.json if exists next to md (optional)
            prev_state_path = state_path  # simplest: use same path from last run if workflow preserves it
            prev_state = read_state(prev_state_path)

        # Build core message
        msg_core = build_message(current)

        # Delta info & gating
        delta_text, score_diff = _delta_block(current, prev_mds[1:], prev_state or current)  # compare with yesterday
        streak = _status_streak(reports_dir, current.get("status", ""))
        if streak > 1:
            delta_text += ("\n" if delta_text else "") + f"📆 *{current.get('status','')}* streak: `{streak} days`"

        full_msg = msg_core + (("\n\n" + delta_text) if delta_text else "")

        # Console table
        try:
            table = Table(title="Daily Summary")
            table.add_column("Field", style="bold"); table.add_column("Value")
            table.add_row("Score", f"{current.get('total_score', 'N/A')}")
            table.add_row("Status", current.get("status", "Unknown") or "Unknown")
            table.add_row("Forming", "Yes" if current.get("forming") else "No")
            when = fmt_local(current.get("as_of") or "", TZ_DISPLAY); table.add_row("As of", when or "-")
            console.print(table)
        except Exception:
            console.print(full_msg)

        # Gate sending (only-on-change)
        only_on_change = args.only_on_change or os.getenv("SEND_ONLY_ON_CHANGE", "1") not in ("0","false","False")
        material = True
        if only_on_change:
            flip_exists = "flip →" in delta_text
            status_changed = "Status changed:" in delta_text
            score_move = abs(score_diff) >= int(args.min_delta)
            material = bool(flip_exists or status_changed or score_move)
            if not material:
                console.print("[yellow]No material change detected → skipping Telegram.[/yellow]")

        if args.no_telegram or not material:
            return 0

        console.print("📨 Sending Telegram notification…")
        _res = send_telegram(full_msg)
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
