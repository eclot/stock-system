#!/usr/bin/env python3
"""
盘中实时监控 — 每10分钟检查信号，飞书推送
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, time as dt_time
import pandas as pd

from scripts.strategy_engine import load_stock_data, ScoringEngine, get_available_symbols, load_all_stocks_info

DATA_DIR = os.path.expanduser("~/stock-system/data")
STATE_FILE = os.path.join(DATA_DIR, "monitor_state.json")
CACHE_FILE = os.path.join(DATA_DIR, "scan_cache.parquet")

MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(15, 0)
MORNING_END = dt_time(11, 30)
AFTERNOON_START = dt_time(13, 0)

def is_trading_time():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    if t < MARKET_OPEN or t > MARKET_CLOSE:
        return False
    if MORNING_END < t < AFTERNOON_START:
        return False
    return True

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_check": None, "notified_signals": [], "last_push": None}

def save_state(state):
    state["last_check"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def check_signals():
    """检查当前信号，返回新发现的信号列表"""
    if not is_trading_time():
        return []
    
    # 加载缓存数据（从收盘扫描结果中取 top 检查）
    if not os.path.exists(CACHE_FILE):
        return []
    
    cached = pd.read_parquet(CACHE_FILE)
    
    # 检查高分股票是否有新信号
    top_stocks = cached[cached["score"] >= 60].head(20)
    state = load_state()
    already_notified = set(state.get("notified_signals", []))
    
    new_alerts = []
    for _, row in top_stocks.iterrows():
        symbol = row["symbol"]
        signals = str(row.get("signals", ""))
        
        # 只关注强买信号
        if "strong_buy" not in signals and "watch_buy" not in signals:
            continue
        
        alert_key = f"{symbol}_{signals}"
        if alert_key in already_notified:
            continue
        
        new_alerts.append({
            "symbol": symbol,
            "name": row.get("name", symbol),
            "score": row["score"],
            "price": row.get("price", 0),
            "signals": signals,
        })
        
        # 标记为已通知
        already_notified.add(alert_key)
    
    # 清理旧记录（保留最近50条）
    if len(already_notified) > 50:
        already_notified = set(list(already_notified)[-50:])
    
    state["notified_signals"] = list(already_notified)
    if new_alerts:
        state["last_push"] = datetime.now().isoformat()
    save_state(state)
    
    return new_alerts


def format_feishu_alert(alerts):
    """格式化飞书推送消息"""
    if not alerts:
        return None
    
    ts = datetime.now().strftime("%H:%M")
    lines = [f"🔔 **盘中信号提醒 — {ts}**", ""]
    
    for a in alerts:
        emoji = "🟢" if "strong_buy" in a["signals"] else "🟡"
        lines.append(f"{emoji} **{a['name']}** ({a['symbol']})")
        lines.append(f"   评分: {a['score']} | 现价: {a['price']:.2f}")
        lines.append("")
    
    lines.append("`#盘中监控`")
    return "\n".join(lines)


if __name__ == "__main__":
    alerts = check_signals()
    if alerts:
        msg = format_feishu_alert(alerts)
        print(msg)
        print(f"\n发现 {len(alerts)} 个新信号，已推送飞书")
    else:
        if is_trading_time():
            print("无新信号")
        else:
            print("非交易时间，跳过")
