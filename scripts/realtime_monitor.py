#!/usr/bin/env python3
"""
盘中实时监控 — 每10分钟检查信号，飞书推送
支持：经典版+增强版双缓存，优先检查自选股
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, time as dt_time
import pandas as pd

from scripts.strategy_engine import load_stock_data, ScoringEngine, get_available_symbols, load_all_stocks_info

DATA_DIR = os.path.expanduser("~/stock-system/data")
STATE_FILE = os.path.join(DATA_DIR, "monitor_state.json")
CACHE_FILE = os.path.join(DATA_DIR, "scan_cache.parquet")
CACHE_ENHANCED = os.path.join(DATA_DIR, "scan_enhanced_cache.parquet")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

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

def load_watchlist_symbols():
    """加载自选股列表"""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item["symbol"] for item in data]
            return [item["symbol"] for item in data.get("items", [])]
    return []

def scan_stock(symbol, mode="enhanced"):
    """单独扫描一只股票，返回评分结果"""
    try:
        df = load_stock_data(symbol)
        if df is None or len(df) < 10:
            return None
        result = ScoringEngine.score_stock(df, mode=mode, symbol=symbol)
        info = load_all_stocks_info()
        name = symbol
        if info is not None:
            match = info[info["symbol"] == symbol]
            if len(match) > 0:
                name = match.iloc[0].get("code_name", symbol)
        return {
            "symbol": symbol,
            "name": name,
            "score": result["score"],
            "price": result["latest"]["close"],
            "change_pct": result["latest"].get("change_pct", 0),
            "rsi": result["latest"].get("rsi", 0),
            "signals": ",".join(result.get("signals", [])),
            "mode": mode,
        }
    except Exception:
        return None

def check_signals():
    """检查当前信号，返回新发现的信号列表"""
    if not is_trading_time():
        return []
    
    state = load_state()
    already_notified = set(state.get("notified_signals", []))
    new_alerts = []
    
    # 1️⃣ 优先检查自选股（单独扫描增强模式）
    watchlist = load_watchlist_symbols()
    if watchlist:
        for symbol in watchlist:
            result = scan_stock(symbol, mode="enhanced")
            if result is None:
                continue
            signals = result["signals"]
            score = result["score"]
            
            # 评分≥80且strong_buy → 买入信号；评分<60或trend_broken → 卖出预警
            has_buy = "strong_buy" in signals
            has_sell = "trend_broken" in signals
            has_overbought = "overbought" in signals
            
            # 买入信号
            if has_buy and score >= 70:
                key = f"{symbol}_buy"
                if key not in already_notified:
                    new_alerts.append({
                        "symbol": symbol,
                        "name": result["name"],
                        "score": score,
                        "price": result["price"],
                        "signals": "strong_buy",
                        "type": "buy",
                        "mode": "enhanced",
                    })
                    already_notified.add(key)
            
            # 卖出/风险信号
            if has_sell or (has_overbought and score < 60):
                key = f"{symbol}_sell"
                if key not in already_notified:
                    new_alerts.append({
                        "symbol": symbol,
                        "name": result["name"],
                        "score": score,
                        "price": result["price"],
                        "signals": signals,
                        "type": "sell" if has_sell else "warning",
                        "mode": "enhanced",
                    })
                    already_notified.add(key)
    
    # 2️⃣ 检查缓存中的高分股票（经典版 + 增强版）
    for cache_file in [CACHE_FILE, CACHE_ENHANCED]:
        if not os.path.exists(cache_file):
            continue
        cached = pd.read_parquet(cache_file)
        mode_name = "enhanced" if "enhanced" in cache_file else "classic"
        
        # 检查高分股票是否有新信号
        top_stocks = cached[cached["score"] >= 60].head(30)
        for _, row in top_stocks.iterrows():
            symbol = row["symbol"]
            signals = str(row.get("signals", ""))
            
            if "strong_buy" not in signals and "watch_buy" not in signals:
                continue
            
            alert_key = f"{symbol}_{signals}_{mode_name}"
            if alert_key in already_notified:
                continue
            
            # 跳过已在自选股中检查过的
            if symbol in watchlist:
                continue
            
            new_alerts.append({
                "symbol": symbol,
                "name": row.get("name", symbol),
                "score": row["score"],
                "price": row.get("price", 0),
                "signals": signals,
                "type": "buy",
                "mode": mode_name,
            })
            already_notified.add(alert_key)
    
    # 清理旧记录（保留最近100条）
    if len(already_notified) > 100:
        already_notified = set(list(already_notified)[-100:])
    
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
    
    # 按类型分组
    buy_alerts = [a for a in alerts if a["type"] == "buy"]
    sell_alerts = [a for a in alerts if a["type"] == "sell"]
    warn_alerts = [a for a in alerts if a["type"] == "warning"]
    
    lines = [f"🔔 **盘中信号提醒 — {ts}**", ""]
    
    if buy_alerts:
        lines.append("**🟢 买入信号**")
        for a in buy_alerts:
            label = "⭐" if a.get("mode") == "enhanced" else ""
            lines.append(f"  {label} {a['name']} ({a['symbol']})")
            lines.append(f"    评分: {a['score']} | 现价: {a['price']:.2f}")
        lines.append("")
    
    if sell_alerts:
        lines.append("**🔴 卖出信号**")
        for a in sell_alerts:
            lines.append(f"  {a['name']} ({a['symbol']})")
            lines.append(f"    评分: {a['score']} | 现价: {a['price']:.2f} | 破位")
        lines.append("")
    
    if warn_alerts:
        lines.append("**⚠️ 风险提醒**")
        for a in warn_alerts:
            lines.append(f"  {a['name']} ({a['symbol']}) — {a['signals']}")
        lines.append("")
    
    lines.append("`#盘中监控 #增强评分`")
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
