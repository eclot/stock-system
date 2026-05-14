#!/usr/bin/env python3
"""
交易模块 — 自选股管理、模拟交易（实盘模拟）、策略回测
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.expanduser("~/stock-system/data")

# ═══════════════════════════════════════════════
#  自选股管理
# ═══════════════════════════════════════════════

WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

def _load_watchlist() -> list:
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def _save_watchlist(wl: list):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def watchlist_list() -> list:
    """获取自选股列表"""
    return _load_watchlist()

def watchlist_add(symbol: str, name: str = "") -> dict:
    """添加自选股"""
    wl = _load_watchlist()
    # 去重
    for item in wl:
        if item["symbol"] == symbol:
            return {"status": "ok", "message": "已在自选股中"}
    wl.append({"symbol": symbol, "name": name, "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    _save_watchlist(wl)
    return {"status": "ok", "message": f"已添加 {name or symbol}"}

def watchlist_remove(symbol: str) -> dict:
    """删除自选股"""
    wl = _load_watchlist()
    new_wl = [item for item in wl if item["symbol"] != symbol]
    if len(new_wl) == len(wl):
        return {"status": "ok", "message": "未在自选股中"}
    _save_watchlist(new_wl)
    return {"status": "ok", "message": f"已移除 {symbol}"}


# ═══════════════════════════════════════════════
#  模拟交易（实盘模拟）
# ═══════════════════════════════════════════════

PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

INITIAL_CASH = 1_000_000  # 初始资金100万

def _default_portfolio():
    return {
        "cash": INITIAL_CASH,
        "holdings": {},  # symbol -> {"shares": int, "avg_price": float}
        "transactions": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

def _load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        pf = _default_portfolio()
        _save_portfolio(pf)
        return pf
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def _save_portfolio(pf: dict):
    pf["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

def portfolio_status(current_prices: dict = None) -> dict:
    """获取投资组合状态"""
    pf = _load_portfolio()
    holdings = pf.get("holdings", {})
    total_market_value = 0
    holdings_detail = []
    
    for symbol, info in holdings.items():
        price = current_prices.get(symbol, info.get("avg_price", 0)) if current_prices else info.get("avg_price", 0)
        market_value = info["shares"] * price
        cost = info["shares"] * info["avg_price"]
        profit = market_value - cost
        profit_pct = (profit / cost * 100) if cost > 0 else 0
        total_market_value += market_value
        
        # 行业归属
        try:
            from scripts.data_loader import get_industry, load_industry_mapping
            _m = load_industry_mapping()
            industry = get_industry(symbol, _m) if _m is not None else None
        except:
            industry = None
        holdings_detail.append({
            "symbol": symbol,
            "name": info.get("name", symbol),
            "industry": industry,
            "shares": info["shares"],
            "avg_price": round(info["avg_price"], 2),
            "current_price": round(price, 2),
            "market_value": round(market_value, 2),
            "cost": round(cost, 2),
            "profit": round(profit, 2),
            "profit_pct": round(profit_pct, 2),
        })
    
    total_assets = pf["cash"] + total_market_value
    total_profit = total_assets - INITIAL_CASH
    total_profit_pct = (total_profit / INITIAL_CASH * 100)
    
    return {
        "cash": round(pf["cash"], 2),
        "holdings": holdings_detail,
        "total_market_value": round(total_market_value, 2),
        "total_assets": round(total_assets, 2),
        "total_profit": round(total_profit, 2),
        "total_profit_pct": round(total_profit_pct, 2),
        "transaction_count": len(pf.get("transactions", [])),
        "created_at": pf.get("created_at", ""),
    }

def portfolio_buy(symbol: str, name: str, shares: int, price: float) -> dict:
    """买入"""
    if shares <= 0:
        return {"status": "error", "message": "数量必须大于0"}
    pf = _load_portfolio()
    cost = shares * price
    if cost > pf["cash"]:
        return {"status": "error", "message": f"资金不足，需要{cost:.2f}，可用{pf['cash']:.2f}"}
    
    pf["cash"] -= cost
    if symbol in pf["holdings"]:
        old = pf["holdings"][symbol]
        total_shares = old["shares"] + shares
        total_cost = old["shares"] * old["avg_price"] + cost
        pf["holdings"][symbol] = {
            "shares": total_shares,
            "avg_price": round(total_cost / total_shares, 2),
            "name": name,
        }
    else:
        pf["holdings"][symbol] = {"shares": shares, "avg_price": round(price, 2), "name": name}
    
    pf["transactions"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "name": name,
        "action": "buy",
        "shares": shares,
        "price": round(price, 2),
        "total": round(cost, 2),
    })
    _save_portfolio(pf)
    return {"status": "ok", "message": f"买入 {name} {shares}股 @ {price:.2f}"}

def portfolio_sell(symbol: str, name: str, shares: int, price: float) -> dict:
    """卖出"""
    if shares <= 0:
        return {"status": "error", "message": "数量必须大于0"}
    pf = _load_portfolio()
    if symbol not in pf["holdings"]:
        return {"status": "error", "message": f"未持有 {symbol}"}
    
    holding = pf["holdings"][symbol]
    if shares > holding["shares"]:
        return {"status": "error", "message": f"持有不足，可卖{holding['shares']}股"}
    
    revenue = shares * price
    pf["cash"] += revenue
    
    if shares == holding["shares"]:
        del pf["holdings"][symbol]
    else:
        pf["holdings"][symbol]["shares"] -= shares
    
    pf["transactions"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "name": name,
        "action": "sell",
        "shares": shares,
        "price": round(price, 2),
        "total": round(revenue, 2),
    })
    _save_portfolio(pf)
    return {"status": "ok", "message": f"卖出 {name} {shares}股 @ {price:.2f}"}

def portfolio_history() -> list:
    """交易历史"""
    pf = _load_portfolio()
    return list(reversed(pf.get("transactions", [])))

def portfolio_reset() -> dict:
    """重置投资组合"""
    _save_portfolio(_default_portfolio())
    return {"status": "ok", "message": "已重置投资组合"}


# ═══════════════════════════════════════════════
#  策略回测
# ═══════════════════════════════════════════════

BACKTEST_DIR = os.path.join(DATA_DIR, "backtests")
os.makedirs(BACKTEST_DIR, exist_ok=True)

def run_backtest(symbol: str, name: str,
                 initial_cash: float = 100000,
                 buy_threshold: float = 70,
                 sell_threshold: float = 55) -> dict:
    """
    简单回测：评分策略
    买入条件：评分 >= buy_threshold
    卖出条件：评分 < sell_threshold
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.strategy_engine import load_stock_data, ScoringEngine
    
    df = load_stock_data(symbol, years=0)
    if df is None or len(df) < 120:
        return {"status": "error", "message": f"数据不足: {symbol}"}
    
    df = df.sort_values("date").reset_index(drop=True)
    
    # 模拟按天回测
    cash = initial_cash
    shares = 0
    transactions = []
    equity_curve = []
    in_position = False
    
    financial = {}
    
    for i in range(120, len(df)):
        chunk = df.iloc[:i+1]
        current = df.iloc[i]
        date = current["date"]
        close = float(current["close"])
        
        result = ScoringEngine.score_stock(chunk, financial)
        score = result["score"]
        
        # 记录权益曲线（每周）
        if i % 5 == 0:
            equity = cash + shares * close
            equity_curve.append({"date": str(date.date()), "equity": round(equity, 2)})
        
        # 买入信号
        if score >= buy_threshold and not in_position and cash > close * 100:
            buy_shares = int(cash * 0.95 / close / 100) * 100  # 95%资金，整手
            if buy_shares >= 100:
                cost = buy_shares * close
                cash -= cost
                shares += buy_shares
                in_position = True
                transactions.append({
                    "date": str(date.date()), "action": "buy",
                    "price": round(close, 2), "shares": buy_shares,
                    "total": round(cost, 2), "score": score,
                })
        
        # 卖出信号
        elif score < sell_threshold and in_position and shares > 0:
            revenue = shares * close
            cash += revenue
            transactions.append({
                "date": str(date.date()), "action": "sell",
                "price": round(close, 2), "shares": shares,
                "total": round(revenue, 2), "score": score,
            })
            shares = 0
            in_position = False
    
    # 最终平仓
    if shares > 0:
        final_close = float(df.iloc[-1]["close"])
        revenue = shares * final_close
        cash += revenue
        transactions.append({
            "date": str(df.iloc[-1]["date"].date()), "action": "sell(平仓)",
            "price": round(final_close, 2), "shares": shares,
            "total": round(revenue, 2), "score": 0,
        })
        shares = 0
    
    # 最终权益曲线
    final_equity = cash
    total_return = (final_equity - initial_cash) / initial_cash * 100
    
    # 买入持有收益
    buy_hold_return = (float(df.iloc[-1]["close"]) / float(df.iloc[120]["close"]) - 1) * 100
    
    result = {
        "symbol": symbol,
        "name": name,
        "initial_cash": initial_cash,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "transaction_count": len([t for t in transactions if "平仓" not in t.get("action","")]),
        "transactions": transactions,
        "equity_curve": equity_curve,
        "strategy": f"评分策略(买入≥{buy_threshold}, 卖出<{sell_threshold})",
        "date_range": f"{df.iloc[120]['date'].date()} ~ {df.iloc[-1]['date'].date()}",
    }
    
    return result


def save_backtest_result(result: dict) -> str:
    """保存回测结果到文件"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result['symbol']}_{ts}.json"
    filepath = os.path.join(BACKTEST_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return filepath


def list_backtest_results() -> list:
    """列出回测结果"""
    if not os.path.exists(BACKTEST_DIR):
        return []
    files = sorted(os.listdir(BACKTEST_DIR), reverse=True)[:20]
    results = []
    for f in files:
        if f.endswith(".json"):
            filepath = os.path.join(BACKTEST_DIR, f)
            with open(filepath, "r") as fh:
                data = json.load(fh)
            results.append({
                "file": f,
                "symbol": data.get("symbol", ""),
                "name": data.get("name", ""),
                "total_return": data.get("total_return", 0),
                "buy_hold_return": data.get("buy_hold_return", 0),
                "date_range": data.get("date_range", ""),
                "strategy": data.get("strategy", ""),
            })
    return results


if __name__ == "__main__":
    # 简单测试
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if cmd == "watchlist":
        print(json.dumps(watchlist_list(), ensure_ascii=False, indent=2))
    elif cmd == "add":
        if len(sys.argv) > 2:
            print(json.dumps(watchlist_add(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""), ensure_ascii=False))
    elif cmd == "remove":
        if len(sys.argv) > 2:
            print(json.dumps(watchlist_remove(sys.argv[2]), ensure_ascii=False))
    elif cmd == "portfolio":
        print(json.dumps(portfolio_status(), ensure_ascii=False, indent=2))
    elif cmd == "reset":
        print(json.dumps(portfolio_reset(), ensure_ascii=False, indent=2))
    elif cmd == "backtest":
        sym = sys.argv[2] if len(sys.argv) > 2 else "sh600519"
        name = sys.argv[3] if len(sys.argv) > 3 else ""
        result = run_backtest(sym, name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("用法: python trading.py [watchlist|add|remove|portfolio|reset|backtest]")
