#!/usr/bin/env python3
"""
自主交易模块 — 盘中信号触发自动买卖

职责:
1. 监听实时信号 (由 cron 调用)
2. 将信号与当前持仓对比，决定买卖
3. 执行模拟交易（通过 trading.py）
4. 输出交易日志，供 cron 推送飞书

用法:
  python auto_trader.py                    # 常规盘中执行：检查信号+自动交易
  python auto_trader.py --morning-scan     # 早盘检查持仓风险
  python auto_trader.py --force-scan       # 强制扫描所有持仓的最新评分
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, time as dt_time
from scripts.trading import (
    portfolio_status, portfolio_buy, portfolio_sell,
    _load_portfolio, _save_portfolio
)
from scripts.realtime_monitor import (
    check_signals, scan_stock, is_trading_time, load_watchlist_symbols
)

MARKET_OPEN = dt_time(9, 30)
MORNING_END = dt_time(11, 30)
AFTERNOON_START = dt_time(13, 0)
MARKET_CLOSE = dt_time(15, 0)

# ── 交易参数 ──────────────────────────────────
MAX_CASH_PER_TRADE = 0.05       # 单次买入不超过可用现金的 5%
MAX_PORTFOLIO_SINGLE = 0.15     # 单只持仓不超过总资产的 15%
SELL_SCORE_THRESHOLD = 60       # 评分低于此值触发卖出（原55→收紧到60）
BUY_SCORE_THRESHOLD = 70        # 评分高于此值触发买入
STOP_LOSS_PCT = -5.0            # 波段仓位 -5% 止损（原-10%→收紧到-5%）
MAX_NEW_POSITIONS = 3           # 每 session 最多开新仓3只
MIN_POSITION_VALUE = 20000      # 新开仓位最低市值 2万元
INDUSTRY_CAPS = {               # 行业持仓上限
    "电子": 3, "机械设备": 2, "国防军工": 2,
    "房地产": 2, "公用事业": 2, "商贸零售": 2,
}


def get_current_prices(holdings: list) -> dict:
    """获取持仓股票的最新价格（从已缓存数据）"""
    prices = {}
    for h in holdings:
        try:
            result = scan_stock(h["symbol"], mode="enhanced")
            if result and result.get("price", 0) > 0:
                prices[h["symbol"]] = {
                    "price": result["price"],
                    "score": result["score"],
                    "signals": result.get("signals", ""),
                }
        except Exception:
            pass
    return prices


def morning_risk_scan() -> dict:
    """早盘检查：扫描所有持仓的最新评分，发现风险信号"""
    status = portfolio_status()
    holdings = status["holdings"]
    if not holdings:
        return {"action": "skip", "message": "空仓，无需检查"}

    alerts = []
    symbols_to_check = [h["symbol"] for h in holdings]

    for symbol in symbols_to_check:
        result = scan_stock(symbol, mode="enhanced")
        if result is None:
            continue
        signals = result.get("signals", "")
        score = result["score"]
        price = result["price"]

        # 评分过低 -> 卖出警告
        if score < SELL_SCORE_THRESHOLD:
            alerts.append({
                "symbol": symbol,
                "name": result["name"],
                "score": score,
                "price": price,
                "reason": f"评分{score}低于卖出阈值{SELL_SCORE_THRESHOLD}",
                "action": "sell_alert",
            })
        # trend_broken -> 趋势破位卖出
        if "trend_broken" in signals:
            alerts.append({
                "symbol": symbol,
                "name": result["name"],
                "score": score,
                "price": price,
                "reason": "趋势破位(trend_broken)",
                "action": "sell_alert",
            })

    return {
        "action": "scan_complete",
        "total_holdings": len(holdings),
        "alerts": alerts,
        "message": f"检查{len(holdings)}只持仓，发现{len(alerts)}个风险" if alerts else f"检查{len(holdings)}只持仓，全部正常",
    }


def execute_auto_trades() -> dict:
    """主逻辑：检查信号 + 执行交易"""
    if not is_trading_time():
        return {"action": "skip", "message": "非交易时间，跳过"}

    # 获取当前状态
    status = portfolio_status()
    holdings = status["holdings"]
    cash = status["cash"]
    total_assets = status["total_assets"]

    # 已持仓的 symbol 集合
    owned_symbols = {h["symbol"] for h in holdings}

    # ═══ 止损检查：先处理已有持仓的风控 ═════════
    stop_loss_trades = []
    for h in holdings:
        sym = h["symbol"]
        profit_pct = h.get("profit_pct", 0)
        if profit_pct <= STOP_LOSS_PCT:
            # 达到止损线 → 立即清仓
            sell_shares = h["shares"]
            result = portfolio_sell(sym, h["name"], sell_shares, h["current_price"])
            if result["status"] == "ok":
                cash += sell_shares * h["current_price"]
                stop_loss_trades.append({
                    "symbol": sym, "name": h["name"],
                    "shares": sell_shares, "price": h["current_price"],
                    "reason": f"止损({profit_pct:+.1f}% ≤ {STOP_LOSS_PCT:.0f}%)",
                })
    # 止损后重新拉取持仓状态
    if stop_loss_trades:
        status = portfolio_status()
        holdings = status["holdings"]
        cash = status["cash"]
        total_assets = status["total_assets"]
        owned_symbols = {h["symbol"] for h in holdings}
    # ═══════════════════════════════════════════

    # 获取新信号
    alerts = check_signals()
    trades = []
    executed_buys = []
    executed_sells = []

    if not alerts:
        return {
            "action": "no_signals",
            "message": "无新信号",
            "holdings_count": len(holdings),
            "cash": cash,
            "total_assets": total_assets,
        }

    # 处理信号
    for a in alerts:
        sym = a["symbol"]
        name = a["name"]
        price = a["price"]
        score = a["score"]
        signal_type = a["type"]
        signals_str = a.get("signals", "")

        # ---- 买入信号 ----
        if signal_type == "buy" and sym not in owned_symbols:
            # ═══ 新增风控检查 ═══════════════════════

            # 1. 每 session 新开仓上限
            if len(executed_buys) >= MAX_NEW_POSITIONS:
                trades.append({
                    "symbol": sym, "name": name, "action": "skip",
                    "reason": f"本批已达最大新开仓数({MAX_NEW_POSITIONS})",
                })
                continue

            # 2. 行业集中度检查
            holdings = portfolio_status()["holdings"]
            industry_counts = {}
            for h in holdings:
                ind = h.get("industry", "")
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
            # 检查候选股票的行业
            signal_industry = a.get("industry", "")
            current_in_ind = industry_counts.get(signal_industry, 0)
            cap = INDUSTRY_CAPS.get(signal_industry, float("inf"))
            if current_in_ind >= cap:
                trades.append({
                    "symbol": sym, "name": name, "action": "skip",
                    "reason": f"行业{signal_industry}已达上限({cap}只)",
                })
                continue

            # ═══════════════════════════════════════

            # 计算买入数量：不超过现金5%，不超过总资产15%
            max_by_cash = cash * MAX_CASH_PER_TRADE
            max_by_portfolio = total_assets * MAX_PORTFOLIO_SINGLE
            max_spend = min(max_by_cash, max_by_portfolio)

            if max_spend < price * 100:
                trades.append({
                    "symbol": sym, "name": name, "action": "skip",
                    "reason": f"资金不足(需至少{price*100:.0f}, 可用{max_spend:.0f})",
                })
                continue

            # 整手买入（100股）: 实际买入股数
            buy_shares = min(int(max_spend / price / 100) * 100, 1000)  # 最多1000股
            if buy_shares < 100:
                trades.append({
                    "symbol": sym, "name": name, "action": "skip",
                    "reason": f"金额不够买1手(需{price*100:.0f})",
                })
                continue

            buy_cost = buy_shares * price
            # 3. 最低仓位市值检查
            if buy_cost < MIN_POSITION_VALUE:
                trades.append({
                    "symbol": sym, "name": name, "action": "skip",
                    "reason": f"买入金额{buy_cost:.0f}<最低仓位{MIN_POSITION_VALUE}",
                })
                continue

            # 执行买入
            result = portfolio_buy(sym, name, buy_shares, price)
            if result["status"] == "ok":
                cash -= buy_cost
                executed_buys.append({
                    "symbol": sym, "name": name,
                    "shares": buy_shares, "price": price,
                    "total": round(buy_cost, 2), "score": score,
                })
                trades.append({
                    "symbol": sym, "name": name, "action": "buy",
                    "shares": buy_shares, "price": price,
                    "total": round(buy_cost, 2), "score": score,
                })

        # ---- 卖出信号 ----
        elif signal_type in ("sell", "warning") and sym in owned_symbols:
            # 查找持仓信息
            holding = next((h for h in holdings if h["symbol"] == sym), None)
            if not holding:
                continue

            sell_shares = holding["shares"]
            result = portfolio_sell(sym, name, sell_shares, price)
            if result["status"] == "ok":
                cash += sell_shares * price
                executed_sells.append({
                    "symbol": sym, "name": name,
                    "shares": sell_shares, "price": price,
                    "total": round(sell_shares * price, 2),
                    "profit_pct": holding.get("profit_pct", 0),
                })
                trades.append({
                    "symbol": sym, "name": name, "action": "sell",
                    "shares": sell_shares, "price": price,
                    "profit_pct": holding.get("profit_pct", 0),
                })

    # 获取交易后最新状态
    new_status = portfolio_status()

    # 合并止损和普通卖出
    all_sells = executed_sells + stop_loss_trades

    return {
        "action": "trades_executed",
        "trades": trades,
        "buys": executed_buys,
        "sells": all_sells,
        "stop_losses": stop_loss_trades,
        "trade_count": len(trades) + len(stop_loss_trades),
        "buy_count": len(executed_buys),
        "sell_count": len(all_sells),
        "holdings_count": len(new_status["holdings"]),
        "cash": new_status["cash"],
        "total_assets": new_status["total_assets"],
        "total_profit": new_status["total_profit"],
        "total_profit_pct": new_status["total_profit_pct"],
        "message": f"执行{len(trades)}笔操作: 买入{len(executed_buys)}, 卖出{len(executed_sells)}" if trades else f"检查{alerts}个信号，无需操作",
    }


def format_trade_report(result: dict) -> str:
    """格式化交易报告供飞书推送"""
    now = datetime.now().strftime("%H:%M")

    if result["action"] in ("skip", "no_signals"):
        return None  # 静默

    lines = [f"📊 **盘中自动交易 — {now}**", ""]

    buys = result.get("buys", [])
    sells = result.get("sells", [])

    if buys:
        lines.append("**🟢 自动买入**")
        for b in buys:
            lines.append(f"  {b['name']} ({b['symbol']})")
            lines.append(f"    买入 {b['shares']}股 @ {b['price']:.2f} = {b['total']:.0f}元 | 评分 {b['score']}")
        lines.append("")

    if sells:
        lines.append("**🔴 自动卖出**")
        for s in sells:
            profit_str = f" | 盈亏 {s['profit_pct']:+.1f}%" if s.get("profit_pct") else ""
            lines.append(f"  {s['name']} ({s['symbol']})")
            lines.append(f"    卖出 {s['shares']}股 @ {s['price']:.2f} = {s['total']:.0f}元{profit_str}")
        lines.append("")

    # 快照
    lines.append(f"**持仓快照**")
    lines.append(f"  持有 {result['holdings_count']} 只 | 总资产 {result['total_assets']:,.0f} 元")
    lines.append(f"  可用现金 {result['cash']:,.0f} 元 | 总盈亏 {result['total_profit']:+,.0f} 元 ({result['total_profit_pct']:+.1f}%)")
    lines.append("")
    lines.append("`#自动交易 #盘中监控 #增强评分`")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--morning-scan" in sys.argv:
        # 早盘风险检查
        result = morning_risk_scan()
        if result.get("alerts"):
            alerts = result["alerts"]
            print(f"⚠️ 早盘风险扫描发现 {len(alerts)} 个风险:\n")
            for a in alerts:
                print(f"  - {a['name']}({a['symbol']}): {a['reason']} 评分{a['score']} 现价{a['price']:.2f}")
        else:
            print(result["message"])

    elif "--force-scan" in sys.argv:
        # 强制扫描所有持仓评分
        from scripts.trading import portfolio_status
        status = portfolio_status()
        holdings = status["holdings"]
        if not holdings:
            print("空仓，无需扫描")
            sys.exit(0)

        print(f"扫描 {len(holdings)} 只持仓的最新评分:\n")
        for h in holdings:
            result = scan_stock(h["symbol"], mode="enhanced")
            if result:
                score = result["score"]
                price = result["price"]
                change = result.get("change_pct", 0)
                signals = result.get("signals", "无")
                profit_pct = h.get("profit_pct", 0)
                status_mark = "⚠️" if score < SELL_SCORE_THRESHOLD else "✅"
                print(f"  {status_mark} {h['name']}({h['symbol']})")
                print(f"    评分:{score:.0f} 现价:{price:.2f} 涨幅:{change:+.2f}% 盈亏:{profit_pct:+.1f}% 信号:{signals}")

    else:
        # 常规执行：自动交易
        result = execute_auto_trades()
        report = format_trade_report(result)
        if report:
            print(report)
            if result["trades"]:
                print(f"\n共执行 {len([t for t in result['trades'] if t['action'] in ('buy','sell')])} 笔交易")
            else:
                print("\n无新信号")
        else:
            if result["action"] == "no_signals":
                print("无新信号")
            elif result["action"] == "skip":
                print("非交易时间，跳过")
            else:
                print("无新信号")
