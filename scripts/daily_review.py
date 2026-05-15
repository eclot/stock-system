#!/usr/bin/env python3
"""
每日复盘脚本 — 分析持仓、系统性能、市场环境

用法:
  python daily_review.py              # 完整复盘 + 持仓分析
  python daily_review.py --brief      # 简版复盘（用于飞书推送）
  python daily_review.py --system     # 系统健康检查
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, timedelta
import pandas as pd
from scripts.trading import portfolio_status, portfolio_history, _load_portfolio
from scripts.strategy_engine import (
    load_stock_data, ScoringEngine, load_all_stocks_info,
    load_financial_data, scan_all_stocks, get_available_symbols
)
from scripts.auto_trader import morning_risk_scan
from scripts.realtime_monitor import scan_stock, is_trading_time


def analyze_holdings() -> dict:
    """深入分析持仓状态"""
    status = portfolio_status()
    holdings = status["holdings"]
    if not holdings:
        return {"status": "empty", "message": "空仓"}

    details = []
    total_cost = 0
    total_market_value = 0
    winners = 0
    losers = 0

    for h in holdings:
        profit = h.get("profit", 0)
        profit_pct = h.get("profit_pct", 0)
        total_cost += h.get("cost", 0)
        total_market_value += h.get("market_value", 0)
        if profit > 0:
            winners += 1
        elif profit < 0:
            losers += 1
        details.append(h)

    # 计算行业分布
    industry_dist = {}
    for h in holdings:
        ind = h.get("industry", "未知")
        if ind not in industry_dist:
            industry_dist[ind] = {"count": 0, "value": 0}
        industry_dist[ind]["count"] += 1
        industry_dist[ind]["value"] += h.get("market_value", 0)

    return {
        "status": "active",
        "holdings": details,
        "holdings_count": len(holdings),
        "winners": winners,
        "losers": losers,
        "cash": status["cash"],
        "total_market_value": total_market_value,
        "total_assets": status["total_assets"],
        "total_profit": status["total_profit"],
        "total_profit_pct": status["total_profit_pct"],
        "industry_distribution": industry_dist,
        "transaction_count": status["transaction_count"],
        "created_at": status["created_at"],
    }


def check_system_health() -> dict:
    """系统健康检查"""
    issues = []
    data_dir = os.path.expanduser("~/stock-system/data")

    # 1. 检查数据完整性
    daily_dir = os.path.join(data_dir, "daily")
    if not os.path.exists(daily_dir):
        issues.append({"severity": "critical", "item": "daily数据目录不存在"})
    else:
        parquet_count = len([f for f in os.listdir(daily_dir) if f.endswith(".parquet")])
        if parquet_count < 100:
            issues.append({"severity": "warning", "item": f"K线数据较少(仅{parquet_count}只)"})

    # 2. 检查缓存
    cache_files = {
        "scan_cache.parquet": "经典版扫描缓存",
        "scan_enhanced_cache.parquet": "增强版扫描缓存",
        "industry_top30.parquet": "行业龙头缓存",
    }
    for fname, desc in cache_files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            issues.append({"severity": "warning", "item": f"{desc}不存在"})

    # 3. 检查持仓数据新鲜度
    portfolio_file = os.path.join(data_dir, "portfolio.json")
    if os.path.exists(portfolio_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(portfolio_file))
        hours_stale = (datetime.now() - mtime).total_seconds() / 3600
        if hours_stale > 24:
            issues.append({
                "severity": "info",
                "item": f"持仓数据更新时间: {mtime.strftime('%Y-%m-%d %H:%M')} ({hours_stale:.0f}小时前)"
            })

    # 4. 检查历史交易数据
    status = portfolio_status()
    if status.get("transaction_count", 0) == 0:
        issues.append({"severity": "info", "item": "模拟交易尚无历史记录"})

    return {
        "issues": issues,
        "critical_count": len([i for i in issues if i["severity"] == "critical"]),
        "warning_count": len([i for i in issues if i["severity"] == "warning"]),
        "info_count": len([i for i in issues if i["severity"] == "info"]),
        "status": "healthy" if not any(i["severity"] == "critical" for i in issues) else "degraded",
    }


def generate_review() -> dict:
    """生成完整复盘报告"""
    now = datetime.now()

    # 1. 持仓分析
    holdings_analysis = analyze_holdings()

    # 2. 系统健康
    system_health = check_system_health()

    # 3. 持仓逐只扫描（最新评分）
    holding_scores = []
    if holdings_analysis["status"] == "active":
        for h in holdings_analysis["holdings"]:
            sym = h["symbol"]
            scan_result = scan_stock(sym, mode="enhanced")
            if scan_result:
                holding_scores.append({
                    "symbol": sym,
                    "name": h["name"],
                    "industry": h.get("industry", ""),
                    "score": scan_result["score"],
                    "price": scan_result["price"],
                    "change_pct": scan_result.get("change_pct", 0),
                    "rsi": scan_result.get("rsi", 0),
                    "signals": scan_result.get("signals", ""),
                    "profit_pct": h.get("profit_pct", 0),
                    "cost": h.get("cost", 0),
                    "market_value": h.get("market_value", 0),
                })

    # 4. 交易历史
    history = portfolio_history()
    today_trades = [t for t in history if t["date"].startswith(now.strftime("%Y-%m-%d"))] if history else []

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "holdings_analysis": holdings_analysis,
        "holding_scores": holding_scores,
        "today_trades": today_trades,
        "system_health": system_health,
        "portfolio_history_count": len(history) if history else 0,
    }


def format_feishu_daily_review(review: dict) -> str:
    """格式化飞书推送的复盘报告"""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    weekday_ch = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]

    lines = [f"📋 **模拟交易每日复盘 — {date} 周{weekday_ch}**", ""]

    ha = review["holdings_analysis"]
    if ha["status"] == "empty":
        lines.append("**持仓状况**")
        lines.append("  ❌ 空仓 — 无持仓，等待买入信号")
        lines.append("")
    else:
        # 总览
        lines.append("**总览**")
        lines.append(f"  持仓 {ha['holdings_count']} 只 | 总资产 {ha['total_assets']:,.0f} 元")
        lines.append(f"  总盈亏 {ha['total_profit']:+,.0f} 元 ({ha['total_profit_pct']:+.2f}%)")
        lines.append(f"  盈利 {ha['winners']} 只 / 亏损 {ha['losers']} 只")
        lines.append(f"  现金 {ha['cash']:,.0f} 元 | 累计交易 {ha['transaction_count']} 笔")
        lines.append("")

        # 行业分布
        if ha.get("industry_distribution"):
            lines.append("**行业分布**")
            for ind, info in sorted(ha["industry_distribution"].items(), key=lambda x: x[1]["value"], reverse=True):
                pct = info["value"] / ha["total_assets"] * 100 if ha["total_assets"] > 0 else 0
                lines.append(f"  {ind}: {info['count']}只, {info['value']:,.0f}元 ({pct:.1f}%)")
            lines.append("")

        # 逐只持仓评分
        if review.get("holding_scores"):
            lines.append("**持仓评分明细**")
            for hs in sorted(review["holding_scores"], key=lambda x: x["score"], reverse=True):
                score_emoji = "🟢" if hs["score"] >= 70 else "🟡" if hs["score"] >= 55 else "🔴"
                lines.append(f"  {score_emoji} **{hs['name']}** ({hs['symbol']})")
                lines.append(f"    评分: {hs['score']:.0f} | 盈亏: {hs['profit_pct']:+.1f}%")
                lines.append(f"    现价: {hs['price']:.2f} ({hs['change_pct']:+.2f}%) | RSI: {hs['rsi']:.0f}")
                if hs.get("signals"):
                    lines.append(f"    信号: {hs['signals']}")
            lines.append("")

    # 今日交易
    today_trades = review.get("today_trades", [])
    if today_trades:
        lines.append("**今日操作**")
        for t in today_trades:
            action_emoji = "🟢" if t["action"] == "buy" else "🔴"
            lines.append(f"  {action_emoji} {t['name']} ({t['symbol']})")
            lines.append(f"    操作: {'买入' if t['action'] == 'buy' else '卖出'} {t['shares']}股 @ {t['price']:.2f}")
            lines.append(f"    金额: {t['total']:,.0f} 元 | 时间: {t['date']}")
        lines.append("")
    else:
        lines.append("**今日操作**")
        lines.append("  无交易")
        lines.append("")

    # 系统健康
    sh = review["system_health"]
    lines.append("**系统状态**")
    issues = sh["issues"]
    severity_map = {"critical": "❌", "warning": "⚠️", "info": "ℹ️"}
    if issues:
        for iss in issues:
            mark = severity_map.get(iss["severity"], "•")
            lines.append(f"  {mark} {iss['item']}")
    else:
        lines.append("  ✅ 全部正常")
    lines.append("")

    # 优化建议
    lines.append("**优化建议**")
    suggestions = []
    if ha["status"] == "active":
        # 检查持仓集中度
        if ha.get("industry_distribution"):
            top_ind = max(ha["industry_distribution"].items(), key=lambda x: x[1]["value"])
            top_pct = top_ind[1]["value"] / ha["total_assets"] * 100 if ha["total_assets"] > 0 else 0
            if top_pct > 60:
                suggestions.append(f"⚠️ 行业集中度偏高: {top_ind[0]}占{top_pct:.0f}%，建议分散")

    if not suggestions:
        suggestions.append("持仓结构健康，维持当前策略")

    for s in suggestions:
        lines.append(f"  {s}")

    lines.append("")
    lines.append("`#每日复盘 #模拟交易 #增强评分`")

    return "\n".join(lines)


if __name__ == "__main__":
    from scripts.auto_trader import format_trade_report, execute_auto_trades
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", action="store_true", help="简版复盘")
    parser.add_argument("--system", action="store_true", help="只做系统健康检查")
    parser.add_argument("--scan-result", type=str, help="传入扫描结果JSON文件路径，合并输出")
    args = parser.parse_args()

    if args.system:
        health = check_system_health()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        sys.exit(0)

    review = generate_review()
    report = format_feishu_daily_review(review)
    print(report)
