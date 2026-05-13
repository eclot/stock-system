#!/usr/bin/env python3
"""
盘中监控 + 定时任务模块
- 全市场扫描（收盘后）
- 盘中实时提醒（交易时段）
- 飞书推送
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
from datetime import datetime, time as dt_time
import pandas as pd

from scripts.strategy_engine import (
    load_stock_data, load_all_stocks_info, load_financial_data,
    ScoringEngine, scan_all_stocks, get_available_symbols
)

# ========== 配置 ==========
DATA_DIR = os.path.expanduser("~/stock-system/data")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# A股交易时间
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(15, 0)
MORNING_CLOSE = dt_time(11, 30)
AFTERNOON_OPEN = dt_time(13, 0)


def is_trading_time() -> bool:
    """判断当前是否在交易时段"""
    now = datetime.now().time()
    if now < MARKET_OPEN or now > MARKET_CLOSE:
        return False
    if MORNING_CLOSE < now < AFTERNOON_OPEN:
        return False
    # 周末
    if datetime.now().weekday() >= 5:
        return False
    return True


def is_market_day() -> bool:
    """判断是否是交易日"""
    return datetime.now().weekday() < 5


def daily_scan_report(top_n: int = 50) -> dict:
    """
    收盘后全市场扫描，生成报告
    返回: {summary, results, chart_paths}
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始全市场扫描...")
    
    results = scan_all_stocks(top_n=top_n)
    
    if results is None or len(results) == 0:
        print("无数据，跳过")
        return None
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_parquet(os.path.join(REPORT_DIR, f"scan_{timestamp}.parquet"))
    
    # 生成图表
    from scripts.visualization import generate_score_chart, generate_kline_chart
    
    score_chart = generate_score_chart(results, top_n=20)
    
    # 生成高评分股票的K线图
    signal_charts = []
    for _, row in results.head(5).iterrows():
        if "strong_buy" in str(row.get("signals", "")):
            df = load_stock_data(row["symbol"])
            if df is not None:
                info = load_all_stocks_info()
                name = row["name"]
                path = generate_kline_chart(df, row["symbol"], name, 
                                           ["strong_buy"])
                if path:
                    signal_charts.append(path)
    
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total": len(results),
        "strong_buy": int((results["score"] >= 70).sum()),
        "watch": int(((results["score"] >= 60) & (results["score"] < 70)).sum()),
        "average": round(float(results["score"].mean()), 1),
        "top5": results.head(5)[["symbol", "name", "score"]].to_dict(orient="records"),
    }
    
    print(f"扫描完成: {summary['total']} 只, "
          f"强买: {summary['strong_buy']}, "
          f"关注: {summary['watch']}")
    
    # 同步更新网页缓存
    try:
        cache_path = os.path.join(DATA_DIR, "scan_cache.parquet")
        time_path = os.path.join(DATA_DIR, "scan_time.txt")
        results.to_parquet(cache_path, compression="zstd")
        with open(time_path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"缓存更新失败: {e}")
    
    return {
        "summary": summary,
        "results": results,
        "charts": {"score": score_chart, "signals": signal_charts},
        "timestamp": timestamp,
    }


def monitor_realtime(interval: int = 30, check_symbols: list = None):
    """
    盘中实时监控（简化版）
    - 每 N 秒检查一次指定股票
    - 检测新信号
    - 推送通知
    
    注意: 本函数为无限循环，用于定时任务
    """
    if not is_trading_time():
        print("当前非交易时间")
        return
    
    if check_symbols is None:
        # 默认检查之前有信号的
        check_symbols = get_available_symbols()[:50]  # 前50只
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始盘中监控 "
          f"({len(check_symbols)} 只, 每{interval}秒)")
    
    # 这里简化实现：实际部署用 cronjob 定时触发，
    # 每次触发只检查一次，不做无限循环
    alerts = []
    
    for symbol in check_symbols:
        df = load_stock_data(symbol)
        if df is None or len(df) < 60:
            continue
        
        result = ScoringEngine.score_stock(df)
        
        if result["score"] >= 70 and "strong_buy" in result["signals"]:
            alerts.append({
                "symbol": symbol,
                "score": result["score"],
                "price": result["latest"].get("close", 0),
                "signals": result["signals"],
            })
    
    if alerts:
        print(f"发现 {len(alerts)} 个信号!")
        for a in alerts[:5]:
            print(f"  {a['symbol']}: 评分 {a['score']}, 信号 {a['signals']}")
    
    return alerts


def run_daily_job():
    """每日收盘后执行"""
    print(f"\n{'='*50}")
    print(f"每日任务: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    if not is_market_day():
        print("非交易日，跳过")
        return
    
    # 全市场扫描
    report = daily_scan_report(top_n=50)
    
    if report:
        print(f"\n扫描结果摘要:")
        print(f"  扫描: {report['summary']['total']} 只")
        print(f"  强买: {report['summary']['strong_buy']}")
        print(f"  关注: {report['summary']['watch']}")
        print(f"  平均分: {report['summary']['average']}")
        
        if report['charts']['score']:
            print(f"  评分图: {report['charts']['score']}")
    
    print(f"{'='*50}\n")
    return report


def format_feishu_daily(report: dict) -> str:
    """格式化每日飞书推送"""
    if not report:
        return "今日无数据"
    
    s = report["summary"]
    
    msg = f"📊 **股票分析日报 — {s['date']}**\n\n"
    msg += f"📈 全市场扫描: {s['total']} 只\n"
    msg += f"🟢 **强烈买入**: {s['strong_buy']} 只\n"
    msg += f"🟡 **关注**: {s['watch']} 只\n"
    msg += f"📊 **平均评分**: {s['average']}\n\n"
    
    msg += "**TOP 5:**\n"
    for i, stock in enumerate(s["top5"]):
        emoji = "🥇🥇🥇🥇🥇"[i]
        msg += f"{emoji} {stock['symbol']} **{stock['name']}** — {stock['score']}分\n"
    
    msg += "\n`#股票分析`"
    return msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股票监控")
    parser.add_argument("mode", choices=["scan", "daily", "monitor"],
                       default="scan", nargs="?")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()
    
    if args.mode == "daily":
        run_daily_job()
    elif args.mode == "monitor":
        alert = monitor_realtime()
    else:
        report = daily_scan_report(top_n=args.top)
        if report:
            print(format_feishu_daily(report))
