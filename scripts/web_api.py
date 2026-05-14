#!/usr/bin/env python3
"""
股票分析 Web 仪表盘 — FastAPI 后端
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.expanduser("~/stock-system/data")

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from scripts.strategy_engine import (
    load_stock_data, load_all_stocks_info, ScoringEngine,
    scan_all_stocks, get_available_symbols
)
from scripts.visualization import (
    generate_kline_chart, format_feishu_message, CHART_DIR
)
from scripts.trading import (
    watchlist_list, watchlist_add, watchlist_remove,
    portfolio_status, portfolio_buy, portfolio_sell,
    portfolio_history, portfolio_reset,
    run_backtest, save_backtest_result, list_backtest_results,
)


def resample_kline(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """将日K线重采样为周/月/年K线
    
    period: w=周, m=月, y=年
    返回重采样后的OHLCV + MA指标
    """
    period_map = {'w': ('W', 5, 20), 'm': ('ME', 12, 24), 'y': ('YE', 5, 10)}
    if period not in period_map:
        return df
    
    freq, ma_short, ma_long = period_map[period]
    df = df.sort_values('date').copy()
    df = df.set_index('date')
    
    # 先尝试新pandas alias (ME/YE)，失败则回退旧alias (M/Y)
    try:
        grouped = df[['open', 'high', 'low', 'close', 'volume']].resample(freq)
    except ValueError:
        fallback = {'w': 'W', 'm': 'M', 'y': 'Y'}
        grouped = df[['open', 'high', 'low', 'close', 'volume']].resample(fallback[period])
    
    resampled = pd.DataFrame({
        'open': grouped['open'].first(),
        'high': grouped['high'].max(),
        'low': grouped['low'].min(),
        'close': grouped['close'].last(),
        'volume': grouped['volume'].sum(),
    }).dropna()
    
    resampled = resampled.reset_index()
    
    # 在聚合后的K线上重新计算均线
    closes = resampled['close'].values.astype(float)
    resampled['ma5'] = pd.Series(closes).rolling(ma_short).mean()
    resampled['ma20'] = pd.Series(closes).rolling(ma_long).mean()
    resampled['ma60'] = pd.Series(closes).rolling(ma_long * 3).mean()
    
    return resampled

# 扫描结果缓存（经典版 + 增强版）
SCAN_CACHE = os.path.join(DATA_DIR, "scan_cache.parquet")
SCAN_CACHE_ENHANCED = os.path.join(DATA_DIR, "scan_enhanced_cache.parquet")
SCAN_TIME_FILE = os.path.join(DATA_DIR, "scan_time.txt")
SCAN_TIME_FILE_ENHANCED = os.path.join(DATA_DIR, "scan_enhanced_time.txt")

def _get_cache_path(mode: str = "classic") -> tuple:
    """获取指定模式的缓存路径 (cache_path, time_path)"""
    if mode == "enhanced":
        return SCAN_CACHE_ENHANCED, SCAN_TIME_FILE_ENHANCED
    return SCAN_CACHE, SCAN_TIME_FILE

def _get_cached_scan(mode: str = "classic") -> pd.DataFrame:
    """读取缓存的扫描结果"""
    cache_path, _ = _get_cache_path(mode)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None

def _save_scan_cache(df: pd.DataFrame, mode: str = "classic"):
    """保存扫描结果"""
    cache_path, time_path = _get_cache_path(mode)
    df.to_parquet(cache_path, compression="zstd")
    with open(time_path, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 扫描线程池（单线程，避免多个扫描同时跑）
_scan_executor = ThreadPoolExecutor(max_workers=1)
_scan_status = {"running": False, "progress": "", "error": None}

async def _run_scan_in_thread(top_n: int = 200, mode: str = "classic") -> pd.DataFrame:
    """在线程池中运行扫描，不阻塞事件循环"""
    global _scan_status
    loop = asyncio.get_event_loop()
    _scan_status = {"running": True, "progress": f"扫描中 ({mode})...", "error": None, "mode": mode}
    try:
        results = await loop.run_in_executor(_scan_executor, scan_all_stocks, top_n, 0, mode)
        if len(results) > 0:
            _save_scan_cache(results, mode=mode)
            _scan_status = {"running": False, "progress": "完成", "error": None, "mode": None}
        return results
    except Exception as e:
        _scan_status = {"running": False, "progress": "", "error": str(e), "mode": None}
        return pd.DataFrame()

app = FastAPI(title="股票分析系统")

@app.get("/api/scan/status")
async def scan_status():
    """获取扫描状态（包含两种模式）"""
    global _scan_status
    cached = _get_cached_scan("classic")
    cached_enhanced = _get_cached_scan("enhanced")
    
    def _read_time(path):
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
        return None
    
    return {
        "scanning": _scan_status["running"],
        "progress": _scan_status["progress"],
        "error": _scan_status.get("error"),
        "mode": _scan_status.get("mode"),
        "classic": {
            "cached_at": _read_time(SCAN_TIME_FILE),
            "has_cache": cached is not None,
            "cached_count": len(cached) if cached is not None else 0,
        },
        "enhanced": {
            "cached_at": _read_time(SCAN_TIME_FILE_ENHANCED),
            "has_cache": cached_enhanced is not None,
            "cached_count": len(cached_enhanced) if cached_enhanced is not None else 0,
        }
    }

# 辅助函数: 安全地转换值为可JSON序列化的格式
def _safe_float(v):
    try:
        if isinstance(v, (np.floating, float)):
            return round(float(v), 2)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return str(v)

def _is_na(v):
    try:
        return pd.isna(v) or v is None
    except:
        return False

# 提供图表静态文件
os.makedirs("web/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("web/index.html", "r") as f:
        return f.read()


@app.get("/api/stocks/search")
async def search_stocks(q: str = Query("", min_length=1)):
    """按名称或代码搜索股票"""
    info = load_all_stocks_info()
    if info is None:
        return {"results": []}
    
    q = q.strip().lower()
    results = info[info["code_name"].str.lower().str.contains(q, na=False) |
                   info["symbol"].str.lower().str.contains(q, na=False)]
    
    return {
        "results": results.head(20)[["symbol", "code_name"]].to_dict(orient="records")
    }


@app.get("/api/stocks")
async def list_stocks():
    """获取股票列表"""
    symbols = get_available_symbols()
    info = load_all_stocks_info()
    
    stocks = []
    for s in symbols[:5000]:  # 最多5000只
        name = s
        if info is not None:
            match = info[info["symbol"] == s]
            if len(match) > 0:
                name = match.iloc[0].get("code_name", s)
        stocks.append({"symbol": s, "name": name})
    
    return {"total": len(symbols), "stocks": stocks}


@app.get("/api/analysis/{symbol}")
async def analyze_stock(symbol: str, period: str = Query("d", regex="^(d|w|m|y)$"),
                        mode: str = Query("classic", regex="^(classic|enhanced)$")):
    """分析单只股票，period: d=日K, w=周K, m=月K, y=年K"""
    df = load_stock_data(symbol, years=0)  # 全量数据
    if df is None:
        return {"error": f"未找到 {symbol}"}
    
    # 评分和信号只在日K下有效（统一用最近5年，与全市场扫描一致）
    result = ScoringEngine.score_stock(
        load_stock_data(symbol),   # 默认years=5
        mode=mode,
        symbol=symbol
    ) if period == "d" else {"score": 0, "details": {}, "signals": [], "latest": {"close": 0}, "mode": mode}
    
    # K线数据（按周期重采样）
    kline_df = df if period == "d" else resample_kline(df, period)
    # 日K默认近3年（~750根），缩放时按需加载更多
    if period == "d" and len(kline_df) > 750:
        kline_df = kline_df.tail(750)
    kline_data = format_kline_json(kline_df)
    
    # 获取名称
    info = load_all_stocks_info()
    name = symbol
    if info is not None:
        match = info[info["symbol"] == symbol]
        if len(match) > 0:
            name = match.iloc[0].get("code_name", symbol)
    
    # 生成K线图（日K才生成PNG）
    chart_path = generate_kline_chart(df, symbol, name, result["signals"]) if period == "d" else None
    
    return {
        "symbol": symbol,
        "name": name,
        "score": result["score"],
        "mode": result.get("mode", "classic"),
        "details": {k: v["score"] for k, v in result["details"].items()},
        "signals": result["signals"],
        "latest": {k: _safe_float(v) 
                   for k, v in result["latest"].items() 
                   if k not in ["symbol", "date"] and not _is_na(v)},
        "chart_url": f"/chart/{symbol}" if chart_path else None,
        "kline_data": kline_data,
        "kline_count": len(kline_data),  # 告诉前端有多少根K线
        "data_range": {  # 当前加载数据的日期范围
            "earliest": int(kline_df["date"].min().timestamp()) if len(kline_df) > 0 else 0,
            "latest": int(kline_df["date"].max().timestamp()) if len(kline_df) > 0 else 0,
        }
    }


@app.get("/api/kline/{symbol}")
async def get_kline_range(symbol: str,
                          before: int = Query(0, description="加载此时间戳之前的K线"),
                          limit: int = Query(500, le=2000)):
    """按需加载更早的K线数据（用于缩放时懒加载）"""
    df = load_stock_data(symbol, years=0)
    if df is None:
        return {"error": f"未找到 {symbol}"}
    
    df = df[df["date"].astype('int64') // 10**6 < before].tail(limit)
    if len(df) == 0:
        return {"kline_data": [], "kline_count": 0}
    
    kline_data = format_kline_json(df)
    return {
        "kline_data": kline_data,
        "kline_count": len(kline_data),
        "data_range": {
            "earliest": int(df["date"].min().timestamp()),
            "latest": int(df["date"].max().timestamp()),
        }
    }


@app.get("/api/realtime/{symbol}")
async def realtime_quote(symbol: str):
    """获取实时行情（从新浪财经）"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
            resp = await client.get(f"https://hq.sinajs.cn/list={symbol}", headers=headers)
            parsed = _parse_sina_quote(resp.content.decode("gbk"))
            if parsed:
                return parsed
            return {"error": "解析失败"}
    except Exception as e:
        return {"error": str(e)}


def _parse_sina_quote(text: str) -> dict:
    """解析新浪财经实时行情数据"""
    import re
    # 如果文本是GBK乱码，尝试修复
    if "hq_str" not in text:
        try:
            text = text.encode("latin1").decode("gbk")
        except:
            return None
    m = re.search(r'"(.+)"', text, re.DOTALL)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) < 10:
        return None
    try:
        price = float(parts[3]) if parts[3] else 0
        prev_close = float(parts[2]) if parts[2] else 0
        return {
            "name": parts[0],
            "open": float(parts[1]) if parts[1] else 0,
            "prev_close": prev_close,
            "price": price,
            "high": float(parts[4]) if parts[4] else 0,
            "low": float(parts[5]) if parts[5] else 0,
            "buy": float(parts[6]) if parts[6] else 0,
            "sell": float(parts[7]) if parts[7] else 0,
            "volume": int(parts[8]) if parts[8] else 0,
            "amount": float(parts[9]) if parts[9] else 0,
            "change": round(price - prev_close, 2) if prev_close > 0 else 0,
            "change_pct": round((price - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0,
        }
    except (ValueError, IndexError):
        return None


@app.get("/api/realtime-batch")
async def realtime_batch(symbols: str = Query("", description="逗号分隔的股票代码，如 sh600519,sz000001")):
    """批量获取实时行情"""
    if not symbols:
        return {"error": "请提供股票代码"}
    import httpx
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
            for i in range(0, len(symbol_list), 50):
                chunk = symbol_list[i:i+50]
                resp = await client.get(f"https://hq.sinajs.cn/list={','.join(chunk)}", headers=headers)
                # 用原始字节解码（新浪返回GBK）
                raw = resp.content.decode("gbk")
                # 每行格式: var hq_str_xxx= "...";
                for line in raw.split(";\n"):
                    line = line.strip()
                    if not line or "hq_str" not in line:
                        continue
                    parsed = _parse_sina_quote(line)
                    if parsed:
                        m2 = re.search(r'hq_str_(\w+)', line)
                        code = m2.group(1) if m2 else "unknown"
                        results[code] = parsed
    except Exception as e:
        return {"error": str(e)}
    return {"quotes": results, "count": len(results)}


@app.get("/api/scan")
async def scan(top_n: int = Query(100, le=200), min_score: float = Query(0),
               mode: str = Query("classic", regex="^(classic|enhanced)$")):
    """全市场扫描（非阻塞，后台线程执行）"""
    global _scan_status
    cache_path, time_path = _get_cache_path(mode)
    
    # 读取缓存时间
    cached_at = None
    if os.path.exists(time_path):
        with open(time_path) as f:
            cached_at = f.read().strip()
    
    cached = _get_cached_scan(mode)
    
    if cached is None and not _scan_status["running"]:
        asyncio.ensure_future(_run_scan_in_thread(top_n=200, mode=mode))
        return {"scanning": True, "mode": mode, "total": 0, "results": [], "cached_at": None, "summary": {"strong_buy": 0, "watch": 0, "average_score": 0}}
    elif _scan_status["running"]:
        return {"scanning": True, "mode": mode, "total": 0, "results": [], "cached_at": None, "summary": {"strong_buy": 0, "watch": 0, "average_score": 0}}
    
    results = cached
    
    # 按评分过滤
    filtered = results[results["score"] >= min_score].head(top_n)
    
    # 清理JSON不兼容的值（NaN → None）
    records = filtered.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                rec[k] = None
    
    return {
        "total": len(filtered),
        "scanned_total": len(results),
        "mode": mode,
        "results": records,
        "cached_at": cached_at,
        "summary": {
            "strong_buy": int((results["score"] >= 70).sum()),
            "watch": int(((results["score"] >= 60) & (results["score"] < 70)).sum()),
            "average_score": round(float(results["score"].mean()), 1),
        }
    }


@app.post("/api/scan/refresh")
async def refresh_scan(mode: str = Query("classic", regex="^(classic|enhanced)$")):
    """强制刷新扫描结果（后台运行，不阻塞）"""
    global _scan_status
    if _scan_status["running"]:
        return {"status": "scanning", "message": "正在扫描中，请稍候"}
    asyncio.ensure_future(_run_scan_in_thread(top_n=200, mode=mode))
    return {"status": "started", "message": f"{'增强版' if mode == 'enhanced' else '经典版'}扫描已启动，请稍候刷新查看结果"}


@app.get("/api/scan/by_industry")
async def scan_by_industry(industry: str = Query(...),
                           mode: str = Query("enhanced", regex="^(classic|enhanced)$"),
                           top_n: int = Query(30, le=50)):
    """按行业筛选股票，返回评分排序前N名（优先读缓存）"""
    try:
        # 优先读行业缓存
        import os
        INDUSTRY_CACHE = os.path.join(DATA_DIR, "industry_top30.parquet")
        if os.path.exists(INDUSTRY_CACHE):
            df_cache = pd.read_parquet(INDUSTRY_CACHE)
            if "industry" in df_cache.columns:
                subset = df_cache[df_cache["industry"] == industry].head(top_n)
                if len(subset) > 0:
                    records = subset.to_dict(orient="records")
                    for rec in records:
                        for k, v in rec.items():
                            if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                                rec[k] = None
                    return {
                        "industry": industry,
                        "total": len(records),
                        "scanned": len(subset),
                        "mode": mode,
                        "cached": True,
                        "results": records,
                    }

        # 缓存未命中，实时扫描
        from scripts.data_loader import load_industry_mapping
        mapping = load_industry_mapping()
        if mapping is None:
            return {"error": "行业数据未加载", "results": []}

        industry_symbols = mapping[mapping["industry"] == industry]["symbol"].tolist()
        if not industry_symbols:
            return {"error": f"未找到行业 [{industry}] 的股票", "results": []}

        loop = asyncio.get_event_loop()
        df_result = await loop.run_in_executor(
            _scan_executor,
            scan_all_stocks, top_n, 0, mode, industry_symbols
        )

        if len(df_result) == 0:
            return {"industry": industry, "total": 0, "results": []}

        records = df_result.head(top_n).to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                    rec[k] = None

        return {
            "industry": industry,
            "total": len(records),
            "scanned": len(industry_symbols),
            "mode": mode,
            "cached": False,
            "results": records,
        }
    except Exception as e:
        logger.error(f"按行业扫描错误: {e}")
        return {"error": str(e), "results": []}


@app.get("/chart/{symbol}")
async def get_chart(symbol: str):
    """获取K线图"""
    import glob
    # 找最新生成的图表
    files = sorted(glob.glob(f"{CHART_DIR}/{symbol}_*.png"))
    if files:
        return FileResponse(files[-1], media_type="image/png")
    return {"error": "图表未生成"}


def format_kline_json(df):
    """格式化K线数据为前端可用格式"""
    if df is None or len(df) == 0:
        return []
    
    result = []
    for _, row in df.iterrows():
        entry = {
            "time": int(row["date"].timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        # 添加指标
        for col in ["ma5", "ma10", "ma20", "ma60", "macd_dif", "macd_dea", 
                     "macd_hist", "rsi", "boll_upper", "boll_mid", "boll_lower",
                     "kdj_k", "kdj_d", "kdj_j"]:
            if col in row and not pd.isna(row[col]):
                entry[col] = float(row[col])
        
        result.append(entry)
    
    return result


# ═══════════════════════════════════════════════
#  自选股 API
# ═══════════════════════════════════════════════

@app.get("/api/watchlist")
async def api_watchlist():
    """获取自选股列表"""
    wl = watchlist_list()
    if not wl:
        return {"items": [], "count": 0}
    
    # 获取最新价格
    info = load_all_stocks_info()
    for item in wl:
        sym = item["symbol"]
        try:
            df = load_stock_data(sym)
            if df is not None:
                item["price"] = round(float(df.iloc[-1]["close"]), 2)
                item["change_pct"] = round(float(df.iloc[-1].get("change_pct", 0)), 2)
        except:
            item["price"] = 0
            item["change_pct"] = 0
        if info is not None:
            match = info[info["symbol"] == sym]
            if len(match) > 0:
                item["name"] = match.iloc[0].get("code_name", item.get("name", sym))
        # 行业归属
        try:
            from scripts.data_loader import get_industry, load_industry_mapping
            _mapping = load_industry_mapping()
            item["industry"] = get_industry(sym, _mapping) if _mapping is not None else None
        except:
            item["industry"] = None
    
    return {"items": wl, "count": len(wl)}

@app.post("/api/watchlist/add")
async def api_watchlist_add(symbol: str = Query(""), name: str = Query("")):
    return watchlist_add(symbol, name)

@app.post("/api/watchlist/remove")
async def api_watchlist_remove(symbol: str = Query("")):
    return watchlist_remove(symbol)


# ═══════════════════════════════════════════════
#  模拟交易 API
# ═══════════════════════════════════════════════

@app.get("/api/portfolio")
async def api_portfolio():
    """投资组合状态"""
    # 获取持仓中所有股票的最新价格
    pf = portfolio_status({})
    symbols = [h["symbol"] for h in pf["holdings"]]
    # 优先获取实时价格
    prices = {}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
            for i in range(0, len(symbols), 50):
                chunk = symbols[i:i+50]
                resp = await client.get(f"https://hq.sinajs.cn/list={','.join(chunk)}", headers=headers)
                raw = resp.content.decode("gbk")
                for line in raw.split(";\n"):
                    if "hq_str" not in line:
                        continue
                    parsed = _parse_sina_quote(line)
                    if parsed and parsed.get("price"):
                        m2 = re.search(r'hq_str_(\w+)', line)
                        code = m2.group(1) if m2 else None
                        if code:
                            prices[code] = parsed["price"]
    except:
        pass
    
    # 实时价格获取失败时降级为本地数据
    for sym in symbols:
        if sym not in prices:
            try:
                df = load_stock_data(sym)
                if df is not None:
                    prices[sym] = float(df.iloc[-1]["close"])
            except:
                pass
    
    return portfolio_status(prices)

@app.post("/api/portfolio/buy")
async def api_portfolio_buy(symbol: str = Query(""), name: str = Query(""),
                             shares: int = Query(100), price: float = Query(0)):
    if price <= 0:
        # 自动获取最新价
        df = load_stock_data(symbol)
        if df is not None:
            price = float(df.iloc[-1]["close"])
        else:
            return {"status": "error", "message": "无法获取价格"}
    return portfolio_buy(symbol, name, shares, price)

@app.post("/api/portfolio/sell")
async def api_portfolio_sell(symbol: str = Query(""), name: str = Query(""),
                              shares: int = Query(100), price: float = Query(0)):
    if price <= 0:
        df = load_stock_data(symbol)
        if df is not None:
            price = float(df.iloc[-1]["close"])
        else:
            return {"status": "error", "message": "无法获取价格"}
    return portfolio_sell(symbol, name, shares, price)

@app.get("/api/portfolio/history")
async def api_portfolio_history():
    return {"transactions": portfolio_history()}

@app.post("/api/portfolio/reset")
async def api_portfolio_reset():
    return portfolio_reset()


# ═══════════════════════════════════════════════
#  回测 API
# ═══════════════════════════════════════════════

@app.post("/api/backtest/run")
async def api_backtest_run(symbol: str = Query(""), name: str = Query(""),
                           buy_threshold: float = Query(70),
                           sell_threshold: float = Query(55)):
    result = run_backtest(symbol, name, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    if result.get("status") != "error":
        save_backtest_result(result)
    return result

@app.get("/api/backtest/list")
async def api_backtest_list():
    return {"results": list_backtest_results()}


@app.get("/api/industries")
async def list_industries():
    """获取所有行业列表"""
    try:
        from scripts.data_loader import load_industry_mapping
        mapping = load_industry_mapping()
        if mapping is None or "industry" not in mapping.columns:
            return {"industries": []}
        industries = sorted(mapping["industry"].unique().tolist())
        return {"industries": industries, "total": len(industries)}
    except Exception as e:
        logger.error(f"行业列表接口错误: {e}")
        return {"industries": []}


@app.get("/api/industry/rankings")
async def industry_rankings():
    """行业板块排名（基于行业动量数据）"""
    try:
        from scripts.data_loader import load_industry_momentum
        rankings = load_industry_momentum()
        if rankings is None or len(rankings) == 0:
            return {"rankings": [], "total": 0, "updated_at": None}
        
        # 按20日涨幅排序（降序）
        rankings = rankings.sort_values("return_20d", ascending=False).reset_index(drop=True)
        rankings["rank"] = range(1, len(rankings) + 1)
        
        records = rankings.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                    rec[k] = None
        
        return {"rankings": records, "total": len(records)}
    except Exception as e:
        logger.error(f"行业排名接口错误: {e}")
        return {"rankings": [], "total": 0, "updated_at": None}


@app.get("/api/industry/{symbol}")
async def stock_industry(symbol: str):
    """获取个股行业归属信息"""
    try:
        from scripts.data_loader import load_industry_mapping, get_industry, load_industry_momentum
        mapping = load_industry_mapping()
        if mapping is None:
            return {"symbol": symbol, "industry": None}
        
        industry_name = get_industry(symbol, mapping)
        if not industry_name:
            return {"symbol": symbol, "industry": None}
        
        # 获取该行业动量
        momentum = load_industry_momentum()
        industry_score = 5.0
        industry_rank = None
        industry_total = None
        if momentum is not None and industry_name in momentum["industry"].values:
            # 计算排名（按20日涨幅）
            sorted_m = momentum.sort_values("return_20d", ascending=False).reset_index(drop=True)
            match_idx = sorted_m[sorted_m["industry"] == industry_name].index[0]
            industry_rank = int(match_idx) + 1
            industry_total = len(sorted_m)
            # 行业动量评分: 排名前30%得8+, 前60%得6+, 其余4+
            pct = industry_rank / industry_total
            if pct <= 0.2:
                industry_score = 9.0
            elif pct <= 0.4:
                industry_score = 7.5
            elif pct <= 0.6:
                industry_score = 6.0
            elif pct <= 0.8:
                industry_score = 4.5
            else:
                industry_score = 3.0
        
        return {
            "symbol": symbol,
            "industry": industry_name,
            "industry_score": industry_score,
            "industry_rank": industry_rank,
            "industry_total": industry_total,
        }
    except Exception as e:
        logger.error(f"个股行业接口错误 ({symbol}): {e}")
        return {"symbol": symbol, "industry": None}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    print(f"启动仪表盘: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
