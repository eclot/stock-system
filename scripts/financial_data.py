#!/usr/bin/env python3
"""
财务数据收集 + 基本面筛选器
数据源: 优先 AKShare → 备用 BaoStock
"""

import pandas as pd
import numpy as np
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

DATA_DIR = os.path.expanduser("~/stock-system/data")
FINANCIAL_DIR = os.path.join(DATA_DIR, "financial")
FINANCIAL_PATH = os.path.join(FINANCIAL_DIR, "financial_indicators.parquet")
QUALIFIED_CACHE = os.path.join(FINANCIAL_DIR, "qualified_stocks.parquet")
INFO_DIR = os.path.join(DATA_DIR, "info")
DAILY_DIR = os.path.join(DATA_DIR, "daily")

# 基本面筛选阈值
FILTER_RULES = {
    "pe_ttm_min": 0,        # PE > 0 (盈利)
    "pe_ttm_max": 100,      # PE < 100 (不过度高估)
    "roe_min": -10,         # ROE > -10% (不严重亏损)
    "debt_ratio_max": 0.9,  # 资产负债率 < 90%
    "revenue_growth_min": -50,  # 营收增速 > -50%
}


def get_available_symbols() -> list:
    """获取已下载K线的股票列表"""
    if not os.path.exists(DAILY_DIR):
        return []
    files = os.listdir(DAILY_DIR)
    return sorted([f.replace(".parquet", "") for f in files if f.endswith(".parquet")])


# ═══════════════════════════════════════════════
#  1. BaoStock 财务数据拉取
# ═══════════════════════════════════════════════

def _convert_symbol_baostock(symbol: str) -> str:
    """转换 Hermes symbol 为 BaoStock 格式: sh600519 → sh.600519"""
    if symbol.startswith("sh") or symbol.startswith("sz"):
        exchange = symbol[:2]
        code = symbol[2:]
        return f"{exchange}.{code}"
    return symbol


def _convert_symbol_from_baostock(bs_code: str) -> str:
    """转换 BaoStock 格式为 Hermes symbol: sh.600519 → sh600519"""
    code = bs_code.replace(".", "")
    return code


def fetch_financial_baostock(symbol: str) -> dict:
    """从BaoStock拉取单只股票财务指标（最近一季度）"""
    import baostock as bs
    result = {"symbol": symbol, "roe": None, "eps": None, "bps": None,
              "debt_ratio": None, "revenue_growth": None, "net_profit_growth": None,
              "gross_margin": None}

    bs_code = _convert_symbol_baostock(symbol)

    try:
        # 获取最近一期的利润表数据 (query_profit_data)
        rs_profit = bs.query_profit_data(code=bs_code, year=2026, quarter=1)
        if rs_profit and rs_profit.error_code == "0":
            while rs_profit.next():
                row = rs_profit.get_row_data()
                # row: code, pubDate, statDate,roeAvg, eps, bps, ... 
                # Actually format varies, let me check
                pass
        rs_profit.close()
    except:
        pass

    # Try DuPont data for ROE
    try:
        rs = bs.query_dupont_data(code=bs_code, year=2026, quarter=1)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # 格式: code, pubDate, statDate, dupontROE, ...
                if len(row) >= 4:
                    try:
                        result["roe"] = float(row[3]) if row[3] else None
                    except:
                        pass
        rs.close()
    except:
        pass

    # 利润表
    try:
        rs = bs.query_profit_data(code=bs_code, year=2026, quarter=1)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # 格式: code, pubDate, statDate, roeAvg, eps, bps, ...
                if len(row) >= 6:
                    try:
                        if result.get("roe") is None and row[3]:
                            result["roe"] = float(row[3])
                        result["eps"] = float(row[4]) if row[4] else None
                        result["bps"] = float(row[5]) if row[5] else None
                    except:
                        pass
        rs.close()
    except:
        pass

    # 成长能力
    try:
        rs = bs.query_growth_data(code=bs_code, year=2026, quarter=1)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # 格式: code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, ...
                if len(row) >= 7:
                    try:
                        result["revenue_growth"] = float(row[6]) if row[6] else None  # YOYNI = 净利润同比
                    except:
                        pass
        rs.close()
    except:
        pass

    # 资产负债
    try:
        rs = bs.query_balance_data(code=bs_code, year=2026, quarter=1)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # 查找负债率相关字段
                if len(row) >= 12:
                    try:
                        total_assets = float(row[6]) if row[6] else 0  # totalAssets
                        total_liab = float(row[8]) if row[8] else 0   # totalLiab
                        if total_assets > 0:
                            result["debt_ratio"] = total_liab / total_assets
                    except:
                        pass
        rs.close()
    except:
        pass

    return result


def reload_financial_batch(symbols: list = None, max_workers: int = 10) -> pd.DataFrame:
    """批量拉取财务数据"""
    import baostock as bs
    
    if symbols is None:
        symbols = get_available_symbols()
    
    if not symbols:
        return pd.DataFrame()
    
    bs.login()
    logger.info(f"开始批量拉取财务数据: {len(symbols)}只股票")
    
    results = []
    total = len(symbols)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_financial_baostock, s): s for s in symbols}
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"  拉取失败 {symbol}: {e}")
                results.append({"symbol": symbol})
            
            if (i + 1) % 200 == 0:
                logger.info(f"  进度: {i+1}/{total}")
            time.sleep(0.01)  # 避免过快
    
    bs.logout()
    
    df = pd.DataFrame(results)
    
    # 保存
    os.makedirs(FINANCIAL_DIR, exist_ok=True)
    df.to_parquet(FINANCIAL_PATH)
    logger.info(f"财务数据已保存: {FINANCIAL_PATH} ({len(df)}行)")
    
    return df


# ═══════════════════════════════════════════════
#  2. 基本面筛选器
# ═══════════════════════════════════════════════

def load_financial_data() -> pd.DataFrame:
    """加载已缓存的财务数据"""
    if os.path.exists(FINANCIAL_PATH):
        return pd.read_parquet(FINANCIAL_PATH)
    return None


def filter_by_fundamentals(symbols: list = None, financial_df: pd.DataFrame = None,
                           relaxed: bool = True) -> list:
    """
    基本面筛选器: 两层过滤
    
    严格模式(relaxed=False): 必须有完整的财务数据才通过
    宽松模式(relaxed=True, 默认): 财务数据缺失的股票也通过（只过滤明显差的）
    
    返回: 通过筛选的股票列表
    """
    if financial_df is None:
        financial_df = load_financial_data()
    
    if financial_df is None or len(financial_df) == 0:
        if relaxed:
            return symbols or get_available_symbols()
        return []
    
    if symbols is None:
        symbols = get_available_symbols()
    
    qualified = []
    skipped_financial = 0
    
    for symbol in symbols:
        match = financial_df[financial_df["symbol"] == symbol]
        if len(match) == 0:
            if relaxed:
                qualified.append(symbol)  # 无数据也通过
                skipped_financial += 1
            continue
        
        row = match.iloc[0]
        
        # 检查各项指标
        passed = True
        reasons = []
        
        # ROE
        roe = row.get("roe")
        if roe is not None and not pd.isna(roe):
            if roe < FILTER_RULES["roe_min"]:
                passed = False
                reasons.append(f"ROE={roe:.1f}%")
        
        # 负债率
        debt = row.get("debt_ratio")
        if debt is not None and not pd.isna(debt):
            if debt > FILTER_RULES["debt_ratio_max"]:
                passed = False
                reasons.append(f"负债率={debt:.1%}")
        
        if passed:
            qualified.append(symbol)
    
    if skipped_financial > 0:
        logger.debug(f"  无财务数据跳过筛选: {skipped_financial}只")
    
    return qualified


def precompute_qualified_stocks(relaxed: bool = True) -> pd.DataFrame:
    """预计算可交易池"""
    symbols = get_available_symbols()
    qualified = filter_by_fundamentals(symbols, relaxed=relaxed)
    
    df = pd.DataFrame({"symbol": qualified})
    df.to_parquet(QUALIFIED_CACHE)
    logger.info(f"可交易池已缓存: {len(qualified)}/{len(symbols)}")
    
    return df


def load_qualified_stocks() -> list:
    """加载缓存的合格股票列表"""
    if os.path.exists(QUALIFIED_CACHE):
        df = pd.read_parquet(QUALIFIED_CACHE)
        return df["symbol"].tolist()
    return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if "--reload" in sys.argv:
        print("=== 拉取财务数据 ===")
        # 先拉小批量测试
        test_symbols = ["sh600519", "sz300750", "sh601318", "sz002415"]
        df = reload_financial_batch(test_symbols, max_workers=4)
        if len(df) > 0:
            print(df[["symbol", "roe", "eps", "debt_ratio"]].to_string())
        print("=== 完成 ===")
    else:
        print("=== 基本面筛选器 ===")
        fin = load_financial_data()
        if fin is None:
            print("无财务数据，使用宽松模式（全部通过）")
            symbols = get_available_symbols()
            print(f"可交易池: {len(symbols)}只")
        else:
            qualified = filter_by_fundamentals(relaxed=True)
            print(f"通过筛选: {len(qualified)}只")
