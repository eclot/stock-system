#!/usr/bin/env python3
"""
统一数据加载层 — 行业归属、财务数据、板块排行

为增强版评分引擎提供数据支持。
所有数据加载函数都带缓存，外部无需关心数据从哪来。
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.expanduser("~/stock-system/data")

# ── 行业数据 ──
INDUSTRY_PATH = os.path.join(DATA_DIR, "info", "industry_mapping.parquet")

# ── 财务数据 ──
FINANCIAL_PATH = os.path.join(DATA_DIR, "financial", "financial_indicators.parquet")

# ── 板块排行缓存（Phase 5用） ──
SECTOR_DIR = os.path.join(DATA_DIR, "sector")
SECTOR_RANK_PATH = os.path.join(SECTOR_DIR, "sector_rankings.parquet")


# ══════════════════════════════════════════════
#  行业归属数据
# ══════════════════════════════════════════════

def load_industry_mapping() -> pd.DataFrame:
    """加载行业归属映射 (symbol → industry_name)"""
    if os.path.exists(INDUSTRY_PATH):
        return pd.read_parquet(INDUSTRY_PATH)
    return None


def get_industry(symbol: str, mapping: pd.DataFrame = None) -> str:
    """获取单只股票的行业名称"""
    if mapping is None:
        mapping = load_industry_mapping()
    if mapping is None:
        return None
    match = mapping[mapping["symbol"] == symbol]
    if len(match) > 0:
        return match.iloc[0]["industry"]
    return None


def reload_industry_mapping() -> pd.DataFrame:
    """从申万行业分类数据构建行业归属映射"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import akshare as ak
    import pandas as pd
    from scripts.industry_code_map import PREFIX_TO_SW1
    
    print("获取申万行业分类数据...")
    df = ak.stock_industry_clf_hist_sw()
    df['update_time'] = pd.to_datetime(df['update_time'])
    
    # 取每只股票最新的行业归属
    latest = df.sort_values('update_time').groupby('symbol').last().reset_index()
    print(f"  共 {len(latest)} 只股票")
    
    # 映射行业代码 → 行业名称
    def code_to_industry(code):
        prefix = str(code)[:2]
        return PREFIX_TO_SW1.get(prefix, "其他")
    
    latest['industry'] = latest['industry_code'].astype(str).apply(code_to_industry)
    
    # 格式化为统一格式: symbol + sh/sz 前缀
    def format_symbol(code):
        code = str(code)
        if code.startswith('6') or code.startswith('9'):
            return f"sh{code}"
        elif code.startswith('0') or code.startswith('3') or code.startswith('2'):
            return f"sz{code}"
        elif code.startswith('4') or code.startswith('8'):
            return f"bj{code}"
        return code
    
    latest['full_symbol'] = latest['symbol'].apply(format_symbol)
    
    result = latest[['full_symbol', 'industry']].rename(columns={'full_symbol': 'symbol'})
    result = result.drop_duplicates(subset=['symbol'])
    
    print(f"  行业分布:")
    for ind, cnt in result['industry'].value_counts().sort_values(ascending=False).items():
        print(f"    {ind}: {cnt}只")
    
    # 保存
    os.makedirs(os.path.dirname(INDUSTRY_PATH), exist_ok=True)
    result.to_parquet(INDUSTRY_PATH, compression="zstd")
    print(f"\n已保存到 {INDUSTRY_PATH}")
    
    return result


# ══════════════════════════════════════════════
#  行业动量数据（预计算）
# ══════════════════════════════════════════════

INDUSTRY_MOMENTUM_PATH = os.path.join(DATA_DIR, "info", "industry_momentum.parquet")


def precompute_industry_momentum(industry_mapping: pd.DataFrame = None) -> pd.DataFrame:
    """
    预计算所有行业的动量指标（5日/20日平均涨幅）
    返回: DataFrame [industry, return_5d, return_20d, rank_5d, rank_20d, stock_count]
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.strategy_engine import load_stock_data, get_available_symbols
    
    if industry_mapping is None:
        industry_mapping = load_industry_mapping()
    if industry_mapping is None:
        print("无行业数据，无法计算行业动量")
        return None
    
    symbols = get_available_symbols()
    
    # 计算每只股票的近5日/20日涨幅
    stock_returns = {}
    for i, symbol in enumerate(symbols):
        if (i + 1) % 1000 == 0:
            print(f"  计算涨幅进度: {i+1}/{len(symbols)}")
        df = load_stock_data(symbol)
        if df is None or len(df) < 20:
            continue
        close = df["close"].values.astype(float)
        ret_5d = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
        ret_20d = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0
        stock_returns[symbol] = {"return_5d": ret_5d, "return_20d": ret_20d}
    
    # 按行业汇总
    industry_data = {}
    for _, row in industry_mapping.iterrows():
        sym = row["symbol"]
        industry = row["industry"]
        if sym not in stock_returns:
            continue
        if industry not in industry_data:
            industry_data[industry] = {"returns_5d": [], "returns_20d": [], "count": 0}
        industry_data[industry]["returns_5d"].append(stock_returns[sym]["return_5d"])
        industry_data[industry]["returns_20d"].append(stock_returns[sym]["return_20d"])
        industry_data[industry]["count"] += 1
    
    # 计算行业指标
    results = []
    for industry, data in industry_data.items():
        if data["count"] == 0:
            continue
        results.append({
            "industry": industry,
            "return_5d": np.median(data["returns_5d"]),
            "return_20d": np.median(data["returns_20d"]),
            "stock_count": data["count"],
        })
    
    df_result = pd.DataFrame(results)
    if len(df_result) == 0:
        return None
    
    # 排名
    df_result["rank_5d"] = df_result["return_5d"].rank(ascending=False, pct=True)
    df_result["rank_20d"] = df_result["return_20d"].rank(ascending=False, pct=True)
    df_result = df_result.sort_values("return_5d", ascending=False).reset_index(drop=True)
    
    # 保存
    os.makedirs(os.path.dirname(INDUSTRY_MOMENTUM_PATH), exist_ok=True)
    df_result.to_parquet(INDUSTRY_MOMENTUM_PATH, compression="zstd")
    print(f"行业动量已保存到 {INDUSTRY_MOMENTUM_PATH}")
    print(f"  共 {len(df_result)} 个行业")
    
    return df_result


def load_industry_momentum() -> pd.DataFrame:
    """加载预计算的行业动量数据"""
    if os.path.exists(INDUSTRY_MOMENTUM_PATH):
        return pd.read_parquet(INDUSTRY_MOMENTUM_PATH)
    return None


def get_industry_momentum_score(industry: str, momentum_data: pd.DataFrame = None) -> float:
    """获取行业的动量评分 (0-10)"""
    if momentum_data is None:
        momentum_data = load_industry_momentum()
    if momentum_data is None or industry not in momentum_data["industry"].values:
        return 5.0
    row = momentum_data[momentum_data["industry"] == industry].iloc[0]
    rank_pct = row["rank_5d"]
    # rank_pct: 0=最好, 1=最差 → 映射到 0-10
    return max(0, min(10, 10 * (1 - rank_pct)))


# ══════════════════════════════════════════════
#  财务数据
# ══════════════════════════════════════════════

def load_financial_indicators() -> pd.DataFrame:
    """加载财务指标"""
    if os.path.exists(FINANCIAL_PATH):
        return pd.read_parquet(FINANCIAL_PATH)
    return None


def get_financial(symbol: str, fin_data: pd.DataFrame = None) -> dict:
    """获取单只股票的财务数据"""
    if fin_data is None:
        fin_data = load_financial_indicators()
    if fin_data is None:
        return {}
    match = fin_data[fin_data["symbol"] == symbol]
    if len(match) > 0:
        return match.iloc[-1].to_dict()
    return {}


# ══════════════════════════════════════════════
#  基础数据检查
# ══════════════════════════════════════════════

def check_data_status() -> dict:
    """检查各种数据是否就绪"""
    status = {
        "industry_mapping": os.path.exists(INDUSTRY_PATH),
        "industry_momentum": os.path.exists(INDUSTRY_MOMENTUM_PATH),
        "financial_indicators": os.path.exists(FINANCIAL_PATH),
        "sector_rankings": os.path.exists(SECTOR_RANK_PATH),
    }
    if status["industry_mapping"]:
        df = pd.read_parquet(INDUSTRY_PATH)
        status["industry_count"] = df["industry"].nunique() if "industry" in df.columns else 0
        status["mapping_count"] = len(df)
    if status["financial_indicators"]:
        df = pd.read_parquet(FINANCIAL_PATH)
        status["financial_count"] = len(df)
    return status


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "reload-industry":
            reload_industry_mapping()
        elif cmd == "precompute-momentum":
            precompute_industry_momentum()
        elif cmd == "status":
            import json
            print(json.dumps(check_data_status(), indent=2, ensure_ascii=False))
        else:
            print("用法: python data_loader.py [reload-industry|precompute-momentum|status]")
    else:
        print(json.dumps(check_data_status(), indent=2, ensure_ascii=False))
