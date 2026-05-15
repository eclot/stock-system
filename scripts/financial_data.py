#!/usr/bin/env python3
"""
财务数据收集 + 基本面筛选器
数据源: BaoStock
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
#  1. BaoStock 财务数据拉取（完整字段）
# ═══════════════════════════════════════════════

def _convert_symbol_baostock(symbol: str) -> str:
    """转换 Hermes symbol 为 BaoStock 格式: sh600519 → sh.600519"""
    if symbol.startswith("sh") or symbol.startswith("sz"):
        exchange = symbol[:2]
        code = symbol[2:]
        return f"{exchange}.{code}"
    return symbol


def _get_latest_close(symbol: str) -> float:
    """从日K Parquet 获取最新收盘价"""
    try:
        import pyarrow.parquet as pq
        fpath = os.path.join(DAILY_DIR, f"{symbol}.parquet")
        if not os.path.exists(fpath):
            return None
        df = pq.read_table(fpath).to_pandas()
        if len(df) == 0:
            return None
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def fetch_financial_baostock(symbol: str, year: int = None, quarter: int = 1) -> dict:
    """从 BaoStock 拉取单只股票的完整财务数据

    返回字段:
      symbol, year, quarter, 
      roe (净资产收益率%), eps_ttm (每股收益TTM), 
      gross_margin (毛利率%), net_profit_margin (净利率%),
      net_profit (净利润), total_share (总股本),
      liability_to_asset (资产负债率%),
      current_ratio (流动比率),
      net_profit_growth (净利润同比%),
      equity_growth (净资产同比%),
      asset_growth (总资产同比%),
      close (最新收盘价, 用于计算PE/PB)
    """
    import baostock as bs

    if year is None:
        from datetime import datetime
        year = datetime.now().year

    bs_code = _convert_symbol_baostock(symbol)
    result = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "roe": None,
        "eps_ttm": None,
        "gross_margin": None,
        "net_profit_margin": None,
        "net_profit": None,
        "total_share": None,
        "liability_to_asset": None,
        "current_ratio": None,
        "net_profit_growth": None,
        "equity_growth": None,
        "asset_growth": None,
    }

    # ── 利润表: ROE, EPS, 毛利率, 净利率 ──
    try:
        rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # fields: code, pubDate, statDate, roeAvg, npMargin, gpMargin, 
                #         netProfit, epsTTM, MBRevenue, totalShare, liqaShare
                if len(row) >= 11:
                    try:
                        result["roe"] = float(row[3]) * 100 if row[3] else None         # roeAvg → %
                        result["net_profit_margin"] = float(row[4]) * 100 if row[4] else None
                        result["gross_margin"] = float(row[5]) * 100 if row[5] else None
                        result["net_profit"] = float(row[6]) if row[6] else None
                        result["eps_ttm"] = float(row[7]) if row[7] else None
                        result["total_share"] = float(row[9]) if row[9] else None
                    except:
                        pass
    except:
        pass

    # ── 成长能力: 净利润同比, 净资产同比, 总资产同比 ──
    try:
        rs = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # fields: code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
                if len(row) >= 8:
                    try:
                        result["equity_growth"] = float(row[3]) * 100 if row[3] else None  # YOYEquity → %
                        result["asset_growth"] = float(row[4]) * 100 if row[4] else None    # YOYAsset → %
                        result["net_profit_growth"] = float(row[5]) * 100 if row[5] else None  # YOYNI → %
                    except:
                        pass
    except:
        pass

    # ── 资产负债表: 负债率, 流动比率 ──
    try:
        rs = bs.query_balance_data(code=bs_code, year=year, quarter=quarter)
        if rs and rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                # fields: code, pubDate, statDate, currentRatio, quickRatio, cashRatio,
                #         YOYLiability, liabilityToAsset, assetToEquity
                if len(row) >= 9:
                    try:
                        result["liability_to_asset"] = float(row[7]) * 100 if row[7] else None  # → %
                        result["current_ratio"] = float(row[3]) if row[3] else None
                    except:
                        pass
    except:
        pass

    # ── 获取最新收盘价（用于计算 PE/PB） ──
    result["close"] = _get_latest_close(symbol)

    return result


def reload_financial_batch(symbols: list = None, max_workers: int = 10,
                           year: int = None, quarter: int = 1) -> pd.DataFrame:
    """批量拉取财务数据（全量重新下载）"""
    import baostock as bs

    if symbols is None:
        symbols = get_available_symbols()

    if not symbols:
        return pd.DataFrame()

    if year is None:
        from datetime import datetime
        year = datetime.now().year

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"BaoStock 登录失败: {lg.error_msg}")
        return pd.DataFrame()

    logger.info(f"开始批量拉取财务数据: {len(symbols)}只股票 (年={year} Q{quarter})")

    results = []
    total = len(symbols)
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            fetch_financial_baostock, s, year, quarter
        ): s for s in symbols}

        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as e:
                logger.warning(f"  拉取失败 {symbol}: {e}")
                results.append({"symbol": symbol, "year": year, "quarter": quarter})
                errors += 1

            if (i + 1) % 500 == 0:
                logger.info(f"  进度: {i+1}/{total} (失败{errors})")

    bs.logout()

    df = pd.DataFrame(results)

    # ── 计算衍生指标 ──
    # PE = 收盘价 / EPS_TTM
    df["pe"] = np.where(
        (df["eps_ttm"].notna()) & (df["eps_ttm"] > 0) & (df["close"].notna()) & (df["close"] > 0),
        (df["close"] / df["eps_ttm"]).round(2),
        None
    )

    # PB ≈ 收盘价 / (净资产/总股本)  — 用 equity = assets - liabilities
    # 从 liability_to_asset 可以推 equity_ratio = 1 - liability_to_asset
    # 但不知道总资产... 所以 PB 暂时无法准确计算。
    # 保留 close 和 eps_ttm 供后续使用

    # ── 新增字段: revenue_growth 别名 (兼容旧版评分引擎) ──
    # 旧版 _score_revenue_growth 取 financial.get("revenue_growth")
    # 但我们没有营收同比数据，用 net_profit_growth 代替作为近似
    if "net_profit_growth" in df.columns and "revenue_growth" not in df.columns:
        df["revenue_growth"] = df["net_profit_growth"]

    # profit_growth 别名
    if "net_profit_growth" in df.columns and "profit_growth" not in df.columns:
        df["profit_growth"] = df["net_profit_growth"]

    # ── 保存 ──
    os.makedirs(FINANCIAL_DIR, exist_ok=True)
    df.to_parquet(FINANCIAL_PATH, index=False)
    logger.info(f"财务数据已保存: {FINANCIAL_PATH} ({len(df)}行, {len(df.columns)}列)")
    logger.info(f"  成功: {len(df) - errors}, 失败: {errors}")

    return df


def update_financial_incremental(symbols: list = None, max_workers: int = 10) -> pd.DataFrame:
    """增量更新财务数据 — 仅下载最新季度，与现有数据合并"""
    import baostock as bs
    from datetime import datetime

    now = datetime.now()
    # 确定最新完整季度
    month = now.month
    if month >= 4:
        year, quarter = now.year, 1  # Q1 (1-3月)
    if month >= 7:
        year, quarter = now.year, 2  # Q2
    if month >= 10:
        year, quarter = now.year, 3  # Q3
    else:
        year, quarter = now.year - 1, 4  # Q4 of previous year

    # 检查现有数据
    existing = None
    if os.path.exists(FINANCIAL_PATH):
        existing = pd.read_parquet(FINANCIAL_PATH)

    # 如果已有同期数据，跳过
    if existing is not None and len(existing) > 0:
        if "year" in existing.columns and "quarter" in existing.columns:
            same_period = existing[(existing["year"] == year) & (existing["quarter"] == quarter)]
            if len(same_period) > 500:  # 已有大部分数据
                logger.info(f"已有 {year}Q{quarter} 数据 ({len(same_period)}只)，跳过下载")
                return existing

    logger.info(f"增量更新: 拉取 {year}Q{quarter} 数据")
    df = reload_financial_batch(symbols, max_workers, year, quarter)

    # 如果之前有旧数据，合并
    if existing is not None and len(df) > 0:
        if "year" in df.columns and "quarter" in df.columns:
            # 合并新旧数据，新数据覆盖旧数据
            combined = existing[~((existing["year"] == year) & (existing["quarter"] == quarter))]
            combined = pd.concat([combined, df], ignore_index=True)
            combined.to_parquet(FINANCIAL_PATH, index=False)
            logger.info(f"合并后总记录: {len(combined)}")
            return combined

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
                qualified.append(symbol)
                skipped_financial += 1
            continue

        row = match.iloc[0]
        passed = True

        # ROE
        roe = row.get("roe")
        if roe is not None and not pd.isna(roe):
            if roe < FILTER_RULES["roe_min"]:
                passed = False

        # 负债率
        debt = row.get("liability_to_asset")
        if debt is not None and not pd.isna(debt):
            if debt > FILTER_RULES["debt_ratio_max"] * 100:  # 转百分比
                passed = False

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


def get_financial_statistics() -> dict:
    """获取财务数据统计概览（前端展示用）"""
    df = load_financial_data()
    if df is None or len(df) == 0:
        return {"total": 0, "fields": []}

    stats = {
        "total": len(df),
        "stocks_with_pe": int(df["pe"].notna().sum()) if "pe" in df.columns else 0,
        "stocks_with_roe": int(df["roe"].notna().sum()) if "roe" in df.columns else 0,
        "stocks_with_eps": int(df["eps_ttm"].notna().sum()) if "eps_ttm" in df.columns else 0,
        "fields": list(df.columns),
        "latest_year": int(df["year"].max()) if "year" in df.columns else None,
        "latest_quarter": int(df[df["year"] == df["year"].max()]["quarter"].max()) 
                          if "year" in df.columns and "quarter" in df.columns else None,
    }

    # 按行业平均ROE
    try:
        from scripts.data_loader import load_industry_mapping, get_industry
        mapping = load_industry_mapping()
        if mapping is not None:
            industry_roes = {}
            for _, row in df.dropna(subset=["roe"]).iterrows():
                ind = get_industry(row["symbol"], mapping)
                if ind:
                    if ind not in industry_roes:
                        industry_roes[ind] = []
                    industry_roes[ind].append(row["roe"])
            stats["industry_avg_roe"] = {
                ind: round(sum(v)/len(v), 1) 
                for ind, v in industry_roes.items() 
                if len(v) >= 5
            }
    except:
        pass

    # PE分布
    if "pe" in df.columns:
        pe_vals = df["pe"].dropna()
        if len(pe_vals) > 0:
            stats["pe_distribution"] = {
                "0-15": int(((pe_vals > 0) & (pe_vals <= 15)).sum()),
                "15-30": int(((pe_vals > 15) & (pe_vals <= 30)).sum()),
                "30-50": int(((pe_vals > 30) & (pe_vals <= 50)).sum()),
                "50-100": int(((pe_vals > 50) & (pe_vals <= 100)).sum()),
                ">100": int((pe_vals > 100).sum()),
                "negative": int((pe_vals <= 0).sum()),
            }

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if "--reload" in sys.argv:
        print("=== 全量拉取财务数据 ===")
        symbols = get_available_symbols()
        print(f"共 {len(symbols)} 只股票")
        df = reload_financial_batch(symbols, max_workers=10)
        if len(df) > 0:
            print(f"\n字段: {list(df.columns)}")
            print(df[["symbol", "pe", "roe", "gross_margin", "liability_to_asset",
                      "net_profit_growth", "eps_ttm"]].head(10).to_string())
    elif "--incremental" in sys.argv:
        print("=== 增量更新财务数据 ===")
        df = update_financial_incremental(max_workers=10)
        if df is not None and len(df) > 0:
            print(f"完成: {len(df)}条记录")
    elif "--stats" in sys.argv:
        print("=== 财务数据统计 ===")
        stats = get_financial_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print("=== 基本面筛选器 ===")
        fin = load_financial_data()
        if fin is None:
            print("无财务数据")
        else:
            print(f"财务数据: {len(fin)}条, 字段: {list(fin.columns)}")
            qualified = filter_by_fundamentals(relaxed=True)
            print(f"通过筛选: {len(qualified)}只")
