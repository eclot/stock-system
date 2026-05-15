#!/usr/bin/env python3
"""
增量更新日K线数据
- 只下载最新交易日至本地已有数据的最后日期之间的数据
- 支持指定股票列表（默认全量）
- 支持只更新持仓股票（--holdings-only）
用法:
    python scripts/update_kline_incremental.py
    python scripts/update_kline_incremental.py --holdings-only
    python scripts/update_kline_incremental.py --symbols sh600519,sz000001
"""
import os
import sys
import time
import json
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

DATA_DIR = os.path.expanduser("~/stock-system/data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

REQUEST_DELAY = 0.1


def bs_login():
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"Baostock登录失败: {lg.error_msg}")
    return lg


def get_new_kline(symbol: str, code: str, start_date: str) -> pd.DataFrame:
    """获取指定日期之后的K线数据"""
    import baostock as bs
    end_date = datetime.now().strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,adjustflag",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    if rs.error_code != "0":
        raise Exception(f"查询失败: {rs.error_msg}")
    
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["date"] != ""].copy()
    if len(df) == 0:
        return None
    
    # 转换类型
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    keep_cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    df = df[[c for c in keep_cols if c in df.columns]]
    
    return df


def get_existing_dates(symbol: str):
    """获取本地已有数据的日期范围"""
    filepath = os.path.join(DAILY_DIR, f"{symbol}.parquet")
    if not os.path.exists(filepath):
        return None, None
    df = pq.read_table(filepath).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df, df["date"].max()


def merge_and_save(df_existing: pd.DataFrame, df_new: pd.DataFrame, symbol: str):
    """合并新旧数据并保存"""
    if df_new is None or len(df_new) == 0:
        return False
    
    # 合并去重
    combined = pd.concat([df_existing, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    # 保存
    filepath = os.path.join(DAILY_DIR, f"{symbol}.parquet")
    cols = [c for c in combined.columns if c not in ["symbol", "adjustflag", "code"]]
    cols = ["symbol", "date"] + [c for c in cols if c != "date"]
    cols = [c for c in cols if c in combined.columns]
    
    table = pa.Table.from_pandas(combined[cols], preserve_index=False)
    pq.write_table(table, filepath, compression="zstd")
    
    return True


def update_symbols(symbols_to_update: list) -> dict:
    """批量更新指定股票"""
    stats = {"updated": 0, "no_change": 0, "failed": 0, "new": 0, "details": []}
    
    # 临时登录
    try:
        bs_login()
    except Exception as e:
        logger.error(f"Baostock登录失败: {e}")
        return stats
    
    try:
        total = len(symbols_to_update)
        for i, symbol in enumerate(symbols_to_update):
            # 构建Baostock code格式
            if symbol.startswith("sh") or symbol.startswith("sz"):
                code = symbol[:2] + "." + symbol[2:]
            else:
                code = symbol  # 保持原样
            
            try:
                df_existing, last_date = get_existing_dates(symbol)
                
                if df_existing is None:
                    # 全新下载
                    start_date = "2024-01-01"
                else:
                    # 增量：从上一次最后日期的第二天开始
                    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    if start_date >= datetime.now().strftime("%Y-%m-%d"):
                        stats["no_change"] += 1
                        if (i + 1) % 200 == 0:
                            logger.info(f"  [{i+1}/{total}] {symbol} - 已是最新")
                        continue
                
                df_new = get_new_kline(symbol, code, start_date)
                
                if df_new is None or len(df_new) == 0:
                    stats["no_change"] += 1
                    if (i + 1) % 200 == 0:
                        logger.info(f"  [{i+1}/{total}] {symbol} - 无新数据")
                    continue
                
                if df_existing is None:
                    merge_and_save(None, df_new, symbol)  # just save new
                    rows = len(df_new)
                    stats["new"] += 1
                    logger.info(f"  [{i+1}/{total}] {symbol} - 新下载 {rows}条 ({df_new['date'].iloc[0].date()}~{df_new['date'].iloc[-1].date()})")
                else:
                    merge_and_save(df_existing, df_new, symbol)
                    rows = len(df_new)
                    stats["updated"] += 1
                    logger.info(f"  [{i+1}/{total}] {symbol} - 更新 {rows}条 ({df_new['date'].iloc[0].date()}~{df_new['date'].iloc[-1].date()})")
                
                stats["details"].append({"symbol": symbol, "rows": rows})
                
            except Exception as e:
                stats["failed"] += 1
                logger.warning(f"  [{i+1}/{total}] {symbol} 失败: {e}")
            
            time.sleep(REQUEST_DELAY)
            
            # 每500个重新登录一次（防超时）
            if (i + 1) % 500 == 0:
                try:
                    import baostock as bs
                    bs.logout()
                except:
                    pass
                try:
                    bs_login()
                except:
                    pass
    finally:
        try:
            import baostock as bs
            bs.logout()
        except:
            pass
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="增量更新日K线数据")
    parser.add_argument("--holdings-only", action="store_true", help="只更新持仓股票")
    parser.add_argument("--symbols", type=str, help="逗号分隔的股票代码列表")
    parser.add_argument("--limit", type=int, default=0, help="限制更新数量（测试用）")
    args = parser.parse_args()
    
    # 获取需要更新的股票列表
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        logger.info(f"指定股票: {len(symbols)} 只")
    elif args.holdings_only:
        with open(PORTFOLIO_FILE) as f:
            pf = json.load(f)
        symbols = list(pf.get("holdings", {}).keys())
        logger.info(f"持仓股票: {len(symbols)} 只")
    else:
        # 全量更新：从data/daily目录读取所有股票
        from scripts.strategy_engine import get_available_symbols
        symbols = get_available_symbols()
        logger.info(f"全量更新: {len(symbols)} 只")
    
    if args.limit > 0:
        symbols = symbols[:args.limit]
        logger.info(f"限制测试: 前{args.limit}只")
    
    logger.info(f"开始增量更新... ({len(symbols)} 只)")
    start = time.time()
    stats = update_symbols(symbols)
    elapsed = time.time() - start
    
    logger.info(f"\n=== 增量更新完成 ===")
    logger.info(f"更新: {stats['updated']} | 无变化: {stats['no_change']} | 失败: {stats['failed']} | 新增: {stats['new']}")
    logger.info(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
