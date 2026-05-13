#!/usr/bin/env python3
"""
可视化模块 — K线图生成 + 飞书推送内容格式化
依赖: matplotlib, mplfinance
"""

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
from datetime import datetime

CHART_DIR = os.path.expanduser("~/stock-system/data/charts")
os.makedirs(CHART_DIR, exist_ok=True)


def generate_kline_chart(df: pd.DataFrame, symbol: str, name: str = "",
                         signals: list = None) -> str:
    """
    生成K线图，返回图片路径
    """
    if df is None or len(df) < 30:
        return None
    
    df = df.copy().sort_values("date")
    df = df.set_index("date")
    
    # 只显示最近60个交易日
    df = df.tail(60)
    
    # 计算技术指标用于显示
    close = df["close"].values.astype(float)
    
    # 准备附加指标
    apds = []
    
    # MA均线
    ma5 = pd.Series(close).rolling(5).mean()
    ma20 = pd.Series(close).rolling(20).mean()
    ma60 = pd.Series(close).rolling(60).mean()
    
    apds.append(mpf.make_addplot(ma5, color="#2C7BE5", width=0.8, label="MA5"))
    apds.append(mpf.make_addplot(ma20, color="#E67E22", width=0.8, label="MA20"))
    if not ma60.isna().all():
        apds.append(mpf.make_addplot(ma60, color="#9B59B6", width=0.8, label="MA60"))
    
    # 成交量
    # 使用 volume panel
    
    # 设置样式 - 使用支持中文的字体
    chinese_font = 'WenQuanYi Zen Hei'
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans", "Noto Sans CJK JP", "AR PL UMing CN"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F0F4F8",
            "axes.facecolor": "#FFFFFF",
        },
        marketcolors=mpf.make_marketcolors(
            up="#E74C3C",
            down="#2ECC71",
            edge="inherit",
            wick="inherit",
            volume="in",
        ),
    )
    
    # 创建图表
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        addplot=apds,
        title=f"{name} ({symbol})",
        ylabel="价格",
        ylabel_lower="成交量",
        figsize=(12, 7),
        returnfig=True,
        tight_layout=True,
    )
    
    # 标注信号
    if signals:
        latest_idx = len(df) - 1
        latest_date = df.index[-1]
        latest_price = df["close"].iloc[-1]
        
        if "strong_buy" in signals:
            axes[0].annotate("🟢 BUY", xy=(latest_date, latest_price),
                            xytext=(latest_date, latest_price * 1.05),
                            fontsize=12, color="#2ECC71", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#2ECC71"))
        
        if "trend_broken" in signals:
            axes[0].annotate("🔴 SELL", xy=(latest_date, latest_price),
                            xytext=(latest_date, latest_price * 0.95),
                            fontsize=12, color="#E74C3C", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#E74C3C"))
        
        if "overbought" in signals:
            axes[0].annotate("⚠️ 过热", xy=(latest_date, latest_price),
                            xytext=(latest_date, latest_price * 1.08),
                            fontsize=10, color="#E67E22")
    
    # 保存
    filepath = os.path.join(CHART_DIR, f"{symbol}_{datetime.now().strftime('%Y%m%d')}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return filepath


def generate_score_chart(results: pd.DataFrame, top_n: int = 20) -> str:
    """
    生成评分排行柱状图，返回图片路径
    """
    if results is None or len(results) == 0:
        return None
    
    df = results.head(top_n).copy()
    
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#FFFFFF")
    
    colors = ["#2ECC71" if s >= 70 else "#F39C12" if s >= 60 else "#E74C3C" 
              for s in df["score"]]
    
    bars = ax.barh(range(len(df)), df["score"], color=colors, height=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{n}\n{s}" for n, s in zip(df["name"], df["symbol"])], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("评分", fontsize=10)
    ax.set_title(f"股票评分排行 ({datetime.now().strftime('%Y-%m-%d')})", 
                 fontsize=13, fontweight="bold")
    ax.axvline(x=70, color="#2ECC71", linestyle="--", alpha=0.5, label="买入线")
    ax.axvline(x=60, color="#F39C12", linestyle="--", alpha=0.5, label="关注线")
    ax.legend(fontsize=9)
    
    # 添加分数标签
    for i, v in enumerate(df["score"]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    filepath = os.path.join(CHART_DIR, f"scores_{datetime.now().strftime('%Y%m%d')}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return filepath


def format_feishu_message(results: pd.DataFrame, title: str = "📊 股票分析日报") -> str:
    """
    格式化飞书消息
    """
    if results is None or len(results) == 0:
        return "暂无数据"
    
    lines = [f"**{title}**"]
    lines.append(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # 统计
    strong_buy = len(results[results["score"] >= 70])
    watch = len(results[(results["score"] >= 60) & (results["score"] < 70)])
    
    lines.append(f"🟢 强烈买入: {strong_buy} | 🟡 关注: {watch} | 📉 观望: {len(results) - strong_buy - watch}")
    lines.append("")
    
    # Top 10
    lines.append("**评分TOP 10:**")
    lines.append("| 排名 | 代码 | 名称 | 评分 | 现价 | 信号 |")
    lines.append("|------|------|------|------|------|------|")
    
    for i, (_, row) in enumerate(results.head(10).iterrows()):
        signals = row.get("signals", "")
        sig_icon = ""
        if "strong_buy" in signals:
            sig_icon = "🟢"
        elif "trend_broken" in signals:
            sig_icon = "🔴"
        
        price = row.get("price", 0)
        lines.append(f"| {i+1} | {row['symbol']} | {row['name']} | **{row['score']:.1f}** | {price:.2f} | {sig_icon} |")
    
    lines.append("")
    lines.append("`#股票分析`")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    from strategy_engine import load_stock_data, ScoringEngine
    
    symbol = "sh600519"
    df = load_stock_data(symbol)
    if df is not None:
        result = ScoringEngine.score_stock(df)
        path = generate_kline_chart(df, symbol, "贵州茅台", result["signals"])
        if path:
            print(f"K线图已保存: {path}")
    
    print("可视化模块测试完成")
