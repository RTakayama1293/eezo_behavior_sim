#!/usr/bin/env python3
"""
EEZO 10百万円達成 収益戦略シミュレーション

revenue_strategy_2026.mdの構成に沿って、価格ジャンプ戦略を組み込んだ
売上目標達成と5年間成長計画のシミュレーション
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# matplotlib設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "docs" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"


def ensure_dirs():
    """出力ディレクトリを作成"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. パラメータ定義（エビデンスベース）
# =============================================================================

@dataclass
class PriceJumpEvidence:
    """価格ジャンプ戦略のエビデンス（exp001_report_20260129.mdより）"""
    threshold: int = 20_000
    jump_rate: float = 0.385  # 38.5%
    aov_change: float = 0.199  # +19.9%
    gross_profit_change: float = 0.037  # +3.7%
    baseline_aov: int = 12_209
    new_aov: int = 14_633


@dataclass
class CostStructureY1:
    """Y1原価構造（北海道発送）"""
    product_cost_rate: float = 0.693  # 商品原価69.3%
    shipping_cost_low: int = 750  # 低価格帯送料
    shipping_cost_high: int = 1_400  # 高価格帯送料
    shipping_cost_avg: int = 1_075  # 平均送料
    insert_card_cost: int = 100  # 同梱カード
    margin_rate: float = 0.15  # 粗利率15%


@dataclass
class CostStructureY2Plus:
    """Y2以降原価構造（大阪在庫）"""
    product_cost_rate: float = 0.693
    shipping_cost_low: int = 500
    shipping_cost_high: int = 900
    shipping_cost_avg: int = 700  # 35%削減
    insert_card_cost: int = 100
    margin_rate: float = 0.20  # 粗利率20%


@dataclass
class TrafficSource:
    """トラフィックソース"""
    name: str
    annual_volume: int
    cvr: float
    active_quarters: List[int]


@dataclass
class BtoBSegment:
    """BtoBセグメント"""
    name: str
    avg_order: int
    orders_per_year: int
    margin_rate: float
    start_year: int


# =============================================================================
# 2. シミュレーションモデル
# =============================================================================

class RevenueStrategySimulator:
    """収益戦略シミュレータ"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.price_jump = PriceJumpEvidence()
        self.cost_y1 = CostStructureY1()
        self.cost_y2 = CostStructureY2Plus()

        # トラフィックソース定義
        self.traffic_sources = {
            "o2o": TrafficSource("O2O（船内QR）", 32_467, 0.0077, [3]),  # Q3のみ
            "newsletter": TrafficSource("メルマガ", 400_000, 0.00043, [2, 3, 4]),
            "sns": TrafficSource("SNS広告", 200_000, 0.0034, [2, 3, 4]),
        }

        # BtoBセグメント定義
        self.btob_segments = {
            "restaurants": BtoBSegment("飲食店・居酒屋", 30_000, 18, 0.12, 2),
            "hotels": BtoBSegment("ホテル・旅館", 80_000, 10, 0.10, 3),
            "retail": BtoBSegment("百貨店・量販店", 300_000, 6, 0.08, 4),
            "corporate": BtoBSegment("法人ギフト", 150_000, 2, 0.15, 3),
        }

        # 5年間目標
        self.yearly_targets = {
            1: {"btoc": 10_000_000, "btob": 0, "mp": 0},
            2: {"btoc": 15_000_000, "btob": 5_000_000, "mp": 0},
            3: {"btoc": 20_000_000, "btob": 15_000_000, "mp": 0},
            4: {"btoc": 35_000_000, "btob": 40_000_000, "mp": 25_000_000},
            5: {"btoc": 50_000_000, "btob": 60_000_000, "mp": 40_000_000},
        }

    # -------------------------------------------------------------------------
    # 2.1 Y1目標達成シミュレーション
    # -------------------------------------------------------------------------

    def simulate_y1_quarterly(self, with_price_jump: bool = True) -> Dict:
        """Y1四半期別シミュレーション"""

        # 基本パラメータ
        base_avg_price = 7_500  # 二峰性価格の平均
        repeat_rate = 0.20
        margin_rate = self.cost_y1.margin_rate

        # 価格ジャンプ効果
        if with_price_jump:
            # AOV向上効果を適用（ただし全員がジャンプするわけではない）
            # 実効AOV = ベース価格 × (1 + ジャンプ率 × AOV増加率)
            effective_aov_multiplier = 1 + (self.price_jump.jump_rate * self.price_jump.aov_change)
            avg_price = int(base_avg_price * effective_aov_multiplier)
            # 粗利率も改善
            margin_rate = margin_rate * (1 + self.price_jump.gross_profit_change)
        else:
            avg_price = base_avg_price

        # 四半期別計画
        quarters = {
            1: {
                "name": "Q1: 移行準備",
                "months": ["1月", "2月", "3月"],
                "new_customers": 0,
                "repeat_customers": 0,
                "activities": ["Shopify構築", "商品選定", "写真撮影"],
            },
            2: {
                "name": "Q2: セット投入期",
                "months": ["4月", "5月", "6月"],
                "new_customers": 153,
                "repeat_customers": 3,
                "activities": ["新サイト稼働", "母の日・父の日ギフト", "レビュー蓄積"],
            },
            3: {
                "name": "Q3: O2O期間",
                "months": ["7月", "8月", "9月"],
                "new_customers": 355,
                "repeat_customers": 11,
                "activities": ["O2O本番", "メルマガ登録獲得", "夏ギフト"],
            },
            4: {
                "name": "Q4: 年末商戦",
                "months": ["10月", "11月", "12月"],
                "new_customers": 633,
                "repeat_customers": 179,
                "activities": ["SNS広告集中", "歳暮ギフト", "リピート促進"],
            },
        }

        # 計算
        results = []
        cumulative_revenue = 0
        cumulative_customers = 0

        for q, data in quarters.items():
            total_orders = data["new_customers"] + data["repeat_customers"]
            revenue = total_orders * avg_price
            gross_profit = int(revenue * margin_rate)
            cumulative_revenue += revenue
            cumulative_customers += data["new_customers"]

            results.append({
                "quarter": q,
                "name": data["name"],
                "new_customers": data["new_customers"],
                "repeat_customers": data["repeat_customers"],
                "total_orders": total_orders,
                "avg_price": avg_price,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "margin_rate": margin_rate,
                "cumulative_revenue": cumulative_revenue,
                "cumulative_customers": cumulative_customers,
                "activities": data["activities"],
            })

        return {
            "quarters": results,
            "annual_summary": {
                "total_revenue": cumulative_revenue,
                "total_new_customers": cumulative_customers,
                "total_orders": sum(r["total_orders"] for r in results),
                "avg_price": avg_price,
                "margin_rate": margin_rate,
                "gross_profit": sum(r["gross_profit"] for r in results),
                "with_price_jump": with_price_jump,
            }
        }

    def simulate_y1_monthly(self, with_price_jump: bool = True) -> List[Dict]:
        """Y1月次シミュレーション"""

        base_avg_price = 7_500
        repeat_rate = 0.20
        margin_rate = self.cost_y1.margin_rate

        if with_price_jump:
            effective_aov_multiplier = 1 + (self.price_jump.jump_rate * self.price_jump.aov_change)
            avg_price = int(base_avg_price * effective_aov_multiplier)
            margin_rate = margin_rate * (1 + self.price_jump.gross_profit_change)
        else:
            avg_price = base_avg_price

        # 月次計画（revenue_strategy_2026.mdに準拠）
        monthly_plan = [
            {"month": 1, "name": "1月", "new": 0, "repeat": 0, "note": "移行準備"},
            {"month": 2, "name": "2月", "new": 0, "repeat": 0, "note": "目玉商品選定・撮影"},
            {"month": 3, "name": "3月", "new": 0, "repeat": 0, "note": "Shopify移行完了"},
            {"month": 4, "name": "4月", "new": 30, "repeat": 0, "note": "新サイト稼働+セット商品"},
            {"month": 5, "name": "5月", "new": 48, "repeat": 0, "note": "母の日ギフトセット"},
            {"month": 6, "name": "6月", "new": 55, "repeat": 0, "note": "父の日+レビュー蓄積"},
            {"month": 7, "name": "7月", "new": 160, "repeat": 15, "note": "O2O開始"},
            {"month": 8, "name": "8月", "new": 180, "repeat": 22, "note": "O2Oピーク"},
            {"month": 9, "name": "9月", "new": 110, "repeat": 20, "note": "O2O終了・敬老の日"},
            {"month": 10, "name": "10月", "new": 120, "repeat": 45, "note": "SNS広告開始"},
            {"month": 11, "name": "11月", "new": 160, "repeat": 64, "note": "歳暮前哨戦"},
            {"month": 12, "name": "12月", "new": 204, "repeat": 100, "note": "歳暮ピーク"},
        ]

        results = []
        cumulative_revenue = 0

        for m in monthly_plan:
            total_orders = m["new"] + m["repeat"]
            revenue = total_orders * avg_price
            gross_profit = int(revenue * margin_rate)
            cumulative_revenue += revenue

            results.append({
                **m,
                "total_orders": total_orders,
                "avg_price": avg_price,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "cumulative_revenue": cumulative_revenue,
            })

        return results

    # -------------------------------------------------------------------------
    # 2.2 原価構造シミュレーション
    # -------------------------------------------------------------------------

    def simulate_cost_structure(self, year: int = 1, with_price_jump: bool = True) -> Dict:
        """原価構造シミュレーション"""

        cost = self.cost_y1 if year == 1 else self.cost_y2

        # 価格帯別原価（二峰性価格）
        price_bands = [
            {"name": "低価格帯", "price": 5_000, "share": 0.50},
            {"name": "高価格帯", "price": 10_000, "share": 0.50},
        ]

        results = []
        for band in price_bands:
            price = band["price"]
            product_cost = int(price * cost.product_cost_rate)
            shipping = cost.shipping_cost_low if price <= 5000 else cost.shipping_cost_high
            card = cost.insert_card_cost
            total_cost = product_cost + shipping + card
            gross_profit = price - total_cost
            margin = gross_profit / price

            results.append({
                "band": band["name"],
                "price": price,
                "share": band["share"],
                "product_cost": product_cost,
                "shipping": shipping,
                "card": card,
                "total_cost": total_cost,
                "gross_profit": gross_profit,
                "margin_rate": margin,
            })

        # 加重平均
        avg_price = sum(b["price"] * b["share"] for b in price_bands)
        avg_cost = sum(r["total_cost"] * r["share"] for r in results)
        avg_profit = avg_price - avg_cost
        avg_margin = avg_profit / avg_price

        # 価格ジャンプ効果
        if with_price_jump and year >= 1:
            # 20K閾値でジャンプする顧客の効果
            jump_effect = {
                "jump_rate": self.price_jump.jump_rate,
                "aov_change": self.price_jump.aov_change,
                "margin_improvement": self.price_jump.gross_profit_change,
                "new_avg_price": int(avg_price * (1 + self.price_jump.jump_rate * self.price_jump.aov_change)),
                "new_margin_rate": avg_margin * (1 + self.price_jump.gross_profit_change),
            }
        else:
            jump_effect = None

        return {
            "year": year,
            "cost_structure": cost,
            "price_bands": results,
            "weighted_average": {
                "price": avg_price,
                "cost": avg_cost,
                "profit": avg_profit,
                "margin": avg_margin,
            },
            "price_jump_effect": jump_effect,
        }

    # -------------------------------------------------------------------------
    # 2.3 5年間成長シミュレーション
    # -------------------------------------------------------------------------

    def simulate_five_years(self, with_price_jump: bool = True) -> List[Dict]:
        """5年間成長シミュレーション"""

        # 年度別パラメータ
        yearly_params = {
            1: {"margin": 0.15, "repeat": 0.20, "avg_price": 7_500, "investment": 520_000},
            2: {"margin": 0.18, "repeat": 0.30, "avg_price": 7_000, "investment": 3_000_000},
            3: {"margin": 0.20, "repeat": 0.35, "avg_price": 6_500, "investment": 5_000_000},
            4: {"margin": 0.18, "repeat": 0.28, "avg_price": 6_000, "investment": 15_000_000},
            5: {"margin": 0.20, "repeat": 0.36, "avg_price": 6_000, "investment": 5_000_000},
        }

        results = []
        cumulative = {"revenue": 0, "profit": 0, "investment": 0}

        for year in range(1, 6):
            target = self.yearly_targets[year]
            params = yearly_params[year]

            # 売上計算
            btoc = target["btoc"]
            btob = target["btob"]
            mp = target["mp"]
            total_revenue = btoc + btob + mp

            # 価格ジャンプ効果（BtoCのみ、Y2以降本格導入）
            price_jump_effect = 0
            if with_price_jump and year >= 2:
                price_jump_effect = int(btoc * self.price_jump.gross_profit_change)

            # 粗利計算
            margin_rate = params["margin"]
            if with_price_jump and year >= 2:
                margin_rate = margin_rate * (1 + self.price_jump.gross_profit_change * 0.5)

            gross_profit = int(total_revenue * margin_rate) + price_jump_effect
            investment = params["investment"]
            operating_profit = gross_profit - investment

            # LTV計算
            repeat_rate = params["repeat"]
            avg_purchases = 1 / (1 - repeat_rate)
            ltv = int(params["avg_price"] * avg_purchases * margin_rate)

            # 累積
            cumulative["revenue"] += total_revenue
            cumulative["profit"] += gross_profit
            cumulative["investment"] += investment

            # 成長率
            if year > 1:
                prev_revenue = results[-1]["total_revenue"]
                growth_rate = (total_revenue - prev_revenue) / prev_revenue
            else:
                growth_rate = 0

            results.append({
                "year": year,
                "btoc": btoc,
                "btob": btob,
                "mp": mp,
                "total_revenue": total_revenue,
                "margin_rate": margin_rate,
                "gross_profit": gross_profit,
                "price_jump_effect": price_jump_effect,
                "investment": investment,
                "operating_profit": operating_profit,
                "repeat_rate": repeat_rate,
                "avg_price": params["avg_price"],
                "ltv": ltv,
                "growth_rate": growth_rate,
                "cumulative_revenue": cumulative["revenue"],
                "cumulative_profit": cumulative["profit"],
                "cumulative_investment": cumulative["investment"],
                "roi": (cumulative["profit"] - cumulative["investment"]) / cumulative["investment"] if cumulative["investment"] > 0 else 0,
            })

        return results

    def simulate_btob_details(self) -> Dict[int, Dict]:
        """BtoB詳細シミュレーション"""

        # セグメント別構成比
        segment_shares = {
            2: {"restaurants": 1.0},
            3: {"restaurants": 0.50, "hotels": 0.25, "corporate": 0.25},
            4: {"restaurants": 0.35, "hotels": 0.25, "retail": 0.20, "corporate": 0.20},
            5: {"restaurants": 0.30, "hotels": 0.25, "retail": 0.25, "corporate": 0.20},
        }

        results = {}
        for year in range(2, 6):
            target_btob = self.yearly_targets[year]["btob"]
            if target_btob == 0:
                continue

            shares = segment_shares.get(year, {})
            segments = {}
            total_accounts = 0

            for seg_id, share in shares.items():
                if share > 0:
                    seg = self.btob_segments[seg_id]
                    segment_revenue = int(target_btob * share)
                    revenue_per_account = seg.avg_order * seg.orders_per_year
                    accounts = max(1, int(segment_revenue / revenue_per_account))

                    segments[seg_id] = {
                        "name": seg.name,
                        "accounts": accounts,
                        "avg_order": seg.avg_order,
                        "orders_per_year": seg.orders_per_year,
                        "revenue": accounts * revenue_per_account,
                        "margin_rate": seg.margin_rate,
                    }
                    total_accounts += accounts

            results[year] = {
                "segments": segments,
                "total_accounts": total_accounts,
                "total_revenue": sum(s["revenue"] for s in segments.values()),
            }

        return results


# =============================================================================
# 3. 可視化
# =============================================================================

def plot_quarterly_revenue(results: Dict, output_path: str):
    """四半期別売上グラフ"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    quarters = results["quarters"]
    x = [f"Q{q['quarter']}" for q in quarters]
    revenues = [q["revenue"] / 10000 for q in quarters]
    new_customers = [q["new_customers"] for q in quarters]
    repeat_customers = [q["repeat_customers"] for q in quarters]

    # 売上棒グラフ
    ax = axes[0]
    bars = ax.bar(x, revenues, color=['#95a5a6', '#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_ylabel('Revenue (10K Yen)')
    ax.set_title('Quarterly Revenue (Y1)')

    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{rev:.0f}', ha='center', fontsize=11, fontweight='bold')

    # 顧客構成
    ax = axes[1]
    width = 0.35
    x_pos = np.arange(len(x))
    ax.bar(x_pos - width/2, new_customers, width, label='New', color='#3498db')
    ax.bar(x_pos + width/2, repeat_customers, width, label='Repeat', color='#e74c3c')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_ylabel('Customers')
    ax.set_title('Customer Composition by Quarter')
    ax.legend()

    plt.suptitle(f'Y1 Target: 10M Yen (Actual: {results["annual_summary"]["total_revenue"]/10000:.0f}K Yen)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_monthly_revenue(monthly: List[Dict], output_path: str):
    """月次売上グラフ"""
    fig, ax = plt.subplots(figsize=(14, 6))

    months = [m["name"] for m in monthly]
    revenues = [m["revenue"] / 10000 for m in monthly]
    cumulative = [m["cumulative_revenue"] / 10000 for m in monthly]

    x = np.arange(len(months))
    width = 0.6

    # 月次売上
    colors = ['#95a5a6'] * 3 + ['#3498db'] * 3 + ['#e74c3c'] * 3 + ['#2ecc71'] * 3
    bars = ax.bar(x, revenues, width, color=colors, edgecolor='black', label='Monthly')

    # 累計線
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, 'k-o', linewidth=2, markersize=6, label='Cumulative')
    ax2.set_ylabel('Cumulative (10K Yen)')

    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Monthly Revenue (10K Yen)')
    ax.set_title('Monthly Revenue Plan (Y1)')

    # Q区切り線
    for q_end in [2.5, 5.5, 8.5]:
        ax.axvline(q_end, color='gray', linestyle='--', alpha=0.5)

    # 凡例統合
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_five_year_growth(results: List[Dict], output_path: str):
    """5年間成長グラフ"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    years = [r["year"] for r in results]
    x = np.arange(len(years))

    # 1. 売上構成（積み上げ）
    ax = axes[0, 0]
    btoc = [r["btoc"] / 1_000_000 for r in results]
    btob = [r["btob"] / 1_000_000 for r in results]
    mp = [r["mp"] / 1_000_000 for r in results]

    ax.bar(x, btoc, 0.6, label='BtoC', color='#3498db')
    ax.bar(x, btob, 0.6, bottom=btoc, label='BtoB', color='#e74c3c')
    ax.bar(x, mp, 0.6, bottom=[b + c for b, c in zip(btoc, btob)], label='MP', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.set_ylabel('Revenue (M Yen)')
    ax.set_title('Revenue by Segment')
    ax.legend()

    for i, r in enumerate(results):
        total = r["total_revenue"] / 1_000_000
        ax.text(i, total + 3, f'{total:.0f}M', ha='center', fontsize=10, fontweight='bold')

    # 2. 粗利・投資
    ax = axes[0, 1]
    profit = [r["gross_profit"] / 1_000_000 for r in results]
    invest = [r["investment"] / 1_000_000 for r in results]
    op_profit = [r["operating_profit"] / 1_000_000 for r in results]

    ax.bar(x - 0.2, profit, 0.4, label='Gross Profit', color='#3498db')
    ax.bar(x + 0.2, invest, 0.4, label='Investment', color='#e74c3c')
    ax.plot(x, op_profit, 'go-', linewidth=2, markersize=8, label='Operating Profit')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.set_ylabel('Amount (M Yen)')
    ax.set_title('Profit vs Investment')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    # 3. LTV推移
    ax = axes[1, 0]
    ltv = [r["ltv"] for r in results]
    repeat = [r["repeat_rate"] * 100 for r in results]

    ax.bar(x, ltv, 0.6, color='#9b59b6', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.set_ylabel('LTV (Yen)')
    ax.set_title('Customer LTV Evolution')

    ax2 = ax.twinx()
    ax2.plot(x, repeat, 'r-o', linewidth=2, markersize=8)
    ax2.set_ylabel('Repeat Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    for i, v in enumerate(ltv):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=10)

    # 4. ROI推移
    ax = axes[1, 1]
    roi = [r["roi"] * 100 for r in results]

    ax.plot(x, roi, 'b-o', linewidth=2, markersize=10)
    ax.fill_between(x, 0, roi, alpha=0.3)
    ax.axhline(100, color='green', linestyle='--', label='100% ROI')
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.set_ylabel('Cumulative ROI (%)')
    ax.set_title('Cumulative ROI')
    ax.legend()

    for i, r in enumerate(roi):
        ax.text(i, r + 5, f'{r:.0f}%', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('EEZO 5-Year Growth Plan (with Price Jump Strategy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cost_structure(cost_result: Dict, output_path: str):
    """原価構造グラフ"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bands = cost_result["price_bands"]

    # 1. 原価内訳（積み上げ棒グラフ）
    ax = axes[0]
    x = np.arange(len(bands))
    width = 0.5

    product = [b["product_cost"] for b in bands]
    shipping = [b["shipping"] for b in bands]
    card = [b["card"] for b in bands]
    profit = [b["gross_profit"] for b in bands]

    ax.bar(x, product, width, label='Product Cost', color='#3498db')
    ax.bar(x, shipping, width, bottom=product, label='Shipping', color='#e74c3c')
    ax.bar(x, card, width, bottom=[p + s for p, s in zip(product, shipping)], label='Insert Card', color='#f39c12')
    ax.bar(x, profit, width, bottom=[p + s + c for p, s, c in zip(product, shipping, card)],
           label='Gross Profit', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels([b["band"] for b in bands])
    ax.set_ylabel('Amount (Yen)')
    ax.set_title(f'Cost Structure by Price Band (Y{cost_result["year"]})')
    ax.legend()

    # 価格ライン
    for i, b in enumerate(bands):
        ax.axhline(b["price"], xmin=i/len(bands) + 0.05, xmax=(i+1)/len(bands) - 0.05,
                   color='black', linestyle='--', linewidth=2)

    # 2. 粗利率比較
    ax = axes[1]
    margins = [b["margin_rate"] * 100 for b in bands]

    bars = ax.bar(x, margins, width, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([b["band"] for b in bands])
    ax.set_ylabel('Margin Rate (%)')
    ax.set_title('Margin Rate by Price Band')
    ax.axhline(15, color='green', linestyle='--', label='Target: 15%')
    ax.legend()

    for bar, m in zip(bars, margins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{m:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 価格ジャンプ効果の注釈
    if cost_result["price_jump_effect"]:
        effect = cost_result["price_jump_effect"]
        fig.text(0.5, 0.02,
                 f'Price Jump Effect: AOV +{effect["aov_change"]*100:.1f}%, Margin +{effect["margin_improvement"]*100:.1f}%',
                 ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 4. レポート生成
# =============================================================================

def generate_report(
    y1_quarterly: Dict,
    y1_monthly: List[Dict],
    cost_y1: Dict,
    cost_y2: Dict,
    five_year: List[Dict],
    btob_details: Dict,
    price_jump: PriceJumpEvidence,
) -> str:
    """revenue_strategy_2026.md形式のレポート生成"""

    report = f"""# EEZO 10百万円達成 収益戦略ドキュメント（価格ジャンプ戦略版）

**作成日**: {datetime.now().strftime('%Y-%m-%d')}
**目標**: 年間売上 10,000,000円 / 粗利 1,500,000円
**期間**: 2026年Q1〜Q4

---

## 1. エグゼクティブサマリー

### 現状の課題
実測ファネルデータから、**商品詳細→カート転換率1.9%** が最大のボトルネック。
配信ベースCVR 0.003%は業界平均の約1,000分の1。

### 価格ジャンプ戦略の効果（新規追加）

20,000円送料無料閾値の導入効果（exp001_report_20260129.mdより）:

| 指標 | 効果 | 備考 |
|------|------|------|
| ジャンプ率 | **{price_jump.jump_rate*100:.1f}%** | 目標30%を達成 |
| AOV変化 | **+{price_jump.aov_change*100:.1f}%** | ¥{price_jump.baseline_aov:,} → ¥{price_jump.new_aov:,} |
| 粗利変化 | **+{price_jump.gross_profit_change*100:.1f}%** | 送料減を商品粗利で相殺 |

### 達成戦略
**Q3でO2O顧客を大量獲得し、Q4年末商戦で売上を刈り取る**二段構え。
- Q3（7-9月）: O2O限定期間で新規顧客獲得
- Q4（10-12月）: Q3顧客のリピート + SNS広告集中 + 歳暮需要で売上ピーク

### 必要投資総額
| 項目 | 金額 | 備考 |
|------|------|------|
| Shopify移行 | 申請済 | エンジニア40h×12ヶ月 |
| 商品写真撮影 | 20,000円 | グループ内カメラマン半日 |
| 同梱カード | 89,000円 | デザイン内製（Canva）+印刷100円×890枚 |
| レビューシステム | 100,000円 | Shopifyアプリ |
| 認証表示整備 | 50,000円 | HACCP等 |
| 予備費 | 40,000円 | |
| **合計** | **約300,000円** | 移行費用除く |

---

## 2. 売上構造（価格ジャンプ戦略反映版）

### 2.1 基本パラメータ

| 項目 | 従来値 | 価格ジャンプ効果後 | 変化 |
|------|--------|------------------|------|
| 平均単価 | ¥7,500 | **¥{y1_quarterly['annual_summary']['avg_price']:,}** | +{(y1_quarterly['annual_summary']['avg_price']/7500-1)*100:.1f}% |
| 粗利率 | 15.0% | **{y1_quarterly['annual_summary']['margin_rate']*100:.1f}%** | +{(y1_quarterly['annual_summary']['margin_rate']/0.15-1)*100:.1f}% |
| リピート率 | 20% | 20% | - |
| 平均購入回数 | 1.25回 | 1.25回 | - |

### 2.2 必要顧客数の再計算

```
目標売上 10,000,000円
÷ 平均単価 {y1_quarterly['annual_summary']['avg_price']:,}円（価格ジャンプ効果後）
÷ 平均購入回数 1.25回
= 必要新規顧客 **{int(10000000 / y1_quarterly['annual_summary']['avg_price'] / 1.25):,}人**
```

**従来比**: 1,067人 → {int(10000000 / y1_quarterly['annual_summary']['avg_price'] / 1.25):,}人（**{(1 - int(10000000 / y1_quarterly['annual_summary']['avg_price'] / 1.25)/1067)*100:.0f}%削減**）

---

## 3. Q別売上推移計画

### 3.1 四半期サマリー

| Q | 期間 | 主要活動 | 新規顧客 | リピート | 売上 | 構成比 |
|---|------|----------|----------|----------|------|--------|
"""

    for q in y1_quarterly["quarters"]:
        share = q["revenue"] / y1_quarterly["annual_summary"]["total_revenue"] * 100 if y1_quarterly["annual_summary"]["total_revenue"] > 0 else 0
        report += f"| Q{q['quarter']} | {q['name'].split(':')[1].strip()} | {', '.join(q['activities'][:2])} | {q['new_customers']}人 | {q['repeat_customers']}人 | **¥{q['revenue']:,}** | {share:.0f}% |\n"

    report += f"| **年間** | | | **{y1_quarterly['annual_summary']['total_new_customers']}人** | **{sum(q['repeat_customers'] for q in y1_quarterly['quarters'])}人** | **¥{y1_quarterly['annual_summary']['total_revenue']:,}** | 100% |\n"

    report += f"""
**整合性検証**:
- 目標 ¥10M に対して **{(y1_quarterly['annual_summary']['total_revenue']/10000000-1)*100:+.1f}%**
- 価格ジャンプ効果による単価向上で達成

### 3.2 月次ブレークダウン

| 月 | 新規 | リピート | 売上 | 累計 | 備考 |
|----|------|----------|------|------|------|
"""

    for m in y1_monthly:
        report += f"| {m['name']} | {m['new']}人 | {m['repeat']}人 | ¥{m['revenue']:,} | ¥{m['cumulative_revenue']:,} | {m['note']} |\n"

    report += f"""
---

## 4. 原価・収益構造

### 4.1 Y1原価構造（北海道発送）

| 価格帯 | 表示価格 | 商品原価 | 送料 | カード | 総原価 | 粗利 | 粗利率 |
|--------|---------|---------|------|--------|--------|------|--------|
"""

    for b in cost_y1["price_bands"]:
        report += f"| {b['band']} | ¥{b['price']:,} | ¥{b['product_cost']:,} | ¥{b['shipping']:,} | ¥{b['card']:,} | ¥{b['total_cost']:,} | ¥{b['gross_profit']:,} | {b['margin_rate']*100:.1f}% |\n"

    report += f"| **加重平均** | ¥{cost_y1['weighted_average']['price']:,.0f} | - | - | - | ¥{cost_y1['weighted_average']['cost']:,.0f} | ¥{cost_y1['weighted_average']['profit']:,.0f} | **{cost_y1['weighted_average']['margin']*100:.1f}%** |\n"

    if cost_y1["price_jump_effect"]:
        effect = cost_y1["price_jump_effect"]
        report += f"""
### 4.2 価格ジャンプ効果

| 指標 | 効果 |
|------|------|
| ジャンプ率 | {effect['jump_rate']*100:.1f}% |
| AOV変化 | +{effect['aov_change']*100:.1f}% |
| 実効平均単価 | ¥{effect['new_avg_price']:,} |
| 実効粗利率 | {effect['new_margin_rate']*100:.1f}% |
"""

    report += f"""
### 4.3 Y2以降（大阪在庫・送料圧縮）

| 価格帯 | 表示価格 | 商品原価 | 送料 | カード | 総原価 | 粗利 | 粗利率 |
|--------|---------|---------|------|--------|--------|------|--------|
"""

    for b in cost_y2["price_bands"]:
        report += f"| {b['band']} | ¥{b['price']:,} | ¥{b['product_cost']:,} | ¥{b['shipping']:,} | ¥{b['card']:,} | ¥{b['total_cost']:,} | ¥{b['gross_profit']:,} | {b['margin_rate']*100:.1f}% |\n"

    report += f"""
**Y2改善ポイント**: 大阪在庫により送料35%削減 → 粗利率15%→20%へ改善

---

## 5. 5年間成長シミュレーション

### 5.1 年次サマリー

| 年度 | BtoC | BtoB | MP | 合計 | 粗利 | 粗利率 | 成長率 |
|------|------|------|-----|------|------|--------|--------|
"""

    for r in five_year:
        growth = f"+{r['growth_rate']*100:.0f}%" if r['growth_rate'] > 0 else "-"
        report += f"| Y{r['year']} | ¥{r['btoc']/1_000_000:.0f}M | ¥{r['btob']/1_000_000:.0f}M | ¥{r['mp']/1_000_000:.0f}M | **¥{r['total_revenue']/1_000_000:.0f}M** | ¥{r['gross_profit']/1_000_000:.1f}M | {r['margin_rate']*100:.0f}% | {growth} |\n"

    final = five_year[-1]
    report += f"""
### 5.2 5年間累計

| 指標 | 金額 |
|------|------|
| 累計売上 | **¥{final['cumulative_revenue']/1_000_000:.0f}M** |
| 累計粗利 | **¥{final['cumulative_profit']/1_000_000:.1f}M** |
| 累計投資 | ¥{final['cumulative_investment']/1_000_000:.1f}M |
| **ROI** | **{final['roi']*100:.0f}%** |

### 5.3 BtoB展開詳細
"""

    for year, details in btob_details.items():
        report += f"\n#### Y{year}: {details['total_accounts']}社 (¥{details['total_revenue']/1_000_000:.1f}M)\n\n"
        report += "| セグメント | アカウント数 | 平均単価 | 年間発注 | 年間売上 |\n"
        report += "|-----------|-------------|---------|---------|----------|\n"
        for seg_id, seg in details["segments"].items():
            report += f"| {seg['name']} | {seg['accounts']}社 | ¥{seg['avg_order']:,} | {seg['orders_per_year']}回 | ¥{seg['revenue']/1_000_000:.1f}M |\n"

    report += f"""
### 5.4 LTV推移

| 年度 | 平均単価 | リピート率 | 購入回数 | 粗利率 | LTV |
|------|---------|-----------|---------|--------|-----|
"""

    for r in five_year:
        avg_purchases = 1 / (1 - r['repeat_rate'])
        report += f"| Y{r['year']} | ¥{r['avg_price']:,} | {r['repeat_rate']*100:.0f}% | {avg_purchases:.2f}回 | {r['margin_rate']*100:.0f}% | **¥{r['ltv']:,}** |\n"

    report += f"""
---

## 6. 価格ジャンプ戦略の実装施策

### 6.1 UI/UX施策

| 施策 | 内容 | 期待効果 |
|------|------|----------|
| カートページ表示 | 「あと○○円で送料無料」を目立つ位置に表示 | ジャンプ率+10% |
| レコメンド機能 | 「送料無料まであと少し」商品を自動提案 | AOV+5% |
| 達成時フィードバック | 20,000円達成時に祝福アニメーション | 顧客満足度向上 |

### 6.2 商品構成の最適化

ジャンプを促進するための追加購入商品：

| 対象顧客 | 追加購入金額 | 推奨商品 |
|---------|-------------|---------|
| 10,000〜12,000円帯 | 8,000〜10,000円 | プレミアムセット、ギフトボックス |
| 12,000〜15,000円帯 | 5,000〜8,000円 | いくら醤油漬け、チーズ詰め合わせ |
| 15,000〜18,000円帯 | 2,000〜5,000円 | 鮭とばセット、エゾシカジャーキー |

---

## 7. リスクと対策

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| **価格ジャンプ効果が想定以下** | 中 | 中 | A/Bテストで検証、閾値調整 |
| **Q3 O2O不振** | 極高 | 中 | SNS広告前倒し、メルマガ強化 |
| **Q4歳暮需要不振** | 極高 | 中 | 早期予約で確保、広告追加投下 |
| **送料コスト増** | 中 | 高 | Y2大阪在庫で圧縮 |

---

## 8. 出力ファイル一覧

| ファイル | 内容 |
|---------|------|
| figures/y1_quarterly_revenue.png | Y1四半期別売上・顧客構成 |
| figures/y1_monthly_revenue.png | Y1月次売上推移 |
| figures/cost_structure_y1.png | Y1原価構造 |
| figures/five_year_growth_v2.png | 5年間成長計画 |
| reports/revenue_strategy_simulation.md | 本レポート |

---

*このレポートは価格ジャンプ戦略のエビデンス（exp001_report_20260129.md）を反映した
シミュレーション結果に基づいています。*

*新日本海商事 EEZO Shopifyリニューアル 商品構造企画*
"""

    return report


# =============================================================================
# 5. メイン実行
# =============================================================================

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("EEZO 10百万円達成 収益戦略シミュレーション")
    print("（価格ジャンプ戦略反映版）")
    print("=" * 60)

    ensure_dirs()
    simulator = RevenueStrategySimulator()

    # 1. Y1四半期シミュレーション
    print("\n【1】Y1四半期シミュレーション")
    print("-" * 40)

    y1_quarterly = simulator.simulate_y1_quarterly(with_price_jump=True)
    summary = y1_quarterly["annual_summary"]

    print(f"年間売上: ¥{summary['total_revenue']:,}")
    print(f"年間粗利: ¥{summary['gross_profit']:,}")
    print(f"平均単価: ¥{summary['avg_price']:,}（価格ジャンプ効果後）")
    print(f"粗利率: {summary['margin_rate']*100:.1f}%")

    plot_quarterly_revenue(y1_quarterly, str(FIGURES_DIR / "y1_quarterly_revenue.png"))
    print("グラフ保存: y1_quarterly_revenue.png")

    # 2. Y1月次シミュレーション
    print("\n【2】Y1月次シミュレーション")
    print("-" * 40)

    y1_monthly = simulator.simulate_y1_monthly(with_price_jump=True)
    final_month = y1_monthly[-1]
    print(f"12月累計売上: ¥{final_month['cumulative_revenue']:,}")

    plot_monthly_revenue(y1_monthly, str(FIGURES_DIR / "y1_monthly_revenue.png"))
    print("グラフ保存: y1_monthly_revenue.png")

    # 3. 原価構造シミュレーション
    print("\n【3】原価構造シミュレーション")
    print("-" * 40)

    cost_y1 = simulator.simulate_cost_structure(year=1, with_price_jump=True)
    cost_y2 = simulator.simulate_cost_structure(year=2, with_price_jump=True)

    print(f"Y1粗利率: {cost_y1['weighted_average']['margin']*100:.1f}%")
    print(f"Y2粗利率: {cost_y2['weighted_average']['margin']*100:.1f}%")

    if cost_y1["price_jump_effect"]:
        effect = cost_y1["price_jump_effect"]
        print(f"価格ジャンプ効果: AOV +{effect['aov_change']*100:.1f}%, 粗利 +{effect['margin_improvement']*100:.1f}%")

    plot_cost_structure(cost_y1, str(FIGURES_DIR / "cost_structure_y1.png"))
    print("グラフ保存: cost_structure_y1.png")

    # 4. 5年間成長シミュレーション
    print("\n【4】5年間成長シミュレーション")
    print("-" * 40)

    five_year = simulator.simulate_five_years(with_price_jump=True)

    print(f"\n{'年':>3} {'売上':>12} {'粗利':>12} {'ROI':>8}")
    print("-" * 40)
    for r in five_year:
        print(f"Y{r['year']:>2} ¥{r['total_revenue']/1_000_000:>10.0f}M ¥{r['gross_profit']/1_000_000:>10.1f}M {r['roi']*100:>7.0f}%")

    plot_five_year_growth(five_year, str(FIGURES_DIR / "five_year_growth_v2.png"))
    print("\nグラフ保存: five_year_growth_v2.png")

    # 5. BtoB詳細
    print("\n【5】BtoB展開詳細")
    print("-" * 40)

    btob_details = simulator.simulate_btob_details()
    for year, details in btob_details.items():
        print(f"Y{year}: {details['total_accounts']}社 (¥{details['total_revenue']/1_000_000:.1f}M)")

    # 6. レポート生成
    print("\n【6】レポート生成")
    print("-" * 40)

    report = generate_report(
        y1_quarterly=y1_quarterly,
        y1_monthly=y1_monthly,
        cost_y1=cost_y1,
        cost_y2=cost_y2,
        five_year=five_year,
        btob_details=btob_details,
        price_jump=simulator.price_jump,
    )

    report_path = REPORTS_DIR / "revenue_strategy_simulation.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"レポート保存: {report_path}")

    print("\n" + "=" * 60)
    print("シミュレーション完了！")
    print("=" * 60)
    print(f"\n出力ディレクトリ: {OUTPUT_DIR}")
    print("生成ファイル:")
    print("  - figures/y1_quarterly_revenue.png")
    print("  - figures/y1_monthly_revenue.png")
    print("  - figures/cost_structure_y1.png")
    print("  - figures/five_year_growth_v2.png")
    print("  - reports/revenue_strategy_simulation.md")

    return {
        "y1_quarterly": y1_quarterly,
        "y1_monthly": y1_monthly,
        "cost_y1": cost_y1,
        "cost_y2": cost_y2,
        "five_year": five_year,
        "btob_details": btob_details,
    }


if __name__ == "__main__":
    main()
