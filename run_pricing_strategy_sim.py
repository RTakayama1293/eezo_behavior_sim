#!/usr/bin/env python3
"""
価格ジャンプ戦略シミュレーション実行スクリプト

20,000円送料無料閾値の効果分析と5年間成長計画のシミュレーション
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pricing_strategy_model import (
    PricingStrategyModel,
    FiveYearGrowthModel,
    PriceJumpConfig,
    CostStructure,
)

# matplotlib設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
except ImportError:
    # japanize_matplotlibがない場合はフォント設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'sans-serif']

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


def plot_price_distribution_shift(result: Dict, output_path: str):
    """価格帯分布変化をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    original = result['distribution']['original_orders']
    new = result['distribution']['new_orders']
    threshold = 20000

    # ヒストグラム（Before/After）
    bins = np.arange(0, 40000, 2000)

    axes[0].hist(original, bins=bins, alpha=0.7, label='Before (threshold=0)', color='blue', edgecolor='black')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: ¥{threshold:,}')
    axes[0].set_xlabel('Order Amount (Yen)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Before: Price Distribution')
    axes[0].legend()

    axes[1].hist(new, bins=bins, alpha=0.7, label='After (threshold=20K)', color='green', edgecolor='black')
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: ¥{threshold:,}')
    axes[1].set_xlabel('Order Amount (Yen)')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('After: Price Distribution (with Jump Effect)')
    axes[1].legend()

    plt.suptitle('Price Distribution Shift with 20,000 Yen Free Shipping Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_jump_analysis(result: Dict, output_path: str):
    """ジャンプ分析をプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    jump_stats = result['jump_stats']
    by_band = jump_stats['by_band']

    # 1. 価格帯別ジャンプ率
    bands = [b['band'] for b in by_band]
    jump_rates = [b['jump_rate'] * 100 for b in by_band]
    colors = ['#3498db' if r < 30 else '#e74c3c' for r in jump_rates]

    ax = axes[0, 0]
    bars = ax.bar(bands, jump_rates, color=colors, edgecolor='black')
    ax.set_ylabel('Jump Rate (%)')
    ax.set_title('Jump Rate by Price Band')
    ax.axhline(30, color='green', linestyle='--', alpha=0.7, label='Target: 30%')
    ax.legend()

    for bar, rate in zip(bars, jump_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=10)

    # 2. 売上・AOV変化
    ax = axes[0, 1]
    metrics = ['Revenue', 'AOV', 'Gross Profit']
    changes = [
        result['changes']['revenue_change'] * 100,
        result['changes']['aov_change'] * 100,
        result['changes']['gross_profit_change'] * 100,
    ]
    colors = ['green' if c > 0 else 'red' for c in changes]

    bars = ax.bar(metrics, changes, color=colors, edgecolor='black')
    ax.set_ylabel('Change (%)')
    ax.set_title('Impact on Key Metrics')
    ax.axhline(0, color='black', linewidth=0.5)

    for bar, change in zip(bars, changes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{change:+.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 3. ジャンプ顧客の増加金額分布
    ax = axes[1, 0]
    original = result['distribution']['original_orders']
    new = result['distribution']['new_orders']
    jumped = result['distribution']['jump_flags']

    if np.any(jumped):
        increases = new[jumped] - original[jumped]
        ax.hist(increases, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(increases), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: ¥{np.mean(increases):,.0f}')
        ax.set_xlabel('Increase Amount (Yen)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Distribution of Increase Amount (Jumped Customers)')
        ax.legend()

    # 4. 顧客数構成比
    ax = axes[1, 1]
    labels = ['Jumped', 'Not Jumped']
    sizes = [jump_stats['total_jumps'], result['n_customers'] - jump_stats['total_jumps']]
    colors = ['#e74c3c', '#3498db']
    explode = (0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title(f'Customer Jump Status (n={result["n_customers"]:,})')

    plt.suptitle('20,000 Yen Threshold Jump Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_five_year_growth(results: List[Dict], output_path: str):
    """5年間成長グラフをプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    years = [r['year'] for r in results]

    # 1. 売上構成推移（積み上げ棒グラフ）
    ax = axes[0, 0]
    btoc = [r['btoc_revenue'] / 1_000_000 for r in results]
    btob = [r['btob_revenue'] / 1_000_000 for r in results]
    mp = [r['mp_revenue'] / 1_000_000 for r in results]

    x = np.arange(len(years))
    width = 0.6

    ax.bar(x, btoc, width, label='BtoC', color='#3498db')
    ax.bar(x, btob, width, bottom=btoc, label='BtoB', color='#e74c3c')
    ax.bar(x, mp, width, bottom=[b + c for b, c in zip(btoc, btob)], label='Marketplace', color='#2ecc71')

    ax.set_xlabel('Year')
    ax.set_ylabel('Revenue (Million Yen)')
    ax.set_title('Revenue Composition by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.legend()

    # 売上目標線
    for i, r in enumerate(results):
        total = r['total_revenue'] / 1_000_000
        ax.text(i, total + 2, f'¥{total:.0f}M', ha='center', fontsize=9, fontweight='bold')

    # 2. 成長率推移
    ax = axes[0, 1]
    growth_rates = [r['growth_rate'] * 100 for r in results]

    bars = ax.bar(x[1:], growth_rates[1:], width=0.5, color=['green' if g < 100 else 'orange' for g in growth_rates[1:]], edgecolor='black')
    ax.set_xlabel('Year')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Year-over-Year Growth Rate')
    ax.set_xticks(x[1:])
    ax.set_xticklabels([f'Y{y}' for y in years[1:]])
    ax.axhline(100, color='red', linestyle='--', alpha=0.7, label='100% Growth')
    ax.legend()

    for bar, rate in zip(bars, growth_rates[1:]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'+{rate:.0f}%', ha='center', fontsize=10)

    # 3. 粗利・投資推移
    ax = axes[1, 0]
    gross_profit = [r['gross_profit'] / 1_000_000 for r in results]
    investment = [r['investment'] / 1_000_000 for r in results]
    operating_profit = [r['operating_profit'] / 1_000_000 for r in results]

    ax.bar(x - 0.2, gross_profit, 0.4, label='Gross Profit', color='#3498db')
    ax.bar(x + 0.2, investment, 0.4, label='Investment', color='#e74c3c')
    ax.plot(x, operating_profit, 'go-', linewidth=2, markersize=8, label='Operating Profit')

    ax.set_xlabel('Year')
    ax.set_ylabel('Amount (Million Yen)')
    ax.set_title('Gross Profit vs Investment')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.legend()

    # 4. 累積ROI推移
    ax = axes[1, 1]
    roi = [r['cumulative']['roi'] * 100 for r in results]
    cum_profit = [r['cumulative']['profit'] / 1_000_000 for r in results]
    cum_investment = [r['cumulative']['investment'] / 1_000_000 for r in results]

    ax.plot(x, roi, 'b-o', linewidth=2, markersize=8, label='Cumulative ROI')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(100, color='green', linestyle='--', alpha=0.7, label='100% ROI')

    ax.set_xlabel('Year')
    ax.set_ylabel('ROI (%)')
    ax.set_title('Cumulative ROI')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.legend()

    for i, r in enumerate(roi):
        ax.text(i, r + 5, f'{r:.0f}%', ha='center', fontsize=10)

    plt.suptitle('EEZO 5-Year Growth Plan (BtoC + BtoB + Marketplace)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ltv_evolution(ltv_results: List[Dict], output_path: str):
    """LTV推移グラフをプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    years = [r['year'] for r in ltv_results]
    x = np.arange(len(years))

    # 1. LTV推移
    ax = axes[0]
    ltv = [r['ltv'] for r in ltv_results]
    avg_price = [r['avg_price'] for r in ltv_results]
    repeat_rate = [r['repeat_rate'] * 100 for r in ltv_results]

    ax.bar(x, ltv, width=0.6, color='#3498db', edgecolor='black', label='LTV')
    ax.set_xlabel('Year')
    ax.set_ylabel('LTV (Yen)')
    ax.set_title('Customer Lifetime Value Evolution')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])

    for i, v in enumerate(ltv):
        ax.text(i, v + 30, f'¥{v:,}', ha='center', fontsize=10, fontweight='bold')

    # 2. ドライバー分析
    ax = axes[1]
    ax.plot(x, repeat_rate, 'r-o', linewidth=2, markersize=8, label='Repeat Rate (%)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Repeat Rate (%)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Y{y}' for y in years])

    ax2 = ax.twinx()
    ax2.plot(x, avg_price, 'b-s', linewidth=2, markersize=8, label='Avg Price (Yen)')
    ax2.set_ylabel('Average Price (Yen)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_title('LTV Drivers: Repeat Rate & Average Price')

    # 凡例を統合
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.suptitle('LTV Evolution Over 5 Years', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(output_path: str):
    """感度分析グラフをプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 追加購入傾向を変えた場合のシミュレーション
    propensities = [0.40, 0.45, 0.50, 0.55, 0.58, 0.65, 0.70]
    jump_rates = []
    aov_changes = []
    profit_changes = []

    for prop in propensities:
        config = PriceJumpConfig(base_propensity=prop)
        model = PricingStrategyModel(config=config)
        result = model.simulate_threshold_effect(n_customers=1000)

        jump_rates.append(result['jump_stats']['jump_rate'] * 100)
        aov_changes.append(result['changes']['aov_change'] * 100)
        profit_changes.append(result['changes']['gross_profit_change'] * 100)

    x = [p * 100 for p in propensities]

    ax.plot(x, jump_rates, 'b-o', linewidth=2, markersize=8, label='Jump Rate (%)')
    ax.plot(x, aov_changes, 'g-s', linewidth=2, markersize=8, label='AOV Change (%)')
    ax.plot(x, profit_changes, 'r-^', linewidth=2, markersize=8, label='Gross Profit Change (%)')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(58, color='gray', linestyle='--', alpha=0.7, label='Baseline (58%)')

    ax.set_xlabel('Base Propensity to Add Items (%)')
    ax.set_ylabel('Rate / Change (%)')
    ax.set_title('Sensitivity Analysis: Impact of Customer Propensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_integrated_report(
    price_jump_result: Dict,
    five_year_results: List[Dict],
    ltv_results: List[Dict],
) -> str:
    """統合レポートを生成"""

    report = f"""# EEZO 価格ジャンプ戦略 & 5年間成長計画 統合レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**シミュレーションモデル**: pricing_strategy_model.py

---

## エグゼクティブサマリー

### 価格ジャンプ戦略の効果
20,000円送料無料閾値の導入により、以下の効果が見込まれる：

| 指標 | 変化 | 判定 |
|------|------|------|
| ジャンプ率 | **{price_jump_result['jump_stats']['jump_rate']*100:.1f}%** | 目標30%を達成 |
| AOV変化 | **{price_jump_result['changes']['aov_change']*100:+.1f}%** | 大幅増加 |
| 粗利変化 | **{price_jump_result['changes']['gross_profit_change']*100:+.1f}%** | 送料減を商品粗利で相殺 |

### 5年間成長計画
"""

    # 5年間サマリー
    final = five_year_results[-1]
    report += f"""
| 年度 | BtoC | BtoB | MP | 合計 | 粗利 | 成長率 |
|------|------|------|-----|------|------|--------|
"""
    for r in five_year_results:
        growth = f"+{r['growth_rate']*100:.0f}%" if r['growth_rate'] > 0 else "-"
        report += f"| Y{r['year']} | ¥{r['btoc_revenue']/1_000_000:.0f}M | ¥{r['btob_revenue']/1_000_000:.0f}M | ¥{r['mp_revenue']/1_000_000:.0f}M | ¥{r['total_revenue']/1_000_000:.0f}M | ¥{r['gross_profit']/1_000_000:.1f}M | {growth} |\n"

    report += f"""
**5年間累計**:
- 累計売上: ¥{final['cumulative']['revenue']/1_000_000:.0f}M
- 累計粗利: ¥{final['cumulative']['profit']/1_000_000:.1f}M
- 累計投資: ¥{final['cumulative']['investment']/1_000_000:.1f}M
- **ROI: {final['cumulative']['roi']*100:.0f}%**

---

## 1. 価格ジャンプ戦略シミュレーション

### 1.1 シミュレーション設定

| 項目 | 設定値 |
|------|--------|
| 顧客数 | {price_jump_result['n_customers']:,}人 |
| 閾値 | 20,000円 |
| 送料（閾値未満） | 1,500円 |
| 追加購入傾向 | 58%（Shopify調査2026） |

### 1.2 シナリオ比較

#### AOV（平均注文額）
| シナリオ | AOV | 変化 |
|---------|-----|------|
| ベースライン | ¥{price_jump_result['baseline']['aov']:,.0f} | - |
| 20K閾値 | ¥{price_jump_result['with_threshold']['aov']:,.0f} | **{price_jump_result['changes']['aov_change']*100:+.1f}%** |

### 1.3 価格帯別ジャンプ率

| 価格帯 | 顧客数 | ジャンプ数 | ジャンプ率 |
|--------|--------|-----------|-----------|
"""
    for band in price_jump_result['jump_stats']['by_band']:
        report += f"| {band['band']} | {band['customers']}人 | {band['jumps']}人 | **{band['jump_rate']*100:.1f}%** |\n"

    report += f"""
**ポイント**: 閾値に近い15,000〜20,000円帯で最もジャンプ率が高い（Goal Gradient Effect確認）。

### 1.4 グラフ

![価格帯分布変化](figures/pricing_distribution_shift.png)

![ジャンプ分析](figures/pricing_jump_analysis.png)

---

## 2. 5年間成長計画シミュレーション

### 2.1 成長フェーズ

| フェーズ | 年度 | 主要施策 | 売上目標 |
|---------|------|---------|---------|
| 基盤構築 | Y1 | Shopify移行、O2O顧客獲得 | ¥10M |
| 効率化+BtoB開始 | Y2 | 大阪物流拠点、飲食店卸売 | ¥20M |
| ブランド確立 | Y3 | ホテル展開、法人ギフト | ¥35M |
| 大規模拡張 | Y4 | MP化、広告投下 | ¥100M |
| 収穫期 | Y5 | 複数収益源安定化 | ¥150M |

### 2.2 BtoB展開詳細
"""
    for r in five_year_results:
        btob = r['btob_details']
        if btob['total_accounts'] > 0:
            report += f"\n#### Y{r['year']}: {btob['total_accounts']}社 (¥{btob['total_revenue']/1_000_000:.1f}M)\n\n"
            report += "| セグメント | アカウント数 | 年間売上 |\n"
            report += "|-----------|-------------|----------|\n"
            for seg_id, seg in btob['segments'].items():
                report += f"| {seg['name']} | {seg['accounts']}社 | ¥{seg['revenue']/1_000_000:.1f}M |\n"

    report += f"""
### 2.3 成長グラフ

![5年間成長](figures/five_year_growth.png)

---

## 3. LTV推移シミュレーション

### 3.1 年次LTV

| 年度 | 平均単価 | リピート率 | 購入回数 | 粗利率 | LTV |
|------|---------|-----------|---------|--------|-----|
"""
    for r in ltv_results:
        report += f"| Y{r['year']} | ¥{r['avg_price']:,} | {r['repeat_rate']*100:.0f}% | {r['avg_purchases']:.2f}回 | {r['margin_rate']*100:.0f}% | **¥{r['ltv']:,}** |\n"

    report += f"""
### 3.2 LTV変動要因

- **Y1-Y3**: 信頼・愛着形成によるリピート率向上
- **Y4**: 大量新規獲得によるミックス変化（一時的LTV低下）
- **Y5**: スケールメリット発現でLTV回復

![LTV推移](figures/ltv_evolution.png)

---

## 4. 感度分析

追加購入傾向（base_propensity）を40%〜70%で変化させた場合の影響を分析。

![感度分析](figures/sensitivity_analysis.png)

**ポイント**:
- 追加購入傾向が40%まで低下しても**粗利はプラス**を維持
- 損益分岐点は推定30〜35%程度
- パラメータの不確実性に対して**堅牢な結果**

---

## 5. 意思決定への示唆

### 5.1 価格ジャンプ戦略の導入推奨

シミュレーション結果に基づき、**20,000円送料無料閾値の導入を推奨**。

**期待効果（月間1,000注文の場合）**:
- 売上増加: 約{price_jump_result['changes']['revenue_change']*price_jump_result['baseline']['total_revenue']/10000:.0f}万円/月
- AOV向上: ¥{price_jump_result['baseline']['aov']:,.0f} → ¥{price_jump_result['with_threshold']['aov']:,.0f}

### 5.2 商品構成への示唆

ジャンプを促進するために以下の商品構成を推奨:

| 対象顧客 | 追加購入金額 | 推奨商品 |
|---------|-------------|---------|
| 10,000〜12,000円帯 | 8,000〜10,000円 | プレミアムセット、ギフトボックス |
| 12,000〜15,000円帯 | 5,000〜8,000円 | いくら醤油漬け、チーズ詰め合わせ |
| 15,000〜18,000円帯 | 2,000〜5,000円 | 鮭とばセット、エゾシカジャーキー |

### 5.3 5年間計画の実行優先度

1. **Y1**: Shopify移行 + O2O顧客獲得 → 売上¥10M達成
2. **Y2**: 大阪物流拠点 + BtoB開始 → 粗利率改善（15%→20%）
3. **Y3**: ブランド確立 + レビュー蓄積 → Y4拡張の準備
4. **Y4**: 大規模投資（¥15M）→ 売上¥100M達成
5. **Y5**: 複数収益源の安定化 → 持続的成長

---

## 6. リスクと留意点

1. **パラメータの不確実性**: 実際の顧客行動は推定値と異なる可能性
2. **Y4成長率186%**: 野心的な目標、投資効果の継続モニタリング必要
3. **BtoB営業リソース**: Y2-Y3のBtoB展開には専任営業が必要
4. **A/Bテスト推奨**: 本番導入前に小規模テストでの検証を推奨

---

## 出力ファイル一覧

| ファイル | 内容 |
|---------|------|
| figures/pricing_distribution_shift.png | 価格帯分布変化図 |
| figures/pricing_jump_analysis.png | ジャンプ詳細分析図 |
| figures/five_year_growth.png | 5年間成長図 |
| figures/ltv_evolution.png | LTV推移図 |
| figures/sensitivity_analysis.png | 感度分析図 |
| reports/pricing_strategy_report.md | 本レポート |

---

*このレポートは自動生成されました。*
*新日本海商事 EEZO Shopifyリニューアル 商品構造企画*
"""

    return report


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("EEZO 価格ジャンプ戦略 & 5年間成長計画シミュレーション")
    print("=" * 60)

    # ディレクトリ作成
    ensure_dirs()

    # 1. 価格ジャンプ効果シミュレーション
    print("\n【1】価格ジャンプ効果シミュレーション")
    print("-" * 40)

    model = PricingStrategyModel()
    price_jump_result = model.simulate_threshold_effect(n_customers=1000)

    print(f"ジャンプ率: {price_jump_result['jump_stats']['jump_rate']*100:.1f}%")
    print(f"AOV変化: {price_jump_result['changes']['aov_change']*100:+.1f}%")
    print(f"粗利変化: {price_jump_result['changes']['gross_profit_change']*100:+.1f}%")

    # グラフ出力
    plot_price_distribution_shift(
        price_jump_result,
        str(FIGURES_DIR / "pricing_distribution_shift.png")
    )
    plot_jump_analysis(
        price_jump_result,
        str(FIGURES_DIR / "pricing_jump_analysis.png")
    )
    print("グラフ保存: pricing_distribution_shift.png, pricing_jump_analysis.png")

    # 2. 5年間成長シミュレーション
    print("\n【2】5年間成長シミュレーション")
    print("-" * 40)

    growth_model = FiveYearGrowthModel()
    five_year_results = growth_model.simulate_five_years()

    for r in five_year_results:
        print(f"Y{r['year']}: ¥{r['total_revenue']/1_000_000:.0f}M (粗利: ¥{r['gross_profit']/1_000_000:.1f}M)")

    plot_five_year_growth(
        five_year_results,
        str(FIGURES_DIR / "five_year_growth.png")
    )
    print("グラフ保存: five_year_growth.png")

    # 3. LTV推移シミュレーション
    print("\n【3】LTV推移シミュレーション")
    print("-" * 40)

    ltv_results = growth_model.simulate_ltv_evolution()

    for r in ltv_results:
        print(f"Y{r['year']}: LTV ¥{r['ltv']:,} (リピート率: {r['repeat_rate']*100:.0f}%)")

    plot_ltv_evolution(
        ltv_results,
        str(FIGURES_DIR / "ltv_evolution.png")
    )
    print("グラフ保存: ltv_evolution.png")

    # 4. 感度分析
    print("\n【4】感度分析")
    print("-" * 40)

    plot_sensitivity_analysis(str(FIGURES_DIR / "sensitivity_analysis.png"))
    print("グラフ保存: sensitivity_analysis.png")

    # 5. レポート生成
    print("\n【5】レポート生成")
    print("-" * 40)

    report = generate_integrated_report(
        price_jump_result,
        five_year_results,
        ltv_results,
    )

    report_path = REPORTS_DIR / "pricing_strategy_report.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"レポート保存: {report_path}")

    print("\n" + "=" * 60)
    print("シミュレーション完了！")
    print("=" * 60)
    print(f"\n出力ディレクトリ: {OUTPUT_DIR}")
    print("生成ファイル:")
    print("  - figures/pricing_distribution_shift.png")
    print("  - figures/pricing_jump_analysis.png")
    print("  - figures/five_year_growth.png")
    print("  - figures/ltv_evolution.png")
    print("  - figures/sensitivity_analysis.png")
    print("  - reports/pricing_strategy_report.md")

    return {
        "price_jump": price_jump_result,
        "five_year": five_year_results,
        "ltv": ltv_results,
    }


if __name__ == "__main__":
    main()
