"""
可視化ユーティリティ
グラフ生成用のヘルパー関数
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
# japanize-matplotlibがインストールされている場合は使用
try:
    import japanize_matplotlib
except ImportError:
    pass

# カラーパレット
COLORS = {
    "baseline": "#4A90A4",      # 落ち着いた青
    "improved": "#5CB85C",      # 緑
    "photo": "#F0AD4E",         # オレンジ
    "review": "#5BC0DE",        # 水色
    "card": "#D9534F",          # 赤
    "ux": "#9B59B6",            # 紫
    "highlight": "#E74C3C",     # 強調赤
    "neutral": "#95A5A6",       # グレー
}


def plot_baseline_vs_improved(
    baseline_data: Dict,
    improved_data: Dict,
    output_path: str,
    title: str = "現状 vs 改善後の比較"
) -> None:
    """
    Sim1: ベースライン vs 改善後の比較棒グラフ

    Args:
        baseline_data: ベースラインデータ
        improved_data: 改善後データ
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 売上比較
    ax1 = axes[0]
    categories = ['Baseline\n(Genjo)', 'Improved\n(Kaizen-go)']
    sales = [baseline_data['total_revenue'], improved_data['total_revenue']]
    bars1 = ax1.bar(categories, sales, color=[COLORS['baseline'], COLORS['improved']], width=0.6)
    ax1.set_ylabel('Annual Sales (Yen)', fontsize=12)
    ax1.set_title('Sales Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(sales) * 1.3)

    # 値ラベル
    for bar, val in zip(bars1, sales):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sales)*0.02,
                f'{val:,.0f} yen', ha='center', va='bottom', fontsize=11)

    # 改善率
    improvement_pct = (improved_data['total_revenue'] / baseline_data['total_revenue'] - 1) * 100
    ax1.text(0.5, 0.95, f'+{improvement_pct:.1f}% improvement',
             transform=ax1.transAxes, ha='center', fontsize=12, color=COLORS['improved'])

    # 右: 粗利比較
    ax2 = axes[1]
    profits = [baseline_data['gross_profit'], improved_data['gross_profit']]
    bars2 = ax2.bar(categories, profits, color=[COLORS['baseline'], COLORS['improved']], width=0.6)
    ax2.set_ylabel('Gross Profit (Yen)', fontsize=12)
    ax2.set_title('Gross Profit Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(profits) * 1.3)

    for bar, val in zip(bars2, profits):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(profits)*0.02,
                f'{val:,.0f} yen', ha='center', va='bottom', fontsize=11)

    profit_improvement = (improved_data['gross_profit'] / baseline_data['gross_profit'] - 1) * 100
    ax2.text(0.5, 0.95, f'+{profit_improvement:.1f}% improvement',
             transform=ax2.transAxes, ha='center', fontsize=12, color=COLORS['improved'])

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_funnel_comparison(
    baseline_funnel: List[Dict],
    improved_funnel: List[Dict],
    output_path: str,
    title: str = "Purchase Funnel Comparison (Before/After)"
) -> None:
    """
    Sim2: ファネル図（Before/After）

    Args:
        baseline_funnel: ベースラインファネルデータ
        improved_funnel: 改善後ファネルデータ
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    def draw_funnel(ax, data, title_text, color):
        stages = [d['name'] for d in data]
        rates = [d['rate'] * 100 for d in data]

        # 横棒グラフでファネルを表現
        y_pos = np.arange(len(stages))
        bars = ax.barh(y_pos, rates, color=color, alpha=0.8, height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Conversion Rate (%)', fontsize=11)
        ax.set_title(title_text, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 110)

        # 値ラベル
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{rate:.2f}%', va='center', fontsize=10)

    draw_funnel(axes[0], baseline_funnel, 'Before (Baseline)', COLORS['baseline'])
    draw_funnel(axes[1], improved_funnel, 'After (Improved)', COLORS['improved'])

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_wtp_sensitivity(
    scenarios: List[Dict],
    output_path: str,
    title: str = "WTP Sensitivity Analysis by Scenario"
) -> None:
    """
    Sim3: WTP感度分析

    Args:
        scenarios: シナリオデータのリスト
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scenario_names = [s['name'] for s in scenarios]
    wtp_stated = [s['wtp_stated'] for s in scenarios]
    wtp_revealed = [s['wtp_revealed'] for s in scenarios]
    changes = [s['stated_change_pct'] for s in scenarios]

    x = np.arange(len(scenario_names))
    width = 0.35

    # 左: WTP金額
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, wtp_stated, width, label='Stated WTP', color=COLORS['baseline'])
    bars2 = ax1.bar(x + width/2, wtp_revealed, width, label='Revealed WTP (Est.)', color=COLORS['improved'])

    ax1.set_ylabel('WTP (Yen)', fontsize=12)
    ax1.set_title('WTP by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=10)
    ax1.legend()
    ax1.set_ylim(0, max(wtp_stated) * 1.3)

    for bar, val in zip(bars1, wtp_stated):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, wtp_revealed):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

    # 右: 変化率
    ax2 = axes[1]
    colors = [COLORS['baseline'] if c == 0 else COLORS['improved'] for c in changes]
    bars3 = ax2.bar(scenario_names, changes, color=colors, width=0.6)
    ax2.set_ylabel('WTP Change (%)', fontsize=12)
    ax2.set_title('WTP Change Rate (Stated)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, val in zip(bars3, changes):
        y_pos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=11)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_ltv_scenarios(
    ltv_data: Dict[str, Dict],
    cumulative_data: Dict[str, List[Dict]],
    output_path: str,
    title: str = "LTV and Repeat Model Analysis"
) -> None:
    """
    Sim4: LTV・リピートモデル

    Args:
        ltv_data: シナリオ別LTVデータ
        cumulative_data: 累積価値データ
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: LTVシナリオ比較
    ax1 = axes[0]
    scenarios = list(ltv_data.keys())
    names = [ltv_data[s]['name'] for s in scenarios]
    ltvs = [ltv_data[s]['ltv'] for s in scenarios]
    avg_purchases = [ltv_data[s]['avg_purchases'] for s in scenarios]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, ltvs, width, label='LTV (Yen)', color=COLORS['improved'])

    ax1.set_ylabel('LTV (Yen)', fontsize=12)
    ax1.set_title('LTV by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)

    # 平均購入回数を右軸に
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, avg_purchases, 'o-', color=COLORS['highlight'], markersize=10, linewidth=2,
                  label='Avg Purchases')
    ax1_twin.set_ylabel('Average Purchases', fontsize=12, color=COLORS['highlight'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['highlight'])

    for bar, val in zip(bars1, ltvs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=10)

    for i, (xi, val) in enumerate(zip(x, avg_purchases)):
        ax1_twin.text(xi, val + 0.05, f'{val:.2f}', ha='center', fontsize=10, color=COLORS['highlight'])

    # 凡例を統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 右: 累積価値の時系列
    ax2 = axes[1]
    colors_list = [COLORS['baseline'], COLORS['photo'], COLORS['improved']]

    for i, (scenario, data) in enumerate(cumulative_data.items()):
        years = [d['year'] for d in data]
        cum_profits = [d['cumulative_profit'] for d in data]
        ax2.plot(years, cum_profits, 'o-', color=colors_list[i % len(colors_list)],
                label=ltv_data[scenario]['name'], linewidth=2, markersize=8)

    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Cumulative Profit (Yen)', fontsize=12)
    ax2.set_title('Cumulative Customer Value (3 Years)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_xticks([1, 2, 3])
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_target_achievement(
    requirements: Dict,
    output_path: str,
    title: str = "10M Yen Achievement Scenario Analysis"
) -> None:
    """
    Sim5: 10百万円達成シナリオ

    Args:
        requirements: 必要条件データ
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = requirements['scenarios']
    names = [s['name'] for s in scenarios]
    cvrs = [s['required_cvr_pct'] for s in scenarios]
    customers = [s['required_customers'] for s in scenarios]
    feasibilities = [s['feasibility'] for s in scenarios]

    # 実現可能性に基づく色
    colors = []
    for f in feasibilities:
        if f == 'High' or f == '高':
            colors.append(COLORS['improved'])
        elif f == 'Med' or f == '中':
            colors.append(COLORS['photo'])
        else:
            colors.append(COLORS['highlight'])

    # 左: 必要CVR
    ax1 = axes[0]
    bars1 = ax1.bar(names, cvrs, color=colors, width=0.6)
    ax1.set_ylabel('Required CVR (%)', fontsize=12)
    ax1.set_title('Required CVR by Scenario', fontsize=14, fontweight='bold')
    ax1.axhline(y=2.5, color='black', linestyle='--', linewidth=1, label='Industry Avg (2.5%)')
    ax1.legend()

    for bar, val, f in zip(bars1, cvrs, feasibilities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%\n({f})', ha='center', va='bottom', fontsize=10)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # 右: 必要顧客数
    ax2 = axes[1]
    bars2 = ax2.bar(names, customers, color=colors, width=0.6)
    ax2.set_ylabel('Required Customers', fontsize=12)
    ax2.set_title('Required New Customers by Scenario', fontsize=14, fontweight='bold')

    for bar, val in zip(bars2, customers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=10)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # 目標情報
    fig.text(0.5, 0.02,
             f'Target: {requirements["target_sales"]:,.0f} yen | Available Traffic: {requirements["available_traffic"]:,}',
             ha='center', fontsize=12, style='italic')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_cvr_waterfall(
    effects: List[Dict],
    output_path: str,
    title: str = "CVR Improvement Breakdown (Waterfall)"
) -> None:
    """
    CVR改善の内訳ウォーターフォール

    Args:
        effects: 効果データのリスト
        output_path: 出力パス
        title: グラフタイトル
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [e['name'] for e in effects]
    values = [e['value'] for e in effects]
    cumulative = [e['cumulative'] for e in effects]

    # ウォーターフォールチャート
    bar_colors = []
    for i, e in enumerate(effects):
        if e.get('is_total'):
            bar_colors.append(COLORS['improved'])
        elif e.get('is_base'):
            bar_colors.append(COLORS['baseline'])
        else:
            bar_colors.append(COLORS['photo'])

    x = np.arange(len(names))
    bottoms = [0] + cumulative[:-1]

    for i, (name, val, bottom, color) in enumerate(zip(names, values, bottoms, bar_colors)):
        if effects[i].get('is_total') or effects[i].get('is_base'):
            ax.bar(i, val, bottom=0, color=color, width=0.6)
        else:
            ax.bar(i, val, bottom=bottom, color=color, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('CVR (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 値ラベル
    for i, (val, cum) in enumerate(zip(values, cumulative)):
        y_pos = cum if not effects[i].get('is_base') else val
        ax.text(i, y_pos + 0.01, f'{val*100:.3f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
