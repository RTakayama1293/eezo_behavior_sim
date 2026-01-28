#!/usr/bin/env python3
"""
EEZO消費者行動×収益シミュレーション
全シミュレーションの実行スクリプト
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.perception_model import PerceptionModel, INTERVENTIONS
from src.models.conversion_model import ConversionModel, TRAFFIC_SOURCES
from src.models.revenue_model import RevenueModel, LTV_SCENARIOS
from src.utils.visualization import (
    plot_baseline_vs_improved,
    plot_funnel_comparison,
    plot_wtp_sensitivity,
    plot_ltv_scenarios,
    plot_target_achievement,
)


# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp001_baseline" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    """出力ディレクトリを作成"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_sim1_baseline_vs_improved() -> Dict:
    """
    Sim1: ベースライン vs 改善後の比較
    """
    print("\n" + "="*60)
    print("Sim1: Baseline vs Improved Comparison")
    print("="*60)

    perception = PerceptionModel()
    conversion = ConversionModel()
    revenue = RevenueModel(avg_order_value=7500, margin_rate=0.15)

    # トラフィック合計
    total_traffic = sum(s.annual_volume for s in TRAFFIC_SOURCES.values())
    print(f"Total annual traffic: {total_traffic:,}")

    # ベースライン計算
    baseline_cvr = 0.0004  # メルマガCVR
    baseline_customers = int(total_traffic * baseline_cvr)
    baseline_revenue = revenue.calculate_annual_revenue(
        customers=baseline_customers,
        repeat_rate=0.10
    )

    print(f"\n[Baseline]")
    print(f"  CVR: {baseline_cvr*100:.4f}%")
    print(f"  Customers: {baseline_customers:,}")
    print(f"  Sales: {baseline_revenue['total_revenue']:,.0f} yen")
    print(f"  Gross Profit: {baseline_revenue['gross_profit']:,.0f} yen")

    # 改善後計算（フル施策）
    interventions = [
        "product_photo_pro",
        "review_system",
        "certification_display",
        "producer_story",
    ]

    # CVR改善の乗数を計算
    cvr_multiplier = perception.calculate_purchase_intent_change(interventions)

    # UX改善効果（カート離脱回復）
    ux_effect = 1 + (0.70 * 0.25 * 0.5)  # 離脱70% × 回復25% × 購買50%
    cvr_multiplier *= ux_effect

    improved_cvr = baseline_cvr * cvr_multiplier
    improved_customers = int(total_traffic * improved_cvr)

    # リピート率向上（同梱カード効果）
    improved_repeat_rate = 0.20

    improved_revenue = revenue.calculate_annual_revenue(
        customers=improved_customers,
        repeat_rate=improved_repeat_rate
    )

    print(f"\n[Improved (Full Package)]")
    print(f"  CVR: {improved_cvr*100:.4f}% (x{cvr_multiplier:.2f})")
    print(f"  Customers: {improved_customers:,}")
    print(f"  Sales: {improved_revenue['total_revenue']:,.0f} yen")
    print(f"  Gross Profit: {improved_revenue['gross_profit']:,.0f} yen")

    improvement_pct = (improved_revenue['total_revenue'] / baseline_revenue['total_revenue'] - 1) * 100
    print(f"\n  Improvement: +{improvement_pct:.1f}%")

    # グラフ出力
    plot_baseline_vs_improved(
        baseline_revenue,
        improved_revenue,
        str(FIGURES_DIR / "01_baseline_vs_improved.png")
    )
    print(f"\nGraph saved: 01_baseline_vs_improved.png")

    return {
        "baseline": baseline_revenue,
        "improved": improved_revenue,
        "cvr_baseline": baseline_cvr,
        "cvr_improved": improved_cvr,
        "cvr_multiplier": cvr_multiplier,
        "improvement_pct": improvement_pct,
    }


def run_sim2_funnel_analysis() -> Dict:
    """
    Sim2: 障壁クリアのファネル可視化
    """
    print("\n" + "="*60)
    print("Sim2: Funnel Analysis (Barrier Clear)")
    print("="*60)

    conversion = ConversionModel()

    # ベースラインファネル
    baseline_funnel = conversion.get_funnel_data(improved=False)
    baseline_final_cvr = baseline_funnel[-1]['rate']

    print("\n[Baseline Funnel]")
    for stage in baseline_funnel:
        print(f"  {stage['name']}: {stage['rate']*100:.4f}%")

    # 改善後ファネル
    improved_funnel = conversion.get_funnel_data(improved=True)
    improved_final_cvr = improved_funnel[-1]['rate']

    print("\n[Improved Funnel]")
    for stage in improved_funnel:
        print(f"  {stage['name']}: {stage['rate']*100:.4f}%")

    print(f"\nFinal CVR: {baseline_final_cvr*100:.4f}% -> {improved_final_cvr*100:.4f}%")
    print(f"Improvement: x{improved_final_cvr/baseline_final_cvr:.1f}")

    # グラフ出力
    plot_funnel_comparison(
        baseline_funnel,
        improved_funnel,
        str(FIGURES_DIR / "02_funnel_comparison.png")
    )
    print(f"\nGraph saved: 02_funnel_comparison.png")

    return {
        "baseline_funnel": baseline_funnel,
        "improved_funnel": improved_funnel,
        "baseline_final_cvr": baseline_final_cvr,
        "improved_final_cvr": improved_final_cvr,
    }


def run_sim3_wtp_sensitivity() -> Dict:
    """
    Sim3: WTP感度分析
    """
    print("\n" + "="*60)
    print("Sim3: WTP Sensitivity Analysis")
    print("="*60)

    perception = PerceptionModel()
    base_wtp = 5000

    # シナリオ定義
    scenarios_def = [
        ("A: Baseline", []),
        ("B: Photo Only", ["product_photo_pro"]),
        ("C: Photo+Review", ["product_photo_pro", "review_system"]),
        ("D: Full Package", ["product_photo_pro", "review_system", "insert_card_unboxing", "producer_story"]),
    ]

    scenarios = []
    for name, interventions in scenarios_def:
        wtp_data = perception.calculate_wtp_change(interventions, base_wtp)
        scenario = {
            "name": name,
            "interventions": interventions,
            **wtp_data
        }
        scenarios.append(scenario)

        print(f"\n[{name}]")
        print(f"  Stated WTP: {wtp_data['wtp_stated']:,.0f} yen (+{wtp_data['stated_change_pct']:.1f}%)")
        print(f"  Revealed WTP (Est.): {wtp_data['wtp_revealed']:,.0f} yen")

    # グラフ出力
    plot_wtp_sensitivity(
        scenarios,
        str(FIGURES_DIR / "03_wtp_sensitivity.png")
    )
    print(f"\nGraph saved: 03_wtp_sensitivity.png")

    return {"scenarios": scenarios}


def run_sim4_ltv_model() -> Dict:
    """
    Sim4: LTV・リピートモデル
    """
    print("\n" + "="*60)
    print("Sim4: LTV and Repeat Model")
    print("="*60)

    revenue = RevenueModel(avg_order_value=7500, margin_rate=0.15)

    # LTVシナリオ計算
    ltv_data = revenue.calculate_ltv_scenarios()

    print("\n[LTV Scenarios]")
    for scenario_id, data in ltv_data.items():
        print(f"\n  {data['name']} (Repeat Rate: {data['repeat_rate']*100:.0f}%)")
        print(f"    Avg Purchases: {data['avg_purchases']:.2f}")
        print(f"    Lifetime Revenue: {data['lifetime_revenue']:,.0f} yen")
        print(f"    LTV: {data['ltv']:,.0f} yen")

    # 累積価値計算（年間100人獲得想定）
    customers_per_year = 100
    cumulative_data = {}

    for scenario_id, scenario in LTV_SCENARIOS.items():
        cumulative = revenue.calculate_cumulative_value(
            customers_per_year=customers_per_year,
            repeat_rate=scenario.repeat_rate,
            years=3
        )
        cumulative_data[scenario_id] = cumulative

    print(f"\n[Cumulative Value (3 Years, {customers_per_year} customers/year)]")
    for scenario_id, data in cumulative_data.items():
        final = data[-1]
        print(f"  {ltv_data[scenario_id]['name']}: {final['cumulative_profit']:,.0f} yen profit")

    # グラフ出力
    plot_ltv_scenarios(
        ltv_data,
        cumulative_data,
        str(FIGURES_DIR / "04_ltv_scenarios.png")
    )
    print(f"\nGraph saved: 04_ltv_scenarios.png")

    return {
        "ltv_data": ltv_data,
        "cumulative_data": cumulative_data,
    }


def run_sim5_target_achievement() -> Dict:
    """
    Sim5: 10百万円達成シナリオ
    """
    print("\n" + "="*60)
    print("Sim5: 10M Yen Achievement Scenario")
    print("="*60)

    revenue = RevenueModel(avg_order_value=7500, margin_rate=0.15)

    # 目標設定
    target_sales = 10_000_000
    target_profit = 1_500_000
    available_traffic = 550_000  # メルマガ400k + O2O 50k + SNS 100k

    print(f"\n[Target]")
    print(f"  Annual Sales: {target_sales:,} yen")
    print(f"  Gross Profit: {target_profit:,} yen")
    print(f"  Available Traffic: {available_traffic:,}")

    # 必要条件の逆算
    requirements = revenue.reverse_calculate_requirements(
        target_sales=target_sales,
        target_profit=target_profit,
        available_traffic=available_traffic
    )

    print(f"\n[Scenarios to Achieve Target]")
    for scenario in requirements['scenarios']:
        print(f"\n  {scenario['name']}:")
        print(f"    Required CVR: {scenario['required_cvr_pct']:.2f}%")
        print(f"    Required Customers: {scenario['required_customers']:,.0f}")
        print(f"    Avg Price: {scenario['avg_price']:,.0f} yen")
        print(f"    Repeat Rate: {scenario['repeat_rate']*100:.0f}%")
        print(f"    Feasibility: {scenario['feasibility']}")

    # グラフ出力
    plot_target_achievement(
        requirements,
        str(FIGURES_DIR / "05_target_achievement.png")
    )
    print(f"\nGraph saved: 05_target_achievement.png")

    return requirements


def generate_report(results: Dict) -> str:
    """
    シミュレーションレポートを生成

    Args:
        results: 全シミュレーション結果

    Returns:
        レポートのMarkdown文字列
    """
    report = f"""# EEZO消費者行動×収益シミュレーション レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## エグゼクティブサマリー

本シミュレーションでは、EEZOの知覚品質向上施策がWTP・CVR・LTVに与える影響をモデル化し、
売上10百万円達成への道筋を分析しました。

### 主要な発見

1. **現状のCVRギャップ**: メルマガCVR 0.04% は業界平均 2-3% の約50-75倍低い
2. **施策投入による改善余地**: フル施策で CVR x{results['sim1']['cvr_multiplier']:.1f} の改善が見込める
3. **LTV向上の重要性**: リピート率を10%→30%に改善することでLTVが約29%向上
4. **目標達成の実現可能性**: バランス型アプローチで達成可能性「高」

---

## Sim1: ベースライン vs 改善後の比較

### 結果

| 指標 | 現状 | 改善後 | 改善率 |
|------|------|--------|--------|
| CVR | {results['sim1']['cvr_baseline']*100:.4f}% | {results['sim1']['cvr_improved']*100:.4f}% | x{results['sim1']['cvr_multiplier']:.1f} |
| 年間売上 | {results['sim1']['baseline']['total_revenue']:,.0f}円 | {results['sim1']['improved']['total_revenue']:,.0f}円 | +{results['sim1']['improvement_pct']:.1f}% |
| 粗利 | {results['sim1']['baseline']['gross_profit']:,.0f}円 | {results['sim1']['improved']['gross_profit']:,.0f}円 | +{results['sim1']['improvement_pct']:.1f}% |

### 施策効果の内訳

- **商品写真強化**: CVR +12.5%
- **レビューシステム**: 購買意図 +30%（CVR換算 +15.3%）
- **認証表示**: 信頼性向上 +10%
- **生産者ストーリー**: 品質知覚 +15%
- **UX改善**: カート離脱回復 +8.75%

![ベースライン vs 改善後](figures/01_baseline_vs_improved.png)

---

## Sim2: 購買ファネル分析

### ファネル各段階の転換率

| 段階 | 障壁 | 現状 | 改善後 | 施策 |
|------|------|------|--------|------|
| 認知→興味 | 「何これ？」 | 30% | 45% | 商品写真強化 |
| 興味→検討 | 「信頼できる？」 | 15% | 25% | レビュー・認証 |
| 検討→購買 | 「高い/面倒」 | 0.9% | 3.6% | UX改善・送料明示 |

### 最終CVR

- **現状**: {results['sim2']['baseline_final_cvr']*100:.4f}%
- **改善後**: {results['sim2']['improved_final_cvr']*100:.4f}%
- **改善倍率**: x{results['sim2']['improved_final_cvr']/results['sim2']['baseline_final_cvr']:.1f}

![ファネル比較](figures/02_funnel_comparison.png)

---

## Sim3: WTP感度分析

### シナリオ別WTP

| シナリオ | 施策 | 表明WTP | 顕示WTP推計 | 変化率 |
|----------|------|---------|-------------|--------|
"""

    for s in results['sim3']['scenarios']:
        report += f"| {s['name']} | {', '.join(s['interventions']) if s['interventions'] else '-'} | {s['wtp_stated']:,.0f}円 | {s['wtp_revealed']:,.0f}円 | +{s['stated_change_pct']:.1f}% |\n"

    report += f"""
### 注意点

- 表明WTP（アンケート回答）の55%が実際の支払い（顕示WTP）として現れる
- 開封体験は57%の顧客でWTP上昇、14%で低下の両刃の剣

![WTP感度分析](figures/03_wtp_sensitivity.png)

---

## Sim4: LTV・リピートモデル

### シナリオ別LTV

| シナリオ | リピート率 | 平均購入回数 | LTV |
|----------|-----------|-------------|-----|
| 現状 | 10% | 1.11回 | {results['sim4']['ltv_data']['baseline']['ltv']:,.0f}円 |
| 同梱カードあり | 20% | 1.25回 | {results['sim4']['ltv_data']['with_card']['ltv']:,.0f}円 |
| 同梱カード+開封体験 | 30% | 1.43回 | {results['sim4']['ltv_data']['with_unboxing']['ltv']:,.0f}円 |

### 累積顧客価値（3年間、年100人獲得）

- **現状**: {results['sim4']['cumulative_data']['baseline'][-1]['cumulative_profit']:,.0f}円
- **同梱カードあり**: {results['sim4']['cumulative_data']['with_card'][-1]['cumulative_profit']:,.0f}円
- **同梱カード+開封体験**: {results['sim4']['cumulative_data']['with_unboxing'][-1]['cumulative_profit']:,.0f}円

![LTVシナリオ](figures/04_ltv_scenarios.png)

---

## Sim5: 10百万円達成シナリオ

### 目標設定

- **年間売上目標**: 10,000,000円
- **粗利目標**: 1,500,000円
- **利用可能トラフィック**: 550,000（メルマガ400k + O2O 50k + SNS 100k）

### 達成シナリオ

| シナリオ | 必要CVR | 必要顧客数 | 平均単価 | リピート率 | 実現可能性 |
|----------|---------|-----------|---------|-----------|-----------|
"""

    for s in results['sim5']['scenarios']:
        report += f"| {s['name']} | {s['required_cvr_pct']:.2f}% | {s['required_customers']:,.0f}人 | {s['avg_price']:,.0f}円 | {s['repeat_rate']*100:.0f}% | {s['feasibility']} |\n"

    report += f"""
### 推奨アプローチ: バランス型

1. **CVR改善**: 0.04% → 0.8%程度を目指す（x20改善）
2. **単価向上**: 平均7,500円 → 9,000円（+20%）
3. **リピート率向上**: 10% → 20%（同梱カード導入）

![目標達成シナリオ](figures/05_target_achievement.png)

---

## 結論と推奨事項

### 達成確率評価

| 項目 | 評価 | 理由 |
|------|------|------|
| CVR改善 | **中** | 業界平均との大きなギャップあり、改善余地は大きいが実行難易度も高い |
| WTP向上 | **中** | 施策効果は実証済みだが、顧客セグメントによる差異に注意 |
| LTV向上 | **高** | 同梱カードは低コストかつ効果が明確 |
| 10百万円達成 | **中** | バランス型アプローチで達成可能だが、複数施策の同時実行が必要 |

### 優先施策（ROI順）

1. **同梱カード導入**（コスト低・効果確実）
2. **商品写真強化**（一度の投資で継続効果）
3. **レビューシステム構築**（信頼構築の基盤）
4. **UX改善（Shopify移行）**（投資大だが効果も大）

### 次のステップ

1. 同梱カードの試作・テスト（1ヶ月）
2. 商品写真のプロ撮影（2週間）
3. レビューシステムの導入検討（1ヶ月）
4. A/Bテストによる効果検証（3ヶ月）

---

*このレポートはシミュレーションに基づく推計であり、実際の結果は市場環境や実行品質により異なる可能性があります。*
"""

    return report


def main():
    """メイン実行関数"""
    print("="*60)
    print("EEZO Consumer Behavior x Revenue Simulation")
    print("="*60)

    # ディレクトリ作成
    ensure_dirs()

    # 全シミュレーション実行
    results = {}

    results['sim1'] = run_sim1_baseline_vs_improved()
    results['sim2'] = run_sim2_funnel_analysis()
    results['sim3'] = run_sim3_wtp_sensitivity()
    results['sim4'] = run_sim4_ltv_model()
    results['sim5'] = run_sim5_target_achievement()

    # レポート生成
    print("\n" + "="*60)
    print("Generating Report...")
    print("="*60)

    report = generate_report(results)
    report_path = OUTPUT_DIR / "simulation_report.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved: {report_path}")

    print("\n" + "="*60)
    print("All simulations completed!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - figures/01_baseline_vs_improved.png")
    print("  - figures/02_funnel_comparison.png")
    print("  - figures/03_wtp_sensitivity.png")
    print("  - figures/04_ltv_scenarios.png")
    print("  - figures/05_target_achievement.png")
    print("  - simulation_report.md")

    return results


if __name__ == "__main__":
    main()
