"""
統合販売シミュレーションモデル

新しいアプローチ:
1. オンライン販売（メルマガ+SNS）の連続的な成長曲線を設計
2. Q単位の目標売上との乖離を計算
3. 乖離をO2Oで埋めるために必要な便数を逆算
4. 全体の整合性を検証
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class MonthlyOnlineProjection:
    """月次オンライン販売予測"""
    month: int
    newsletter_customers: int
    sns_customers: int
    repeat_customers: int
    total_online_customers: int
    online_revenue: int
    cumulative_revenue: int


@dataclass
class QuarterlyTarget:
    """四半期目標"""
    quarter: int
    target_revenue: int
    target_customers: int


@dataclass
class O2ORequirement:
    """O2O必要量"""
    customers_needed: int
    contact_points_needed: int  # 必要接点数
    voyages_per_weekend_day: float  # 週末1日あたりの便数
    scan_rate: float
    conversion_rate: float


class IntegratedSalesModel:
    """統合販売シミュレーションモデル"""

    def __init__(
        self,
        avg_unit_price: int = 7500,
        margin_rate: float = 0.15,
        repeat_rate: float = 0.20,
    ):
        self.avg_unit_price = avg_unit_price
        self.margin_rate = margin_rate
        self.repeat_rate = repeat_rate

        # オンラインチャネルの初期CVR
        self.newsletter_cvr_initial = 0.025  # 2.5%（改善後目標）
        self.sns_cvr_initial = 0.005  # 0.5%

        # O2Oパラメータ
        self.o2o_passengers_per_voyage = 400  # 1便あたり乗客数
        self.o2o_scan_rate = 0.07  # QRスキャン率 7%
        self.o2o_conversion_rate = 0.11  # スキャン→購入 11%

    def project_online_continuous_growth(
        self,
        months: List[int] = None,
        newsletter_clicks_per_month: int = 910,
        sns_clicks_start: int = 5000,  # SNS初期クリック数
        sns_growth_rate: float = 0.08,  # 月次成長率
        cvr_improvement_monthly: float = 0.02,  # CVR月次改善率
        o2o_customers_q3: int = 0,  # Q3のO2O顧客数（Q4リピート計算用）
    ) -> List[MonthlyOnlineProjection]:
        """
        オンライン販売の連続的成長をモデル化

        Args:
            months: 対象月（4-12月）
            newsletter_clicks_per_month: メルマガ月次クリック数
            sns_clicks_start: SNS初月クリック数
            sns_growth_rate: SNS月次成長率
            cvr_improvement_monthly: CVR月次改善率
            o2o_customers_q3: Q3のO2O顧客数（Q4リピート計算用）

        Returns:
            月次オンライン予測リスト
        """
        if months is None:
            months = list(range(4, 13))  # 4月〜12月

        projections = []
        cumulative_revenue = 0
        cumulative_online_customers = 0  # オンライン新規のみ

        # CVR推移（改善効果を反映）
        newsletter_cvr = self.newsletter_cvr_initial
        sns_cvr = self.sns_cvr_initial

        for i, month in enumerate(months):
            # CVR改善（時間経過とともに向上）
            if i > 0:
                newsletter_cvr = min(0.035, newsletter_cvr * (1 + cvr_improvement_monthly))
                sns_cvr = min(0.012, sns_cvr * (1 + cvr_improvement_monthly))

            # メルマガ顧客（安定した配信）
            newsletter_customers = int(newsletter_clicks_per_month * newsletter_cvr)

            # SNS顧客（成長曲線：Q4に向けて増加）
            month_index = i
            # Q4に集中投下するため、10-12月は加速
            if month >= 10:
                # Q4: SNS広告集中投下（予算の70%をQ4に）
                sns_multiplier = 3.0 + (month - 10) * 0.8  # 10月3.0倍、11月3.8倍、12月4.6倍
            elif month >= 7:
                sns_multiplier = 0.2  # Q3はO2Oに注力、SNSは最小限
            else:
                sns_multiplier = 1.0  # Q2は通常

            sns_clicks = int(sns_clicks_start * ((1 + sns_growth_rate) ** month_index) * sns_multiplier)
            sns_customers = int(sns_clicks * sns_cvr)

            # リピート顧客の計算
            repeat_customers = 0

            # オンライン顧客からのリピート
            if cumulative_online_customers > 0:
                monthly_repeat_base = self.repeat_rate / 9  # 9ヶ月で分散
                if month == 12:
                    monthly_repeat_base *= 2.5  # 歳暮需要
                elif month >= 10:
                    monthly_repeat_base *= 1.5
                repeat_customers += int(cumulative_online_customers * monthly_repeat_base)

            # Q3 O2O顧客からのQ4リピート（重要！）
            if month >= 10 and o2o_customers_q3 > 0:
                # O2O顧客は歳暮需要でリピート率が高い
                o2o_repeat_rate = 0.25 if month == 12 else (0.15 if month == 11 else 0.10)
                repeat_customers += int(o2o_customers_q3 * o2o_repeat_rate)

            total_customers = newsletter_customers + sns_customers + repeat_customers
            monthly_revenue = total_customers * self.avg_unit_price
            cumulative_revenue += monthly_revenue
            cumulative_online_customers += (newsletter_customers + sns_customers)  # 新規のみ累積

            projections.append(MonthlyOnlineProjection(
                month=month,
                newsletter_customers=newsletter_customers,
                sns_customers=sns_customers,
                repeat_customers=repeat_customers,
                total_online_customers=total_customers,
                online_revenue=monthly_revenue,
                cumulative_revenue=cumulative_revenue,
            ))

        return projections

    def calculate_quarterly_gap(
        self,
        online_projections: List[MonthlyOnlineProjection],
        quarterly_targets: List[QuarterlyTarget],
    ) -> Dict[int, Dict]:
        """
        四半期目標とオンライン売上の乖離を計算

        Args:
            online_projections: オンライン予測
            quarterly_targets: 四半期目標

        Returns:
            四半期別の乖離情報
        """
        gaps = {}

        for qt in quarterly_targets:
            q = qt.quarter
            q_months = {
                1: [1, 2, 3],
                2: [4, 5, 6],
                3: [7, 8, 9],
                4: [10, 11, 12],
            }[q]

            # 該当四半期のオンライン売上合計
            q_online_revenue = sum(
                p.online_revenue for p in online_projections
                if p.month in q_months
            )
            q_online_customers = sum(
                p.total_online_customers for p in online_projections
                if p.month in q_months
            )

            gap_revenue = qt.target_revenue - q_online_revenue
            gap_customers = max(0, int(gap_revenue / self.avg_unit_price))

            gaps[q] = {
                "quarter": q,
                "target_revenue": qt.target_revenue,
                "online_revenue": q_online_revenue,
                "gap_revenue": gap_revenue,
                "online_customers": q_online_customers,
                "gap_customers": gap_customers,
                "needs_o2o": gap_revenue > 0,
            }

        return gaps

    def calculate_o2o_requirements(
        self,
        customers_needed: int,
        operating_weeks: int = 13,  # Q3: 13週
        weekend_days: int = 2,
    ) -> O2ORequirement:
        """
        必要なO2O便数を逆算

        Args:
            customers_needed: 必要顧客数
            operating_weeks: 運航週数
            weekend_days: 週あたりの運航日数

        Returns:
            O2O必要量
        """
        # 逆算：必要顧客数 → 必要スキャン数 → 必要接点数 → 必要便数
        scans_needed = customers_needed / self.o2o_conversion_rate
        contact_points_needed = scans_needed / self.o2o_scan_rate

        total_operating_days = operating_weeks * weekend_days
        contact_points_per_day = contact_points_needed / total_operating_days
        voyages_per_day = contact_points_per_day / self.o2o_passengers_per_voyage

        return O2ORequirement(
            customers_needed=customers_needed,
            contact_points_needed=int(contact_points_needed),
            voyages_per_weekend_day=voyages_per_day,
            scan_rate=self.o2o_scan_rate,
            conversion_rate=self.o2o_conversion_rate,
        )

    def verify_consistency(
        self,
        online_projections: List[MonthlyOnlineProjection],
        o2o_customers: int,
        annual_target: int = 10_000_000,
    ) -> Dict:
        """
        全体の整合性を検証

        Args:
            online_projections: オンライン予測
            o2o_customers: O2O顧客数
            annual_target: 年間売上目標

        Returns:
            整合性検証結果
        """
        # オンライン合計
        total_online_revenue = sum(p.online_revenue for p in online_projections)
        total_online_customers = sum(
            p.newsletter_customers + p.sns_customers
            for p in online_projections
        )
        total_repeat = sum(p.repeat_customers for p in online_projections)

        # O2O売上
        o2o_revenue = o2o_customers * self.avg_unit_price

        # 合計
        total_revenue = total_online_revenue + o2o_revenue
        total_new_customers = total_online_customers + o2o_customers
        total_orders = total_new_customers + total_repeat

        # 目標との差分（正=超過、負=不足）
        gap_from_target = total_revenue - annual_target
        gap_percentage = gap_from_target / annual_target * 100

        # 整合性チェック
        checks = {
            "revenue_within_tolerance": abs(gap_percentage) < 10,  # ±10%以内
            "revenue_exceeds_target": total_revenue >= annual_target,
            "implied_avg_price": total_revenue / total_orders if total_orders > 0 else 0,
            "implied_repeat_rate": total_repeat / total_new_customers if total_new_customers > 0 else 0,
            "gross_profit": total_revenue * self.margin_rate,
        }

        return {
            "total_revenue": total_revenue,
            "online_revenue": total_online_revenue,
            "o2o_revenue": o2o_revenue,
            "total_new_customers": total_new_customers,
            "total_repeat_orders": total_repeat,
            "total_orders": total_orders,
            "annual_target": annual_target,
            "gap_from_target": gap_from_target,  # 正の値=超過
            "gap_percentage": gap_percentage,
            "checks": checks,
        }


class FiveYearPlanModel:
    """5か年計画モデル（BtoB含む）"""

    def __init__(self):
        # 基本パラメータ
        self.base_margin_rate = 0.15

    def simulate_five_year_plan(self) -> List[Dict]:
        """
        5か年計画をシミュレーション（BtoB含む）

        Returns:
            年次計画リスト
        """
        plans = []

        # Y1: 基盤構築（現行計画）
        y1 = {
            "year": 1,
            "phase": "基盤構築",
            "btoc_revenue": 10_000_000,
            "btob_revenue": 0,
            "marketplace_revenue": 0,
            "total_revenue": 10_000_000,
            "margin_rate": 0.15,
            "gross_profit": 1_500_000,
            "new_customers": 1067,
            "btob_accounts": 0,
            "key_initiatives": [
                "Shopify移行",
                "レビューシステム導入",
                "同梱カード開始",
                "O2O顧客獲得",
            ],
            "risks": ["O2O不振", "Q4歳暮需要不振"],
        }
        plans.append(y1)

        # Y2: 効率化 + BtoB開始
        y2 = {
            "year": 2,
            "phase": "効率化 + BtoB開始",
            "btoc_revenue": 15_000_000,  # BtoC成長
            "btob_revenue": 5_000_000,   # 飲食店・小売店への卸開始
            "marketplace_revenue": 0,
            "total_revenue": 20_000_000,
            "margin_rate": 0.18,  # BtoBは粗利率低めだが安定
            "gross_profit": 3_600_000,
            "new_customers": 1800,
            "btob_accounts": 20,  # 20社取引開始
            "key_initiatives": [
                "大阪物流拠点構築",
                "BtoB営業チーム立上げ",
                "飲食店・居酒屋への卸売開始",
                "定期購入プラン導入",
            ],
            "risks": ["BtoB営業リソース不足", "物流コスト増"],
        }
        plans.append(y2)

        # Y3: ブランド確立 + BtoB拡大
        y3 = {
            "year": 3,
            "phase": "ブランド確立 + BtoB拡大",
            "btoc_revenue": 20_000_000,
            "btob_revenue": 15_000_000,  # ホテル・旅館への展開
            "marketplace_revenue": 0,
            "total_revenue": 35_000_000,
            "margin_rate": 0.20,
            "gross_profit": 7_000_000,
            "new_customers": 2500,
            "btob_accounts": 80,  # 80社へ拡大
            "key_initiatives": [
                "ホテル・旅館への提案開始",
                "企業ギフト（お歳暮）受注",
                "プラットフォーム開発着手",
                "レビュー3,000件達成",
            ],
            "risks": ["競合参入", "BtoB与信管理"],
        }
        plans.append(y3)

        # Y4: 大規模拡張 + マーケットプレイス
        y4 = {
            "year": 4,
            "phase": "大規模拡張 + MP化",
            "btoc_revenue": 35_000_000,
            "btob_revenue": 40_000_000,  # 百貨店・量販店への展開
            "marketplace_revenue": 25_000_000,  # 他事業者出店
            "total_revenue": 100_000_000,
            "margin_rate": 0.18,
            "gross_profit": 18_000_000,
            "new_customers": 8000,
            "btob_accounts": 200,
            "key_initiatives": [
                "商品拡充（取扱ブランド1社→10社）",
                "広告大規模投下（8M）",
                "マーケットプレイス本格稼働",
                "百貨店・量販店への納入開始",
            ],
            "risks": ["投資回収遅延", "品質管理"],
        }
        plans.append(y4)

        # Y5: 収穫期
        y5 = {
            "year": 5,
            "phase": "収穫期",
            "btoc_revenue": 50_000_000,
            "btob_revenue": 60_000_000,
            "marketplace_revenue": 40_000_000,  # 手数料収入
            "total_revenue": 150_000_000,
            "margin_rate": 0.20,
            "gross_profit": 30_000_000,
            "new_customers": 12000,
            "btob_accounts": 350,
            "key_initiatives": [
                "複数収益源の安定化",
                "海外展開検討開始",
                "サブスクリプションモデル本格化",
                "自社ブランド商品開発",
            ],
            "risks": ["市場飽和", "人材確保"],
        }
        plans.append(y5)

        return plans

    def calculate_btob_details(self) -> List[Dict]:
        """
        BtoB詳細計画（5か年計画の目標値に整合）

        5か年計画のBtoB目標:
        - Y2: ¥5M
        - Y3: ¥15M
        - Y4: ¥40M
        - Y5: ¥60M

        Returns:
            BtoB年次計画
        """
        btob_plans = []

        # 年度別のBtoB目標売上に基づいて逆算
        btob_targets = {
            2: 5_000_000,
            3: 15_000_000,
            4: 40_000_000,
            5: 60_000_000,
        }

        # 顧客セグメント別の構成比と特性
        # 各年度で売上目標を達成するために必要なアカウント数を逆算
        segments = {
            "restaurants": {
                "name": "飲食店・居酒屋",
                "avg_order_value": 30_000,  # 月2回の発注、単価控えめ
                "orders_per_year": 18,  # 月1.5回
                "margin_rate": 0.12,
                "revenue_share": {2: 1.0, 3: 0.50, 4: 0.35, 5: 0.30},  # 年度別構成比
            },
            "hotels": {
                "name": "ホテル・旅館",
                "avg_order_value": 80_000,  # 月1回、中規模注文
                "orders_per_year": 10,
                "margin_rate": 0.10,
                "revenue_share": {2: 0, 3: 0.25, 4: 0.25, 5: 0.25},
            },
            "retail": {
                "name": "百貨店・量販店",
                "avg_order_value": 300_000,  # 大口、2ヶ月に1回
                "orders_per_year": 6,
                "margin_rate": 0.08,
                "revenue_share": {2: 0, 3: 0, 4: 0.20, 5: 0.25},
            },
            "corporate_gifts": {
                "name": "法人ギフト",
                "avg_order_value": 150_000,  # 中元・歳暮
                "orders_per_year": 2,
                "margin_rate": 0.15,
                "revenue_share": {2: 0, 3: 0.25, 4: 0.20, 5: 0.20},
            },
        }

        for year in range(2, 6):
            target_revenue = btob_targets[year]
            year_plan = {
                "year": year,
                "target_revenue": target_revenue,
                "segments": {},
                "total_accounts": 0,
                "total_revenue": 0,
                "total_profit": 0,
            }

            for seg_id, seg in segments.items():
                share = seg["revenue_share"].get(year, 0)
                if share > 0:
                    segment_revenue = int(target_revenue * share)
                    revenue_per_account = seg["avg_order_value"] * seg["orders_per_year"]
                    accounts = max(1, int(segment_revenue / revenue_per_account))

                    actual_revenue = accounts * revenue_per_account
                    actual_profit = int(actual_revenue * seg["margin_rate"])

                    year_plan["segments"][seg_id] = {
                        "name": seg["name"],
                        "accounts": accounts,
                        "annual_revenue": actual_revenue,
                        "annual_profit": actual_profit,
                    }
                    year_plan["total_accounts"] += accounts
                    year_plan["total_revenue"] += actual_revenue
                    year_plan["total_profit"] += actual_profit

            btob_plans.append(year_plan)

        return btob_plans

    def verify_five_year_consistency(self, plans: List[Dict]) -> Dict:
        """
        5か年計画の整合性検証

        Args:
            plans: 年次計画リスト

        Returns:
            整合性検証結果
        """
        checks = []

        # 成長率の妥当性チェック
        for i in range(1, len(plans)):
            prev = plans[i-1]
            curr = plans[i]
            growth_rate = (curr["total_revenue"] - prev["total_revenue"]) / prev["total_revenue"]

            # 成長率が200%を超える場合は警告
            check = {
                "year": curr["year"],
                "growth_rate": growth_rate,
                "is_reasonable": growth_rate <= 2.0,
                "warning": None if growth_rate <= 2.0 else f"Y{curr['year']}の成長率{growth_rate*100:.0f}%は高すぎる可能性",
            }
            checks.append(check)

        # 累積計算
        cumulative_revenue = sum(p["total_revenue"] for p in plans)
        cumulative_profit = sum(p["gross_profit"] for p in plans)
        cumulative_investment = sum([
            500_000,    # Y1
            3_000_000,  # Y2（物流拠点）
            5_000_000,  # Y3（プラットフォーム）
            15_000_000, # Y4（拡張投資）
            5_000_000,  # Y5（運用改善）
        ])

        return {
            "checks": checks,
            "cumulative_revenue": cumulative_revenue,
            "cumulative_profit": cumulative_profit,
            "cumulative_investment": cumulative_investment,
            "roi": (cumulative_profit - cumulative_investment) / cumulative_investment if cumulative_investment > 0 else 0,
            "payback_achieved": cumulative_profit > cumulative_investment,
        }


def run_integrated_simulation():
    """
    統合シミュレーションを実行（年間目標ピッタリに調整）
    """
    print("=" * 60)
    print("EEZO 統合販売シミュレーション（目標調整版）")
    print("=" * 60)

    # モデル初期化
    model = IntegratedSalesModel()
    annual_target = 10_000_000

    print("\n【Step 1】オンラインのみの年間売上を予測")
    print("-" * 40)

    # Step 1: O2Oなしでオンライン予測を計算
    online_projections_no_o2o = model.project_online_continuous_growth(o2o_customers_q3=0)
    online_only_revenue = sum(p.online_revenue for p in online_projections_no_o2o)

    print(f"  オンラインのみ年間売上: ¥{online_only_revenue:,}")
    print(f"  年間目標: ¥{annual_target:,}")
    print(f"  乖離: ¥{annual_target - online_only_revenue:,}")

    print("\n【Step 2】O2O顧客数を反復計算で最適化")
    print("-" * 40)

    # Step 2: 反復計算でO2O顧客数を最適化
    # O2O顧客はQ3で獲得し、Q4でリピートも発生するため、
    # 単純な乖離÷単価ではなく、リピート効果も考慮して調整

    best_o2o = 0
    best_gap = float('inf')

    for o2o_test in range(0, 500, 10):  # 0〜500人を10人刻みでテスト
        projections = model.project_online_continuous_growth(o2o_customers_q3=o2o_test)
        online_revenue = sum(p.online_revenue for p in projections)
        o2o_revenue = o2o_test * model.avg_unit_price
        total_revenue = online_revenue + o2o_revenue
        gap = abs(total_revenue - annual_target)

        if gap < best_gap:
            best_gap = gap
            best_o2o = o2o_test

    # 微調整（1人刻み）
    for o2o_test in range(max(0, best_o2o - 15), best_o2o + 15):
        projections = model.project_online_continuous_growth(o2o_customers_q3=o2o_test)
        online_revenue = sum(p.online_revenue for p in projections)
        o2o_revenue = o2o_test * model.avg_unit_price
        total_revenue = online_revenue + o2o_revenue
        gap = abs(total_revenue - annual_target)

        if gap < best_gap:
            best_gap = gap
            best_o2o = o2o_test

    o2o_customers_needed = best_o2o
    print(f"  最適O2O顧客数: {o2o_customers_needed}人")
    print(f"  目標との乖離: ¥{best_gap:,}")

    print("\n【Step 3】最終予測（O2Oリピート反映）")
    print("-" * 40)

    # Step 3: 最適化されたO2O顧客数で再計算
    online_projections = model.project_online_continuous_growth(o2o_customers_q3=o2o_customers_needed)

    print(f"{'月':>3} {'メルマガ':>8} {'SNS':>8} {'リピート':>8} {'合計':>8} {'売上':>12} {'累計':>12}")
    print("-" * 70)
    for p in online_projections:
        print(f"{p.month:>3}月 {p.newsletter_customers:>8} {p.sns_customers:>8} "
              f"{p.repeat_customers:>8} {p.total_online_customers:>8} "
              f"¥{p.online_revenue:>10,} ¥{p.cumulative_revenue:>10,}")

    # 四半期目標との比較（参考）
    quarterly_targets = [
        QuarterlyTarget(quarter=2, target_revenue=1_000_000, target_customers=133),
        QuarterlyTarget(quarter=3, target_revenue=3_800_000, target_customers=507),
        QuarterlyTarget(quarter=4, target_revenue=5_200_000, target_customers=693),
    ]

    print("\n【4】四半期別売上（O2O含む）")
    print("-" * 40)

    gaps = model.calculate_quarterly_gap(online_projections, quarterly_targets)

    # Q3にO2O売上を加算
    o2o_revenue = o2o_customers_needed * model.avg_unit_price

    for q, gap in gaps.items():
        actual_revenue = gap['online_revenue']
        if q == 3:
            actual_revenue += o2o_revenue
        print(f"  Q{q}: オンライン ¥{gap['online_revenue']:,}" +
              (f" + O2O ¥{o2o_revenue:,} = ¥{actual_revenue:,}" if q == 3 else f" = ¥{actual_revenue:,}"))

    # O2O必要量の逆算
    print("\n【5】O2O必要量")
    print("-" * 40)

    o2o_req = model.calculate_o2o_requirements(o2o_customers_needed)
    if o2o_customers_needed > 0:
        print(f"  必要顧客数: {o2o_req.customers_needed}人")
        print(f"  必要接点数: {o2o_req.contact_points_needed:,}人")
        print(f"  週末1日あたり必要便数: {o2o_req.voyages_per_weekend_day:.1f}便")
        print(f"  （スキャン率 {o2o_req.scan_rate*100:.0f}%, CVR {o2o_req.conversion_rate*100:.0f}%）")

        if o2o_req.voyages_per_weekend_day <= 5:
            print("  ✓ 現実的な便数（5便/日以下）")
        elif o2o_req.voyages_per_weekend_day <= 8:
            print("  △ やや多い便数（追加施策要検討）")
        else:
            print("  ✗ 非現実的な便数（オンライン強化が必要）")

    # 全体整合性の検証
    print("\n【6】全体整合性の検証")
    print("-" * 40)

    consistency = model.verify_consistency(online_projections, o2o_customers_needed)

    print(f"  総売上: ¥{consistency['total_revenue']:,}")
    print(f"    - オンライン: ¥{consistency['online_revenue']:,}")
    print(f"    - O2O: ¥{consistency['o2o_revenue']:,}")
    print(f"  目標: ¥{annual_target:,}")

    gap_pct = consistency['gap_percentage']
    if abs(gap_pct) < 1:
        print(f"  ✓ 目標とほぼ一致（乖離 {gap_pct:+.1f}%）")
    elif gap_pct > 0:
        print(f"  目標超過: +¥{consistency['gap_from_target']:,} (+{gap_pct:.1f}%)")
    else:
        print(f"  目標不足: ¥{consistency['gap_from_target']:,} ({gap_pct:.1f}%)")

    print(f"  新規顧客数: {consistency['total_new_customers']}人")
    print(f"  リピート注文数: {consistency['total_repeat_orders']}人")
    print(f"  総注文数: {consistency['total_orders']}人")
    print(f"  暗黙の平均単価: ¥{consistency['checks']['implied_avg_price']:,.0f}")
    print(f"  暗黙のリピート率: {consistency['checks']['implied_repeat_rate']*100:.1f}%")
    print(f"  粗利: ¥{consistency['checks']['gross_profit']:,.0f}")

    # 6. 5か年計画（BtoB含む）
    print("\n" + "=" * 60)
    print("【6】5か年計画シミュレーション（BtoB含む）")
    print("=" * 60)

    five_year_model = FiveYearPlanModel()
    five_year_plans = five_year_model.simulate_five_year_plan()

    print(f"\n{'年':>3} {'フェーズ':>16} {'BtoC':>12} {'BtoB':>12} {'MP':>12} {'合計':>12} {'粗利':>12} {'粗利率':>6}")
    print("-" * 90)

    for plan in five_year_plans:
        print(f"Y{plan['year']:>2} {plan['phase']:>16} "
              f"¥{plan['btoc_revenue']:>10,} ¥{plan['btob_revenue']:>10,} "
              f"¥{plan['marketplace_revenue']:>10,} ¥{plan['total_revenue']:>10,} "
              f"¥{plan['gross_profit']:>10,} {plan['margin_rate']*100:>5.0f}%")

    # BtoB詳細
    print("\n【6-1】BtoB展開詳細")
    print("-" * 40)

    btob_plans = five_year_model.calculate_btob_details()

    for bp in btob_plans:
        print(f"\nY{bp['year']}: 総アカウント数 {bp['total_accounts']}社, 売上 ¥{bp['total_revenue']:,}")
        for seg_id, seg_data in bp['segments'].items():
            print(f"  - {seg_data['name']}: {seg_data['accounts']}社 (¥{seg_data['annual_revenue']:,})")

    # 5か年整合性検証
    print("\n【6-2】5か年計画の整合性検証")
    print("-" * 40)

    consistency_5y = five_year_model.verify_five_year_consistency(five_year_plans)

    print(f"  累積売上: ¥{consistency_5y['cumulative_revenue']:,}")
    print(f"  累積粗利: ¥{consistency_5y['cumulative_profit']:,}")
    print(f"  累積投資: ¥{consistency_5y['cumulative_investment']:,}")
    print(f"  ROI: {consistency_5y['roi']*100:.0f}%")
    print(f"  投資回収: {'達成' if consistency_5y['payback_achieved'] else '未達'}")

    for check in consistency_5y['checks']:
        status = "✓" if check['is_reasonable'] else "⚠"
        print(f"  {status} Y{check['year']} 成長率: {check['growth_rate']*100:.0f}%")
        if check['warning']:
            print(f"      警告: {check['warning']}")

    return {
        "online_projections": online_projections,
        "gaps": gaps,
        "o2o_requirements": o2o_req if o2o_customers_needed > 0 else None,
        "consistency": consistency,
        "five_year_plans": five_year_plans,
        "btob_plans": btob_plans,
    }


if __name__ == "__main__":
    results = run_integrated_simulation()
