"""
Layer2: 行動転換モデル
購買意図 → 実購買（CVR）
※意図→行動の転換率 r=0.51 を適用
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


# 転換率パラメータ
INTENT_TO_BEHAVIOR_RATE = 0.51  # 購買意図→実購買の転換率


@dataclass
class TrafficSource:
    """トラフィックソース"""
    name: str
    annual_volume: int
    cvr_baseline: float
    cvr_industry_avg: float


# トラフィックソース定義
TRAFFIC_SOURCES: Dict[str, TrafficSource] = {
    "email": TrafficSource(
        name="メルマガ",
        annual_volume=400000,
        cvr_baseline=0.0004,
        cvr_industry_avg=0.025
    ),
    "o2o": TrafficSource(
        name="O2O（船内→EC）",
        annual_volume=50000,
        cvr_baseline=0.02,
        cvr_industry_avg=0.03
    ),
    "sns": TrafficSource(
        name="SNS",
        annual_volume=100000,
        cvr_baseline=0.005,
        cvr_industry_avg=0.02
    ),
}


@dataclass
class FunnelStage:
    """ファネル段階"""
    name: str
    conversion_rate_baseline: float
    conversion_rate_improved: float
    barrier: str
    intervention: str


# ファネル定義
FUNNEL_STAGES: List[FunnelStage] = [
    FunnelStage(
        name="認知→興味",
        conversion_rate_baseline=0.30,
        conversion_rate_improved=0.45,
        barrier="何これ？",
        intervention="商品写真強化"
    ),
    FunnelStage(
        name="興味→検討",
        conversion_rate_baseline=0.15,
        conversion_rate_improved=0.25,
        barrier="信頼できる？",
        intervention="レビュー・認証"
    ),
    FunnelStage(
        name="検討→購買",
        conversion_rate_baseline=0.009,  # 0.04% / (30% * 15%)
        conversion_rate_improved=0.036,  # 改善後
        barrier="高い/面倒",
        intervention="UX改善・送料明示"
    ),
]


class ConversionModel:
    """行動転換モデル"""

    def __init__(self):
        self.intent_to_behavior_rate = INTENT_TO_BEHAVIOR_RATE
        self.traffic_sources = TRAFFIC_SOURCES
        self.funnel_stages = FUNNEL_STAGES

    def calculate_cvr_improvement(
        self,
        base_cvr: float,
        intervention_effects: Dict[str, float]
    ) -> float:
        """
        施策によるCVR改善を計算

        Args:
            base_cvr: ベースCVR
            intervention_effects: 施策効果の辞書

        Returns:
            改善後のCVR
        """
        cvr = base_cvr

        # 各施策効果を乗算
        for effect_type, multiplier in intervention_effects.items():
            if effect_type == "cvr_multiplier":
                cvr *= multiplier
            elif effect_type == "purchase_intent_multiplier":
                # 購買意図改善 → 行動転換率を適用
                intent_improvement = multiplier - 1
                behavior_improvement = intent_improvement * self.intent_to_behavior_rate
                cvr *= (1 + behavior_improvement)
            elif effect_type == "cart_recovery_rate":
                # カート回復率（離脱70%のX%を回復）
                cart_abandonment = 0.70
                recovered = cart_abandonment * multiplier
                # CVRへの影響を計算
                cvr *= (1 + recovered * 0.5)  # 回復した分の50%が購買

        return cvr

    def calculate_funnel_conversion(
        self,
        improved: bool = False
    ) -> Dict[str, float]:
        """
        ファネル各段階の転換率を計算

        Args:
            improved: 改善後かどうか

        Returns:
            各段階の転換率と累積転換率
        """
        cumulative = 1.0
        results = {"stages": [], "cumulative_rates": []}

        for stage in self.funnel_stages:
            rate = stage.conversion_rate_improved if improved else stage.conversion_rate_baseline
            cumulative *= rate
            results["stages"].append({
                "name": stage.name,
                "rate": rate,
                "barrier": stage.barrier,
                "intervention": stage.intervention,
            })
            results["cumulative_rates"].append(cumulative)

        results["final_cvr"] = cumulative
        return results

    def calculate_traffic_conversion(
        self,
        cvr_multipliers: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """
        トラフィックソース別の転換を計算

        Args:
            cvr_multipliers: ソース別CVR乗数

        Returns:
            ソース別の転換結果
        """
        if cvr_multipliers is None:
            cvr_multipliers = {}

        results = {}
        total_baseline_customers = 0
        total_improved_customers = 0

        for source_id, source in self.traffic_sources.items():
            multiplier = cvr_multipliers.get(source_id, 1.0)
            cvr_improved = source.cvr_baseline * multiplier

            customers_baseline = source.annual_volume * source.cvr_baseline
            customers_improved = source.annual_volume * cvr_improved

            results[source_id] = {
                "name": source.name,
                "traffic": source.annual_volume,
                "cvr_baseline": source.cvr_baseline,
                "cvr_improved": cvr_improved,
                "customers_baseline": customers_baseline,
                "customers_improved": customers_improved,
                "improvement_pct": (cvr_improved / source.cvr_baseline - 1) * 100,
            }

            total_baseline_customers += customers_baseline
            total_improved_customers += customers_improved

        results["total"] = {
            "customers_baseline": total_baseline_customers,
            "customers_improved": total_improved_customers,
            "improvement_pct": (total_improved_customers / total_baseline_customers - 1) * 100 if total_baseline_customers > 0 else 0,
        }

        return results

    def get_funnel_data(self, improved: bool = False) -> List[Dict]:
        """
        ファネルデータを取得（可視化用）

        Args:
            improved: 改善後かどうか

        Returns:
            ファネルデータのリスト
        """
        stages = [
            {"name": "認知（サイト訪問）", "rate": 1.0},
        ]

        cumulative = 1.0
        for stage in self.funnel_stages:
            rate = stage.conversion_rate_improved if improved else stage.conversion_rate_baseline
            cumulative *= rate

            stage_name = stage.name.split("→")[1] if "→" in stage.name else stage.name
            if stage_name == "興味":
                stage_name = "興味（商品ページ閲覧）"
            elif stage_name == "検討":
                stage_name = "検討（カート追加）"
            elif stage_name == "購買":
                stage_name = "購買（決済完了）"

            stages.append({
                "name": stage_name,
                "rate": cumulative,
                "barrier": stage.barrier,
                "intervention": stage.intervention,
            })

        return stages
