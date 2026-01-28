"""
Layer1: 知覚品質モデル
施策投入 → 知覚品質変化 → 購買意図変化
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EffectSize:
    """効果量データクラス"""
    factor: str
    effect_size_r: float
    effect_type: str
    source: str
    note: str = ""


# メタ分析からの効果量（CLAUDE.mdより）
EFFECT_SIZES: Dict[str, EffectSize] = {
    "trust": EffectSize(
        factor="trust_to_purchase_intent",
        effect_size_r=0.67,
        effect_type="correlation",
        source="Handoyo 2024",
        note="最強の予測因子"
    ),
    "package_aesthetics": EffectSize(
        factor="package_aesthetics_to_purchase_intent",
        effect_size_r=0.65,
        effect_type="correlation",
        source="Gunaratne 2019",
        note="信頼に次ぐ効果"
    ),
    "review_valence": EffectSize(
        factor="review_valence_to_purchase_intent",
        effect_size_r=0.563,
        effect_type="correlation",
        source="Ismagilova 2020",
        note="量より質が重要"
    ),
    "product_photo": EffectSize(
        factor="product_photo_to_sales",
        effect_size_r=0.125,
        effect_type="direct_effect",
        source="Deliveroo Field",
        note="+10-15%の中央値"
    ),
    "origin_story": EffectSize(
        factor="origin_story_to_quality_perception",
        effect_size_r=0.30,
        effect_type="correlation",
        source="Peterson 1995",
        note="品質知覚経由"
    ),
    "unboxing_positive": EffectSize(
        factor="unboxing_positive_to_wtp",
        effect_size_r=0.57,
        effect_type="proportion",
        source="Joutsela 2016",
        note="57%でWTP上昇"
    ),
    "unboxing_negative": EffectSize(
        factor="unboxing_negative_to_wtp",
        effect_size_r=-0.14,
        effect_type="proportion",
        source="Joutsela 2016",
        note="14%でWTP低下"
    ),
}


@dataclass
class Intervention:
    """施策データクラス"""
    name: str
    cost_initial: float
    cost_per_unit: float
    effect_metric: str
    effect_value: float
    confidence: str


# 施策パラメータ
INTERVENTIONS: Dict[str, Intervention] = {
    "product_photo_pro": Intervention(
        name="商品写真強化",
        cost_initial=400000,
        cost_per_unit=0,
        effect_metric="cvr_multiplier",
        effect_value=1.125,
        confidence="high"
    ),
    "review_system": Intervention(
        name="レビューシステム",
        cost_initial=100000,
        cost_per_unit=0,
        effect_metric="purchase_intent_multiplier",
        effect_value=1.30,
        confidence="high"
    ),
    "insert_card": Intervention(
        name="同梱カード",
        cost_initial=0,
        cost_per_unit=65,
        effect_metric="repeat_rate_delta",
        effect_value=0.075,
        confidence="medium"
    ),
    "insert_card_unboxing": Intervention(
        name="同梱カード+開封体験",
        cost_initial=50000,
        cost_per_unit=80,
        effect_metric="wtp_multiplier",
        effect_value=1.20,
        confidence="medium"
    ),
    "ux_shopify_migration": Intervention(
        name="UX改善（Shopify移行）",
        cost_initial=1500000,
        cost_per_unit=0,
        effect_metric="cart_recovery_rate",
        effect_value=0.25,
        confidence="high"
    ),
    "certification_display": Intervention(
        name="認証表示",
        cost_initial=50000,
        cost_per_unit=0,
        effect_metric="trust_signal",
        effect_value=1.10,
        confidence="medium"
    ),
    "producer_story": Intervention(
        name="生産者ストーリー",
        cost_initial=200000,
        cost_per_unit=0,
        effect_metric="quality_perception_multiplier",
        effect_value=1.15,
        confidence="medium"
    ),
}


class PerceptionModel:
    """知覚品質モデル"""

    def __init__(self, base_purchase_intent: float = 0.10):
        """
        Args:
            base_purchase_intent: ベースの購買意図（デフォルト10%）
        """
        self.base_purchase_intent = base_purchase_intent
        self.effect_sizes = EFFECT_SIZES
        self.interventions = INTERVENTIONS

    def calculate_perception_change(
        self,
        intervention_ids: List[str]
    ) -> Dict[str, float]:
        """
        施策投入による知覚品質変化を計算

        Args:
            intervention_ids: 施策IDのリスト

        Returns:
            知覚品質変化の辞書
        """
        changes = {
            "quality_perception_delta": 0.0,
            "trust_delta": 0.0,
            "aesthetics_delta": 0.0,
        }

        for int_id in intervention_ids:
            if int_id not in self.interventions:
                continue
            intervention = self.interventions[int_id]

            if int_id == "product_photo_pro":
                changes["aesthetics_delta"] += 0.15  # 審美性向上
            elif int_id == "producer_story":
                changes["quality_perception_delta"] += 0.15
            elif int_id == "certification_display":
                changes["trust_delta"] += 0.10
            elif int_id == "review_system":
                changes["trust_delta"] += 0.20

        return changes

    def calculate_purchase_intent_change(
        self,
        intervention_ids: List[str]
    ) -> float:
        """
        施策による購買意図変化率を計算

        Args:
            intervention_ids: 施策IDのリスト

        Returns:
            購買意図の乗数（1.0 = 変化なし）
        """
        multiplier = 1.0

        for int_id in intervention_ids:
            if int_id not in self.interventions:
                continue
            intervention = self.interventions[int_id]

            if intervention.effect_metric == "purchase_intent_multiplier":
                multiplier *= intervention.effect_value
            elif intervention.effect_metric == "cvr_multiplier":
                # CVR改善も購買意図に影響
                multiplier *= intervention.effect_value
            elif intervention.effect_metric == "trust_signal":
                # 信頼シグナルは購買意図に間接的に影響
                multiplier *= intervention.effect_value
            elif intervention.effect_metric == "quality_perception_multiplier":
                # 品質知覚も購買意図に影響（係数0.7で減衰）
                multiplier *= 1 + (intervention.effect_value - 1) * 0.7

        return multiplier

    def calculate_wtp_change(
        self,
        intervention_ids: List[str],
        base_wtp: float = 5000
    ) -> Dict[str, float]:
        """
        WTP（支払意思額）の変化を計算

        Args:
            intervention_ids: 施策IDのリスト
            base_wtp: ベースのWTP

        Returns:
            WTP変化の辞書（表明値と顕示推計値）
        """
        wtp_multiplier = 1.0

        for int_id in intervention_ids:
            if int_id not in self.interventions:
                continue
            intervention = self.interventions[int_id]

            if intervention.effect_metric == "wtp_multiplier":
                wtp_multiplier *= intervention.effect_value
            elif intervention.effect_metric == "quality_perception_multiplier":
                # 品質知覚向上はWTPにも影響
                wtp_multiplier *= 1 + (intervention.effect_value - 1) * 0.5

        wtp_stated = base_wtp * wtp_multiplier
        # 表明→顕示の割引（55%が実際に支払う）
        wtp_revealed = wtp_stated * 0.55

        return {
            "wtp_stated": wtp_stated,
            "wtp_revealed": wtp_revealed,
            "wtp_multiplier": wtp_multiplier,
            "stated_change_pct": (wtp_multiplier - 1) * 100,
            "revealed_change_pct": (wtp_multiplier - 1) * 100 * 0.55,
        }
