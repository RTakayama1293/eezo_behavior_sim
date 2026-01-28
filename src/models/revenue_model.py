"""
Layer3: 収益構造モデル
初回購買 → リピート → LTV → 累積売上
※リピート率は同梱カード・開封体験の関数
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class PriceBand:
    """価格帯"""
    name: str
    price: int
    margin_rate: float
    margin_per_unit: int


# 価格帯定義
PRICE_BANDS: Dict[str, PriceBand] = {
    "band_a": PriceBand(
        name="価格帯A",
        price=5000,
        margin_rate=0.15,
        margin_per_unit=750
    ),
    "band_b": PriceBand(
        name="価格帯B",
        price=10000,
        margin_rate=0.15,
        margin_per_unit=1500
    ),
}


@dataclass
class LTVScenario:
    """LTVシナリオ"""
    name: str
    repeat_rate: float
    description: str


# LTVシナリオ定義
LTV_SCENARIOS: Dict[str, LTVScenario] = {
    "baseline": LTVScenario(
        name="現状",
        repeat_rate=0.10,
        description="同梱カードなし"
    ),
    "with_card": LTVScenario(
        name="同梱カードあり",
        repeat_rate=0.20,
        description="話のネタによる愛着形成"
    ),
    "with_unboxing": LTVScenario(
        name="同梱カード+開封体験",
        repeat_rate=0.30,
        description="徹底した開封体験設計"
    ),
}


class RevenueModel:
    """収益構造モデル"""

    def __init__(
        self,
        avg_order_value: float = 7500,
        margin_rate: float = 0.15,
        base_repeat_rate: float = 0.10
    ):
        """
        Args:
            avg_order_value: 平均注文額
            margin_rate: 粗利率
            base_repeat_rate: ベースリピート率
        """
        self.avg_order_value = avg_order_value
        self.margin_rate = margin_rate
        self.base_repeat_rate = base_repeat_rate
        self.price_bands = PRICE_BANDS
        self.ltv_scenarios = LTV_SCENARIOS

    def calculate_ltv(self, repeat_rate: float) -> Dict[str, float]:
        """
        LTVを計算

        LTV = 平均注文額 × (1 + リピート率 / (1 - リピート率)) × 粗利率

        Args:
            repeat_rate: リピート率

        Returns:
            LTV関連指標の辞書
        """
        if repeat_rate >= 1.0:
            repeat_rate = 0.99  # 上限設定

        # 平均購入回数の計算
        avg_purchases = 1 / (1 - repeat_rate)

        # 生涯売上
        lifetime_revenue = self.avg_order_value * avg_purchases

        # LTV（粗利ベース）
        ltv = lifetime_revenue * self.margin_rate

        return {
            "repeat_rate": repeat_rate,
            "avg_purchases": avg_purchases,
            "lifetime_revenue": lifetime_revenue,
            "ltv": ltv,
        }

    def calculate_ltv_scenarios(self) -> Dict[str, Dict]:
        """
        全LTVシナリオを計算

        Returns:
            シナリオ別LTV結果
        """
        results = {}

        for scenario_id, scenario in self.ltv_scenarios.items():
            ltv_data = self.calculate_ltv(scenario.repeat_rate)
            results[scenario_id] = {
                "name": scenario.name,
                "description": scenario.description,
                **ltv_data
            }

        return results

    def calculate_annual_revenue(
        self,
        customers: int,
        repeat_rate: float
    ) -> Dict[str, float]:
        """
        年間売上を計算

        Args:
            customers: 新規顧客数
            repeat_rate: リピート率

        Returns:
            売上関連指標
        """
        # 初回購入
        initial_revenue = customers * self.avg_order_value

        # リピート購入（簡易計算：初年度のみ）
        repeat_customers = customers * repeat_rate
        repeat_revenue = repeat_customers * self.avg_order_value

        total_revenue = initial_revenue + repeat_revenue
        gross_profit = total_revenue * self.margin_rate

        return {
            "customers": customers,
            "initial_revenue": initial_revenue,
            "repeat_customers": repeat_customers,
            "repeat_revenue": repeat_revenue,
            "total_revenue": total_revenue,
            "gross_profit": gross_profit,
        }

    def calculate_cumulative_value(
        self,
        customers_per_year: int,
        repeat_rate: float,
        years: int = 3
    ) -> List[Dict]:
        """
        累積顧客価値を計算（複数年）

        Args:
            customers_per_year: 年間新規顧客数
            repeat_rate: リピート率
            years: 計算年数

        Returns:
            年別累積価値
        """
        results = []
        cumulative_customers = 0
        cumulative_revenue = 0
        cumulative_profit = 0

        for year in range(1, years + 1):
            # 新規顧客
            new_customers = customers_per_year
            cumulative_customers += new_customers

            # 既存顧客からのリピート
            if year > 1:
                repeat_from_existing = (cumulative_customers - new_customers) * repeat_rate
            else:
                repeat_from_existing = 0

            # 年間購入数
            annual_purchases = new_customers + repeat_from_existing
            annual_revenue = annual_purchases * self.avg_order_value
            annual_profit = annual_revenue * self.margin_rate

            cumulative_revenue += annual_revenue
            cumulative_profit += annual_profit

            results.append({
                "year": year,
                "new_customers": new_customers,
                "repeat_purchases": repeat_from_existing,
                "total_purchases": annual_purchases,
                "annual_revenue": annual_revenue,
                "annual_profit": annual_profit,
                "cumulative_revenue": cumulative_revenue,
                "cumulative_profit": cumulative_profit,
            })

        return results

    def reverse_calculate_requirements(
        self,
        target_sales: float,
        target_profit: float,
        available_traffic: int,
    ) -> Dict[str, Dict]:
        """
        目標達成に必要な条件を逆算

        Args:
            target_sales: 目標売上
            target_profit: 目標粗利
            available_traffic: 利用可能トラフィック

        Returns:
            必要条件のシナリオ
        """
        scenarios = []

        # シナリオ1: CVR改善重視
        required_customers = target_sales / self.avg_order_value
        required_cvr = required_customers / available_traffic
        scenarios.append({
            "name": "CVR改善重視",
            "required_cvr": required_cvr,
            "required_cvr_pct": required_cvr * 100,
            "required_customers": required_customers,
            "avg_price": self.avg_order_value,
            "repeat_rate": self.base_repeat_rate,
            "feasibility": "中" if required_cvr < 0.02 else "低",
        })

        # シナリオ2: 単価アップ重視
        higher_price = self.avg_order_value * 1.5
        required_customers_2 = target_sales / higher_price
        required_cvr_2 = required_customers_2 / available_traffic
        scenarios.append({
            "name": "単価アップ重視",
            "required_cvr": required_cvr_2,
            "required_cvr_pct": required_cvr_2 * 100,
            "required_customers": required_customers_2,
            "avg_price": higher_price,
            "repeat_rate": self.base_repeat_rate,
            "feasibility": "中" if required_cvr_2 < 0.015 else "低",
        })

        # シナリオ3: リピート重視（LTV向上）
        higher_repeat = 0.30
        ltv_data = self.calculate_ltv(higher_repeat)
        effective_value = self.avg_order_value * ltv_data["avg_purchases"]
        required_customers_3 = target_sales / effective_value
        required_cvr_3 = required_customers_3 / available_traffic
        scenarios.append({
            "name": "リピート重視",
            "required_cvr": required_cvr_3,
            "required_cvr_pct": required_cvr_3 * 100,
            "required_customers": required_customers_3,
            "avg_price": self.avg_order_value,
            "repeat_rate": higher_repeat,
            "avg_purchases": ltv_data["avg_purchases"],
            "feasibility": "高" if required_cvr_3 < 0.01 else "中",
        })

        # シナリオ4: バランス型
        balanced_price = self.avg_order_value * 1.2
        balanced_repeat = 0.20
        ltv_balanced = self.calculate_ltv(balanced_repeat)
        effective_value_4 = balanced_price * ltv_balanced["avg_purchases"]
        required_customers_4 = target_sales / effective_value_4
        required_cvr_4 = required_customers_4 / available_traffic
        scenarios.append({
            "name": "バランス型",
            "required_cvr": required_cvr_4,
            "required_cvr_pct": required_cvr_4 * 100,
            "required_customers": required_customers_4,
            "avg_price": balanced_price,
            "repeat_rate": balanced_repeat,
            "avg_purchases": ltv_balanced["avg_purchases"],
            "feasibility": "高" if required_cvr_4 < 0.01 else "中",
        })

        return {
            "target_sales": target_sales,
            "target_profit": target_profit,
            "available_traffic": available_traffic,
            "scenarios": scenarios,
        }
