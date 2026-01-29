"""
価格ジャンプ戦略シミュレーションモデル

20,000円送料無料閾値の効果をモデル化:
- Goal Gradient Effect: 閾値に近い顧客ほどジャンプ率が高い
- AOV向上効果: 10,000〜20,000円帯の顧客が20,000円超えにジャンプ
- 粗利改善: 送料粗利減を商品粗利増で相殺

参照: docs/outputs/reports/exp001_report_20260129.md
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import random


@dataclass
class PriceJumpConfig:
    """価格ジャンプ戦略の設定"""
    threshold: int = 20_000  # 送料無料閾値
    shipping_cost_below: int = 1_500  # 閾値未満の送料
    shipping_cost_above: int = 0  # 閾値以上の送料
    base_propensity: float = 0.58  # 追加購入傾向（Shopify調査2026）
    overspend_mean: float = 0.10  # 閾値超過率の平均
    overspend_std: float = 0.05  # 閾値超過率の標準偏差


@dataclass
class CostStructure:
    """原価構造（年度別）"""
    year: int
    product_cost_rate: float  # 商品原価率（売上比）
    shipping_cost_avg: int  # 平均送料
    insert_card_cost: int  # 同梱カード単価
    margin_rate: float  # 粗利率

    @classmethod
    def y1_hokkaido(cls) -> "CostStructure":
        """Y1: 北海道発送（粗利15%）"""
        return cls(
            year=1,
            product_cost_rate=0.693,  # 商品原価69.3%
            shipping_cost_avg=1_075,  # 北海道→本州送料平均
            insert_card_cost=100,
            margin_rate=0.15
        )

    @classmethod
    def y2_plus_osaka(cls, year: int = 2) -> "CostStructure":
        """Y2以降: 大阪在庫（送料圧縮、粗利20%）"""
        return cls(
            year=year,
            product_cost_rate=0.693,  # 商品原価は同じ
            shipping_cost_avg=700,  # 大阪→本州送料（35%削減）
            insert_card_cost=100,
            margin_rate=0.20
        )


@dataclass
class PriceBandDistribution:
    """価格帯分布"""
    band_name: str
    min_price: int
    max_price: int
    baseline_share: float  # ベースライン構成比
    after_jump_share: float = 0  # ジャンプ後構成比


class PricingStrategyModel:
    """価格ジャンプ戦略モデル"""

    def __init__(
        self,
        config: PriceJumpConfig = None,
        seed: int = 42
    ):
        self.config = config or PriceJumpConfig()
        self.rng = np.random.default_rng(seed)

        # 価格帯分布（ベースライン）
        self.price_bands = [
            PriceBandDistribution("〜5,000円", 0, 5_000, 0.10),
            PriceBandDistribution("5,000〜10,000円", 5_000, 10_000, 0.35),
            PriceBandDistribution("10,000〜15,000円", 10_000, 15_000, 0.30),
            PriceBandDistribution("15,000〜20,000円", 15_000, 20_000, 0.15),
            PriceBandDistribution("20,000円以上", 20_000, float('inf'), 0.10),
        ]

    def generate_customer_orders(
        self,
        n_customers: int = 1000,
        avg_order_value: int = 12_209
    ) -> np.ndarray:
        """
        顧客の注文金額を生成（対数正規分布）

        Args:
            n_customers: 顧客数
            avg_order_value: 平均注文額

        Returns:
            注文金額の配列
        """
        # 対数正規分布のパラメータ
        sigma = 0.5
        mu = np.log(avg_order_value) - sigma**2 / 2

        orders = self.rng.lognormal(mu, sigma, n_customers)
        return orders

    def simulate_price_jump(
        self,
        original_amount: float,
    ) -> Tuple[float, bool]:
        """
        個別顧客の価格ジャンプをシミュレーション

        Args:
            original_amount: 元の注文金額

        Returns:
            (新しい注文金額, ジャンプしたかどうか)
        """
        threshold = self.config.threshold

        # 閾値以上の場合はジャンプ不要
        if original_amount >= threshold:
            return original_amount, False

        # ジャンプ可能な金額帯かチェック（5,000円以上〜20,000円未満）
        if original_amount < 5_000:
            return original_amount, False

        # 閾値までの差額
        gap = threshold - original_amount

        # Goal Gradient Effect: 閾値に近いほどジャンプ率が高い
        # 参照: exp001_report_20260129.md の感度分析
        distance_factor = 1 - (gap / threshold)  # 0〜1
        propensity = self.config.base_propensity * (0.5 + 0.5 * distance_factor)

        # ジャンプするかどうかを決定
        if self.rng.random() < propensity:
            # ジャンプ時の超過額を計算（閾値を少し超える）
            overspend_rate = self.rng.normal(
                self.config.overspend_mean,
                self.config.overspend_std
            )
            overspend_rate = max(0, overspend_rate)  # 負の場合は0
            new_amount = threshold * (1 + overspend_rate)
            return new_amount, True
        else:
            return original_amount, False

    def simulate_threshold_effect(
        self,
        n_customers: int = 1000
    ) -> Dict:
        """
        送料無料閾値効果のシミュレーション

        Args:
            n_customers: 顧客数

        Returns:
            シミュレーション結果
        """
        # 顧客注文金額を生成
        original_orders = self.generate_customer_orders(n_customers)

        # ベースライン計算
        baseline_revenue = np.sum(original_orders)
        baseline_aov = np.mean(original_orders)
        baseline_shipping_revenue = n_customers * self.config.shipping_cost_below

        # 価格ジャンプシミュレーション
        new_orders = []
        jump_flags = []

        for order in original_orders:
            new_order, jumped = self.simulate_price_jump(order)
            new_orders.append(new_order)
            jump_flags.append(jumped)

        new_orders = np.array(new_orders)
        jump_flags = np.array(jump_flags)

        # ジャンプ後の計算
        new_revenue = np.sum(new_orders)
        new_aov = np.mean(new_orders)

        # 送料計算（閾値以上は無料）
        shipping_after = np.sum([
            0 if order >= self.config.threshold else self.config.shipping_cost_below
            for order in new_orders
        ])

        # 価格帯別ジャンプ率
        jump_by_band = []
        for band in self.price_bands:
            mask = (original_orders >= band.min_price) & (original_orders < band.max_price)
            band_customers = np.sum(mask)
            band_jumps = np.sum(jump_flags & mask)
            jump_rate = band_jumps / band_customers if band_customers > 0 else 0
            jump_by_band.append({
                "band": band.band_name,
                "customers": int(band_customers),
                "jumps": int(band_jumps),
                "jump_rate": jump_rate
            })

        # 粗利計算（商品粗利 - 送料負担）
        margin_rate = 0.15  # Y1粗利率
        baseline_product_margin = baseline_revenue * margin_rate
        baseline_gross_profit = baseline_product_margin + baseline_shipping_revenue * margin_rate

        new_product_margin = new_revenue * margin_rate
        new_shipping_cost = shipping_after  # 送料は全額負担
        new_gross_profit = new_product_margin - (baseline_shipping_revenue - shipping_after) * margin_rate

        return {
            "n_customers": n_customers,
            "baseline": {
                "total_revenue": baseline_revenue,
                "aov": baseline_aov,
                "shipping_revenue": baseline_shipping_revenue,
                "gross_profit": baseline_gross_profit,
            },
            "with_threshold": {
                "total_revenue": new_revenue,
                "aov": new_aov,
                "shipping_cost": shipping_after,
                "gross_profit": new_gross_profit,
            },
            "changes": {
                "revenue_change": (new_revenue - baseline_revenue) / baseline_revenue,
                "aov_change": (new_aov - baseline_aov) / baseline_aov,
                "gross_profit_change": (new_gross_profit - baseline_gross_profit) / baseline_gross_profit,
            },
            "jump_stats": {
                "total_jumps": int(np.sum(jump_flags)),
                "jump_rate": np.mean(jump_flags),
                "avg_increase": np.mean(new_orders[jump_flags] - original_orders[jump_flags]) if np.any(jump_flags) else 0,
                "by_band": jump_by_band,
            },
            "distribution": {
                "original_orders": original_orders,
                "new_orders": new_orders,
                "jump_flags": jump_flags,
            }
        }


class FiveYearGrowthModel:
    """5年間成長モデル（BtoC + BtoB + MP）"""

    def __init__(self):
        # 年度別目標（docs/revenue_strategy_2026.mdより）
        self.targets = {
            1: {"btoc": 10_000_000, "btob": 0, "mp": 0},
            2: {"btoc": 15_000_000, "btob": 5_000_000, "mp": 0},
            3: {"btoc": 20_000_000, "btob": 15_000_000, "mp": 0},
            4: {"btoc": 35_000_000, "btob": 40_000_000, "mp": 25_000_000},
            5: {"btoc": 50_000_000, "btob": 60_000_000, "mp": 40_000_000},
        }

        # 年度別粗利率
        self.margin_rates = {
            1: 0.15,
            2: 0.18,  # BtoB混合
            3: 0.20,
            4: 0.18,  # 大規模投資期
            5: 0.20,
        }

        # 年度別投資額
        self.investments = {
            1: 520_000,     # 基盤構築（エンジニア除く）
            2: 3_000_000,   # 大阪物流拠点
            3: 5_000_000,   # プラットフォーム開発
            4: 15_000_000,  # 商品拡充+広告+MP化
            5: 5_000_000,   # 運用改善
        }

        # BtoBセグメント定義
        self.btob_segments = {
            "restaurants": {
                "name": "飲食店・居酒屋",
                "avg_order": 30_000,
                "orders_per_year": 18,
                "margin_rate": 0.12,
                "start_year": 2,
            },
            "hotels": {
                "name": "ホテル・旅館",
                "avg_order": 80_000,
                "orders_per_year": 10,
                "margin_rate": 0.10,
                "start_year": 3,
            },
            "retail": {
                "name": "百貨店・量販店",
                "avg_order": 300_000,
                "orders_per_year": 6,
                "margin_rate": 0.08,
                "start_year": 4,
            },
            "corporate_gifts": {
                "name": "法人ギフト",
                "avg_order": 150_000,
                "orders_per_year": 2,
                "margin_rate": 0.15,
                "start_year": 3,
            },
        }

    def simulate_year(self, year: int) -> Dict:
        """
        単年シミュレーション

        Args:
            year: 年度（1-5）

        Returns:
            年度シミュレーション結果
        """
        target = self.targets[year]
        margin_rate = self.margin_rates[year]
        investment = self.investments[year]

        total_revenue = target["btoc"] + target["btob"] + target["mp"]
        gross_profit = int(total_revenue * margin_rate)
        operating_profit = gross_profit - investment

        # 価格ジャンプ効果の適用（BtoCのみ）
        price_jump_effect = 0
        if year >= 2:  # Y2から価格ジャンプ戦略を本格導入
            # AOV +19.9%、粗利+3.7%の効果（シミュレーション結果より）
            price_jump_effect = int(target["btoc"] * 0.037 * margin_rate)

        return {
            "year": year,
            "btoc_revenue": target["btoc"],
            "btob_revenue": target["btob"],
            "mp_revenue": target["mp"],
            "total_revenue": total_revenue,
            "margin_rate": margin_rate,
            "gross_profit": gross_profit,
            "price_jump_effect": price_jump_effect,
            "investment": investment,
            "operating_profit": operating_profit,
        }

    def simulate_btob_details(self, year: int) -> Dict:
        """
        BtoB詳細シミュレーション

        Args:
            year: 年度

        Returns:
            BtoB詳細結果
        """
        if year < 2:
            return {"year": year, "segments": {}, "total_revenue": 0, "total_accounts": 0}

        target_btob = self.targets[year]["btob"]
        if target_btob == 0:
            return {"year": year, "segments": {}, "total_revenue": 0, "total_accounts": 0}

        # セグメント別構成比（年度によって変化）
        segment_shares = {
            2: {"restaurants": 1.0},
            3: {"restaurants": 0.50, "hotels": 0.25, "corporate_gifts": 0.25},
            4: {"restaurants": 0.35, "hotels": 0.25, "retail": 0.20, "corporate_gifts": 0.20},
            5: {"restaurants": 0.30, "hotels": 0.25, "retail": 0.25, "corporate_gifts": 0.20},
        }

        shares = segment_shares.get(year, {})
        segments = {}
        total_revenue = 0
        total_accounts = 0

        for seg_id, share in shares.items():
            if share > 0:
                seg = self.btob_segments[seg_id]
                segment_revenue = int(target_btob * share)
                revenue_per_account = seg["avg_order"] * seg["orders_per_year"]
                accounts = max(1, int(segment_revenue / revenue_per_account))
                actual_revenue = accounts * revenue_per_account

                segments[seg_id] = {
                    "name": seg["name"],
                    "accounts": accounts,
                    "revenue": actual_revenue,
                    "profit": int(actual_revenue * seg["margin_rate"]),
                }
                total_revenue += actual_revenue
                total_accounts += accounts

        return {
            "year": year,
            "segments": segments,
            "total_revenue": total_revenue,
            "total_accounts": total_accounts,
        }

    def simulate_five_years(self) -> List[Dict]:
        """
        5年間シミュレーション

        Returns:
            年次シミュレーション結果のリスト
        """
        results = []
        cumulative_revenue = 0
        cumulative_profit = 0
        cumulative_investment = 0

        for year in range(1, 6):
            year_result = self.simulate_year(year)
            btob_details = self.simulate_btob_details(year)

            cumulative_revenue += year_result["total_revenue"]
            cumulative_profit += year_result["gross_profit"]
            cumulative_investment += year_result["investment"]

            year_result["btob_details"] = btob_details
            year_result["cumulative"] = {
                "revenue": cumulative_revenue,
                "profit": cumulative_profit,
                "investment": cumulative_investment,
                "roi": (cumulative_profit - cumulative_investment) / cumulative_investment if cumulative_investment > 0 else 0,
            }

            # 成長率計算
            if year > 1:
                prev_revenue = self.targets[year-1]["btoc"] + self.targets[year-1]["btob"] + self.targets[year-1]["mp"]
                year_result["growth_rate"] = (year_result["total_revenue"] - prev_revenue) / prev_revenue
            else:
                year_result["growth_rate"] = 0

            results.append(year_result)

        return results

    def simulate_ltv_evolution(self) -> List[Dict]:
        """
        5年間のLTV推移シミュレーション

        Returns:
            年次LTV結果
        """
        # 年度別パラメータ（docs/revenue_strategy_2026.md 付録Eより）
        ltv_params = [
            {"year": 1, "avg_price": 7500, "repeat_rate": 0.20, "margin_rate": 0.15},
            {"year": 2, "avg_price": 7000, "repeat_rate": 0.30, "margin_rate": 0.20},
            {"year": 3, "avg_price": 6500, "repeat_rate": 0.35, "margin_rate": 0.22},
            {"year": 4, "avg_price": 6000, "repeat_rate": 0.28, "margin_rate": 0.18},
            {"year": 5, "avg_price": 6000, "repeat_rate": 0.36, "margin_rate": 0.20},
        ]

        results = []
        for params in ltv_params:
            # LTV = 平均単価 × 平均購入回数 × 粗利率
            # 平均購入回数 = 1 / (1 - リピート率)
            repeat_rate = params["repeat_rate"]
            avg_purchases = 1 / (1 - repeat_rate)
            ltv = params["avg_price"] * avg_purchases * params["margin_rate"]

            results.append({
                **params,
                "avg_purchases": avg_purchases,
                "ltv": int(ltv),
            })

        return results


def run_pricing_strategy_simulation():
    """価格戦略シミュレーションを実行"""
    print("=" * 60)
    print("EEZO 価格ジャンプ戦略シミュレーション")
    print("=" * 60)

    # 1. 価格ジャンプ効果シミュレーション
    print("\n【1】20,000円送料無料閾値の効果")
    print("-" * 40)

    model = PricingStrategyModel()
    result = model.simulate_threshold_effect(n_customers=1000)

    print(f"シミュレーション顧客数: {result['n_customers']:,}人")
    print(f"\n[ベースライン]")
    print(f"  総売上: ¥{result['baseline']['total_revenue']:,.0f}")
    print(f"  AOV: ¥{result['baseline']['aov']:,.0f}")

    print(f"\n[20K閾値導入後]")
    print(f"  総売上: ¥{result['with_threshold']['total_revenue']:,.0f}")
    print(f"  AOV: ¥{result['with_threshold']['aov']:,.0f}")

    print(f"\n[変化率]")
    print(f"  売上変化: {result['changes']['revenue_change']*100:+.1f}%")
    print(f"  AOV変化: {result['changes']['aov_change']*100:+.1f}%")
    print(f"  粗利変化: {result['changes']['gross_profit_change']*100:+.1f}%")

    print(f"\n[ジャンプ統計]")
    print(f"  ジャンプ率: {result['jump_stats']['jump_rate']*100:.1f}%")
    print(f"  平均増加額: ¥{result['jump_stats']['avg_increase']:,.0f}")

    print("\n[価格帯別ジャンプ率]")
    for band in result['jump_stats']['by_band']:
        print(f"  {band['band']}: {band['jump_rate']*100:.1f}% ({band['jumps']}/{band['customers']}人)")

    # 2. 5年間成長シミュレーション
    print("\n" + "=" * 60)
    print("【2】5年間成長シミュレーション（BtoC + BtoB + MP）")
    print("=" * 60)

    growth_model = FiveYearGrowthModel()
    five_year_results = growth_model.simulate_five_years()

    print(f"\n{'年':>3} {'BtoC':>12} {'BtoB':>12} {'MP':>12} {'合計':>14} {'粗利':>12} {'成長率':>8}")
    print("-" * 80)

    for r in five_year_results:
        growth = f"+{r['growth_rate']*100:.0f}%" if r['growth_rate'] > 0 else "-"
        print(f"Y{r['year']:>2} ¥{r['btoc_revenue']:>10,} ¥{r['btob_revenue']:>10,} "
              f"¥{r['mp_revenue']:>10,} ¥{r['total_revenue']:>12,} "
              f"¥{r['gross_profit']:>10,} {growth:>8}")

    # 累積計算
    final = five_year_results[-1]
    print("\n[5年間累計]")
    print(f"  累計売上: ¥{final['cumulative']['revenue']:,}")
    print(f"  累計粗利: ¥{final['cumulative']['profit']:,}")
    print(f"  累計投資: ¥{final['cumulative']['investment']:,}")
    print(f"  ROI: {final['cumulative']['roi']*100:.0f}%")

    # 3. BtoB詳細
    print("\n【3】BtoB展開詳細")
    print("-" * 40)

    for r in five_year_results:
        btob = r['btob_details']
        if btob['total_accounts'] > 0:
            print(f"\nY{r['year']}: {btob['total_accounts']}社 (¥{btob['total_revenue']:,})")
            for seg_id, seg in btob['segments'].items():
                print(f"  - {seg['name']}: {seg['accounts']}社 (¥{seg['revenue']:,})")

    # 4. LTV推移
    print("\n【4】LTV推移シミュレーション")
    print("-" * 40)

    ltv_results = growth_model.simulate_ltv_evolution()
    print(f"\n{'年':>3} {'平均単価':>10} {'リピート率':>10} {'購入回数':>8} {'粗利率':>8} {'LTV':>10}")
    print("-" * 60)

    for r in ltv_results:
        print(f"Y{r['year']:>2} ¥{r['avg_price']:>8,} {r['repeat_rate']*100:>9.0f}% "
              f"{r['avg_purchases']:>7.2f}回 {r['margin_rate']*100:>7.0f}% ¥{r['ltv']:>8,}")

    return {
        "price_jump": result,
        "five_year": five_year_results,
        "ltv": ltv_results,
    }


if __name__ == "__main__":
    results = run_pricing_strategy_simulation()
