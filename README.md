# EEZO消費者行動×収益シミュレーション

## 概要

北海道食材専門商社のEC事業「EEZO」において、**知覚品質向上施策**が**WTP・CVR・LTV**に与える影響を多層モデルでシミュレーションする。

**Claude Code on the Web** での利用を想定。

## 問題設定

**現状の課題**:
- CVR: 0.04%（業界平均2-3%の約50分の1）
- 「商品は見ているが魅力不足」
- 高級商品が置いてあるだけで売れない

**問い**:
- 同梱カード、商品写真、レビュー、UX改善でどの障壁がクリアされるか？
- 知覚品質の向上はWTP・LTVにどう影響するか？
- 売上10百万円達成には何が必要か？

## モデル構造

```
Layer 1: 知覚品質モデル
  施策 → 信頼/審美性/ストーリー → 購買意図

Layer 2: 行動転換モデル
  購買意図 → 実購買（r=0.51で減衰）

Layer 3: 収益構造モデル
  初回購買 → リピート → LTV → 累積売上
```

## エビデンス基盤

パラメータはメタ分析の効果量を使用：

| 要因 | 効果量(r) | 出典 |
|------|----------|------|
| 信頼→購買意図 | 0.67 | Handoyo 2024 |
| パッケージ審美性 | 0.65 | Gunaratne 2019 |
| レビュー正負 | 0.563 | Ismagilova 2020 |
| 開封体験→WTP | +57%/-14% | Joutsela 2016 |

## 使い方

### 1. GitHubにプッシュ

```bash
git init
git add .
git commit -m "Initial setup"
git remote add origin https://github.com/[your-username]/eezo-behavior-sim.git
git push -u origin main
```

### 2. Claude Code on the Web でリポジトリ選択

https://claude.ai/code にアクセスし、このリポジトリを選択

### 3. 指示を入力

```
CLAUDE.mdを読んで、全シミュレーションを実行してレポートにまとめて
```

## 出力物

- シミュレーションレポート（Markdown）
- 可視化グラフ（PNG）
  - ベースライン vs 改善後比較
  - ファネル分析（Before/After）
  - WTP感度分析
  - LTVシナリオ比較
  - 10百万円達成条件マトリクス

## 技術スタック

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- japanize-matplotlib（日本語フォント対応）

---

*新日本海商事 EEZO事業戦略検討用*
