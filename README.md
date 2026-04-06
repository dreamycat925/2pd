# Two-Point Discrimination

2点識別覚の検査者補助用 Streamlit アプリです。

## 概要

- 練習 / 本番 / 事後 の 3 フェーズ
- 練習で 5 問連続正答してから本番に進む想定
- 本番は 1点/2点 の 100 trial 系列
- 2点 trial のみ 2-down 1-up staircase を更新
- 開始 30 mm
- 固定ラダー上を隣の段へ移動
- 下限 1 mm / 上限 50 mm
- 閾値は最後 6 reversals の中央値

## 本番系列

- 100 trial
- 1点 40 trial
- 2点 60 trial
- 1点 / 2点 は 4 回以上連続しない
- 系列 1 / 系列 2 / ランダム を選択可能

## 本番停止条件

- PASS: 1 mm の 2点 trial で 4 連続正答
- FAIL: 50 mm の 2点 trial で 2 連続誤答
- 収束完了: 10 reversals 到達
- 収束不良: 100 trial 到達

## 練習 / 事後

- 2点のみ 30 mm から開始
- 5 問連続正答で PASS
- 同一 mm で 2 問誤答すると上の mm に進む
- 50 mm で 2 問誤答すると FAIL

通常は、まず練習で 5 問連続正答となったことを確認してから本番に進みます。

## 実行方法

```bash
pip install -r requirements.txt
streamlit run 2pd_discrimination_streamlit_app.py
```

## 推奨手順

1. 練習を実施する
2. 5 問連続正答を確認して本番に進む
3. 本番を実施する
4. 必要に応じて事後を実施する

## 注意

- 検査者補助アプリです
- 医療機器ではありません
