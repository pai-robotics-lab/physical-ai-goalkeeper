# AIロボットゴールキーパー

Physical AI応用1講座の最終課題として、ボールの初期軌道からゴールライン通過位置を
MLPで予測し、Genesis上の1自由度ロボットをPD制御して迎撃するシステムです。

- [実演動画](videos/PAI_final_goalkeeper.mp4)
- [技術報告書](docs/technical_report.pdf)
- [検証結果](results/goalkeeper_validation.json)

## 課題概要

抗力とスピンを含む非線形な2次元ボール軌道を生成し、時刻0.00、0.20、0.40、0.60秒の
4点の座標から、ゴールライン `x=0` の通過位置を予測します。AIの出力をロボットの
関節目標へ変換し、GenesisのURDF関節をPD位置制御します。

AI軌道予測は初期4点に対して一度だけ実行します。閉ループなのは、Genesisの実関節位置を
フィードバックするPD関節制御です。

## システム構成

```text
非線形軌道生成
      │
      ▼
4時点の座標観測 ──► 8次元入力 (x0,y0,...,x3,y3)
      │
      ▼
MLPまたは等速予測 ──► ゴール通過位置 y_pred
      │
      ▼
速度制限・関節限界 ──► 関節目標 q*
      │
      ▼
Genesis PD制御 ──► 実関節位置 q
      │
      ▼
|q - ball_crossing_y| <= 0.18 m で位置差評価
```

ロボットは固定レールと可動台車をprismatic jointで接続した1自由度URDFモデルです。
黄色いパドルは台車と一体で移動します。

## AIモデルと学習方法

- 入力: 4時点の `(x, y)` 座標を並べた8次元ベクトル
- 出力: ゴールライン通過位置1値
- 構造: `Linear(8,32) - Tanh - Linear(32,32) - Tanh - Linear(32,1)`
- 損失: 平均二乗誤差
- 最適化: Adam、learning rate 0.01、full batch
- 学習データ: 4000軌道
- 検証データ: 1000軌道
- 学習回数: 300 epochs
- 再現条件: seed 7、CPU

保存済みモデルには、学習済み重み、入力・出力の正規化値、軌道設定、評価値が含まれます。

## Genesisロボット制御

MLPの予測値を可動範囲 `-2.35 m` から `+2.35 m` に制限し、1ステップ当たりの目標移動量にも
上限を設けます。Genesisでは `Kp=2200`、`Kv=95` のPD位置制御を使用します。

成功判定はボールとパドルの接触イベントではありません。ボールがゴールラインを通過した時点の
ロボット実関節位置 `q` と、ボール通過位置 `ball_crossing_y` の差が0.18 m以内かを評価します。

## 動作環境

検証済み環境:

- Windows 10
- Python 3.12.14（対応範囲: 3.10から3.13）
- PyTorch 2.14.1+cpu
- Genesis 1.2.1
- CPUバックエンド

## 環境構築

PowerShellで実行します。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

環境確認:

```powershell
python -m pai_final.check_env
python -m pai_final.smoke_test
```

Genesis/Quadrantsのキャッシュはリポジトリ内の `.cache/` に作成されます。

## 学習・評価・動画再現

### 学習

```powershell
.\.venv\Scripts\python.exe -m pai_final.train_goalkeeper --seed 7
```

このコマンドは `artifacts/goalkeeper_model.pt` と評価JSONを更新します。配布済みモデルを保全する
場合は、`--output` で別の保存先を指定してください。

### 提出結果の一括検証

```powershell
.\.venv\Scripts\python.exe scripts\validate_goalkeeper_submission.py
```

この検証は公開成果物のSHA-256を確認し、MLPの決定的再学習、1000軌道の予測評価、
Genesis実関節によるAI・等速予測各10球の迎撃評価を実行します。結果は
`output/validation/` に生成されます。

### Genesisデモ

```powershell
# AI
.\.venv\Scripts\python.exe -m pai_final.demo_goalkeeper --shots 10 --seed 21

# 等速予測
.\.venv\Scripts\python.exe -m pai_final.demo_goalkeeper `
    --controller baseline --shots 10 --seed 21
```

### 比較動画の再生成

```powershell
.\.venv\Scripts\python.exe -m pai_final.demo_goalkeeper `
    --shots 10 --seed 21 --record videos\goalkeeper_ai_raw.mp4

.\.venv\Scripts\python.exe -m pai_final.demo_goalkeeper `
    --controller baseline --shots 10 --seed 21 `
    --record videos\goalkeeper_baseline_raw.mp4

.\.venv\Scripts\python.exe scripts\create_submission_video.py
```

## 実験結果

| 評価 | AI (MLP) | 等速予測 | 条件 |
|---|---:|---:|---|
| MAE | 0.078313 m | 0.388345 m | 未学習1000軌道 |
| 予測代理成功率 | 91.0% | 26.0% | 予測位置差 <= 0.18 m |
| Genesis実関節成功率 | 100.0% (10/10) | 10.0% (1/10) | seed 21、位置差判定 |

91%対26%はロボットの動特性を含まない予測代理指標です。100%対10%はGenesisの実関節位置を
使った各10球の結果です。標本数と定義が異なるため、同じ母集団の結果として直接比較しません。

## 実装上の制約

- ボール軌道は独自の2次元運動モデルで生成し、計算位置をGenesisシーンへ反映しています。
  Genesisの接触・摩擦ソルバでボール運動を生成しているわけではありません。
- 成功判定は実接触ではなく、ボール通過時の実関節位置との位置差です。
- 観測値はシミュレーション内部座標であり、カメラ画像認識は実装していません。
- ロボットは1自由度の専用機構です。実機およびSim-to-Realは未検証です。
- Genesis実関節評価は各10球であり、統計的評価には追加試行が必要です。

## テスト

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

## リポジトリ構成

```text
artifacts/   学習済みMLPと評価値
assets/      1自由度ゴールキーパーURDF
docs/        技術報告書
results/     検証済みJSON、CSV、グラフ
scripts/     一括検証、比較動画生成
src/         Pythonパッケージ
tests/       決定的な自動テスト
videos/      最終実演動画
```
