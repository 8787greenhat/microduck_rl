# 雙鴨 v3：能實際接近、可觀測的停穩出腳流程與 PPO 訓練

使用者於 2026-11-24 授權開始新一輪訓練。本版基於 PR #8 `042b56d`，保留 v1、v2、PR #8 和單鴨 B/C 的程式與結果，新增獨立 v3 入口。

## 本輪做什麼

- **高階 PPO 對固定規則 A**，沒有 reset curriculum、沒有 self-play。
- 正式只跑 seed 1，1024 個平行環境、2000 iterations，從新網路初始化。不載入舊 v2、PR #8 的凍結 actor 或短測試權重。
- 延用 v2 的 MLP 128×128、24 rollout steps、4 minibatches、5 epochs、固定 learning rate 3e-4、entropy 0.002、gamma 0.99、lambda 0.95，以及全部獎勵與原遊戲計分規則。
- 官方走路、右腳踢球 ONNX、61D 低階觀測、BAM／碰撞／物理步長不變。

## 為什麼不能直接啟動舊入口

PR #8 是凍結舊 actor 的控制器短測候選版，原 `train_duel_v2.py` 不會套用它。它也沒有把控制器記憶完整暴露給 PPO。
因此新增 `DuelTrainV3` 和專用 runner／訓練／評測入口；舊訓練指令依然保持原行為。

速度校準使用固定官方走路技能、64 個獨立環境、各指令8個複本、2秒模擬。此名義場景中 0.08～0.20 m/s 的前進指令只產生毫米級移動，0.30 m/s 才穩定走動。這是該場景的實測，未宣稱所有方向與動態條件都有相同門檻。

v3 保留遠處的 PPO 走位；只有距準備點12cm內且收到出腳請求才接手，以校準後0.30m/s的平面指令朝準備點移動。到位才送零走路速度，由官方技能保持平衡。
沿用 PR #8 的停穩、意圖逾時／取消、踢球鎖定與冷卻機制；原位置、角度、身體速度<0.05m/s與球速<0.15m/s的出腳檢查仍保留，沒有放寬。

## 新觀測／checkpoint 契約

原70維之外，actor與critic各增加6維：FOLLOW／ALIGN／SETTLE one-hot、意圖時間、停穩時間、冷卻時間，合計 **76維**。PPO可觀察控制器的記憶與下一步動作條件，實際出腳／reset後回傳的是最終控制狀態。
契約為 `duel-v3-actor76-critic76-normalized4`；v1、v2和B/C權重不能直接載入。
checkpoint每100次及最後保存，採暫存檔再原子更名；沒有宣稱能精確恢復物理狀態／RNG。

## 執行

依原Python3.12、`uv.lock`與官方技能，不升級套件。

```bash
uv sync --frozen --python 3.12
uv run --frozen hf download pollen-robotics/microduck-policies alpha_walking.onnx ball_kick_right.onnx --local-dir policies
uv run --frozen python tests/test_duel_v3.py

# 必須先短測，檢查實際 optimizer 更新與 checkpoint，不只是能載入。
uv run --frozen python scripts/train_duel_v3.py --num-envs 64 --iterations 5 --seed 1 --out logs/duel_v3_smoke_new

# 本機授權的正式設定；不要把短測模型當成正式初始化權重。
uv run --frozen python scripts/train_duel_v3.py --num-envs 1024 --iterations 2000 --seed 1 --out logs/duel_v3_seed1_new
uv run --frozen python scripts/eval_duel_v3.py --checkpoint logs/duel_v3_seed1_new/checkpoints/model_1999.pt --out logs/duel_v3_seed1_fixed600_new
```

和穆如要跑另一個 training seed，須使用同一個 v3 commit、設定和技能；舊 v2 seed 2 不可算成 v3 的第二個 seed。這台電腦本輪不自動啟動 seed 2／3。

評測仍用原100固定場景×3次×旋轉換邊，共600局，RNG seed0，每個方向獨立程序。正式結果以全600局為分母，也保存踢球次數與最後觸球，不把推球進門全部稱作成功射門。
控制器的準備位置正向測試、64×5或1024×5訓練通過，都不等於完整對抗學會；正式表現需等最終 checkpoint 的固定評測。

## 2026-11-24 seed 1 完訓結果

正式1024環境／2000次更新與最終固定600局均已完成（returncode 0），最終checkpoint索引1999。
相同600局中，v2進球106/600（17.67%），v3進球167/600（27.83%）；失球13→7；成功局平均時間5.655→3.768秒。
v3有521局啟動踢球技能，但仍402局球出界。各版本只有seed 1，且v3含控制器與觀測修改，不能歸因單一因素。
model_100中途評測worker於PyTorch import時segfault，model_500／1000未評測；中途學習曲線尚未完成。

完整21個checkpoint、600局原始資料、v2對照與失敗紀錄：[結果包README](../results/duel/v3_seed1_20261124/README.md)。
訓練來源仍是 `e8b9756c03ae66a693275ee84b61a36aab367b2f`；本次僅新增既有結果與文件。
