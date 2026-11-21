# 雙鴨對戰 v1：固定對戰評測與 PPO 對固定 A

這是獨立於單鴨 B/C 實驗的新任務。先驗證環境和評測，再決定長訓練預算。
目前程式支援規則 A 對 A，以及新高階 PPO 對固定 A；沒有 self-play。
單鴨 B/C 的 checkpoint 輸入不同，不能直接冒充雙鴨 PPO。

## 環境約定

- 同一場地兩隻完整碰撞 BAM Microduck、一顆球、兩個球門。獨立場地間距 4 m。
- 場地 2 × 1.2 m；球半徑 0.035 m；球門淨寬 0.4 m、高 0.2 m。
- 藍色攻 +X，橘色攻 −X。訓練先固定藍色為 PPO、橘色為規則 A。
- 物理與接觸取樣 200 Hz；官方低階技能 50 Hz；高階決策 10 Hz。
- 官方走路與右腳踢球權重固定，兩隻各自保持 61 維低階觀測、14 維關節動作及 2 秒踢球鎖定。
- 高階動作 4 維：前後、左右、轉向速度與踢球觸發。actor 54 維、critic 61 維，包含對手相對位置、速度、方向。
- 每回合最多 20 秒。整顆球穿越任一球門有效開口即進球，允許烏龍球，不要求指定腳觸球才計分。
- 任一鴨傾斜超過 70°、任一鴨中心出場、球完全出界，便提前結束。這是簡化的安全回合規則，不是完整足球賽制。
- 無 curriculum、觀測噪聲或質量／摩擦等額外 domain randomization。沿用 BAM 致動器啟動時參數展開；評測使用共同 RNG seed 0。

每次重設都依各場地原點重新放球門與場地，避免平行環境球門留在原點。
接觸感測在每個物理子步累積，避免控制步結束的 forward 覆蓋短暫接觸。
接觸數是「有接觸的物理取樣數」，不是獨立踢球次數。最後觸球 −1 表示未觀測到，−2 表示同一子步雙方觸球；不據此猜誰踢的。

## 固定評測：100 × 3 × 2 = 600 局

先讀 [固定協定](../benchmarks/duel/v1/manifest.json) 與 [場景](../benchmarks/duel/v1/layouts.json)。
場景生成 seed 20261121，不能看結果後換題。球初始靜止，x∈[−0.3,0.3]、y∈[−0.25,0.25] m；
雙方從右腳準備姿勢後方 0.25–0.45 m 出發，橫向偏移 ±0.1 m、朝向偏移 ±45°。
這套先測訓練分布內的基本對抗，尚未測移動球起始、特殊邊角、不同對手或真實機器人。

每組測 3 次，再把整個場景旋轉 180°並交換隊色，讓待測策略分別攻 +X／−X。
旋轉保留右腳，不做會把右腳變成左腳的鏡射。每個方向一次跑 300 個環境，只記每個環境的第一局。
不同硬體／版本仍可能有數值差異；比較時要附 manifest、hash 與逐局資料，不能保證逐位元一致。
重複局共享初始幾何，可能受 BAM reset RNG／GPU 數值影響；600 局不是 600 個獨立訓練 seed。

主要結果互斥，優先順序為：進球、失球、雙方／自己／對手摔倒、雙方／自己／對手出界、球出界、超時。
原始重疊旗標一併保存。進球率和失球率分母都包含全部 600 局，不剔除提早結束；超時與其他中止另列。
另列各方向分數、每局淨進球、成功進球平均／中位／p90 秒數。進球時間只描述成功局，不能單獨判定策略較好。

## 安裝與執行

沿用 repo 的 Python 3.12 與 uv.lock，不任意升級。WSL 使用 Windows NVIDIA 驅動提供的 CUDA，先確認 PyTorch CUDA 可用。

```bash
uv sync --frozen --python 3.12
uv run --frozen hf download pollen-robotics/microduck-policies alpha_walking.onnx ball_kick_right.onnx --local-dir policies
uv run --frozen python tests/test_duel.py
uv run --frozen python scripts/eval_duel.py --prepare

# 8 局工具短測試，不是正式分數
uv run --frozen python scripts/eval_duel.py --baseline --smoke --out logs/duel_A_smoke

# 固定規則 A 對 A，共 600 局，不訓練
uv run --frozen python scripts/eval_duel.py --baseline --out logs/duel_A_fixed_v1

# 新 PPO 的最小訓練驗證；必須確認權重更新、checkpoint 和 verification.json
uv run --frozen python scripts/train_duel.py --num-envs 64 --iterations 5 --seed 1 --out logs/duel_PPO_smoke

# 完整訓練完成後，用相同工具評測自己的雙鴨模型
uv run --frozen python scripts/eval_duel.py --checkpoint logs/duel_PPO_smoke/checkpoints/model_4.pt --out logs/duel_PPO_fixed_v1
```

最後一行使用短測試模型僅供驗證載入流程，不能宣稱它已學會對抗。
所有 out 必須是新資料夾，避免覆蓋。訓練入口預設只有 64 環境／5 次 iteration；增加預算前請先檢查短測試。
不要用通用 `play` 驅動此任務：底層是 28 維關節動作，高階是 4 維。
任務有登錄 `Mjlab-Duel-Flat-MicroDuck`，分享實驗優先使用上述腳本，讓 manifest 與檢查一起保存。

模型雜湊必須符合：

| 官方技能 | SHA256 |
|---|---|
| alpha_walking.onnx | e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c |
| ball_kick_right.onnx | 147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af |

## PPO 設定與後續分工

沿用現有高階 PPO：MLP 128×128、觀測正規化、每環境 24 步、4 minibatches、5 epochs、初始 learning rate 3e−4（adaptive）、γ=0.99、λ=0.95。
兩個固定技能不更新。新網路從頭學高階控制，不能直接載入單鴨 B/C。此 v1 沒有精確物理／RNG resume；訓練腳本也不提供 resume，避免誤認為可無縫續跑。
獎勵：己方進球 +10、失球 −10；任一安全中止 −10、超時 −2、每秒 −0.1、每次踢球啟動 −0.5。
踢球前以準備位置／角度改善量給有限 shaping，第一次己方踢球後停止 shaping，站在目標不會持續得分。
這是待驗證的初版獎勵，不代表已找到最好的對抗策略。

和睦與其他機器可先在相同 commit／lock／技能下各跑短測試，回傳 manifest、verification、checkpoint hash。
正式訓練前共同決定環境／獎勵版本與預算，再分配不同 seed 各自訓練；這不是兩台 GPU 共同更新同一模型。
回傳固定評測的 manifest.json、summary.json、per_episode.csv 及 checkpoint/hash，才能做配對比較。
原單鴨 B/C 的訓練來源、參數、100×3 場景與既有結果均保留；這個雙鴨協定是另一個新任務。
