# 雙鴨 v2：修正動作、獎勵與有限回合後的 PPO

v1 seed 1 已完成 2000 iterations，但固定 600 局只進 1 球。診斷顯示少出腳、動作分布尺度持續擴張，以及第一次嘗試後永久關閉準備回饋。
v2 是多項工程修正合併的實驗，**PPO 對固定規則 A，沒有 curriculum，也沒有 self-play**。不能用本次比較單獨歸因某一項修正。

## 保留不變

- v1 分支、模型、原始評測與失敗紀錄保留；原單鴨 B/C 不改。
- 場地、球門、碰撞、BAM、200 Hz 物理／接觸、50 Hz 官方技能與 10 Hz 高階決策不變。
- 官方走路與右腳踢球固定為 [v1 文件](duel.md) 的 SHA256。每隻仍是 61 維低階觀測與 14 維關節動作。
- [固定場景](../benchmarks/duel/v1/layouts.json) SHA256 仍為 `02e8cafd3413af366f94f5940e67c3ba03ba8c3aa374b0ae69a241e97dfcadd9`。
- 100 場景 × 3 次 × 旋轉換邊，共 600 局，eval RNG seed 0。進球、出界、摔倒、20 秒超時的計分／結束優先序不變，所有局都保留在分母。

## v2 改動

### 動作與 PPO

Actor 輸出 4 維 normalized 動作。使用 tanh-normal 分布，走位三維再乘 `[0.4,0.3,1.0]` 成為實際速度。取樣、log-prob、KL、entropy 與 deterministic／ONNX 路徑使用一致的轉換；不再靠環境硬截斷無限大的 raw 指令。
latent mean 限於 ±2；latent std 以平滑 sigmoid 限於 `[0.05,0.6]`，初始 0.4。記錄的 latent std 不能直接當作物理速度標準差。
MLP 仍是 128×128、每次 24 高階步、4 minibatches、5 epochs、gamma=0.99、lambda=0.95。
learning rate 固定 3e−4，不使用 v1 的 adaptive 上調；entropy coefficient 改為 0.002，計算的是經 tanh 轉換後的 entropy。

### 固定出腳檢查，不是課程

PPO 仍決定走位與是否請求踢球。只有距離準備點 <5cm、朝向誤差 <0.15rad、身體速度 <0.05m/s、球速 <0.15m/s，並維持 0.2 秒，才接受出腳請求。
踢球仍有 2 秒鎖定。未準備好的請求會被忽略並記錄，不扣探索嘗試費。這組條件全程固定，不隨訓練漸變，所以不是 curriculum。
對手維持原本固定 A 的控制流程；上述檢查是 v2 我方控制器的一部分，必須隨 checkpoint 一起使用。

### 回饋與終局

| 事件 | 我方 base reward |
|---|---:|
| 我方／對方進球 | +10／−10 |
| 我方摔倒或出界 | −5 |
| 只有對方摔倒或出界 | −1，沒有正獎勵 |
| 球出界且最後觀測觸球方是我方 | −5 |
| 球出界且最後觸球為對方、未知或同時 | 0 |
| 20 秒超時 | −5 |
| 時間 | 每秒 −0.1 |
| 踢球嘗試 | 0 |

同時發生事件時沿用進球優先；安全事件內先處理我方、再對方、最後球出界，避免疊加罰分。
最後觸球使用既有 200 Hz 身體接觸 latch；不把未知觸球當作我方失誤。

準備回饋改成有界 potential：`Phi=2*exp(-prep_distance/0.25)+0.25*(1+cos(heading_error))`，每個高階步給 `0.99*Phi(next)-Phi(current)`，終局 potential 為 0。
第一次踢空後仍能學回準備點；站在目標不會持續領正回饋。折扣與 PPO gamma 一致，CPU 測試檢查 discounted telescoping，不能靠反覆遠離／靠近刷回饋。
20 秒是完整有限回合；不再用 `time_outs` 對這些終局額外 bootstrap。

### 可觀測狀態

Actor／critic 均為 70 維：保留 v1 的 54 維，加入準備位移／朝向、身體速度、剩餘時間、出腳準備計時、對手技能／準備階段與最後觸球方。
v2 不以 `ever_kicked` 影響回饋。v1 的 54 維 checkpoint 與單鴨 B/C checkpoint 不能直接載入 v2。

## 執行

沿用 Python 3.12、原 uv.lock 與相同官方技能，不新增或升級依賴。

```bash
uv sync --frozen --python 3.12
uv run --frozen hf download pollen-robotics/microduck-policies alpha_walking.onnx ball_kick_right.onnx --local-dir policies
uv run --frozen python tests/test_duel_v2.py
uv run --frozen python scripts/train_duel_v2.py --num-envs 64 --iterations 5 --seed 1 --out logs/duel_v2_smoke
uv run --frozen python scripts/eval_duel_v2.py --checkpoint logs/duel_v2_smoke/checkpoints/model_4.pt --smoke --out logs/duel_v2_eval_smoke

# 正式完整 seed 1，從新網路開始；不載入 v1 或短測試權重。
uv run --frozen python scripts/train_duel_v2.py --num-envs 1024 --iterations 2000 --seed 1 --out logs/duel_v2_seed1
uv run --frozen python scripts/eval_duel_v2.py --checkpoint logs/duel_v2_seed1/checkpoints/model_1999.pt --out logs/duel_v2_seed1_fixed600
```

每 100 iterations 與最後保存 checkpoint，manifest 記錄完整設定、來源與技能 hash；verification.json 檢查 actor/critic 更新、optimizer 次數、有限值、固定學習率、std 範圍與技能保持固定。
新版評測每個方向使用獨立程序，避免 v1 換邊重建環境時曾遇到的 segmentation fault；不宣稱已定位該底層崩潰的根因。
out 必須是新資料夾，避免覆蓋資料；目前入口不提供精確 physics/RNG resume。

本次只授權啟動 v2 seed 1。和穆若跑 v2 seed 2，必須用同一個 v2 commit／設定；已存在的 v1 seed 2 仍保留為 v1，不混成同版本多 seed 結果。
正式評測看進球、失球與中止原因，不能只因 reward 上升、可存檔或正向控制案例進球就宣稱學會對抗。
