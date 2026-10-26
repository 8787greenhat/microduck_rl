# 方法 B seed 1 結果包

方法 B（高階 PPO，無課程）seed 1 的設定、全部存檔、評估結果、TensorBoard 紀錄，以及和方法 C 的比對。放進 repo，讓組員不用互傳壓縮檔就能直接使用。

固定考題 300 局的正式結果：**B 95.3%**；同一台電腦上，方法 A 40.3%、方法 C（seed 1）92.3%。

下面的指令都在 **repo 根目錄**執行，並假設 `P` 是這包的路徑：

```bash
P=results/soccer/b_seed1_comparison_20261025
```

## 準備（第一次才需要）

```bash
uv sync

# 兩個凍結技能不在 git 裡；固定版本下載，確保和大家用的是同一份
uv run hf download pollen-robotics/microduck-policies \
  alpha_walking.onnx ball_kick_right.onnx \
  --revision 1b56c396825c052a4e26e95cf2b8d8298af9e9b4 --local-dir policies
sha256sum policies/alpha_walking.onnx policies/ball_kick_right.onnx
# e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c  policies/alpha_walking.onnx
# 147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af  policies/ball_kick_right.onnx

# 確認這包檔案完整（應該 50 個都 OK）
(cd $P && sha256sum -c SHA256SUMS.txt --quiet) && echo OK
```

評估和觀看需要 NVIDIA GPU；沒有 GPU 也能跑，只是很慢。

## 1. 看鴨子射門

```bash
uv run scripts/view_soccer_policy.py --checkpoint $P/checkpoints/model_1999.pt --num-envs 4
```

在 MuJoCo 視窗按 `]` / `[` 切換鴨子、`Enter` 重置、空白鍵暫停。每局結束時終端機會印出結束原因（`valid_goal` 是有效進球）。

想看還沒學會的樣子，換成 `model_100.pt`：它在這個階段完全不踢，每局都撐到 12 秒超時。

`uv run play` 不能用在足球任務，它只會送 14 個馬達指令，不懂高階的 4 個數字。

## 2. 在固定考題上評估

```bash
uv run scripts/eval_soccer.py --checkpoint $P/checkpoints/model_1999.pt \
  --repeats 3 --out logs/eval_soccer/my_B_seed1
```

100 個固定配置各考 3 次，約 30 秒。輸出在 `logs/eval_soccer/my_B_seed1/summary.json`，終端機會印出各項比例。

**重測會有 1～2 個百分點的差異是正常的**（GPU 計算順序、隨機化）。例如 `model_1999.pt` 原本是 95.3%，重測過一次是 96.3%。這包裡的正式數字在 `$P/results/B_seed1/summary.json`。

## 3. 畫學習曲線（固定考題版）

訓練時每 100 輪存了一次檔（`model_0.pt` … `model_1999.pt`，共 21 個）。全部拿去考同一份題，就能得到「訓練量 vs 真正的進球率」，可以拿來和方法 C 比學習速度。

```bash
for f in $P/checkpoints/model_*.pt; do
  name=$(basename $f .pt)
  uv run scripts/eval_soccer.py --checkpoint $f --repeats 3 --out logs/eval_soccer/B_seed1_curve/$name
done

uv run python - <<'EOF'
import json, re
from pathlib import Path
rows = sorted(
    (int(re.search(r"model_(\d+)", str(f)).group(1)), json.loads(f.read_text())["rates"]["goal"])
    for f in Path("logs/eval_soccer/B_seed1_curve").glob("model_*/summary.json")
)
for it, goal in rows:
    print(f"{it:5d}  {goal:.3f}")
EOF
```

21 個存檔約 10 分鐘。實測：`model_100` 是 0.000，`model_1999` 是 0.963。

**為什麼不直接看訓練曲線？** 方法 C 前 174 輪練的是比較簡單的配置，訓練中的成功率本來就比較高，和 B 不能直接比。放到同一份考題上評估才公平。C 也要對自己的每個存檔做同樣的事。

## 4. 看訓練曲線

```bash
uv run tensorboard --logdir $P/raw_logs
```

瀏覽器打開它印出的網址（通常是 http://localhost:6006）。重點曲線：

| 曲線 | 意思 |
| --- | --- |
| `Train/mean_episode_length` | 每局撐幾步，**一步 0.1 秒**，120 = 超時。第 5–140 輪貼在 120，就是「不踢階段」 |
| `Episode_Termination/valid_goal` | 每輪有幾局以有效進球結束 |
| `Episode_Termination/ball_out_of_bounds`、`time_out`、`fell_over` | 各種失敗原因的次數 |
| `Episode_Reward/*` | 每條評分規則的得分，已乘權重、換算成每秒；扣分項必須 ≤ 0 |

不想開 TensorBoard 的話，`$P/learning_curves/B_seed1.csv` 是同樣的資料，每列一輪，可以用 Excel 打開。

## 5. 和方法 A 比較、自己訓練

```bash
# 方法 A（幾何規則），踢球門檻 5 cm，約 40%
uv run scripts/eval_soccer.py --baseline --kick-tol 0.05 --repeats 3 --out logs/eval_soccer/my_A

# 自己訓練方法 B 的另一個 seed（2048 環境約需 4 GB 顯示卡記憶體、約 5 小時）
uv run train Mjlab-Soccer-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max_iterations 2000 --agent.seed 2 --agent.run-name seed2

# 方法 C（加課程）
uv run train Mjlab-Soccer-Curriculum-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max_iterations 2000 --agent.seed 2 --agent.run-name seed2
```

要和 seed 1 比較時，環境數、輪數一定要相同（2048、2000）。

## 這包裡有什麼

| 路徑 | 內容 |
| --- | --- |
| `00_START_HERE_給組員.txt` | 一頁總結與 B／C 核對結果 |
| `run_info_B_seed1.json` | commit `7fa6298`、seed 1、2048 × 24 × 2000、未中斷、啟動指令 |
| `training_configs/` | 實際的 `agent.yaml`、`env.yaml`，以及和方法 C 的逐行比對（只差課程設定） |
| `checkpoints/` | 21 個存檔（每 100 輪一個）＋訓練時匯出的高階 ONNX |
| `evaluation_set/` | 100 個固定配置，SHA256 `3a59dee2…`，和方法 C 用的相同 |
| `results/` | A、B、C（在同一台電腦重測）的 `summary.json` 與逐局 `per_episode.csv`／`.pt`，以及方法 A 的門檻掃描 |
| `learning_curves/B_seed1.csv` | 每輪的所有訓練曲線 |
| `raw_logs/` | TensorBoard 原始檔 |
| `analysis/B_vs_C_seed1_analysis.md` | B 和 C 的完整比較分析 |
| `figures/` | 報告用的圖 |
| `reproducibility/` | 套件版本、GPU、技能雜湊、`uv.lock` |

`per_episode.csv` 每列一局：`layout`（0–99）、`repeat`（0–2）、各結果欄是 0／1、`time_s` 是秒數。想找失敗的局，篩選 `goal = 0`。

## 注意

- `.gitignore` 預設不收 `.pt`、`.onnx`。這包的存檔是用 `git add -f` 加進來的；之後要放新的模型進 `results/` 也要加 `-f`。
- pytest 或其他指令出現 `No module named 'lark'` 之類的錯誤，是 ROS 的 `PYTHONPATH` 在干擾，指令前面加 `env -u PYTHONPATH`。
- 更完整的足球任務說明、方法 C 的訓練接力步驟，見 `docs/soccer.md`。
