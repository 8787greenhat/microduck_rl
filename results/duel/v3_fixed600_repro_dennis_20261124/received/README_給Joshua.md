# PR #9 v3 seed 1：和穆的固定 600 局重測（2026-11-24）

照 `results/duel/v3_ball_out_review_20261124/teammate_retest.md` 執行：官方入口 `scripts/eval_duel_v3.py`，不加 `--smoke`，沒有訓練、沒有改任何程式或參數。只跑了一次。

## 核對

- commit `6908e64d`，分支 `joshua/duel-v3-training`，working tree clean
- `verify_package.py`：verified，75 個檔案、21 個 checkpoint
- checkpoint `f50f7fad…`、固定場景 `02e8cafd…`、`uv.lock`、兩個官方技能的 SHA256 都與說明一致
- 輸出 `manifest.json` 的 `checkpoint_sha256`、`protocol_manifest_sha256`、`uv_lock_sha256` 和你的正式結果相同
- `eval_duel_v3.py` returncode 0，600 局全部完成，約 129 秒

## 結果對照

| 600 局 | 你的正式結果 | 和穆重測 |
| --- | --- | --- |
| 進球 | 167（27.8%） | 160（26.7%） |
| 失球 | 7 | 15 |
| 球出界 | 402 | 405 |
| 超時 | 22 | 16 |
| 對手跌倒 | 2 | 2 |
| 對手出界 | 0 | 2 |
| 曾啟動我方踢球的局數 | 521 | 520 |
| 進球（side 0 ／ side 1） | 87 ／ 80 | 78 ／ 82 |

- **總進球率差 −1.2 個百分點**，兩者差的標準誤約 2.6 個百分點，約 0.45 倍，總量上一致。
- **出腳局數幾乎相同**（521 vs 520），球出界也幾乎相同（402 vs 405）。
- **失球 7 → 15** 是差最多的一項。次數少，波動本來就大（約 1.7 倍標準誤），但值得記一下。

## 逐局一致性

用 `(scene_id, repeat, learner_side)` 對齊兩份 `per_episode.csv`：

- 結束原因相同：**475 ／ 600（79%）**
- 兩邊都進球：128 局；只有你進球：39 局；只有我進球：32 局

總數接近，但大約五分之一的局結果不同。這和你在出界分析裡看到的「291／300 局至少一項紀錄不同、65 局終局不同」是同一類現象：同一模型、同一 RNG seed 在不同機器上，逐局軌跡並不一致。比較時應該看 600 局的總量，不能拿單局去對。

## 兩台機器的差異

| | 你 | 我 |
| --- | --- | --- |
| GPU | RTX 4060 Laptop 8 GB | RTX 4050 Laptop 6 GB（驅動 595.99.02） |
| 系統 | WSL Ubuntu | 原生 Ubuntu，kernel 6.8.0-149 |
| Python | 3.12.14 | 3.12.3 |
| torch／mjlab／mujoco-warp／warp | 2.9.1／1.3.0／3.8.1／1.12.0 | 相同 |

評測開始前，我的 GPU 已經有約 1.4 GB 被其他程式占用，評測本身沒有出錯。

## 檔案

- `dennis_v3_seed1_fixed600_20261124/`：官方入口的完整輸出，未經修改（根層 `summary.json`、`manifest.json`、`per_episode.csv`、`status.json`，以及 `side_0/`、`side_1/` 原始 JSON 與日誌）
- `console/`：`verify_package.py` 與 600 局評測的完整終端機輸出
- `versions.txt`：commit、執行指令、各項 SHA256、nvidia-smi、Python 完整版本與套件版本
