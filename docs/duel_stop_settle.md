# 雙鴨：到位減速、停穩再踢（短測候選版）

原 v2 的最終模型在固定 600 局中沒有觸發我方踢球技能。診斷顯示，位置與角度正確時仍要求快速走動，實際速度中位數約 0.22 m/s，無法通過原本 <0.05 m/s 的出腳檢查。

這次新增獨立、可選用的 `DuelStopSettle`，**不修改原 `DuelV2`、舊 checkpoint、官方技能、獎勵、進球規則或原 600 局結果**。使用既有 v2 actor，觀測維持 70 維；尚未訓練新 PPO，也沒有開啟 curriculum 或 self-play。

## 控制流程

1. **FOLLOW**：PPO 繼續決定走位；遠處的踢球請求不會直接送進技能。
2. **ALIGN**：距準備點 <20 cm、朝向誤差 <0.5 rad、球速 <0.15 m/s 時，鎖存附近的出腳意圖。用既有準備位移／角度，以慢速比例控制對準，平面指令最高 0.12 m/s、轉向最高 0.5 rad/s。
3. **SETTLE**：準備點距離 <4.5 cm、角度 <0.12 rad 時，送出零走路速度指令，由原官方走路技能保持平衡；沒有把關節動作歸零。
4. 原位置、角度、鴨子速度、球速檢查全數通過且停穩取樣持續至少 0.3 秒，才送出踢球請求。**原 v2 guard 仍是最後檢查，沒有放寬或繞過。**
5. 踢球沿用官方右腳技能與原 2 秒鎖定。漂移時回到 ALIGN；球離開、滾太快或 3 秒內未完成準備時取消意圖；踢球期間及結束後短暫冷卻，避免連續誤觸發。

這是手寫的高階控制器修正，並不表示原 PPO 自己學會停步或射門。原始 PPO 動作、實際套用動作、控制階段、出腳當下的真實速度與觸球紀錄都會保存。

## 短測方式

在 Linux／WSL2＋CUDA 環境執行。PR 尚未合併，先切結果分支；依原 Python 3.12 和 `uv.lock` 安裝，不升級依賴。
本分支已附這次使用的 v2 seed 1 最終 checkpoint，不需要重新訓練或另外向他人索取。單鴨 B/C 或雙鴨 v1 的 checkpoint 不能替代它。

```bash
git fetch origin
git switch joshua/duel-stop-settle
uv sync --frozen --python 3.12

# 官方低階技能仍須放在 policies/，與原 v2 相同；啟動時會驗證原 SHA256。
uv run --frozen hf download pollen-robotics/microduck-policies alpha_walking.onnx ball_kick_right.onnx --local-dir policies

# 只用 Python 標準函式庫，核對結果包、checkpoint 與控制器來源；不使用 GPU。
uv run --frozen python results/duel/stop_settle_validation_20261122/verify_package.py
uv run --frozen python tests/test_duel_stop_settle.py
uv run --frozen python tests/test_duel_v2.py

# 上限是 64 個環境、200 個高階步（每局最多20秒），沒有 learn()/PPO更新。
uv run --frozen python scripts/check_duel_stop_settle.py \
  --checkpoint results/duel/stop_settle_validation_20261122/inputs/model_1999.pt \
  --source synthetic_request --layout approach --side 0 --out logs/stop_settle_control_01

# 載入相同、凍結的 PPO actor；控制器套用到 PPO 真實請求。
uv run --frozen python scripts/check_duel_stop_settle.py \
  --checkpoint results/duel/stop_settle_validation_20261122/inputs/model_1999.pt \
  --source policy --layout approach --side 0 --out logs/stop_settle_policy_01

# 原固定場景的前32組各2次，64局對戰短測；不是正式600局評測。
uv run --frozen python scripts/check_duel_stop_settle.py \
  --checkpoint results/duel/stop_settle_validation_20261122/inputs/model_1999.pt \
  --source policy --layout fixed --side 0 --out logs/stop_settle_fixed64_01
```

輸出資料夾必須不存在；重新測試時換一個新名稱。上述都只有推論，每次最多64個環境、200個高階步，沒有 optimizer 更新。
物理接觸可能使跨機器／不同執行的個別終局略有差異，請保留每次 `result.json`，不要覆蓋或只回傳最好的一次。
若官方下載模型的 hash 不符，請保留錯誤並核對版本，不要用其他技能或升級套件來繞過檢查。

請和穆回傳：三個短測資料夾的 `result.json`、執行 commit、GPU／driver／Python 及 torch、mjlab、mujoco-warp、warp 版本；若失敗，附完整 console。模型 hash 已包含在結果中。

`--controller original_v2` 可測原控制流程。`--side 1` 使用旋轉換邊後的右腳技能情境，沒有把動作鏡射成左腳。

受控案例把球設為靜止、對手放遠，包含精確準備位置與後退 4、10、16 cm 的四組起點。它們驗證控制流程，**不能拿其進球率冒充真實對戰表現**。
結果同時列出全部進球，以及「出腳後有右腳接觸、最後觀測觸球為我方、且進球」的局數；兩者不同，不把所有進球都稱為射門成功。

## 版本與後續工作

- 基於 v2 commit `3c6bcc768790e1bd4a0f3f492eba0df23fe92e65`，新增 `stop_settle.py`、`stop_settle_env.py`、短測腳本與 CPU 測試。
- 凍結 actor 為既有 seed 1 `model_1999.pt`，SHA256 `52fcc4eed3e7428f78e4b006365e49ebb1f1dc8c982259f3f6e88192bda8823a`。
- 該模型包含 actor、normalizer 與原 checkpoint state，放在結果包 `inputs/model_1999.pt`；是舊模型的逐位元複本，沒有重新訓練。
- 每次短測驗證 actor／normalizer 未改、optimizer state 沒有更新、技能 hash 一致、所有出腳都通過原 guard。
- 這個候選控制器只由新短測入口明確選用。原 `train_duel_v2.py` 不會自動套用它，也沒有新增長訓練入口。
- 未來訓練前，仍需決定控制器狀態的觀測契約、訓練／推論一致性，再做固定對戰評測。不能沿用舊 v2 的 106/600 當作此候選版的新分數。

## 本次實測

| 短測案例 | 出腳局數／64 | 全部進球 | 出腳後有右腳接觸且進球 |
|---|---:|---:|---:|
| 原 v2＋凍結 PPO：準備位置 | 0 | 43 | 0 |
| 新流程＋固定請求：準備位置 | 56 | 41 | 41 |
| 新流程＋凍結 PPO：準備位置 | 52 | 49 | 43 |
| 新流程＋固定請求：旋轉換邊 | 58 | 39 | 39 |
| 新流程＋凍結 PPO：前32固定場景各2次 | 27 | 1 | 1 |

完整逐局與出腳事件見 [短測結果包](../results/duel/stop_settle_validation_20261122/README.md)。

`scripts/check_duel_stop_settle.py` 的短程序目前停用循環垃圾回收（GC），用來避開本機曾在 PyTorch import 時遇到的程序中止；沒有改 actor、Torch RNG 或物理設定。這項隔離程序處理尚未在隊友機器驗證。
