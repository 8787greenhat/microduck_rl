# PR #8 停穩再踢：和穆的跨機器短測（2026-11-23）

照 `docs/duel_stop_settle.md` 執行，commit `042b56d4`，沒有改任何程式、沒有訓練。

## 結論

三組都完整跑完（64／64 局結束、0 未完成），控制器來源與 checkpoint 的 SHA256 和你的結果完全相同，actor／normalizer 未改、原 v2 guard 沒有被繞過。**出腳行為在兩台機器上一致**；進球數有差異，屬於物理接觸的跨機器變異範圍。

## 對照

| 短測 | 出腳／64 | 全部進球 | 出腳＋右腳接觸＋進球 |
| --- | --- | --- | --- |
| 1 固定請求＋準備位置 | 你 56 ／ 我 57 | 41 ／ 34 | 41 ／ 34 |
| 2 凍結 PPO＋準備位置 | 52 ／ 55 | 49 ／ 48 | 43 ／ 44 |
| 3 凍結 PPO＋固定場景前 32 組×2 | 27 ／ 26 | 1 ／ 4 | 1 ／ 1 |

結束原因（`episodes[].end_reason`）：

| 短測 | 你 | 我 |
| --- | --- | --- |
| 1 | goal_for 41、ball_out 19、learner_oob 3、goal_against 1 | goal_for 34、ball_out 26、learner_oob 3、goal_against 1 |
| 2 | goal_for 49、ball_out 14、timeout 1 | goal_for 48、ball_out 15、goal_against 1 |
| 3 | ball_out 54、goal_against 6、goal_for 1、learner_oob 1、opponent_oob 1、timeout 1 | ball_out 55、goal_against 3、goal_for 4、opponent_oob 1、timeout 1 |

## 觀察

- **出腳數幾乎相同**（差 1–3 局），控制器的狀態流程在兩台機器上行為一致。
- **短測 1 的進球少 7 局**（34 vs 41），多出來的都變成 ball_out。出腳數一樣，差在踢出去之後的方向或觸球品質；兩者的轉換率（60% vs 73%）差約 1.5 個標準誤，以 64 局來說不算異常，但值得留意。
- **短測 3 我這邊 goal_for 4 局，但只有 1 局符合「出腳＋右腳接觸」**，另外 3 局不是我方射門進的（可能是碰撞或對手觸球後滾進）。兩邊符合射門條件的都是 1 局。
- **真實對戰的主要問題一樣是球出界**：兩邊都約 55／64，和 v1 A 對 A 的 413／600 是同一個問題。

## 其他

- `logs/stop_settle_control_01/` 在我電腦上是一個空資料夾，是先前一次沒跑完的嘗試留下的。我沒有覆蓋它，短測 1 改用 `stop_settle_control_02`。
- CPU 檢查三項全部通過：`verify_package.py`、`test_duel_stop_settle.py`（8 項）、`test_duel_v2.py`（12 項）。
- 這台是原生 Ubuntu，不是 WSL；GC 停用的處理在這台沒有遇到問題。

## 檔案

- `results/<資料夾>/result.json`：三組短測的完整輸出
- `console/`：六個步驟的完整終端機輸出（3 個 CPU 檢查、3 個短測）
- `versions.txt`：commit、GPU／驅動、Python、torch、mjlab、mujoco-warp、warp、uv.lock 與技能的雜湊
