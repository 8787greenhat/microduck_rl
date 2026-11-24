# 雙鴨 v3 seed 1：完訓與固定 600 局結果

先看本檔，再看 `comparison/comparison.json` 和 `evaluation/fixed600/per_episode.csv`。
本包是 **高階 PPO 對固定規則 A**；沒有 curriculum、沒有 self-play，對手沒有同步學習。
這是雙鴨實驗，與單鴨方法 B／C 的評測分開。

## 結果

| 相同固定評測 | v2 seed 1 | v3 seed 1 |
|---|---:|---:|
| 進球局數／全部局數 | 106/600（17.67%） | 167/600（27.83%） |
| 失球局數 | 13 | 7 |
| 球出界 | 461 | 402 |
| 超時 | 9 | 22 |
| 成功局進球平均秒數 | 5.655 | 3.768 |
| 成功局進球中位秒數 | 4.920 | 3.600 |
| 成功局進球 p90 秒數 | 8.290 | 4.192 |
| 我方曾啟動踢球技能的局數 | 0 | 521 |

進球比例增加 **10.17 個百分點**。v3 的 167 個進球局中，164 局曾啟動踢球技能，167 局曾記錄我方右腳觸球。
這是「同一局曾發生」的紀錄，不能直接斷言每個進球都由踢球技能造成。
成功進球時間只計成功局，兩版的成功場景集合不完全相同，不能視為每一題都加速。
主要問題仍是 **402/600（67%）球出界**，尚未達到穩定控球對抗。

固定100場景 × 每場3次 × 旋轉換邊2個方向，共600局；每個方向300局。
沿用原 `microduck-duel-fixed-v1`，eval RNG seed 0、20秒上限、相同官方技能／名義物理／進球規則。
靜止球開場，策略用凍結正規化的 deterministic mean；只統計每個 world 的第一局。
所有出界、跌倒、超時都保留在600局分母，不只挑有進球的局。
v3 正向87/300、換邊80/300；完整終局分類保存在官方 summary 中。

## 訓練與來源

- RTX 4060 Laptop；Python 3.12.14、既有 `uv.lock`；未升級套件。
- source commit：`e8b9756c03ae66a693275ee84b61a36aab367b2f`（發布結果前的 PR #9 程式）。
- 新初始化 seed 1，1024環境、2000 iterations；最後索引1999。2026-11-24約09:18至14:18（台灣時間），最終評測約14:20完成。
- train 與最終固定600局 eval returncode 均為0；actor/critic確實更新，最終40000 optimizer steps；checkpoint張量與67994筆TensorBoard scalar均有限。
- 21個checkpoint均保存：model_0、100、200…1900、1999。索引從0開始，model_100是101次更新後，並非100次。
- v3高階actor/critic各76維，低階走路／踢球61維契約不變；不要用v2的70維入口載入v3模型。
- 具體PPO／控制器設定、來源及模型hash見 `training/manifest.json`、`run_config.json`、`provenance.json`。
- 最終checkpoint SHA256：`f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2`。

v3加快近球準備點接近，並讓PPO觀察停穩流程狀態；保留到位、角度、身體速度與球速出腳檢查。
本輪是整體控制器／觀測改版，各版只有一個training seed，不能單獨歸因走路速度或宣稱統計顯著。
三次場景重複不是三個training seeds；不能用此結果宣稱curriculum有效。

## 尚未完成：中途學習曲線

**完整訓練與最終600局已完成，中途checkpoint評測未完成。**
model_100 的 side 0 worker 在載入 PyTorch 階段發生 segmentation fault（-11），父程序returncode 1，根因尚未確定。
model_500／1000的評測未啟動。原始日誌與狀態保留於 `failure_evidence/`；本包不補造中途分數，也未重訓／重跑評測。
`training/status.json` 的 `state: complete` 需搭配 `intermediate_evaluation_complete: false` 解讀。
checkpoint可供後續重測，但本次只能報告最終分數，不能聲稱有完整v3學習曲線。

## 檔案與使用方式

- `checkpoints/`、`checkpoint_index.json`：21個權重及SHA256／更新次數／評測狀態。
- `evaluation/fixed600/`：最終官方summary、600局CSV、雙方向原始JSON／日誌。
- `comparison/v2_fixed600/`：未改寫的v2評測輸入；`comparison.json`由兩版CSV重算。
- `evaluation_set/`：原固定場景與規則manifest，layouts SHA256 `02e8cafd3413af366f94f5940e67c3ba03ba8c3aa374b0ae69a241e97dfcadd9`。
- `training/`：原始console、TensorBoard、初始actor、設定／完成狀態／CPU核對證據。
- `file_manifest.json`、`verify_package.py`：套件內所有檔案的SHA256及純Python核對程式。

在repo根目錄，只核對資料（不啟動GPU、不訓練）：

```bash
python results/duel/v3_seed1_20261124/verify_package.py
```

如需另外重測，先依 `docs/duel_v3.md` 準備鎖定依賴及相同官方技能，使用新的輸出目錄：

```bash
uv run --frozen python scripts/eval_duel_v3.py --checkpoint results/duel/v3_seed1_20261124/checkpoints/model_1999.pt --out logs/duel_v3_seed1_recheck_new
```

此命令會執行新的推論評測；不會恢復訓練。checkpoint不包含完整物理/RNG狀態，不承諾跨機器逐位一致。
v2最終權重已在本分支繼承的 `results/duel/stop_settle_validation_20261122/inputs/model_1999.pt`，SHA見comparison manifest。
本次上傳沒有新拍攝影片，600局資料也不是影片rollout的代稱。
