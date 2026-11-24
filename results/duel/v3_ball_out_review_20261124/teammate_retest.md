# 和穆：用現有v3 seed 1模型重測，不訓練

本次要確認另一台機器的評測結果；不用重訓，也不要換成PR #8／v2的70維模型。
取得PR #9分支（如果有本機未提交修改，先保留，不要reset或覆蓋）：

```bash
git fetch origin
git switch joshua/duel-v3-training
git pull --ff-only
git rev-parse HEAD
python results/duel/v3_seed1_20261124/verify_package.py
```

沿用Python3.12、原uv.lock與相同官方walking／right-kick技能；既有環境不要重裝／升級。
checkpoint SHA256應為 `f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2`；固定場景SHA256應為 `02e8cafd3413af366f94f5940e67c3ba03ba8c3aa374b0ae69a241e97dfcadd9`。

在repo根目錄執行（輸出資料夾必須是新的，存在就換名稱）：

```bash
uv run --frozen python scripts/eval_duel_v3.py --checkpoint results/duel/v3_seed1_20261124/checkpoints/model_1999.pt --out logs/dennis_v3_seed1_fixed600_20261124
```

不要加 `--smoke`；原eval自帶100場景×3次×雙方向、RNG seed0與相同計分，沒有要調參數。
這只做推論，不更新模型／optimizer，也不是訓練seed2。
請把整個新輸出目錄壓縮回傳，至少包括根層summary.json、manifest.json、per_episode.csv、status.json，及兩個方向的原始JSON／日誌。
另附git commit、Python完整版本、GPU／驅動（nvidia-smi）、套件版本。不要只回報進球數，方便檢查場景／模型hash、終局分類和出腳數。

Joshua原正式結果：167/600進球、7失球、402球出界、521局曾啟動我方踢球。
另一次帶記錄的診斷推論有分歧，不承諾相同seed逐局一致；重測用原官方入口，保留不同結果即可，不用反覆跑到數字一樣。
