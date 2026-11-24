# v3 seed 1：三種規則對手，384局探索性測試

先看本檔，再看 `raw/comparison_summary.json` 與 `raw/all_384_episodes.csv`。
使用同一個v3最終模型，不訓練、不更新actor／normalizer。橘色都是固定規則，尚未測PPO對PPO，也不是self-play。

| 橘色對手（各128局） | 藍色進球 | 橘色進球 | 球出界 | 超時 |
|---|---:|---:|---:|---:|
| 原版規則A | 36（28.1%） | 2 | 87 | 1 |
| 縮短停穩等待A | 37（28.9%） | 2 | 86 | 3 |
| 遠處直接追球A | 22（17.2%） | 8 | 96 | 1 |

其餘為跌倒或機器人出界，完整互斥分類見summary；所有128局都在分母。
縮短等待後結果接近；直接追球時藍色進球下降、失球增加，顯示這一輪觀察受對手風格影響。三種條件下藍色進球仍多於對手，但大量球出界尚未解決。
新規則沒有獨立校準強度，不能直接稱為更強；單一training seed與少量規則變體，不能證明對任意對手都強或宣稱統計顯著。
藍色是PPO加上v3到位停穩控制器，橘色使用幾何規則；這比較整套控制系統，優勢不能全歸功PPO演算法。

## 測試設定與原600局的差別

- 原固定場景0–31、repeat 0與1、兩個方向，共32×2×2＝128局／對手；三種對手總計384局。
- 每方向64個平行環境，eval RNG seed0、20秒上限；沿用原技能、名義物理、進球／終局優先規則、凍結正規化的deterministic actor。
- 這是**探索性子集測試**，不是原100×3×2＝600局（每方向300個平行環境）。只比較本輪三個條件；不拿新36/128直接與舊167/600當成同題同批次比較。
- 原正式600局結果與原始資料均保留在相鄰 `../v3_seed1_20261124/`，沒有改寫。
- 固定模型為training seed1；兩次場景重複不代表兩個training seeds。

## 對手具體改了什麼

1. `original_A`：原幾何準備位置控制，停穩等待0.5秒。
2. `short_dwell_A`：等待0.5→0.2秒，原位置／角度／身體速度檢查保留；沒有調藍色的出腳guard。
3. `direct_chase_A`：距球超過0.16m時直接追球；接近後回到原準備位置控制與出腳檢查。僅更改橘色指令，保持原速度上限。

原始wrapper在隔離程序中替換幾何對手函式；未編輯repo訓練／正式評測來源。GPU測試前5項CPU檢查通過。
六個worker returncode均0，384局完整、不重複；模型與正規化未變、原訓練來源乾淨。
訓練及本輪來源commit：`e8b9756c03ae66a693275ee84b61a36aab367b2f`。
checkpoint SHA256：`f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2`。
Python3.12.14、原uv.lock與官方walking／right-kick模型，完整版本與hash見六份manifest。

## 檔案與核對

- `raw/`：完整原始45檔（含其原file_manifest），逐局CSV／JSON、六組manifest／summary／日誌、當時執行的原始wrapper；原樣保存。
- `raw/結果說明.txt`的「尚未推送GitHub」是本機分析當時的歷史狀態；本README說明此次發布。
- `reproduce/`：去除本機絕對路徑的推論入口、相同對手模組及CPU測試；沒有修改對手公式、場景或模型推論邏輯。
- `verify_package.py`、頂層`file_manifest.json`：純Python核對所有檔案hash、原包hash與逐局摘要。

在repo根目錄核對已保存結果，不用GPU：

```bash
python results/duel/v3_opponent_probe_20261124/verify_package.py
```

## 隊友要重測時

切PR #9分支 `joshua/duel-v3-training`，沿用Python3.12、uv.lock、相同官方技能與現有CUDA環境，不任意升級套件。
先核對上方結果與相鄰模型包，再執行CPU檢查：

```bash
uv run --frozen python results/duel/v3_opponent_probe_20261124/reproduce/test_duel_opponent_variants.py
```

僅在需要重新推論時，使用新的輸出目錄：

```bash
uv run --frozen python results/duel/v3_opponent_probe_20261124/reproduce/run_probe.py --out logs/duel_opponent_probe_recheck_new
```

此入口依序執行三種對手／兩個方向，輸出每組逐局JSON／CSV、summary／manifest／日誌；不訓練、不使用optimizer。
它會核對模型、原訓練來源檔案及uv.lock hash，接受同一程式的後續資料／文件提交，不要求把Git退回舊commit。
原始384局由`raw/eval_duel_opponent_variants.py`執行；移植入口另外通過CPU檢查、CLI與來源hash核對，發布時沒有再重跑GPU384局。
不同機器或新的推論程序不保證逐位一致，請保留不同結果；不要反覆跑到數字相同。
本次發布沒有啟動新訓練、新seed或新GPU評測。
