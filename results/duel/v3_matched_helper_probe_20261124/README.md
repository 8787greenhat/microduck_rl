# v3 seed 1：相同到位停穩控制器的512局比較

先看本檔，再看 `raw/comparison_summary.json` 與 `raw/all_512_episodes.csv`。
本輪只推論，沒有重新訓練或更新模型。P＝凍結v3 seed1 PPO決策；R＝幾何準備位置規則；H＝原v3 FastStopController。

| 每組128局 | 藍色進球 | 橘色進球 | 球出界 | 超時 |
|---|---:|---:|---:|---:|
| P＋H 對 原版A | 44（34.4%） | 3 | 75 | 4 |
| P＋H 對 R＋H | 23（18.0%） | 1 | 99 | 4 |
| R＋H 對 原版A | 38（29.7%） | 6 | 79 | 4 |
| R＋H 對 R＋H | 11（8.6%） | 13 | 102 | 1 |

其餘為跌倒或機器人出界；完整互斥終局分類見summary。所有128局都在分母，並非只計有進球的局。

## 這輪能說什麼

相同控制器時，固定PPO對此規則對手仍有23局進球、1局失球的優勢跡象；但99/128球出界，尚未穩定完成對戰。
R＋H對舊A也有38局進球、6局失球，顯示規則搭配新流程的整套基準本身就能對舊A取得領先，先前優勢不能全部歸功PPO。
R＋H使用幾何走位並持續提出出腳意圖，交由H與原guard判斷是否接受；它是新的規則基準，不能視為原A只切一個開關的純因果消融。
這些結果不能相減成「幾％來自AI、幾％來自控制器」，也沒有統計顯著性或普遍優越的主張。
PPO只有一個training seed，且原本對舊A訓練；換成R＋H也同時改變了對手分布。兩邊均使用同一套官方走路／右腳踢球技能，沒有第二隻學習模型或self-play。

## 相同的控制器，具體相同在哪

- P＋H與R＋H直接使用同一個 `mjlab_microduck.duel.v3_env.FastStopController` 類別與設定，未另寫近似版。
- 捕捉距準備點0.12m、接近指令0.30m/s、位置deadband0.045m、角度deadband0.12rad、停穩0.3秒、意圖逾時3秒、冷卻0.5秒。
- 真正出腳仍通過原guard：距離<0.05m、角度誤差<0.15rad、身體速度<0.05m/s、球速<0.15m/s，ready hold 0.2秒。
- 規則對手的每次技能啟動都檢查guard、SETTLE階段與零走路指令；違規次數為0。reset會清除對手計時／意圖，換邊時只接管對手的欄位。
- 全模擬guard計數包含部分world完成首局後的後續回合，不能當成正式128局出腳數；正式局數用每局kick_attempts統計。

## 設定與前兩輪分開

原固定場景0–31、repeat 0與1、兩個方向，32×2×2＝128局／組，四組共512局；每方向64平行環境，eval RNG seed0、20秒上限。
保持原名義物理、技能與進球／終局規則。進球允許非技能碰球與烏龍球，進球局比例不是射門準確率。
本輪P＋H對舊A新對照為44/128；先前384局探測的36/128、正式600局的167/600各自保留，沒有改寫或混算。
不同程序／接觸數值敏感性尚未隔離，相同seed不保證逐位重播；只比較本輪四組，不把跨輪差異稱作訓練進步。

來源commit：`e8b9756c03ae66a693275ee84b61a36aab367b2f`。模型SHA256：`f50f7fade3a12abea06f12acdfc904d2029a57238ed864aaa940e73a1df9cfc2`。
Python3.12.14、原uv.lock及官方技能，完整套件版本／來源hash見各manifest。R＋H條件也載入相同actor以維持初始化程序，但不使用actor決策；manifest的actor_used已區分。

## 檔案與驗證

- `raw/`：原始57檔，包含512局CSV、八組逐局JSON／manifest／summary／日誌、原始wrapper與原file_manifest，位元組不改寫。
- `raw/公平控制器比較結果.txt`中「本機、未上傳」是產生報告時的歷史狀態。
- `reproduce/`：去除本機絕對路徑的推論入口、原樣共用控制器模組與7項CPU測試。路徑／來源hash檢查以外的推論邏輯不變。
- `verify_package.py`與頂層`file_manifest.json`：純Python重算512局與核對包內hash，不需GPU。

原測試7項CPU檢查通過、8個worker returncode0、512局完整且唯一，actor／normalizer與checkpoint前後未變。
可攜入口另驗證CPU檢查、CLI及來源hash，發布時沒有再重跑GPU512局。

在repo根目錄核對已保存結果：

```bash
python results/duel/v3_matched_helper_probe_20261124/verify_package.py
```

## 隊友需要重測時

使用PR #9分支 `joshua/duel-v3-training`，沿用Python3.12、uv.lock、相同官方技能與既有CUDA環境，依 `docs/duel_v3.md` 準備，不任意升級套件。
先執行CPU檢查，再按需要用新的輸出目錄推論：

```bash
uv run --frozen python results/duel/v3_matched_helper_probe_20261124/reproduce/test_duel_matched_helper.py
uv run --frozen python results/duel/v3_matched_helper_probe_20261124/reproduce/run_probe.py --out logs/duel_matched_helper_recheck_new
```

入口會核對既有最終checkpoint、原來源檔案與uv.lock，依序跑四組／兩方向並保存全部輸出；不訓練、不使用optimizer。輸出資料夾若已存在，請換新名稱。
本次發布沒有修改repo訓練／正式評測來源、PPO、低階技能或原600局資料，也沒有啟動新訓練或其他seed。
