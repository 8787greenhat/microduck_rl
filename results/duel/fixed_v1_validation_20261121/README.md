# 雙鴨固定對戰 v1：工具驗證與規則基準結果

先讀 [環境／指令](../../../docs/duel.md)，再看本頁與 baseline_A_vs_A/summary.json、per_episode.csv。
固定題目在 benchmarks/duel/v1/；後續所有模型沿用同一版本與 hash。

## 實際完成

- CPU：11 項新雙鴨檢查 + 原有 30 項單鴨足球檢查，共 41 項通過。
- 原 MDP 的 275 個函式／類別 AST 完全相同；單鴨 B/C cfg、adapter、eval、uv.lock 與依賴宣告未改。
- A 對 A：100 場景 × 3 次 × 2 進攻方向，共 600 局，returncode=0。
- 可分享的 PPO 入口：64 環境／5 iterations；actor、critic 各 6 個 MLP tensor 改變，100 optimizer steps，有限值與 checkpoint 通過。
- 短模型也完成同樣 600 局；確認 deterministic actor 與 normalization 沒被評測修改。它放在 PPO_5_iterations_validation_only/，僅驗證工具，不當作對抗學習成果。
- 沒有啟動長訓練、其他 seed 或 self-play。

短模型的 600 局實測是己方進球 1、失球 182、球出界 263、超時 154。這只有 5 次更新，尚未學成，不用這組分數評斷 PPO 方法好壞。

## 規則 A 對 A 的固定 600 局

| 結束原因（互斥） | 局數 | 占全部局數 |
|---|---:|---:|
| 待測 A 進球 | 47 | 7.83% |
| 固定對手 A 進球 | 59 | 9.83% |
| 球出界 | 413 | 68.83% |
| 超時 | 74 | 12.33% |
| 待測鴨摔倒 | 1 | 0.17% |
| 對手摔倒 | 3 | 0.50% |
| 待測鴨出界 | 1 | 0.17% |
| 對手出界 | 2 | 0.33% |

雙方合計 106/600 局（17.67%）有人進球；球出界 413/600（68.83%）是最大類別。
待測 A 的成功進球時間平均 6.225 秒、中位 5.080 秒、p90 8.836 秒；這些時間只描述 47 個成功局。
這次評測本身的 rollout／環境建立耗時約 132.3 秒，Python／CUDA 初始化另計。

106 個進球局都有觀測到右腳接觸，也都有至少一次踢球觸發。
其中 54 局最後觀測到的身體觸球方是該球門的守方，值得回放追查；這個感測分類不代表已證明 54 次主動烏龍踢球。
進球規則允許烏龍球，不用觸球感測當作硬計分門檻；raw overlap flags、觸球取樣數與 kick attempts 都在 CSV。

這是「固定規則在新對抗環境的基準」，不是兩個 RL 模型的比較。
同一份題目含重複幾何／BAM 樣本，不把 600 局當作 600 個獨立 training seeds。
此版只涵蓋靜止球起始、基本場內位置與固定 A；移動球起始、其他對手、self-play 尚未測。
雙鴨與原單鴨 B/C 的分數分開呈現。

## 版本與來源

- 測試程式 commit：d3c06eda69022ee950f5e7ec8b6a3a29fda3ec2b（之後的提交只補結果／文件與檔案保存規則）。
- 分支：joshua/duel-fixed-evaluation-v1；base：1a8a1053533210d6695454cf51026551f3e4dbbc。
- 固定 layouts SHA256：02e8cafd3413af366f94f5940e67c3ba03ba8c3aa374b0ae69a241e97dfcadd9。
- Python 3.12.14、PyTorch 2.9.1+cu128／CUDA 12.8、mjlab 1.3.0、MuJoCo 3.10.0、mujoco-warp 3.8.1、Warp 1.12.0、rsl-rl 5.0.1。
- RTX 4060 Laptop 8GB、Windows NVIDIA driver 610.62、Ubuntu WSL2。沿用已驗證的本機 driver library path，沒有將使用者路徑寫進 repo 工具。
- 官方 walking／right kick ONNX 的 SHA256 見各 manifest；兩個檔案與記憶體權重未更新。
- uv.lock 的 Git canonical SHA256：2eeeb680025baa737e7ccf3a87da3695a9d49b228de0bb60d8af0c022ab5f1aa。
  Windows checkout 的 raw SHA 不同是 CRLF 換行；已核對 normalize 後完全相同，沒有升級依賴。
- 64×5 checkpoint SHA256：484276d4aa6474e5dfb949720f1543892a2aeea7c00f42c016a9d34d5e589d0c。短模型二進位留在本機，PR 提供驗證資料，隊友可用相同指令自行生成。

baseline manifest 的 source_dirty=true 來自 Windows／WSL Git 換行設定不同；git diff --ignore-space-at-eol 為空，明確設 local core.autocrlf=true 後狀態乾淨。各測試 source SHA256 完全相同，詳見 extended_verification.json。
第一次 portable 短測試在 import torch 時發生 SIGILL（−4），尚未建立環境或更新；保存證據後用原環境重試成功，原因未定位。
未改套件、未將這次例外包裝成已修復的驅動問題；先保留此紀錄供長訓練前檢查。

## 下一步再決定長訓練

1. 先請和睦 pull 同分支，跑 8 局 eval smoke 與 64×5 training smoke，核對版本／技能／checkpoint。
2. 選幾個球出界或守方最後觸球的場景回放，確認是規則追球、干擾、踢球姿勢或場地條件造成。
3. 一起決定是否維持 v1、訓練預算與 seed 分工；若調環境或計分，另開 v2 固定協定，保留這份 v1 結果。
4. 正式跑 PPO 對固定 A 後，用相同協定評測、比較進球／失球／安全中止和時間；self-play 留到這一步可靠後。
