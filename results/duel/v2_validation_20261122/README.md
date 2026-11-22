# v2 長訓練前驗證

先看 docs/duel_v2.md。這些是程式／模型更新／出腳檢查／評測流程驗證，尚不能宣稱 PPO 學會對抗。

- 新增 12 項 CPU 測試，加上既有 v1／soccer 41 項回歸檢查通過。
- v1 與單鴨 B/C 原有 289 個 MDP 函式／類別保持一致；依賴、固定場景與舊 task source 未改。
- 64×5 與 1024×5 PPO 真正更新 actor/critic，100 optimizer steps；值有限、std 在 [0.05,0.6]、learning rate 固定 3e-4、技能未更新。
- GPU 驗證遠距離請求不亂踢、20 秒正確終局且不 bootstrap、時鐘與計時器重設。
- 人工放到準備姿勢、由幾何控制器送指令的正向技能案例，64/64 能出腳並進球。這不是學習型 PPO 的成績，也不是固定 600 題。
- 官方 mjlab ONNX exporter 已驗證 actor70 正規化與 bounded deterministic 路徑，ONNX Runtime 與 Torch 相符。
- eval_smoke8/ 只是 4 場景 × 1 次 × 2 方向，共 8 局的腳本檢查，不是正式分數。
- 正式 seed 1 仍從頭學 1024 環境／2000 iterations，不載入這些短測試 checkpoint；完成後才用原 v1 的 600 題比較。

部分 pre-commit manifest 的 source_dirty=true 如實保留；正式工作使用已發布 commit 的乾淨獨立 Linux checkout。原 v1 checkpoint 與結果完全保留。
