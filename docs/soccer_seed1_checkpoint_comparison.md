# B/C seed 1 固定評測的 checkpoint 比較

已使用原 100 場景各 3 次，在 RTX 4060 上評測 B/C 各 21 個 checkpoint，沒有重新訓練。

- model_100：B 0%，C 38.7%；C 較早在這個存檔點出現進展。
- 第一次測得至少 70%：B/C 都是 model_200。
- 第一次測得至少 90%：B model_200，C model_400。
- 最終結果保留 B 95.7%、C 94.0%，成功進球平均 3.180／3.321 秒。

這是單一 training seed 的描述性比較。原 evaluator 的 checkpoint 步數恢復、共用 CoM 課程，以及 C 的中斷續訓都保留並揭露；不能由這條曲線推論方法的一般性優劣。

![B/C fixed-test learning curves](../results/soccer/bc_seed1_checkpoint_eval_20261025/learning_speed.png)

[先看結果包 README](../results/soccer/bc_seed1_checkpoint_eval_20261025/README.md)，內含完整逐局資料、checkpoint manifest、C 的 21 份模型、核對工具及重現限制。
