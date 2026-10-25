這份曲線包含初始 run 的 iteration 0–200，加上恢復 run 的 201–1999，共2000列。
原 run 的201–267因失敗後沒有對應checkpoint而捨棄，未混入曲線。
effective_high_level_transitions 是有效學習軌跡的預算，不包含捨棄67次所耗的3,293,184筆transitions或smoke／診斷。
recorded_at_utc 保留實際時間；跨run有修復空檔。兩台不同GPU不能只用wall time直接歸因於curriculum。
Curriculum/reset_layout/goal_rate 是訓練階段累計統計，不是固定集正式成功率；恢復時舊計數重新初始化。
空白欄位代表該iteration未記錄此指標（例如尚無完成episode），不補成0；每列均核對PPO loss與collection time存在。
