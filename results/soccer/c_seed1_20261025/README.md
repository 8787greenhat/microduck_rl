# C seed 1 result artifacts

- [Verified results and B/C comparison](../../../docs/soccer_c_seed1_results.md)
- [Full comparison ZIP, including checkpoint and videos](microduck_C_seed1_comparison_20261025.zip)
- [Original evaluation summary](evaluation/summary.json) and [300 per-episode rows](evaluation/per_episode.csv)
- [Fixed evaluation JSON](evaluation_set/soccer_eval_set.json)
- [Effective training budget](effective_training_settings.json)
- [Actual initial configs](training_configs/initial) and [actual resume configs](training_configs/resume)
- [Accepted 2,000-iteration learning curve](learning_curves/accepted_seed1.csv)
- [Package versions and skill hashes](reproducibility/versions_and_models.json)
- [Full audit](evaluation/final_audit.json), [checkpoint verification](evaluation/checkpoint_verified.json), and [recovery](recovery)

The B default config under `reference_configs` is a source-code reference, not the teammate's actual run settings. The original C evaluation is 94.0%; PR #1 separately reports 92.3% for C on the teammate machine. Preserve both provenance records.

The final `.pt` checkpoint is inside the ZIP. Low-level skill ONNX files are not bundled; their official download revision and exact hashes are recorded. The original package README explains how to compare B.
