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

The B default config under `reference_configs` is still a source-code reference. The actual B archive has now been received through Discord and audited: see [the verification record and figures](comparison_audit). It confirms B286/300 and teammate-C277/300; the original C evaluation remains282/300. Raw B files remain in the supplied archive rather than being copied into this C result folder.

The final `.pt` checkpoint is inside the ZIP. Low-level skill ONNX files are not bundled; their official download revision and exact hashes are recorded. The original package README explains how to compare B.

The subsequent B audit and one local RTX4060 rerun are in `comparison_audit/`. B scored287/300 here against the original C282/300. The original C ZIP is an unchanged historical delivery snapshot; its older notes saying B was not yet supplied describe the state at packaging time. Use the current comparison document/audit for the latest status.
