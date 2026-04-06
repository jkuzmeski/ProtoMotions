 wsl commands

  Full end-to-end biomechanics pipeline with the dedicated PyRoki env:

  python HumanRetargeting/biomechanics_retarget/pipeline.py \
    HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    --subject-profile HumanRetargeting/biomechanics_retarget/profiles/S_GENERIC.yaml \
    --pyroki-python ./.venvs/pyroki/bin/python \
    --pyroki-script ./pyroki/batch_retarget_to_smpl_lower_body.py \
    --visualize-pipeline

  If you want the shortest resume path, this also works now:

  python HumanRetargeting/biomechanics_retarget/pipeline.py \
    HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    --subject-profile HumanRetargeting/biomechanics_retarget/profiles/S_GENERIC.yaml \
    --step convert

  Then package:

  python HumanRetargeting/biomechanics_retarget/pipeline.py \
    HumanRetargeting/biomechanics_retarget/treadmill_data/S_GENERIC \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC \
    --subject-profile HumanRetargeting/biomechanics_retarget/profiles/S_GENERIC.yaml \
    --pyroki-python ./.venvs/pyroki/bin/python \
    --step package

  Playback command stays:

  python examples/env_kinematic_playback.py \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/motion_files/S02_30ms_Long.motion \
    --robot-name smpl_lower_body_subject_S_GENERIC \
    --simulator newton \
    --num-envs 1

  Speed-conditioned masked-mimic experiment commands:

  Package step now also writes explicit experiment manifests under
  `HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/`.

  If you need to regenerate those manifests from the master YAML:

  python scripts/generate_experiment_manifests.py \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/motions_S_GENERIC.yaml \
    HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix

  Train the teacher for one subset:

  python protomotions/train_agent.py \
    --robot-name smpl_lower_body_subject_S_GENERIC \
    --simulator newton \
    --experiment-path examples/experiments/mimic/mlp_bm.py \
    --experiment-name s_generic_teacher_every_other \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --num-envs 64 \
    --batch-size 256

  Visual check the trained teacher (viewer on, no --headless):

  python protomotions/inference_agent.py \
    --checkpoint results/s_generic_teacher_every_other/last.ckpt \
    --simulator newton \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --num-envs 1

  python protomotions/inference_agent.py \
    --checkpoint results/s_generic_teacher_every_other/last.ckpt \
    --simulator newton \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --num-envs 1 \
    

  Train the speed-conditioned student on the same subset:

  python protomotions/train_agent.py \
    --robot-name smpl_lower_body_subject_S_GENERIC \
    --simulator newton \
    --experiment-path examples/experiments/masked_mimic/transformer_bm_speed.py \
    --experiment-name s_generic_student_every_other \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --expert-model-path results/s_generic_teacher_every_other/last.ckpt \
    --num-envs 64 \
    --batch-size 256

  Visual check the trained student (viewer on, no --headless):

  python protomotions/inference_agent.py \
    --checkpoint results/s_generic_student_every_other/last.ckpt \
    --simulator newton \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --num-envs 1

  Run motion-backed biomechanics evaluation for a trained student:

  python protomotions/inference_agent.py \
    --checkpoint results/s_generic_student_every_other/last.ckpt \
    --simulator newton \
    --motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
    --num-envs 8 \
    --full-eval \
    --headless

  Run speed-only deployment inference with no motion library:

  python protomotions/inference_agent.py \
    --checkpoint results/s_generic_student_every_other/last.ckpt \
    --simulator newton \
    --deployment-mode \
    --target-speed 3.5 \
    --num-envs 1

  Swap the manifest path and experiment name for other subsets:

  - `all_8.yaml`
  - `anchor_3.yaml`
  - `speed_2.yaml`
  - `leave_edge_low.yaml`
  - `leave_edge_high.yaml`
  - `loo_15.yaml`
  - `loo_20.yaml`
  - `loo_25.yaml`
  - `loo_30.yaml`
  - `loo_35.yaml`
  - `loo_40.yaml`
  - `loo_45.yaml`
  - `loo_50.yaml`

windows powershell commands

  Full end-to-end biomechanics pipeline with the dedicated PyRoki env:

  python .\HumanRetargeting\biomechanics_retarget\pipeline.py `
    .\HumanRetargeting\biomechanics_retarget\treadmill_data\S_GENERIC `
    .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC `
    --subject-profile .\HumanRetargeting\biomechanics_retarget\profiles\S_GENERIC.yaml `
    --pyroki-python .\.venvs\pyroki\Scripts\python.exe `
    --pyroki-script .\pyroki\batch_retarget_to_smpl_lower_body.py `
    --visualize-pipeline

  If you want the shortest resume path, this also works now:

  python .\HumanRetargeting\biomechanics_retarget\pipeline.py `
    .\HumanRetargeting\biomechanics_retarget\treadmill_data\S_GENERIC `
    .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC `
    --subject-profile .\HumanRetargeting\biomechanics_retarget\profiles\S_GENERIC.yaml `
    --step convert

  Then package:

  python .\HumanRetargeting\biomechanics_retarget\pipeline.py `
    .\HumanRetargeting\biomechanics_retarget\treadmill_data\S_GENERIC `
    .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC `
    --subject-profile .\HumanRetargeting\biomechanics_retarget\profiles\S_GENERIC.yaml `
    --pyroki-python .\.venvs\pyroki\Scripts\python.exe `
    --step package

  Playback command stays:

  python .\examples\env_kinematic_playback.py `
    --experiment-path .\examples\experiments\mimic\mlp.py `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\motion_files\S02_30ms_Long.motion `
    --robot-name smpl_lower_body_subject_S_GENERIC `
    --simulator newton `
    --num-envs 1

  Speed-conditioned masked-mimic experiment commands:

  Package step now also writes explicit experiment manifests under
  `.\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\`.

  If you need to regenerate those manifests from the master YAML:

  python .\scripts\generate_experiment_manifests.py `
    .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\motions_S_GENERIC.yaml `
    .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix

  Train the teacher for one subset:

  python .\protomotions\train_agent.py `
    --robot-name smpl_lower_body_subject_S_GENERIC `
    --simulator newton `
    --experiment-path .\examples\experiments\mimic\mlp_bm.py `
    --experiment-name s_generic_teacher_every_other `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\every_other.yaml `
    --num-envs 64 `
    --batch-size 256

  Visual check the trained teacher (viewer on, no --headless):

  python .\protomotions\inference_agent.py `
    --checkpoint .\results\s_generic_teacher_every_other\last.ckpt `
    --simulator newton `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\every_other.yaml `
    --num-envs 1

  Train the speed-conditioned student on the same subset:

  python .\protomotions\train_agent.py `
    --robot-name smpl_lower_body_subject_S_GENERIC `
    --simulator newton `
    --experiment-path .\examples\experiments\masked_mimic\transformer_bm_speed.py `
    --experiment-name s_generic_student_every_other `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\every_other.yaml `
    --expert-model-path .\results\s_generic_teacher_every_other\last.ckpt `
    --num-envs 64 `
    --batch-size 256

  Visual check the trained student (viewer on, no --headless):

  python .\protomotions\inference_agent.py `
    --checkpoint .\results\s_generic_student_every_other\last.ckpt `
    --simulator newton `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\every_other.yaml `
    --num-envs 1

  Run motion-backed biomechanics evaluation for a trained student:

  python .\protomotions\inference_agent.py `
    --checkpoint .\results\s_generic_student_every_other\last.ckpt `
    --simulator newton `
    --motion-file .\HumanRetargeting\biomechanics_retarget\processed_data\S_GENERIC\yaml_data\experiment_matrix\every_other.yaml `
    --num-envs 8 `
    --full-eval `
    --headless

  Run speed-only deployment inference with no motion library:

  python .\protomotions\inference_agent.py `
    --checkpoint .\results\s_generic_student_every_other\last.ckpt `
    --simulator newton `
    --deployment-mode `
    --target-speed 3.5 `
    --num-envs 1

  Swap the manifest path and experiment name for other subsets:

  - `all_8.yaml`
  - `anchor_3.yaml`
  - `speed_2.yaml`
  - `leave_edge_low.yaml`
  - `leave_edge_high.yaml`
  - `loo_15.yaml`
  - `loo_20.yaml`
  - `loo_25.yaml`
  - `loo_30.yaml`
  - `loo_35.yaml`
  - `loo_40.yaml`
  - `loo_45.yaml`
  - `loo_50.yaml`


    
