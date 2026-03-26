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


    
