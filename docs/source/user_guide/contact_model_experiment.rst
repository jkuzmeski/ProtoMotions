Contact Model Experiment
========================

This experiment compares three SMPL lower-body contact models in ProtoMotions running on a
custom Newton checkout with foot-only pressure-field contact enabled:

* ``smpl_lower_body`` with the default box-foot collision geometry and point contacts
* ``smpl_lower_body_ellipsoid_feet`` with ellipsoid feet and point contacts
* ``smpl_lower_body_ellipsoid_feet`` with ellipsoid feet and pressure-field contacts

The current scope is flat-ground motion imitation only. Ground remains rigid. Only the
ankle and toe bodies participate in the compliant foot-ground path. The main analysis targets
are stance-force variance and center-of-pressure (CoP) continuity.

Experiment Summary
------------------

**Working branches**

* Newton: ``jkuzm/protomotions-smpl-pressure-feet``
* ProtoMotions: ``jkuzm/smpl-pressure-feet``

**Key implementation points**

* Pressure-field contact is wired through ``protomotions/simulator/newton/simulator.py`` and
  ``protomotions/simulator/newton/config.py``.
* The ellipsoid-foot asset lives at
  ``protomotions/data/assets/mjcf/smpl_humanoid_lower_body_ellipsoid_feet.xml``.
* The default ``smpl_lower_body`` asset is the box-foot baseline because the ankle and toe
  collision geoms are boxes in
  ``HumanRetargeting/rescale/smpl_humanoid_lower_body_adjusted_pd.xml``.
* Contact analysis exports now include:

  * XYZ ground-reaction-force waveforms
  * normalized CoP waveforms
  * foot-ground pressure maps

Comparison Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Variant
     - Training config
     - Biomechanics-eval config
   * - Box feet, point contact
     - ``examples/experiments/mimic/smpl_lower_body_box_feet.py``
     - ``examples/experiments/mimic/smpl_lower_body_box_feet_biomech_eval.py``
   * - Ellipsoid feet, point contact
     - ``examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_point.py``
     - ``examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_point_biomech_eval.py``
   * - Ellipsoid feet, pressure fields
     - ``examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_pressure.py``
     - ``examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_pressure_biomech_eval.py``

Step 1: Set Up the Environment
------------------------------

The ProtoMotions WSL setup script now prefers a sibling or explicit local Newton checkout
instead of forcing the old pinned Newton clone.

.. code-block:: bash

   cd /mnt/d/Biomotions/ProtoMotions
   ./scripts/setup_wsl_newton_env.sh
   source .venv/bin/activate

Check that the virtual environment is using the local Newton checkout:

.. code-block:: bash

   python -m pip show newton

The reported editable location should point at your pressure-field Newton checkout, for example
``/mnt/d/newton``.

Step 2: Choose a Motion Set
---------------------------

The default motion file used throughout the commands below is the S_GENERIC
``every_other`` matrix manifest:

.. code-block:: bash

   /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml

For biomechanics evaluation, motion-derived speed control expects per-trial speed metadata.
The current speed-control path resolves speed from either:

* ``<motion_dir>/metadata/<motion_stem>.json`` with a ``speed_mps`` field
* or a filename pattern that ``HumanRetargeting.biomechanics_retarget.subject_profiles``
  can parse into a speed

Step 3: Validate Config Generation
----------------------------------

This is the fastest way to verify that each contact-model variant still resolves cleanly:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_box_feet.py \
       --experiment-name smpl_lb_box_cfg \
       --create-config-only

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_point.py \
       --experiment-name smpl_lb_point_cfg \
       --create-config-only

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_pressure.py \
       --experiment-name smpl_lb_pressure_cfg \
       --pressure-field-foot-kh 2.5e7 \
       --pressure-field-foot-sdf-max-resolution 32 \
       --create-config-only

Step 4: Train the Three Contact Variants
----------------------------------------

Use the same motion file and similar optimization settings for all three runs.

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body \
       --simulator newton \
       --num-envs 512 \
       --batch-size 16384 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_box_feet.py \
       --experiment-name smpl_lb_box

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 512 \
       --batch-size 16384 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_point.py \
       --experiment-name smpl_lb_point

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 512 \
       --batch-size 16384 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_pressure.py \
       --experiment-name smpl_lb_pressure \
       --pressure-field-foot-kh 2.5e7 \
       --pressure-field-foot-sdf-max-resolution 32

Step 4b: Run the S_GENERIC Teacher and Student
----------------------------------------------

These are the default local CLI commands for the speed-conditioned pressure-field experiment
from this branch:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_subject_S_GENERIC \
       --simulator newton \
       --num-envs 1024 \
       --batch-size 4096 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/mlp_bm_pressure_feet.py \
       --experiment-name s_generic_teacher_every_other_pressure \
       --pressure-field-foot-kh 2.5e7 \
       --pressure-field-foot-sdf-max-resolution 32

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_subject_S_GENERIC \
       --simulator newton \
       --num-envs 1024 \
       --batch-size 4096 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/masked_mimic/transformer_bm_speed_pressure_feet.py \
       --experiment-name s_generic_student_every_other_pressure \
       --expert-model-path results/s_generic_teacher_every_other_pressure/last.ckpt \
       --pressure-field-foot-kh 2.5e7 \
       --pressure-field-foot-sdf-max-resolution 32

Step 5: Smoke-Test Each Trained Checkpoint
------------------------------------------

Before exporting biomechanics plots, verify that each trained policy boots on Newton:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_lb_pressure/last.ckpt \
       --simulator newton \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --num-envs 1 \
       --headless

Repeat this for ``results/smpl_lb_box/last.ckpt`` and ``results/smpl_lb_point/last.ckpt``.

Step 6: Build Biomechanics Evaluation Configs
---------------------------------------------

The contact-analysis exports are produced by ``BiomechanicsEvaluator``. The dedicated
``*_biomech_eval.py`` configs keep the same lower-body mimic policy architecture, add a
motion-derived speed-control component, and swap the evaluator to the biomechanics path.

Generate config-only results for each variant:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_box_feet_biomech_eval.py \
       --experiment-name smpl_lb_box_eval_cfg \
       --biomechanics-episodes-per-speed 8 \
       --create-config-only

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_point_biomech_eval.py \
       --experiment-name smpl_lb_point_eval_cfg \
       --biomechanics-episodes-per-speed 8 \
       --create-config-only

   python protomotions/train_agent.py \
       --robot-name smpl_lower_body_ellipsoid_feet \
       --simulator newton \
       --num-envs 1 \
       --batch-size 1 \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --experiment-path examples/experiments/mimic/smpl_lower_body_ellipsoid_feet_pressure_biomech_eval.py \
       --experiment-name smpl_lb_pressure_eval_cfg \
       --pressure-field-foot-kh 2.5e7 \
       --pressure-field-foot-sdf-max-resolution 32 \
       --biomechanics-episodes-per-speed 8 \
       --create-config-only

Create disposable evaluation copies of the trained runs and replace the inference config:

.. code-block:: bash

   cp -r results/smpl_lb_box results/smpl_lb_box_eval
   cp results/smpl_lb_box_eval_cfg/resolved_configs_inference.pt results/smpl_lb_box_eval/
   cp results/smpl_lb_box_eval_cfg/experiment_config.py results/smpl_lb_box_eval/

   cp -r results/smpl_lb_point results/smpl_lb_point_eval
   cp results/smpl_lb_point_eval_cfg/resolved_configs_inference.pt results/smpl_lb_point_eval/
   cp results/smpl_lb_point_eval_cfg/experiment_config.py results/smpl_lb_point_eval/

   cp -r results/smpl_lb_pressure results/smpl_lb_pressure_eval
   cp results/smpl_lb_pressure_eval_cfg/resolved_configs_inference.pt results/smpl_lb_pressure_eval/
   cp results/smpl_lb_pressure_eval_cfg/experiment_config.py results/smpl_lb_pressure_eval/

Step 7: Run Full Biomechanics Evaluation
----------------------------------------

Run the evaluation path on the copied checkpoint directories:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_lb_box_eval/last.ckpt \
       --simulator newton \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --num-envs 1 \
       --headless \
       --full-eval

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_lb_point_eval/last.ckpt \
       --simulator newton \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --num-envs 1 \
       --headless \
       --full-eval

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_lb_pressure_eval/last.ckpt \
       --simulator newton \
       --motion-file /mnt/d/Biomotions/ProtoMotions/HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/yaml_data/experiment_matrix/every_other.yaml \
       --num-envs 1 \
       --headless \
       --full-eval

Each run writes per-speed exports under:

.. code-block:: text

   results/<experiment_name>_eval/results/biomechanics/
   ├── summary.json
   └── <speed_tag>/
       ├── contact_analysis.npz
       ├── contact_waveforms.png
       └── pressure_maps.png

Step 8: Regenerate Per-Run Plots
--------------------------------

Use the plotting script on a speed directory or directly on ``contact_analysis.npz``:

.. code-block:: bash

   python scripts/plot_biomechanics_contact_analysis.py \
       results/smpl_lb_pressure_eval/results/biomechanics/1p25

This regenerates:

* ``contact_waveforms.png`` with left and right XYZ GRF plus normalized CoP
* ``pressure_maps.png`` with left and right normalized foot-ground pressure maps

Step 9: Compare Runs Side by Side
---------------------------------

Overlay the three contact models and write summary metrics:

.. code-block:: bash

   python scripts/compare_biomechanics_contact_runs.py \
       --run box=results/smpl_lb_box_eval/results/biomechanics/1p25 \
       --run point=results/smpl_lb_point_eval/results/biomechanics/1p25 \
       --run pressure=results/smpl_lb_pressure_eval/results/biomechanics/1p25 \
       --output-dir results/contact_model_compare_1p25

The comparison script writes:

* ``comparison_contact_waveforms.png``
* ``comparison_pressure_maps.png``
* ``comparison_summary.json``

The summary JSON includes per-foot proxies for:

* mean GRF standard deviation in X, Y, and Z
* normalized CoP path length
* CoP valid fraction
* peak pressure

Contact Analysis Details
------------------------

**CoP normalization**

CoP is normalized against the inferred support bounds of the active foot geometry. For the
lower-body assets this support is reconstructed from the ankle and toe collision bodies, then
mapped into the range ``[-1, 1]`` along the fore-aft and medial-lateral directions.

**Pressure maps**

The current pressure map is an estimated foot-ground interface map built by binning
per-contact normal force samples over the normalized foot support. It is not a direct export of
Newton's internal pressure-field volume.

**Current scope limits**

* Flat ground only
* Rigid terrain with compliant feet
* Foot-only compliant contacts
* Ellipsoid-foot pressure fields are the current working pressure-contact geometry

Related Files
-------------

* ``protomotions/simulator/newton/simulator.py``
* ``protomotions/simulator/newton/config.py``
* ``protomotions/agents/evaluators/biomechanics_evaluator.py``
* ``scripts/plot_biomechanics_contact_analysis.py``
* ``scripts/compare_biomechanics_contact_runs.py``
* ``scripts/submit_s_generic_teacher_pressure_slurm.py``
* ``scripts/submit_s_generic_student_pressure_slurm.py``

Slurm Launchers for S_GENERIC
-----------------------------

The current checkout also includes two Slurm helper scripts for the speed-conditioned
S_GENERIC pressure-field experiment:

* ``scripts/submit_s_generic_teacher_pressure_slurm.py`` submits the teacher now
* ``scripts/submit_s_generic_student_pressure_slurm.py`` submits the student with
  ``#SBATCH --begin=now+12hours`` by default

Both wrappers target the ``every_other`` subset by default, share a stable remote code
directory, and pass the pressure-field foot settings through ``protomotions/train_slurm.py``.

Example teacher submit:

.. code-block:: bash

   python scripts/submit_s_generic_teacher_pressure_slurm.py \
       --user my_cluster_username \
       --subset every_other \
       --remote-dir-name s_generic_every_other_pressure_suite \
       --slurm-time 12:00:00 \
       --partition gpu

Example student submit 12 hours later:

.. code-block:: bash

   python scripts/submit_s_generic_student_pressure_slurm.py \
       --user my_cluster_username \
       --subset every_other \
       --remote-dir-name s_generic_every_other_pressure_suite \
       --slurm-time 12:00:00 \
       --partition gpu

If the teacher checkpoint already exists locally and needs to be copied into the shared
remote directory before launching the student, add:

.. code-block:: bash

   --sync-local-teacher-results
