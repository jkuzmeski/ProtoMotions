# Robot Scaling Master Script

This directory contains a master script that coordinates the scaling of a robot to a new height by running multiple scaling scripts in the correct order.

## Files

- `master_scaling.py` - Main master script that coordinates all scaling operations
- `scaling_xml.py` - Scales XML files to new height
- `scaling_usda.py` - Generates USDA files from scaled XML
- `scaling_yaml.py` - Creates YAML configuration files
- `scaling_robot.py` - Updates robot configurations in robots.py
- `example_usage.py` - Example script showing how to use the master script

## Usage

### Basic Usage

Scale a robot to 1.8 meters height:

```bash
python master_scaling.py --height 1.8
```

Scale using the adjusted torque control assets instead of the default PD assets:

```bash
python protomotions\data\assets\rescale\master_scaling.py --height 1.8 --asset_variant adjusted_torque
```

### Advanced Usage

```bash
# Scale with custom file paths
python master_scaling.py --height 2.0 \
    --input_xml /path/to/robot.xml \
    --ref_usda /path/to/reference.usda \
    --template_yaml /path/to/config.yaml

# Skip robot configuration updates (only generate files)
python master_scaling.py --height 1.5 --skip_robot_config

# Run in step-by-step mode (pause between each step)
python master_scaling.py --height 1.9 --step_by_step

# Check for existing files without generating new ones
python master_scaling.py --height 1.8 --check_only

# Force overwrite existing files without prompting
python master_scaling.py --height 1.8 --force_overwrite
```

## Process Overview

The master script runs the following steps in order:

1. **Scale XML** (`scaling_xml.py`)
   - Takes the selected base XML file and scales it to the new height
   - Generates: `<robot_variant>_height_XXXcm.xml`

2. **Generate USDA** (`scaling_usda.py`)
   - Creates a USDA file from the scaled XML using a reference USDA as template
   - Generates: `<robot_variant>_height_XXXcm.usda`

3. **Create YAML Config** (`scaling_yaml.py`)
   - Creates a YAML configuration file that points to the scaled assets
   - Generates: `<robot_variant>_height_XXXcm.yaml`

4. **Update Robot Configs** (`scaling_robot.py`) [Optional]
   - Adds new robot configurations to `robots.py`
   - Updates imports in `scene.py`

## Default File Paths

The script uses these default paths (can be overridden with command line arguments):

```text
Input XML:     ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid_lower_body_base.xml
Reference USD: ProtoMotions/protomotions/data/assets/usd/smpl_humanoid_lower_body_base.usda
Template YAML: ProtoMotions/protomotions/config/robot/smpl_humanoid_lower_body.yaml
Robots File:   ProtoMotions/protomotions/simulator/isaaclab/utils/robots.py
Scene File:    ProtoMotions/protomotions/simulator/isaaclab/utils/scene.py
```

Passing `--asset_variant adjusted_pd` or `--asset_variant adjusted_torque` switches the source XML/USDA to the pre-tuned files in `data/assets/rescale` while still writing the scaled outputs into the standard `data/assets/mjcf` and `data/assets/usd` folders.

## Command Line Options

- `--height` (required): Target height in meters
- `--input_xml`: Path to input XML file
- `--ref_usda`: Path to reference USDA file
- `--template_yaml`: Path to template YAML configuration
- `--robots_file`: Path to robots.py file
- `--scene_file`: Path to scene.py file
- `--asset_variant`: Choose which asset variant to scale (`base`, `adjusted_pd`, `adjusted_torque`)
- `--skip_robot_config`: Skip updating robot configurations
- `--step_by_step`: Pause between each step for verification
- `--force_overwrite`: Overwrite existing files without prompting
- `--check_only`: Only check for existing files and exit (no generation)

## Example

```bash
# Scale robot to 1.8m height
python master_scaling.py --height 1.8

# This will generate:
# - smpl_humanoid_lower_body_height_180cm.xml (or the chosen variant prefix)
# - smpl_humanoid_lower_body_height_180cm.usda (or the chosen variant prefix)
# - smpl_humanoid_lower_body_height_180cm.yaml (or the chosen variant prefix)
# - New robot configuration in robots.py
```

## Error Handling

The script will:

- Validate that required files exist before starting
- Check for existing files and prompt user for action
- Stop if any step fails
- Provide clear error messages
- Create backups of modified files

## Duplicate Prevention

The master script automatically checks for existing files before starting:

- **Existing Files Detected**: Shows which files already exist and offers options:
  - `y` - Overwrite existing files and continue
  - `n` - Abort the process
  - `s` - Skip existing files and only generate missing ones

- **Check Only Mode**: Use `--check_only` to see what files exist without generating anything

- **Force Overwrite**: Use `--force_overwrite` to automatically overwrite without prompting

## Tips

1. **Test First**: Use `--skip_robot_config` to generate files without modifying robot configurations
2. **Step by Step**: Use `--step_by_step` to verify each step before proceeding
3. **Backups**: The script automatically creates backups of modified files
4. **Height Range**: Recommended height range is 1.0m to 3.0m

## Troubleshooting

### Import Errors

Make sure all scaling scripts are in the same directory:

- `scaling_xml.py`
- `scaling_usda.py`
- `scaling_yaml.py`
- `scaling_robot.py`

### File Not Found

Check that the default file paths exist or provide custom paths using command line arguments.

### Permission Errors

Ensure you have write permissions to the output directories.
