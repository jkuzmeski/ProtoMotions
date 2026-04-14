#!/usr/bin/env python3
"""
Master Robot Scaling Script

This script coordinates the scaling of a robot by running the following scripts in order:
1. scaling_xml.py - Scale the XML file to the new height
2. scaling_usda.py - Generate the corresponding USDA file
3. scaling_yaml.py - Create the YAML configuration file
4. scaling_robot.py - Update the robot configurations in robots.py

Usage:
    python master_scaling.py --height 1.8 --input_xml path/to/input.xml --template_yaml path/to/template.yaml

Author: John Kuzmeski
Date: 2025-09-23
"""

import os
import sys
import argparse
import re
import shutil

# Import the scaling modules
try:
    from scaling_xml import copy_and_scale_xml
    from scaling_usda import extract_body_data_from_xml, update_usda_file
    from scaling_yaml import create_scaled_yaml_config
    from scaling_robot import (
        generate_scaled_robot_configs,
        parse_joint_gains_from_usda,
        parse_joint_gains_from_xml,
    )
except ImportError as e:
    print(f"Error importing scaling modules: {e}")
    print("Make sure all scaling scripts are in the same directory as this master script.")
    sys.exit(1)


class RobotScaler:
    """Class to handle the complete robot scaling process."""
    
    def __init__(self, height: float, base_paths: dict, variant: str = 'base'):
        """
        Initialize the robot scaler.
        
        Args:
            height: Target height in meters
            base_paths: Dictionary containing base file paths
        """
        self.height = height
        self.base_paths = base_paths
        self.variant = (variant or base_paths.get('variant', 'base')).lower()
        self.generated_files = {}
        self.skip_existing = False  # Flag to skip existing files
        
        # Generate height suffix for consistent naming
        self.height_int = round(height * 100)  # Convert to centimeters
        self.height_suffix = f"_height_{self.height_int}cm"

        # Variant-specific naming information
        self.robot_type_base = base_paths.get('robot_type_base', 'smpl_humanoid_lower_body')
        self.config_prefix = base_paths.get('config_prefix', 'SMPL_LOWER_BODY')
        self.usd_stem = base_paths.get('usd_stem', self.robot_type_base)
        self.base_robot_config_name = base_paths.get('base_robot_config', 'SMPL_LOWER_BODY_CFG')
        self.output_xml_dir = base_paths.get('output_xml_dir', os.path.dirname(base_paths['input_xml']))
        self.output_usda_dir = base_paths.get('output_usda_dir', os.path.dirname(base_paths['ref_usda']))
        self.yaml_output_dir = os.path.dirname(base_paths['template_yaml'])
        self.asset_output_stem = base_paths.get('asset_stem', self.robot_type_base)
        self.xml_extension = os.path.splitext(base_paths['input_xml'])[1]
        ref_usda_path = base_paths.get('ref_usda')
        self.usda_extension = os.path.splitext(ref_usda_path)[1] if ref_usda_path else '.usda'
        
        print(f"=== Robot Scaling Process - Target Height: {height:.1f}m ===")
        print(f"Height suffix: {self.height_suffix}")
        if self.variant != 'base':
            print(f"Asset variant: {self.variant} ({self.robot_type_base})")

    def _expected_xml_path(self) -> str:
        """Compute the expected output path for the scaled XML."""
        return os.path.join(
            self.output_xml_dir,
            f"{self.asset_output_stem}{self.height_suffix}{self.xml_extension}"
        )

    def _expected_usda_path(self) -> str:
        """Compute the expected output path for the generated USDA."""
        return os.path.join(
            self.output_usda_dir,
            f"{self.usd_stem}{self.height_suffix}{self.usda_extension}"
        )

    def _expected_yaml_path(self) -> str:
        """Compute the expected output path for the YAML config."""
        return os.path.join(
            self.yaml_output_dir,
            f"{self.robot_type_base}{self.height_suffix}.yaml"
        )
        
    def check_existing_files(self) -> dict:
        """
        Check which files already exist for this height.
        
        Returns:
            Dictionary with file types as keys and (exists, path) tuples as values
        """
        existing = {}
        
        # Check XML file
        xml_path = self._expected_xml_path()
        existing['xml'] = (os.path.exists(xml_path), xml_path)
        
        # Check USDA file
        usda_path = self._expected_usda_path()
        existing['usda'] = (os.path.exists(usda_path), usda_path)
        
        # Check YAML file
        yaml_path = self._expected_yaml_path()
        existing['yaml'] = (os.path.exists(yaml_path), yaml_path)
        
        # Check robot configuration (if applicable)
        if 'robots_file' in self.base_paths:
            robots_file = self.base_paths['robots_file']
            config_name = f"{self.config_prefix}_HEIGHT_{self.height_int}CM_CFG"
            config_exists = self._check_robot_config_exists(robots_file, config_name)
            existing['robot_config'] = (config_exists, config_name)
        
        return existing
    
    def _check_robot_config_exists(self, robots_file: str, config_name: str) -> bool:
        """
        Check if a robot configuration already exists in robots.py.
        
        Args:
            robots_file: Path to the robots.py file
            config_name: Name of the configuration to check for
            
        Returns:
            True if configuration exists, False otherwise
        """
        if not os.path.exists(robots_file):
            return False
        
        try:
            with open(robots_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for the configuration name
            pattern = rf"^{re.escape(config_name)}\s*="
            return bool(re.search(pattern, content, re.MULTILINE))
        except Exception:
            return False
    
    def show_existing_files_status(self, existing_files: dict, force_overwrite: bool = False) -> bool:
        """
        Show the status of existing files and ask user what to do.
        
        Args:
            existing_files: Dictionary from check_existing_files()
            force_overwrite: If True, skip prompts and overwrite existing files
            
        Returns:
            True if should proceed, False if should abort
        """
        any_exist = any(exists for exists, _ in existing_files.values())
        
        if not any_exist:
            print("✓ No existing files found for this height - safe to proceed")
            return True
        
        print(f"\n⚠️  Found existing files for height {self.height:.1f}m:")
        
        file_labels = {
            'xml': 'XML file',
            'usda': 'USDA file',
            'yaml': 'YAML config',
            'robot_config': 'Robot configuration'
        }
        
        for file_type, (exists, path) in existing_files.items():
            if exists:
                display_path = os.path.basename(path) if file_type != 'robot_config' else path
                print(f"  • {file_labels[file_type]}: {display_path}")
        
        if force_overwrite:
            print("🔄 Force overwrite mode - will replace existing files")
            return True
        
        print("\nOptions:")
        print("  y - Overwrite existing files and continue")
        print("  n - Abort process")
        print("  s - Skip existing files and only generate missing ones")
        
        while True:
            choice = input("\nHow would you like to proceed? (y/n/s): ").lower().strip()
            if choice in ['y', 'n', 's']:
                if choice == 'n':
                    print("Process aborted by user.")
                    return False
                elif choice == 's':
                    print("Will skip existing files and only generate missing ones.")
                    self.skip_existing = True
                else:
                    print("Will overwrite existing files.")
                    self.skip_existing = False
                return True
            else:
                print("Please enter 'y', 'n', or 's'")
        
    def step1_scale_xml(self) -> bool:
        """
        Step 1: Scale the XML file.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n--- Step 1: Scaling XML file ---")
        input_xml = self.base_paths['input_xml']
        
        if not os.path.exists(input_xml):
            print(f"Error: Input XML file not found: {input_xml}")
            return False
        
        expected_xml = self._expected_xml_path()

        if os.path.exists(expected_xml) and self.skip_existing:
            print(f"⏭️  Skipping - XML file already exists: {os.path.basename(expected_xml)}")
            self.generated_files['xml'] = expected_xml
            return True
        
        print(f"Input XML: {os.path.basename(input_xml)}")
        
        # Use copy_and_scale_xml to generate the scaled XML file
        output_xml = copy_and_scale_xml(
            input_xml,
            self.height,
            self.height_suffix,
            output_dir=self.output_xml_dir,
        )

        if not output_xml:
            print("✗ Failed to generate scaled XML file")
            return False
        
        # Rename to expected filename if necessary (primarily for base assets)
        if os.path.normcase(output_xml) != os.path.normcase(expected_xml):
            try:
                if os.path.exists(expected_xml):
                    os.remove(expected_xml)
                shutil.move(output_xml, expected_xml)
                output_xml = expected_xml
                print(f"Renamed XML output to: {os.path.basename(expected_xml)}")
            except Exception as e:
                print(f"Warning: Could not move XML file to expected location: {e}")

        self.generated_files['xml'] = output_xml
        print(f"✓ Successfully generated: {os.path.basename(output_xml)}")
        return True
    
    def step2_generate_usda(self) -> bool:
        """
        Step 2: Generate the USDA file from the scaled XML.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n--- Step 2: Generating USDA file ---")
        
        if 'xml' not in self.generated_files:
            print("Error: No scaled XML file available from Step 1")
            return False
        
        xml_file = self.generated_files['xml']
        ref_usda = self.base_paths['ref_usda']
        
        if not os.path.exists(ref_usda):
            print(f"Error: Reference USDA file not found: {ref_usda}")
            return False
        
        output_usda = self._expected_usda_path()
        
        # Check if USDA file already exists
        if os.path.exists(output_usda) and self.skip_existing:
            print(f"⏭️  Skipping - USDA file already exists: {os.path.basename(output_usda)}")
            self.generated_files['usda'] = output_usda
            return True
        
        print(f"Scaled XML: {os.path.basename(xml_file)}")
        print(f"Reference USDA: {os.path.basename(ref_usda)}")
        
        # Extract body data from the scaled XML
        body_data, hierarchy = extract_body_data_from_xml(xml_file)
        
        if not body_data or not hierarchy:
            print("✗ Failed to extract body data from scaled XML")
            return False
        
        # Update the USDA file
        try:
            update_usda_file(ref_usda, body_data, hierarchy, output_usda)
            self.generated_files['usda'] = output_usda
            print(f"✓ Successfully generated: {os.path.basename(output_usda)}")
            return True
        except Exception as e:
            print(f"✗ Failed to generate USDA file: {e}")
            return False
    
    def step3_create_yaml_config(self) -> bool:
        """
        Step 3: Create the YAML configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n--- Step 3: Creating YAML configuration ---")
        
        template_yaml = self.base_paths['template_yaml']
        
        if not os.path.exists(template_yaml):
            print(f"Error: Template YAML file not found: {template_yaml}")
            return False
        
        # Generate output YAML path
        output_yaml = self._expected_yaml_path()
        
        # Check if YAML file already exists
        if os.path.exists(output_yaml) and self.skip_existing:
            print(f"⏭️  Skipping - YAML config already exists: {os.path.basename(output_yaml)}")
            self.generated_files['yaml'] = output_yaml
            return True
        
        print(f"Template YAML: {os.path.basename(template_yaml)}")
        
        # Extract filenames for the generated files
        xml_filename = os.path.basename(self.generated_files.get('xml', ''))
        usda_filename = os.path.basename(self.generated_files.get('usda', ''))
        
        # Create the scaled YAML config
        success = create_scaled_yaml_config(
            template_yaml,
            output_yaml,
            self.height,
            xml_filename,
            usda_filename,
            robot_type_base=self.robot_type_base,
        )
        
        if success:
            self.generated_files['yaml'] = output_yaml
            print(f"✓ Successfully generated: {os.path.basename(output_yaml)}")
            return True
        else:
            print("✗ Failed to generate YAML configuration")
            return False
    
    def step4_update_robot_configs(self) -> bool:
        """
        Step 4: Update robot configurations in robots.py.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n--- Step 4: Updating robot configurations ---")
        
        robots_file = self.base_paths.get('robots_file')
        scene_file = self.base_paths.get('scene_file')
        
        if not robots_file or not os.path.exists(robots_file):
            print(f"Error: robots.py file not found: {robots_file}")
            return False
        
        # Check if robot configuration already exists
        config_name = f"{self.config_prefix}_HEIGHT_{self.height_int}CM_CFG"
        if self._check_robot_config_exists(robots_file, config_name) and self.skip_existing:
            print(f"⏭️  Skipping - Robot config already exists: {config_name}")
            return True
        
        print(f"Robots file: {os.path.basename(robots_file)}")
        if scene_file and os.path.exists(scene_file):
            print(f"Scene file: {os.path.basename(scene_file)}")
        
        joint_gains_map = None
        xml_asset = self.generated_files.get('xml')
        usda_asset = self.generated_files.get('usda')

        joint_gains = {}
        if xml_asset and os.path.exists(xml_asset):
            joint_gains = parse_joint_gains_from_xml(xml_asset)

        if not joint_gains and usda_asset and os.path.exists(usda_asset):
            joint_gains = parse_joint_gains_from_usda(usda_asset)

        if joint_gains:
            joint_gains_map = {self.height_int: joint_gains}
            print(f"Found joint gain overrides for {len(joint_gains)} joints")
        else:
            print("Warning: Could not extract joint gains from generated assets; using base config defaults")

        # Generate scaled robot configurations (for a single height)
        success = generate_scaled_robot_configs(
            robots_file,
            [self.height],  # Single height in a list
            base_config=self.base_robot_config_name,
            create_backup_file=True,
            scene_file=scene_file if scene_file and os.path.exists(scene_file) else None,
            update_scene_file=True,
            config_prefix=self.config_prefix,
            usd_stem=self.usd_stem,
            joint_gains_per_height=joint_gains_map,
        )
        
        if success:
            print("✓ Successfully updated robot configurations")
            return True
        else:
            print("✗ Failed to update robot configurations")
            return False
    
    def step5_update_scene_initialization(self) -> bool:
        """
        Step 5: Update scene.py to include initialization for the new robot type.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n--- Step 5: Updating scene initialization ---")
        
        scene_file = self.base_paths.get('scene_file')
        
        if not scene_file or not os.path.exists(scene_file):
            print(f"Error: scene.py file not found: {scene_file}")
            return False
        
        robot_type = f"{self.robot_type_base}{self.height_suffix}"
        config_name = f"{self.config_prefix}_HEIGHT_{self.height_int}CM_CFG"
        
        # Check if initialization already exists
        if self._check_scene_initialization_exists(scene_file, robot_type) and self.skip_existing:
            print(f"⏭️  Skipping - Scene initialization already exists: {robot_type}")
            return True
        
        print(f"Scene file: {os.path.basename(scene_file)}")
        print(f"Adding initialization for robot type: {robot_type}")
        
        success = self._add_scene_initialization(scene_file, robot_type, config_name)
        
        if success:
            print(f"✓ Successfully added scene initialization for {robot_type}")
            return True
        else:
            print(f"✗ Failed to add scene initialization for {robot_type}")
            return False
    
    def _check_scene_initialization_exists(self, scene_file: str, robot_type: str) -> bool:
        """
        Check if scene initialization already exists for a robot type.
        
        Args:
            scene_file: Path to the scene.py file
            robot_type: Robot type to check for
            
        Returns:
            True if initialization exists, False otherwise
        """
        if not os.path.exists(scene_file):
            return False
        
        try:
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for the robot type in elif statement
            pattern = rf'elif robot_type == "{robot_type}":'
            return bool(re.search(pattern, content))
            
        except Exception:
            return False
    
    def _add_scene_initialization(self, scene_file: str, robot_type: str, config_name: str) -> bool:
        """
        Add scene initialization block for a new robot configuration.
        
        Args:
            scene_file: Path to the scene.py file
            robot_type: Robot type identifier
            config_name: Configuration name to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the insertion point (after the last smpl_humanoid_lower_body_height block)
            pattern = r'(elif robot_type == "smpl_humanoid_lower_body.*?":\s*.*?filter_prim_paths_expr=\[.*?\],\s*\))'
            matches = list(re.finditer(pattern, content, re.DOTALL))
            
            if not matches:
                # If no height-specific blocks found, insert after the base smpl_humanoid_lower_body block
                pattern = r'(elif robot_type == "smpl_humanoid_lower_body":\s*.*?filter_prim_paths_expr=\[.*?\],\s*\))'
                matches = list(re.finditer(pattern, content, re.DOTALL))
            
            if not matches:
                print("Could not find insertion point in scene.py")
                return False
            
            # Find the last match (to insert after the last height variant)
            last_match = matches[-1]
            insertion_point = last_match.end()
            
            # Create the new initialization block
            new_block = f'''
        elif robot_type == "{robot_type}":
            self.robot: ArticulationCfg = {config_name}.replace(
                prim_path="/World/envs/env_.*/Robot"
            )
            self.contact_sensor: ContactSensorCfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/bodies/.*",
                filter_prim_paths_expr=[f"/World/objects/object_{{i}}" for i in range(0)],
            )
            self.imu_sensor: ImuCfg = ImuCfg(
                prim_path="/World/envs/env_.*/Robot/bodies/Pelvis",
                update_period=0.0,
                history_length=3,
                debug_vis=True,
                offset=ImuCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0)
                ),
                gravity_bias=(0.0, 0.0, 9.81)
            )'''
            
            # Insert the new block
            new_content = content[:insertion_point] + new_block + content[insertion_point:]
            
            # Create backup
            backup_path = f"{scene_file}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created backup: {os.path.basename(backup_path)}")
            
            # Write the updated content
            with open(scene_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
            
        except Exception as e:
            print(f"Error updating scene.py: {e}")
            return False
    
    def run_complete_scaling(self, force_overwrite: bool = False) -> bool:
        """
        Run the complete scaling process.
        
        Args:
            force_overwrite: If True, overwrite existing files without prompting
        
        Returns:
            True if all steps successful, False otherwise
        """
        print(f"Starting complete robot scaling process for height {self.height:.1f}m")
        
        # Check for existing files
        existing_files = self.check_existing_files()
        if not self.show_existing_files_status(existing_files, force_overwrite):
            return False
        
        # Step 1: Scale XML
        if not self.step1_scale_xml():
            return False
        
        # Step 2: Generate USDA
        if not self.step2_generate_usda():
            return False
        
        # Step 3: Create YAML config
        if not self.step3_create_yaml_config():
            return False
        
        # Step 4: Update robot configs (optional, can be skipped)
        if 'robots_file' in self.base_paths:
            if not self.step4_update_robot_configs():
                print("Warning: Failed to update robot configurations, but other steps succeeded")
        
        # Step 5: Update scene initialization (optional, can be skipped)
        if 'scene_file' in self.base_paths:
            if not self.step5_update_scene_initialization():
                print("Warning: Failed to update scene initialization, but other steps succeeded")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 SCALING COMPLETE for height {self.height:.1f}m!")
        print(f"{'='*60}")
        print("Generated files:")
        for file_type, file_path in self.generated_files.items():
            print(f"  {file_type.upper()}: {os.path.basename(file_path)}")
        
        return True


def get_default_paths(variant: str = 'base') -> dict:
    """Get default file paths for the scaling process."""
    base_dir = "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\data\\assets"
    mjcf_dir = os.path.join(base_dir, "mjcf")
    usd_dir = os.path.join(base_dir, "usd")
    rescale_dir = os.path.join(base_dir, "rescale")

    variant_key = (variant or 'base').lower()

    variant_options = {
        'base': {
            'input_xml': os.path.join(mjcf_dir, "smpl_humanoid_lower_body_base.xml"),
            'ref_usda': os.path.join(usd_dir, "smpl_humanoid_lower_body_base.usda"),
            'robot_type_base': 'smpl_humanoid_lower_body',
            'config_prefix': 'SMPL_LOWER_BODY',
            'usd_stem': 'smpl_humanoid_lower_body',
            'asset_stem': 'smpl_humanoid_lower_body',
            'base_robot_config': 'SMPL_LOWER_BODY_CFG',
        },
        'adjusted_pd': {
            'input_xml': os.path.join(rescale_dir, "smpl_humanoid_lower_body_adjusted_pd.xml"),
            'ref_usda': os.path.join(rescale_dir, "smpl_humanoid_lower_body_adjusted_pd.usda"),
            'robot_type_base': 'smpl_humanoid_lower_body_adjusted_pd',
            'config_prefix': 'SMPL_LOWER_BODY_ADJUSTED_PD',
            'usd_stem': 'smpl_humanoid_lower_body_adjusted_pd',
            'asset_stem': 'smpl_humanoid_lower_body_adjusted_pd',
            'base_robot_config': 'SMPL_LOWER_BODY_CFG',
        },
        'adjusted_torque': {
            'input_xml': os.path.join(rescale_dir, "smpl_humanoid_lower_body_adjusted_torque.xml"),
            'ref_usda': os.path.join(rescale_dir, "smpl_humanoid_lower_body_adjusted_torque.usda"),
            'robot_type_base': 'smpl_humanoid_lower_body_adjusted_torque',
            'config_prefix': 'SMPL_LOWER_BODY_ADJUSTED_TORQUE',
            'usd_stem': 'smpl_humanoid_lower_body_adjusted_torque',
            'asset_stem': 'smpl_humanoid_lower_body_adjusted_torque',
            'base_robot_config': 'SMPL_LOWER_BODY_ADJUSTED_TORQUE_HEIGHT_180CM_CFG',
        },
    }

    variant_info = variant_options.get(variant_key, variant_options['base'])

    return {
        'input_xml': variant_info['input_xml'],
        'ref_usda': variant_info['ref_usda'],
        'template_yaml': "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\config\\robot\\smpl_humanoid_lower_body.yaml",
        'robots_file': "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\simulator\\isaaclab\\utils\\robots.py",
        'scene_file': "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\simulator\\isaaclab\\utils\\scene.py",
        'output_xml_dir': mjcf_dir,
        'output_usda_dir': usd_dir,
        'variant': variant_key,
        'robot_type_base': variant_info['robot_type_base'],
        'config_prefix': variant_info['config_prefix'],
        'usd_stem': variant_info['usd_stem'],
        'asset_stem': variant_info['asset_stem'],
        'base_robot_config': variant_info['base_robot_config'],
    }


def main():
    """Main function to run the master scaling script."""
    parser = argparse.ArgumentParser(
        description="Master script to scale a robot to a new height",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--height', type=float, required=True,
        help='Target height in meters (e.g., 1.8)'
    )

    parser.add_argument(
        '--asset_variant', type=str, choices=['base', 'adjusted_pd', 'adjusted_torque'],
        default='base', help='Select which asset variant to scale from'
    )
    
    # Optional file paths (will use defaults if not provided)
    parser.add_argument(
        '--input_xml', type=str, default=None,
        help='Path to the input XML file'
    )
    parser.add_argument(
        '--ref_usda', type=str, default=None,
        help='Path to the reference USDA file'
    )
    parser.add_argument(
        '--template_yaml', type=str, default=None,
        help='Path to the template YAML configuration'
    )
    parser.add_argument(
        '--robots_file', type=str, default=None,
        help='Path to the robots.py file'
    )
    parser.add_argument(
        '--scene_file', type=str, default=None,
        help='Path to the scene.py file'
    )
    
    # Options
    parser.add_argument(
        '--skip_robot_config', action='store_true',
        help='Skip updating robot configurations (steps 1-3 only)'
    )
    parser.add_argument(
        '--step_by_step', action='store_true',
        help='Pause between each step for verification'
    )
    parser.add_argument(
        '--force_overwrite', action='store_true',
        help='Overwrite existing files without prompting'
    )
    parser.add_argument(
        '--check_only', action='store_true',
        help='Only check for existing files and exit (no generation)'
    )
    
    args = parser.parse_args()
    
    # Validate height
    if args.height <= 0 or args.height > 3.0:
        print(f"Error: Height {args.height}m seems unrealistic. Please use a value between 0 and 3.0 meters.")
        return 1
    
    # Get base paths
    base_paths = get_default_paths(args.asset_variant)
    
    # Override with provided paths
    if args.input_xml:
        base_paths['input_xml'] = args.input_xml
    if args.ref_usda:
        base_paths['ref_usda'] = args.ref_usda
    if args.template_yaml:
        base_paths['template_yaml'] = args.template_yaml
    if args.robots_file:
        base_paths['robots_file'] = args.robots_file
    if args.scene_file:
        base_paths['scene_file'] = args.scene_file
    
    # Remove robot config files if skipping that step
    if args.skip_robot_config:
        base_paths.pop('robots_file', None)
        base_paths.pop('scene_file', None)
    
    # Validate required files exist
    required_files = ['input_xml', 'ref_usda', 'template_yaml']
    for file_key in required_files:
        if file_key in base_paths and not os.path.exists(base_paths[file_key]):
            print(f"Error: Required file not found: {base_paths[file_key]}")
            return 1
    
    # Create the scaler and run the process
    scaler = RobotScaler(args.height, base_paths, variant=args.asset_variant)
    
    # Handle check-only mode
    if args.check_only:
        print("\n--- CHECK ONLY MODE ---")
        existing_files = scaler.check_existing_files()
        scaler.show_existing_files_status(existing_files, force_overwrite=True)  # Just show status
        return 0
    
    if args.step_by_step:
        print("\n--- STEP-BY-STEP MODE ---")
        print("You will be prompted to continue after each step.\n")
        
        # Check for existing files first
        existing_files = scaler.check_existing_files()
        if not scaler.show_existing_files_status(existing_files, args.force_overwrite):
            return 0
        
        # Step 1
        if not scaler.step1_scale_xml():
            return 1
        if input("\nContinue to Step 2? (y/N): ").lower() != 'y':
            print("Process stopped by user.")
            return 0
        
        # Step 2
        if not scaler.step2_generate_usda():
            return 1
        if input("\nContinue to Step 3? (y/N): ").lower() != 'y':
            print("Process stopped by user.")
            return 0
        
        # Step 3
        if not scaler.step3_create_yaml_config():
            return 1
        
        # Step 4 (if not skipped)
        if not args.skip_robot_config:
            if input("\nContinue to Step 4 (robot configs)? (y/N): ").lower() != 'y':
                print("Skipping robot configuration update.")
            else:
                scaler.step4_update_robot_configs()
                
                # Step 5 (scene initialization)
                if 'scene_file' in scaler.base_paths:
                    if input("\nContinue to Step 5 (scene initialization)? (y/N): ").lower() != 'y':
                        print("Skipping scene initialization update.")
                    else:
                        scaler.step5_update_scene_initialization()
        
        print("\nStep-by-step process completed!")
        
    else:
        # Run complete process
        success = scaler.run_complete_scaling(args.force_overwrite)
        return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
