import os
import yaml
from typing import List, Optional, Dict, Any
import argparse


def load_yaml_template(template_path: str) -> Optional[Dict[Any, Any]]:
    """
    Load a YAML template file.
    
    Args:
        template_path: Path to the template YAML file
        
    Returns:
        Dictionary containing the YAML data, or None if failed
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML template from {template_path}: {e}")
        return None


def create_scaled_yaml_config(
    template_path: str,
    output_path: str,
    height: float,
    xml_filename: str = None,
    usda_filename: str = None,
    robot_type_base: Optional[str] = None,
) -> bool:
    """
    Create a new YAML configuration file for a scaled robot.
    
    Args:
        template_path: Path to the base YAML template
        output_path: Path where the new YAML file will be saved
        height: Height of the scaled robot in meters
        xml_filename: Name of the scaled XML file (optional, will be inferred if not provided)
        usda_filename: Name of the scaled USDA file (optional, will be inferred if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    # Load the template
    config_data = load_yaml_template(template_path)
    if config_data is None:
        return False
    
    height_int = round(height * 100)  # Convert to centimeters to avoid decimal points

    # Determine base robot type for naming
    original_robot_type = config_data['robot']['asset']['robot_type']
    base_robot_type = robot_type_base or original_robot_type

    # Generate default filenames if not provided
    if xml_filename is None:
        xml_filename = f"{base_robot_type}_height_{height_int}cm.xml"
    if usda_filename is None:
        usda_filename = f"{base_robot_type}_height_{height_int}cm.usda"
    
    # Update the robot type to include the height
    new_robot_type = f"{base_robot_type}_height_{height_int}cm"
    config_data['robot']['asset']['robot_type'] = new_robot_type
    
    # Update the asset file paths
    config_data['robot']['asset']['asset_file_name'] = f"mjcf/{xml_filename}"
    config_data['robot']['asset']['usd_asset_file_name'] = f"usd/{usda_filename}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the new YAML file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write("# @package _global_\n")
            f.write(f"# Auto-generated configuration for scaled robot (height: {height:.1f}m)\n")
            f.write(f"# Based on template: {os.path.basename(template_path)}\n\n")
            
            # Write the YAML data (excluding the first line if it's a comment)
            yaml_content = yaml.dump(config_data, default_flow_style=False, sort_keys=False, width=120)
            f.write(yaml_content)
        
        print(f"Successfully created YAML config: {output_path}")
        print(f"  Robot type: {new_robot_type}")
        print(f"  XML file: {xml_filename}")
        print(f"  USDA file: {usda_filename}")
        return True
        
    except Exception as e:
        print(f"Error saving YAML config to {output_path}: {e}")
        return False


def generate_config_filename(
    template_path: str,
    height: float,
    output_dir: str = None,
    name_prefix: Optional[str] = None,
) -> str:
    """
    Generate an appropriate filename for the scaled robot config.
    
    Args:
        template_path: Path to the template file
        height: Height of the scaled robot
        output_dir: Output directory (defaults to same as template)
        
    Returns:
        Full path for the new config file
    """
    if output_dir is None:
        output_dir = os.path.dirname(template_path)
    
    # Extract base name from template
    if name_prefix:
        base_name = name_prefix
    else:
        base_name = os.path.splitext(os.path.basename(template_path))[0]
    
    # Create new filename with height suffix
    height_int = round(height * 100)  # Convert to centimeters to avoid decimal points
    new_filename = f"{base_name}_height_{height_int}cm.yaml"
    
    return os.path.join(output_dir, new_filename)


def batch_create_yaml_configs(
    template_path: str,
    heights: List[float],
    output_dir: str = None,
    name_prefix: Optional[str] = None,
    robot_type_base: Optional[str] = None,
) -> List[str]:
    """
    Create multiple YAML configuration files for different robot heights.
    
    Args:
        template_path: Path to the base YAML template
        heights: List of heights in meters to create configs for
        output_dir: Output directory (defaults to same as template)
        
    Returns:
        List of successfully created config file paths
    """
    created_files = []
    
    for height in heights:
        output_path = generate_config_filename(template_path, height, output_dir, name_prefix)
        
        if create_scaled_yaml_config(
            template_path,
            output_path,
            height,
            robot_type_base=robot_type_base,
        ):
            created_files.append(output_path)
    
    return created_files


def update_existing_config(config_path: str, xml_path: str = None, usda_path: str = None) -> bool:
    """
    Update an existing YAML config to point to specific XML and USDA files.
    
    Args:
        config_path: Path to the YAML config file to update
        xml_path: Path to the XML file (relative to assets directory)
        usda_path: Path to the USDA file (relative to assets directory)
        
    Returns:
        True if successful, False otherwise
    """
    config_data = load_yaml_template(config_path)
    if config_data is None:
        return False
    
    # Update file paths if provided
    if xml_path:
        config_data['robot']['asset']['asset_file_name'] = xml_path
    if usda_path:
        config_data['robot']['asset']['usd_asset_file_name'] = usda_path
    
    # Save the updated config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            # Preserve header
            f.write("# @package _global_\n\n")
            yaml_content = yaml.dump(config_data, default_flow_style=False, sort_keys=False, width=120)
            f.write(yaml_content)
        
        print(f"Updated YAML config: {config_path}")
        return True
        
    except Exception as e:
        print(f"Error updating YAML config {config_path}: {e}")
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Generate YAML configuration files for scaled robots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--template', type=str,
        default='D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\config\\robot\\smpl_humanoid_lower_body.yaml',
        help='Path to the template YAML configuration file.'
    )
    parser.add_argument(
        '--heights', type=float, nargs='+',
        default=[1.5, 1.8, 2.0, 2.2],
        help='List of heights in meters to create configs for.'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for the new config files. Defaults to same directory as template.'
    )
    parser.add_argument(
        '--single_height', type=float, default=None,
        help='Create config for a single height instead of batch processing.'
    )
    parser.add_argument(
        '--xml_file', type=str, default=None,
        help='Specific XML filename to use (for single height mode).'
    )
    parser.add_argument(
        '--usda_file', type=str, default=None,
        help='Specific USDA filename to use (for single height mode).'
    )
    
    args = parser.parse_args()
    
    print("=== YAML Configuration Generator ===\n")
    
    if not os.path.exists(args.template):
        print(f"Error: Template file not found: {args.template}")
        return
    
    if args.single_height is not None:
        # Single height mode
        output_path = generate_config_filename(args.template, args.single_height, args.output_dir)
        success = create_scaled_yaml_config(
            args.template, output_path, args.single_height,
            args.xml_file, args.usda_file
        )
        if success:
            print(f"\nSuccessfully created config for height {args.single_height}m")
        else:
            print(f"\nFailed to create config for height {args.single_height}m")
    else:
        # Batch mode
        print(f"Creating configs for heights: {', '.join(f'{h:.1f}m' for h in args.heights)}")
        created_files = batch_create_yaml_configs(args.template, args.heights, args.output_dir)
        
        print(f"\n{'='*50}")
        print(f"Successfully created {len(created_files)} configuration files:")
        for config_file in created_files:
            print(f"  - {os.path.basename(config_file)}")
        
        if len(created_files) < len(args.heights):
            failed_count = len(args.heights) - len(created_files)
            print(f"\nWarning: {failed_count} configuration(s) failed to create.")
    
    print("\nDone!")


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:  # No command line arguments
        print("=== YAML Configuration Generator - Example Mode ===\n")
        
        template_path = "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\config\\robot\\smpl_humanoid_lower_body.yaml"
        
        if not os.path.exists(template_path):
            print(f"Template file not found: {template_path}")
            print("Please update the path in the script or provide command line arguments.")
        else:
            # Test with a few different heights
            test_heights = [1.6, 1.9, 2.1]
            
            print(f"Template: {os.path.basename(template_path)}")
            print(f"Creating configs for heights: {', '.join(f'{h:.1f}m' for h in test_heights)}")
            
            created_files = batch_create_yaml_configs(template_path, test_heights)
            
            print(f"\n{'='*50}")
            print(f"Created {len(created_files)} configuration files:")
            for config_file in created_files:
                print(f"  - {os.path.basename(config_file)}")
            
            print("\nExample usage with command line:")
            print("python scaling_yaml.py --heights 1.5 2.0 2.5")
            print("python scaling_yaml.py --single_height 1.8 --xml_file my_robot.xml --usda_file my_robot.usda")
    else:
        main()
