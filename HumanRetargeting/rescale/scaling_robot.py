#!/usr/bin/env python3
"""
Robot Configuration Scaling Script

This script creates new robot configurations in robots.py for scaled robots.
It copies the base SMPL_LOWER_BODY_CFG and creates new configurations with
updated USD paths and appropriate naming to match the generated YAML configs.

Author: Auto-generated
Date: 2025-09-23
"""

import os
import re
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


JointGains = Dict[str, Dict[str, float]]


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_joint_gains_from_xml(xml_path: str) -> JointGains:
    """Parse joint stiffness and damping values from an MJCF XML file."""

    if not os.path.exists(xml_path):
        return {}

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}

    joint_gains: JointGains = {}

    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if not name:
            continue

        stiffness = _to_float(joint.get('stiffness'))
        damping = _to_float(joint.get('damping'))

        # Skip joints that don't explicitly specify gains.
        if stiffness is None and damping is None:
            continue

        gains: Dict[str, float] = {}
        if stiffness is not None:
            gains['stiffness'] = stiffness
        if damping is not None:
            gains['damping'] = damping
        joint_gains[name] = gains

    return joint_gains


def parse_joint_gains_from_usda(usda_path: str) -> JointGains:
    """Parse joint stiffness and damping values from a USDA file."""

    if not os.path.exists(usda_path):
        return {}

    joint_gains: JointGains = {}

    current_joint: Optional[str] = None
    brace_depth = 0
    axis_names: Dict[str, str] = {}
    axis_gains: Dict[str, Dict[str, float]] = {}

    mjcf_axis_pattern = re.compile(r'mjcf:(rot[XYZ]):name\s*=\s*"([^"]+)"')
    drive_pattern = re.compile(
        r'drive:(rot[XYZ]):physics:(stiffness|damping)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)'
    )

    try:
        with open(usda_path, 'r', encoding='utf-8') as fh:
            for raw_line in fh:
                line = raw_line.strip()

                if line.startswith('def PhysicsJoint'):
                    # Flush any pending joint before starting a new one.
                    if current_joint and axis_names:
                        for axis_key, joint_name in axis_names.items():
                            gains = axis_gains.get(axis_key, {})
                            if not gains:
                                continue
                            stored: Dict[str, float] = {}
                            if 'stiffness' in gains:
                                stored['stiffness'] = gains['stiffness']
                            if 'damping' in gains:
                                stored['damping'] = gains['damping']
                            if stored:
                                joint_gains[joint_name] = stored

                    # Reset trackers for the new joint.
                    parts = line.split('"')
                    current_joint = parts[1] if len(parts) > 1 else None
                    brace_depth = 0
                    axis_names = {}
                    axis_gains = {}
                    continue

                if current_joint is None:
                    continue

                brace_depth += raw_line.count('{') - raw_line.count('}')

                if brace_depth < 0:
                    # Block closed unexpectedly; reset state.
                    current_joint = None
                    axis_names = {}
                    axis_gains = {}
                    brace_depth = 0
                    continue

                if brace_depth == 0:
                    # Leaving the joint block; flush collected values.
                    for axis_key, joint_name in axis_names.items():
                        gains = axis_gains.get(axis_key, {})
                        if not gains:
                            continue
                        stored: Dict[str, float] = {}
                        if 'stiffness' in gains:
                            stored['stiffness'] = gains['stiffness']
                        if 'damping' in gains:
                            stored['damping'] = gains['damping']
                        if stored:
                            joint_gains[joint_name] = stored

                    current_joint = None
                    axis_names = {}
                    axis_gains = {}
                    continue

                name_match = mjcf_axis_pattern.search(line)
                if name_match:
                    axis_key = name_match.group(1)
                    axis_names[axis_key] = name_match.group(2)
                    continue

                drive_match = drive_pattern.search(line)
                if drive_match:
                    axis_key = drive_match.group(1)
                    quantity = drive_match.group(2)
                    value = _to_float(drive_match.group(3))
                    if value is None:
                        continue
                    axis_gains.setdefault(axis_key, {})[quantity] = value

    except OSError:
        return {}

    if current_joint and axis_names:
        for axis_key, joint_name in axis_names.items():
            gains = axis_gains.get(axis_key, {})
            if not gains:
                continue
            stored: Dict[str, float] = {}
            if 'stiffness' in gains:
                stored['stiffness'] = gains['stiffness']
            if 'damping' in gains:
                stored['damping'] = gains['damping']
            if stored:
                joint_gains[joint_name] = stored

    return joint_gains


def _split_joint_name(name: str) -> Tuple[str, Optional[str]]:
    if '_' not in name:
        return name, None
    base, axis = name.rsplit('_', 1)
    return base, axis


def _format_gain_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def _build_gain_mappings(joint_gains: JointGains) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Convert per-joint gains into dictionaries keyed by joint expressions."""

    stiffness_map: Dict[str, float] = {}
    damping_map: Dict[str, float] = {}

    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    for joint_name, gains in joint_gains.items():
        base, axis = _split_joint_name(joint_name)
        grouped.setdefault(base, {})[axis or ''] = gains

    for base, axes in grouped.items():
        stiffness_values = {
            gains['stiffness']
            for gains in axes.values()
            if 'stiffness' in gains
        }
        damping_values = {
            gains['damping']
            for gains in axes.values()
            if 'damping' in gains
        }

        if axes and len(stiffness_values) == 1:
            stiffness_map[f"{base}_."] = stiffness_values.pop()
        else:
            for axis, gains in axes.items():
                if 'stiffness' in gains:
                    key = base if not axis else f"{base}_{axis}"
                    stiffness_map[key] = gains['stiffness']

        if axes and len(damping_values) == 1:
            damping_map[f"{base}_."] = damping_values.pop()
        else:
            for axis, gains in axes.items():
                if 'damping' in gains:
                    key = base if not axis else f"{base}_{axis}"
                    damping_map[key] = gains['damping']

    return stiffness_map, damping_map


def _format_gain_block(key: str, mapping: Dict[str, float], indent: str) -> str:
    inner_indent = indent + "    "
    if not mapping:
        return f"{indent}{key}={{\n{indent}}},"

    lines = [f"{indent}{key}={{"]
    for name in sorted(mapping):
        lines.append(f'{inner_indent}"{name}": {_format_gain_value(mapping[name])},')
    lines.append(f"{indent}}},")
    return "\n".join(lines)


def _replace_dict_block(config_text: str, key: str, mapping: Dict[str, float]) -> str:
    search_token = f"{key}="
    start_idx = config_text.find(search_token)
    if start_idx == -1:
        return config_text

    brace_start = config_text.find('{', start_idx)
    if brace_start == -1:
        return config_text

    depth = 0
    idx = brace_start
    brace_end = -1
    while idx < len(config_text):
        if config_text[idx] == '{':
            depth += 1
        elif config_text[idx] == '}':
            depth -= 1
            if depth == 0:
                brace_end = idx
                break
        idx += 1

    if brace_end == -1:
        return config_text

    block_end = brace_end + 1
    while block_end < len(config_text) and config_text[block_end] in ' \t':
        block_end += 1
    if block_end < len(config_text) and config_text[block_end] == ',':
        block_end += 1

    line_start = config_text.rfind('\n', 0, start_idx)
    indent_start = line_start + 1 if line_start != -1 else 0
    indent = config_text[indent_start:start_idx]

    new_block = _format_gain_block(key, mapping, indent)

    return config_text[:start_idx] + new_block + config_text[block_end:]


def inject_joint_gains(config_text: str, joint_gains: JointGains) -> str:
    stiffness_map, damping_map = _build_gain_mappings(joint_gains)
    updated = config_text
    if stiffness_map:
        updated = _replace_dict_block(updated, 'stiffness', stiffness_map)
    if damping_map:
        updated = _replace_dict_block(updated, 'damping', damping_map)
    return updated


def extract_robot_config(file_path: str, config_name: str = "SMPL_LOWER_BODY_CFG") -> Optional[str]:
    """
    Extract a robot configuration from robots.py file.
    
    Args:
        file_path: Path to the robots.py file
        config_name: Name of the configuration to extract
        
    Returns:
        The configuration string or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match the configuration block
        pattern = rf"^{re.escape(config_name)}\s*=\s*ArticulationCfg\("
        
        lines = content.split('\n')
        config_lines = []
        in_config = False
        paren_count = 0
        
        for line in lines:
            if re.match(pattern, line.strip()):
                in_config = True
                config_lines.append(line)
                paren_count += line.count('(') - line.count(')')
            elif in_config:
                config_lines.append(line)
                paren_count += line.count('(') - line.count(')')
                
                # End of configuration when parentheses are balanced
                if paren_count <= 0:
                    break
        
        if config_lines:
            return '\n'.join(config_lines)
        else:
            print(f"Configuration '{config_name}' not found in {file_path}")
            return None
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def create_scaled_config(
    base_config: str,
    height: float,
    base_config_name: str = "SMPL_LOWER_BODY_CFG",
    config_prefix: Optional[str] = None,
    usd_stem: Optional[str] = None,
    joint_gains: Optional[JointGains] = None,
) -> str:
    """
    Create a scaled robot configuration from a base configuration.
    
    Args:
        base_config: The base configuration string
        height: Height of the scaled robot
        base_config_name: Name of the base configuration
        config_prefix: Optional override for the generated configuration prefix
        usd_stem: Optional override for the USD filename stem used in paths
        
    Returns:
        The new scaled configuration string
    """
    # Create new configuration name
    height_int = round(height * 100)  # Convert to centimeters to avoid decimal points
    prefix = config_prefix or base_config_name.replace('_CFG', '')
    new_config_name = f"{prefix}_HEIGHT_{height_int}CM_CFG"
    
    # Replace the configuration name
    scaled_config = base_config.replace(base_config_name, new_config_name, 1)
    
    # Update the USD path to point to the scaled robot
    usd_stem = usd_stem or "smpl_humanoid_lower_body"
    new_usd_value = f"protomotions/data/assets/usd/{usd_stem}_height_{height_int}cm.usda"
    usd_pattern = r'(usd_path=")([^"]+)(")'
    scaled_config = re.sub(usd_pattern, r"\1" + new_usd_value + r"\3", scaled_config, count=1)
    
    # Optionally update the initial position based on height scaling
    # Keep the same relative ground clearance
    original_height = 1.7  # Original model height
    scale_factor = height / original_height
    new_pos_z = 0.95 * scale_factor
    
    # Update the initial position
    old_pos_pattern = r'pos=\(0\.0,\s*0\.0,\s*0\.95\)'
    new_pos = f'pos=(0.0, 0.0, {new_pos_z:.2f})'
    scaled_config = re.sub(old_pos_pattern, new_pos, scaled_config)
    
    if joint_gains:
        scaled_config = inject_joint_gains(scaled_config, joint_gains)

    return scaled_config


def insert_configs_into_file(file_path: str, new_configs: List[str],
                             insert_after: str = "SMPL_LOWER_BODY_CFG") -> bool:
    """
    Insert new configurations into the robots.py file after a specified configuration.
    
    Args:
        file_path: Path to the robots.py file
        new_configs: List of new configuration strings
        insert_after: Configuration name after which to insert
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the end of the specified configuration
        pattern = rf"^{re.escape(insert_after)}\s*=\s*ArticulationCfg\("
        lines = content.split('\n')
        insert_line = -1
        in_config = False
        paren_count = 0
        
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                in_config = True
                paren_count += line.count('(') - line.count(')')
            elif in_config:
                paren_count += line.count('(') - line.count(')')
                
                # End of configuration when parentheses are balanced
                if paren_count <= 0:
                    insert_line = i + 1
                    break
        
        if insert_line == -1:
            print(f"Could not find end of configuration '{insert_after}'")
            return False
        
        # Insert new configurations
        new_lines = lines[:insert_line]
        
        for config in new_configs:
            new_lines.append("")  # Empty line before new config
            new_lines.append("")  # Second empty line for spacing
            new_lines.extend(config.split('\n'))
        
        new_lines.extend(lines[insert_line:])
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        
        print(f"Successfully inserted {len(new_configs)} configurations into {file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def find_config_with_prefix(file_path: str, config_prefix: str) -> Optional[str]:
    """Return the first configuration definition that matches the given prefix."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = rf'^({re.escape(config_prefix)}_HEIGHT_\d+CM_CFG)\s*=\s*ArticulationCfg\('
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
            return matches[0]
    except Exception as exc:
        print(f"Error scanning {file_path} for existing configurations: {exc}")
    return None


def create_backup(file_path: str) -> str:
    """
    Create a backup of the robots.py file.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    backup_path = f"{file_path}.backup"
    try:
        with open(file_path, 'r', encoding='utf-8') as original:
            with open(backup_path, 'w', encoding='utf-8') as backup:
                backup.write(original.read())
        print(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {e}")
        return ""


def update_scene_imports(scene_file: str, new_config_names: List[str],
                         create_backup_file: bool = True) -> bool:
    """
    Update the imports in scene.py to include new robot configurations.
    
    Args:
        scene_file: Path to the scene.py file
        new_config_names: List of new configuration names to add to imports
        create_backup_file: Whether to create a backup file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup if requested
        if create_backup_file:
            backup_path = create_backup(scene_file)
            if not backup_path:
                print("Failed to create backup of scene.py. Aborting.")
                return False
        
        with open(scene_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the import block for robots
        import_pattern = r'from protomotions\.simulator\.isaaclab\.utils\.robots import \((.*?)\)'
        match = re.search(import_pattern, content, re.DOTALL)
        
        if not match:
            print("Could not find robot imports in scene.py")
            return False
        
        # Extract current imports
        current_imports = match.group(1)
        current_import_list = [imp.strip().rstrip(',') for imp in current_imports.split('\n') if imp.strip()]
        
        # Add new imports (avoid duplicates)
        for config_name in new_config_names:
            if config_name not in current_import_list:
                current_import_list.append(config_name)
        
        # Format the new import block
        formatted_imports = ',\n    '.join(current_import_list)
        new_import_block = f'from protomotions.simulator.isaaclab.utils.robots import (\n    {formatted_imports},\n)'
        
        # Replace the import block
        new_content = re.sub(import_pattern, new_import_block, content, flags=re.DOTALL)
        
        # Write back to file
        with open(scene_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Successfully updated imports in {os.path.basename(scene_file)}")
        print(f"Added {len(new_config_names)} new configuration imports")
        return True
        
    except Exception as e:
        print(f"Error updating scene.py imports: {e}")
        return False


def generate_scaled_robot_configs(
    robots_file: str,
    heights: List[float],
    base_config: str = "SMPL_LOWER_BODY_CFG",
    create_backup_file: bool = True,
    scene_file: Optional[str] = None,
    update_scene_file: bool = True,
    config_prefix: Optional[str] = None,
    usd_stem: Optional[str] = None,
    joint_gains_per_height: Optional[Dict[int, JointGains]] = None,
) -> bool:
    """
    Generate scaled robot configurations and insert them into robots.py.
    Optionally update scene.py imports.
    
    Args:
        robots_file: Path to the robots.py file
        heights: List of heights to create configurations for
        base_config: Name of the base configuration to copy
        create_backup_file: Whether to create a backup file
        scene_file: Path to the scene.py file (optional)
        update_scene_file: Whether to update scene.py imports
        config_prefix: Optional override for the generated configuration prefix
        usd_stem: Optional override for the USD filename stem
        joint_gains_per_height: Optional mapping of height (cm) to joint gain data
        
    Returns:
        True if successful, False otherwise
    """
    # Create backup if requested
    if create_backup_file:
        backup_path = create_backup(robots_file)
        if not backup_path:
            print("Failed to create backup. Aborting.")
            return False
    
    # Determine which configuration to use as the template for duplication
    template_config_name = base_config
    base_config_text = extract_robot_config(robots_file, template_config_name)

    # If a variant prefix is provided, prefer to copy an existing config with that prefix
    if config_prefix:
        preferred_template = find_config_with_prefix(robots_file, config_prefix)
        if preferred_template:
            if preferred_template != template_config_name:
                alt_config_text = extract_robot_config(robots_file, preferred_template)
                if alt_config_text:
                    print(f"Using existing configuration '{preferred_template}' as template for new variants.")
                    template_config_name = preferred_template
                    base_config_text = alt_config_text
            elif not base_config_text:
                base_config_text = extract_robot_config(robots_file, preferred_template)

    if not base_config_text:
        print(f"Error: Could not locate template configuration '{template_config_name}'.")
        return False
    
    # Create scaled configurations
    new_configs = []
    new_config_names = []
    effective_prefix = config_prefix or template_config_name.replace('_CFG', '')
    if '_HEIGHT_' in effective_prefix:
        effective_prefix = effective_prefix.split('_HEIGHT_')[0]

    for height in heights:
        height_int = round(height * 100)
        joint_gains = None
        if joint_gains_per_height:
            joint_gains = joint_gains_per_height.get(height_int)
            if joint_gains is None:
                print(f"Warning: No joint gain overrides found for {height_int}cm; using template values")

        scaled_config = create_scaled_config(
            base_config_text,
            height,
            template_config_name,
            config_prefix=effective_prefix,
            usd_stem=usd_stem,
            joint_gains=joint_gains,
        )
        new_configs.append(scaled_config)

        config_name = f"{effective_prefix}_HEIGHT_{height_int}CM_CFG"
        new_config_names.append(config_name)
        print(f"Generated configuration for height {height:.1f}m")

    # Insert configurations into file
    success = insert_configs_into_file(robots_file, new_configs, template_config_name)
    
    if not success:
        return False
    
    # Update scene.py imports if requested
    if update_scene_file and scene_file:
        if os.path.exists(scene_file):
            scene_success = update_scene_imports(scene_file, new_config_names, create_backup_file)
            if not scene_success:
                print("Warning: Failed to update scene.py imports, but robot configs were added successfully")
        else:
            print(f"Warning: Scene file not found: {scene_file}")
    
    if success:
        print(f"\nSuccessfully added {len(new_configs)} scaled robot configurations!")
        print("The following configurations were added:")
        for config_name in new_config_names:
            print(f"  - {config_name}")
        
        if update_scene_file and scene_file and os.path.exists(scene_file):
            print("\nScene.py imports have been updated to include the new configurations.")
    
    return success


def list_existing_configs(robots_file: str) -> List[str]:
    """
    List all existing robot configurations in the robots.py file.
    
    Args:
        robots_file: Path to the robots.py file
        
    Returns:
        List of configuration names
    """
    try:
        with open(robots_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all configuration definitions
        pattern = r'^([A-Z_]+_CFG)\s*=\s*ArticulationCfg\('
        matches = re.findall(pattern, content, re.MULTILINE)
        
        return matches
        
    except Exception as e:
        print(f"Error reading {robots_file}: {e}")
        return []


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Generate scaled robot configurations for robots.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--robots_file', type=str,
        default='D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\simulator\\isaaclab\\utils\\robots.py',
        help='Path to the robots.py file'
    )
    parser.add_argument(
        '--scene_file', type=str,
        default='D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\simulator\\isaaclab\\utils\\scene.py',
        help='Path to the scene.py file'
    )
    parser.add_argument(
        '--heights', type=float, nargs='+',
        default=[1.5, 1.8, 2.0, 2.2],
        help='List of heights to create configurations for'
    )
    parser.add_argument(
        '--base_config', type=str,
        default='SMPL_LOWER_BODY_CFG',
        help='Name of the base configuration to copy'
    )
    parser.add_argument(
        '--no_backup', action='store_true',
        help='Skip creating a backup file'
    )
    parser.add_argument(
        '--no_scene_update', action='store_true',
        help='Skip updating scene.py imports'
    )
    parser.add_argument(
        '--list_configs', action='store_true',
        help='List existing configurations and exit'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.robots_file):
        print(f"Error: robots.py file not found: {args.robots_file}")
        return 1
    
    print("=== ROBOT CONFIGURATION SCALING ===")
    print(f"Target file: {os.path.basename(args.robots_file)}")
    print(f"Scene file: {os.path.basename(args.scene_file)}")
    
    if args.list_configs:
        print("\nExisting configurations:")
        configs = list_existing_configs(args.robots_file)
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config}")
        return 0
    
    print(f"Base config: {args.base_config}")
    print(f"Heights: {', '.join(f'{h:.1f}m' for h in args.heights)}")
    print(f"Create backup: {not args.no_backup}")
    print(f"Update scene.py: {not args.no_scene_update}")
    
    # Check if files exist
    if not os.path.exists(args.robots_file):
        print(f"\nError: robots.py file not found: {args.robots_file}")
        return 1
    
    if not args.no_scene_update and not os.path.exists(args.scene_file):
        print(f"\nWarning: scene.py file not found: {args.scene_file}")
        print("Scene import updates will be skipped.")
    
    # Check if base configuration exists
    existing_configs = list_existing_configs(args.robots_file)
    if args.base_config not in existing_configs:
        print(f"\nError: Base configuration '{args.base_config}' not found!")
        print("Available configurations:")
        for config in existing_configs:
            print(f"  - {config}")
        return 1
    
    # Generate scaled configurations
    success = generate_scaled_robot_configs(
        args.robots_file,
        args.heights,
        args.base_config,
        not args.no_backup,
        args.scene_file if not args.no_scene_update else None,
        not args.no_scene_update
    )
    
    if success:
        print(f"\n{'='*50}")
        print("SUCCESS! Scaled robot configurations added to robots.py")
        print("\nYou can now use these configurations in your code:")
        for height in args.heights:
            height_int = round(height * 100)  # Convert to centimeters to avoid decimal points
            config_name = f"SMPL_LOWER_BODY_HEIGHT_{height_int}CM_CFG"
            print(f"  from robots import {config_name}")
        print(f"\n{'='*50}")
        return 0
    else:
        print("\nFailed to generate scaled configurations.")
        return 1


if __name__ == "__main__":
    import sys
    
    # Example usage when run directly
    if len(sys.argv) == 1:  # No command line arguments
        print("=== ROBOT CONFIGURATION SCALING - Example Mode ===\n")
        
        robots_file = "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\simulator\\isaaclab\\utils\\robots.py"
        
        if not os.path.exists(robots_file):
            print(f"robots.py file not found: {robots_file}")
            print("Please update the path in the script or provide command line arguments.")
        else:
            print(f"Target file: {os.path.basename(robots_file)}")
            print("Listing existing configurations...")
            
            configs = list_existing_configs(robots_file)
            print(f"\nFound {len(configs)} existing configurations:")
            for i, config in enumerate(configs, 1):
                print(f"  {i}. {config}")
            
            print("\nExample usage:")
            print("python scaling_robot_configs.py --heights 1.6 1.9 2.1")
            print("python scaling_robot_configs.py --base_config SMPL_LOWER_BODY_CFG --heights 2.0")
            print("python scaling_robot_configs.py --list_configs")
            print("python scaling_robot_configs.py --no_scene_update --heights 1.8 2.1")
    else:
        sys.exit(main())
