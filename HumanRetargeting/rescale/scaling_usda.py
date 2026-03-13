import xml.etree.ElementTree as ET
import re
import argparse
import os
from collections import OrderedDict
import math


def extract_body_data_from_xml(xml_path):
    """
    Parses a MuJoCo XML file to extract the full body hierarchy, including names,
    parent-child relationships, relative positions, geometry types, and detailed capsule info.

    Args:
        xml_path (str): The file path to the input XML file.

    Returns:
        tuple: A tuple containing:
            - dict: `body_data` mapping body names to their parent, position, geom type, and capsule info.
            - dict: `hierarchy` mapping parent names to a list of their children.
        Returns (None, None) if the file cannot be parsed.
    """
    body_data = OrderedDict()
    hierarchy = OrderedDict()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def traverse_bodies(element, parent_name=None):
            for body in element.findall('body'):
                name = body.get('name')
                pos_str = body.get('pos')
                if name and pos_str:
                    pos_values = [float(p) for p in pos_str.split()]
                    geom_element = body.find('geom')
                    geom_type = geom_element.get('type') if geom_element is not None else 'none'
                    
                    # Extract capsule information if it's a capsule
                    capsule_info = None
                    if geom_type == 'capsule' and geom_element is not None:
                        fromto_str = geom_element.get('fromto')
                        if fromto_str:
                            fromto_values = [float(v) for v in fromto_str.split()]
                            # Calculate from and to points
                            from_point = fromto_values[:3]
                            to_point = fromto_values[3:]
                            
                            # Calculate height (distance between points)
                            capsule_height = math.sqrt(sum((a - b)**2 for a, b in zip(to_point, from_point)))
                            
                            # Calculate center point (midpoint of from and to)
                            center_point = [(a + b) / 2 for a, b in zip(from_point, to_point)]
                            
                            capsule_info = {
                                'height': capsule_height,
                                'center': center_point,
                                'from_point': from_point,
                                'to_point': to_point
                            }

                    body_data[name] = {
                        'parent': parent_name,
                        'pos': pos_values,
                        'geom': geom_type,
                        'capsule_info': capsule_info
                    }
                    
                    if parent_name:
                        if parent_name not in hierarchy:
                            hierarchy[parent_name] = []
                        hierarchy[parent_name].append(name)
                    traverse_bodies(body, name)

        worldbody = root.find('worldbody')
        if worldbody is not None:
            traverse_bodies(worldbody)

    except (FileNotFoundError, ET.ParseError) as e:
        print(f"Error processing XML file at '{xml_path}': {e}")
        return None, None

    return body_data, hierarchy


def update_usda_file(ref_usda_path, body_data, hierarchy, output_usda_path):
    """
    Updates a USDA file with new body transforms, joint positions, capsule heights, and capsule transforms.

    Args:
        ref_usda_path (str): Path to the reference USDA template file.
        body_data (dict): Dictionary containing data for each body.
        hierarchy (dict): Dictionary representing the parent-to-child hierarchy.
        output_usda_path (str): Path to save the newly created USDA file.
    """
    try:
        with open(ref_usda_path, 'r') as f:
            usda_content = f.read()
    except FileNotFoundError:
        print(f"Error: Reference USDA file not found at '{ref_usda_path}'")
        return

    absolute_positions = {}
    total_updates = 0

    root_name = next((name for name, data in body_data.items() if data['parent'] is None), None)
    if not root_name:
        print("Error: Could not determine the root body.")
        return

    def process_body_recursively(body_name, parent_abs_pos):
        nonlocal usda_content, total_updates

        relative_pos = body_data[body_name]['pos']
        current_abs_pos = [p + r for p, r in zip(parent_abs_pos, relative_pos)]
        absolute_positions[body_name] = current_abs_pos

        # --- 1. Update body's Xform transform (absolute position) ---
        new_transform_translation = f"{current_abs_pos[0]}, {current_abs_pos[1]}, {current_abs_pos[2]}, 1"
        transform_pattern = re.compile(
            r'(def Xform "' + re.escape(body_name) + r'".*?matrix4d xformOp:transform\s*=\s*\([^)]+\),\s*\([^)]+\),\s*\([^)]+\),\s*\()'
            r'[^)]+'
            r'(\))',
            re.DOTALL
        )
        if transform_pattern.search(usda_content):
            usda_content = transform_pattern.sub(r'\g<1>' + new_transform_translation + r'\g<2>', usda_content)
            print(f"Updated transform for body '{body_name}'.")
            total_updates += 1

        # --- 2. Update joint's localPos0 (relative position) ---
        new_joint_pos = f"({', '.join(map(str, relative_pos))})"
        joint_pattern = re.compile(
            r'(def PhysicsJoint "' + re.escape(body_name) + r'".*?point3f physics:localPos0 = ).*?(\n)',
            re.DOTALL
        )
        if joint_pattern.search(usda_content):
            usda_content = joint_pattern.sub(r'\1' + new_joint_pos + r'\2', usda_content)
            print(f"Updated position for joint '{body_name}'.")
            total_updates += 1

        # --- 3. Update capsule height and transform if applicable ---
        if body_data[body_name]['geom'] == 'capsule' and body_data[body_name]['capsule_info'] is not None:
            capsule_info = body_data[body_name]['capsule_info']
            new_height = capsule_info['height']
            new_center = capsule_info['center']
            
            # First, find the body section boundaries
            body_start_pattern = re.compile(
                r'def Xform "' + re.escape(body_name) + r'" \([^\)]*\)\s*\{',
                re.DOTALL
            )
            
            start_match = body_start_pattern.search(usda_content)
            if start_match:
                start_pos = start_match.start()
                
                # Find the matching closing brace for this body
                brace_count = 0
                pos = start_match.end()
                end_pos = len(usda_content)
                
                for i in range(pos, len(usda_content)):
                    if usda_content[i] == '{':
                        brace_count += 1
                    elif usda_content[i] == '}':
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                        brace_count -= 1
                
                # Extract the body section
                body_section = usda_content[start_pos:end_pos]
                original_section = body_section
                
                # Update capsule heights
                height_pattern = re.compile(r'(\s*double height = )[\d\.\-e\+]+')
                
                def replace_height(match):
                    nonlocal total_updates
                    total_updates += 1
                    return match.group(1) + str(new_height)
                
                body_section = height_pattern.sub(replace_height, body_section)
                
                # Update capsule transforms (Z-translation should be the center Z coordinate)
                transform_pattern = re.compile(
                    r'(\s*matrix4d xformOp:transform = \( \(1, 0, 0, 0\), \(0, 1, 0, 0\), \(0, 0, 1, 0\), \(0, 0, )'
                    r'[\d\.\-e\+]+'
                    r'(, 1\) \))'
                )
                
                def replace_transform(match):
                    nonlocal total_updates
                    total_updates += 1
                    return match.group(1) + str(new_center[2]) + match.group(2)
                
                body_section = transform_pattern.sub(replace_transform, body_section)
                
                if body_section != original_section:
                    # Replace the body section in the main content
                    usda_content = usda_content[:start_pos] + body_section + usda_content[end_pos:]
                    height_count = len(height_pattern.findall(original_section))
                    transform_count = len(transform_pattern.findall(original_section))
                    print(f"Updated {height_count} capsule height(s) and {transform_count} capsule transform(s) for body '{body_name}' to height={new_height:.6f}, center_z={new_center[2]:.6f}.")

        # Recurse for any children
        if body_name in hierarchy:
            for child_name in hierarchy[body_name]:
                process_body_recursively(child_name, current_abs_pos)

    # Process root body and initiate recursion
    root_pos = body_data[root_name]['pos']
    new_root_translation = f"{root_pos[0]}, {root_pos[1]}, {root_pos[2]}, 1"
    root_transform_pattern = re.compile(
        r'(def Xform "' + re.escape(root_name) + r'".*?matrix4d xformOp:transform\s*=\s*\([^)]+\),\s*\([^)]+\),\s*\([^)]+\),\s*\()'
        r'[^)]+'
        r'(\))', re.DOTALL
    )
    if root_transform_pattern.search(usda_content):
        usda_content = root_transform_pattern.sub(r'\g<1>' + new_root_translation + r'\g<2>', usda_content)
        print(f"Updated transform for root body '{root_name}'.")
        total_updates += 1
    
    if root_name in hierarchy:
        for child_name in hierarchy[root_name]:
            process_body_recursively(child_name, root_pos)

    try:
        with open(output_usda_path, 'w') as f:
            f.write(usda_content)
        print(f"\nSuccessfully created new USDA file at: '{output_usda_path}'")
        print(f"Total updates made: {total_updates}")
    except IOError as e:
        print(f"Error: Could not write to output file '{output_usda_path}'. Reason: {e}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Update body transforms, joint positions, capsule heights and transforms in a USDA file from a MuJoCo XML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--xml', type=str, default='D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\data\\assets\\mjcf\\smpl_humanoid_lower_body_base.xml',
        help='Path to the input XML file with new body positions.'
    )
    parser.add_argument(
        '--usda_ref', type=str, default='D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\data\\assets\\usd\\smpl_humanoid_lower_body_base.usda',
        help='Path to the reference USDA file to use as a template.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path for the output USDA file. Defaults to the XML filename with a .usda extension.'
    )
    args = parser.parse_args()

    output_path = args.output or f"{os.path.splitext(args.xml)[0]}.usda"

    print("--- Starting Complete USD Update (Heights + Transforms) ---")
    body_data, hierarchy = extract_body_data_from_xml(args.xml)

    if body_data and hierarchy:
        # Print debug information about capsules found
        capsule_bodies = {name: data for name, data in body_data.items() if data['geom'] == 'capsule'}
        print(f"Found {len(capsule_bodies)} bodies with capsule geometry:")
        for name, data in capsule_bodies.items():
            info = data['capsule_info']
            print(f"  {name}: height = {info['height']:.6f}, center = ({info['center'][0]:.6f}, {info['center'][1]:.6f}, {info['center'][2]:.6f})")
        
        update_usda_file(args.usda_ref, body_data, hierarchy, output_path)
    else:
        print("Could not extract body data. Exiting.")
    
    print("--- Script finished ---")


if __name__ == '__main__':
    main()