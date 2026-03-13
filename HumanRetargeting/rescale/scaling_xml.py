import os
import re
from typing import List, Tuple, Optional

ORIGINAL_MODEL_HEIGHT_M = 1.7  # kept for external uses

_BODY_REGEX = re.compile(r"<body name=\"([^\"]+)\"")
_BODY_TAG_REGEX = re.compile(r"<body\b([^>]*)>")  # capture attribute section of each opening body tag (fixed word boundary)
_ATTR_REGEX = re.compile(r"(\w+)=\"([^\"]*)\"")  # generic attribute extractor
_GEOM_TAG_REGEX = re.compile(r"<geom\b([^>]*)>", re.IGNORECASE)  # capture attribute section of each opening geom tag
_FROMTO_ATTR_REGEX = re.compile(r'fromto="([^"]+)"')  # raw fromto attribute values only


def find_body_names(xml_text: str):  # core extractor
    return _BODY_REGEX.findall(xml_text)


def get_body_names_from_file(path: str):  # thin file helper
    try:
        with open(path, encoding="utf-8") as f:
            return find_body_names(f.read())
    except OSError:
        return []


def extract_body_names(source: str):  # unified convenience (path or xml string)
    return get_body_names_from_file(source) if os.path.isfile(source) else find_body_names(source)


def find_bodies_with_positions(xml_text: str):
    """Return list of tuples (name, pos_list_or_None) for each <body> tag.

    pos_list_or_None is a list of 3 floats if a pos attribute exists, else None.
    Robust to attribute order; ignores nested structure (only start tags parsed).
    """
    results = []
    for attr_str in _BODY_TAG_REGEX.findall(xml_text):
        attrs = dict(_ATTR_REGEX.findall(attr_str))
        name = attrs.get("name")
        if not name:
            continue
        pos_raw = attrs.get("pos")
        if pos_raw is not None:
            # Split on whitespace, filter empty, convert to float if possible
            parts = [p for p in pos_raw.strip().split() if p]
            try:
                pos_vals = [float(p) for p in parts]
            except ValueError:
                pos_vals = None  # malformed numbers
        else:
            pos_vals = None
        results.append((name, pos_vals))
    return results


def find_geoms_with_fromto(xml_text: str):
    """Return list of tuples (geom_name_or_None, fromto_values_list) for each <geom> with fromto attribute."""
    results = []
    for attr_str in _GEOM_TAG_REGEX.findall(xml_text):
        attrs = dict(_ATTR_REGEX.findall(attr_str))
        fromto_raw = attrs.get("fromto")
        if not fromto_raw:
            continue
        parts = [p for p in fromto_raw.strip().split() if p]
        if len(parts) != 6:
            continue
        try:
            fromto_vals = [float(p) for p in parts]
        except ValueError:
            continue
        results.append((attrs.get("name"), fromto_vals))
    return results


def get_geoms_with_fromto_from_file(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return find_geoms_with_fromto(f.read())
    except OSError:
        return []


def extract_ordered_fromtos(source: str, names_order=("L_Hip", "L_Knee", "R_Hip", "R_Knee")):
    """Return ordered list of (name, fromto_list) using first appearances of fromto attributes.

    We rely on the known fixed order in the file: L_Hip, L_Knee, R_Hip, R_Knee. If more fromto
    attributes exist (feet, toes, etc.), they are appended with auto-index names (e.g. extra_5...).
    """
    if os.path.isfile(source):
        try:
            with open(source, encoding="utf-8") as f:
                xml_text = f.read()
        except OSError:
            return []
    else:
        xml_text = source
    raw_values = []
    for value in _FROMTO_ATTR_REGEX.findall(xml_text):
        parts = [p for p in value.strip().split() if p]
        if len(parts) != 6:
            continue
        try:
            raw_values.append([float(p) for p in parts])
        except ValueError:
            continue
    named = []
    # Assign known names first
    for idx, name in enumerate(names_order):
        if idx < len(raw_values):
            named.append((name, raw_values[idx]))
    # Append any extras beyond the known list
    for extra_idx in range(len(names_order), len(raw_values)):
        named.append((f"extra_{extra_idx+1}", raw_values[extra_idx]))
    return named


def get_geoms_with_fromto(source: str):
    """Unified convenience: pass path or xml, returns (name, fromto_list) tuples."""
    if os.path.isfile(source):
        return get_geoms_with_fromto_from_file(source)
    return find_geoms_with_fromto(source)


# map_ordered_fromtos removed: merged into extract_ordered_fromtos


def get_body_positions_from_file(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return find_bodies_with_positions(f.read())
    except OSError:
        return []


# scale the z values to be the length of the new height* fractions
def scale_z_values(bodies: list, new_height: float):
    # if the names of the body contains hip use HIP_FRAC
    for name, pos in bodies:
        if pos is not None and len(pos) == 3:
            if "Hip" in name:
                pos[2] = round(pos[2] * (new_height / ORIGINAL_MODEL_HEIGHT_M), 4)
            elif "Knee" in name:
                pos[2] = round(pos[2] * (new_height / ORIGINAL_MODEL_HEIGHT_M), 4)
            elif "Ankle" in name:
                pos[2] = round(pos[2] * (new_height / ORIGINAL_MODEL_HEIGHT_M), 4)
    return bodies


def scale_fromto(fromto: list, scaled_bodies: list):
    """Adjust z endpoints of fromto capsules to track remapped body z positions.

    Steps implemented (interpreting the provided instructions):
      1. Build a map of scaled body z positions.
      2. Remap joint target z values:
           L_Hip -> L_Knee,   R_Hip -> R_Knee,
           L_Knee -> L_Ankle, R_Knee -> R_Ankle.
         (If a remap target body isn't found, fall back to the joint's own body z if available.)
            3. For each fromto entry (expects list of tuples (name, [x1,y1,z1,x2,y2,z2])) where name matches Hip/Knee:
                     - total = z1 + z2 (your instructions say: "Add the z1 and z2 values")
                     - diff = target_body_z - total (compare body z to that sum)
                     - delta_each = diff / 2 ("Divide the subtracted value by 2")
                     - new z1 = z1 + delta_each; new z2 = z2 + delta_each (shift both equally)
                 This yields (new_z1 + new_z2) == target_body_z and avoids doubling the target.
      4. Return a new list with updated fromto arrays (others unchanged).
    """
    if not fromto:
        return fromto

    # Build body z lookup
    body_z = {name: pos[2] for name, pos in scaled_bodies if pos and len(pos) == 3}

    def side(name: str):
        return name.split("_")[0] if "_" in name else ""

    def remap_target(joint_name: str):
        s = side(joint_name)
        if "Hip" in joint_name:
            return f"{s}_Knee"
        if "Knee" in joint_name:
            return f"{s}_Ankle"
        return joint_name

    updated = []
    for name, vals in fromto:
        if not (isinstance(vals, (list, tuple)) and len(vals) == 6):
            updated.append((name, vals))
            continue
        # Only adjust hip/knee capsules per description
        if "Hip" in name or "Knee" in name:
            target_body = remap_target(name)
            target_z = body_z.get(target_body)
            if target_z is None:
                # fallback to own body z if present
                target_z = body_z.get(name)
            if target_z is not None:
                z1 = float(vals[2])
                z2 = float(vals[5])
                total = z1 + z2
                # Shift each endpoint by half the difference so new sum matches target_z
                delta = (target_z - total)
                new_vals = list(vals)
                new_vals[5] = round(z2 + delta, 4)
                updated.append((name, new_vals))
                continue
        # default (no change)
        updated.append((name, vals))
    return updated


def replace_body_positions_in_xml(xml_text: str, scaled_bodies: List[Tuple[str, Optional[list]]]) -> str:
    """Replace body position values in XML text with scaled values."""
    if not scaled_bodies:
        return xml_text
    
    # Create a map of body names to their scaled positions
    body_pos_map = {name: pos for name, pos in scaled_bodies if pos and len(pos) == 3}
    
    def replace_body_pos(match):
        attr_str = match.group(1)
        attrs = dict(_ATTR_REGEX.findall(attr_str))
        name = attrs.get("name")
        
        if name in body_pos_map:
            new_pos = body_pos_map[name]
            pos_str = f"{new_pos[0]:.4f} {new_pos[1]:.4f} {new_pos[2]:.4f}"
            
            # Replace the pos attribute in the attribute string
            new_attr_str = re.sub(r'pos="[^"]*"', f'pos="{pos_str}"', attr_str)
            return f"<body{new_attr_str}>"
        
        return match.group(0)  # No change if body not found
    
    return _BODY_TAG_REGEX.sub(replace_body_pos, xml_text)


def replace_fromto_in_xml(xml_text: str, scaled_fromtos: List[Tuple[str, list]]) -> str:
    """Replace fromto values in XML text with scaled values."""
    if not scaled_fromtos:
        return xml_text
    
    # Create a list of scaled fromto values in order (L_Hip, L_Knee, R_Hip, R_Knee)
    fromto_values = []
    for name, vals in scaled_fromtos:
        if isinstance(vals, (list, tuple)) and len(vals) == 6:
            fromto_str = " ".join(f"{v:.4f}" for v in vals)
            fromto_values.append(fromto_str)
    
    fromto_index = 0
    
    def replace_geom_fromto(match):
        nonlocal fromto_index
        attr_str = match.group(1)
        attrs = dict(_ATTR_REGEX.findall(attr_str))
        
        # Check if this geom has a fromto attribute and is a capsule (the ones we want to scale)
        if "fromto" in attrs and attrs.get("type") == "capsule":
            if fromto_index < len(fromto_values):
                new_fromto = fromto_values[fromto_index]
                new_attr_str = re.sub(r'fromto="[^"]*"', f'fromto="{new_fromto}"', attr_str)
                fromto_index += 1
                return f"<geom{new_attr_str}>"
        
        return match.group(0)  # No change if not a capsule fromto
    
    return _GEOM_TAG_REGEX.sub(replace_geom_fromto, xml_text)


def scale_xml_file(input_path: str, output_path: str, new_height: float) -> bool:
    """
    Scale an XML file and save the result to a new file.
    
    Args:
        input_path: Path to the input XML file
        output_path: Path where the scaled XML file will be saved
        new_height: New height in meters to scale to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the original XML file
        with open(input_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()
        
        # Extract and scale body positions
        bodies_pos = find_bodies_with_positions(xml_text)
        scaled_bodies = scale_z_values(bodies_pos, new_height)
        
        # Extract and scale fromto values
        fromtos = extract_ordered_fromtos(xml_text)
        scaled_fromtos = scale_fromto(fromtos, scaled_bodies)
        
        # Replace values in XML text
        scaled_xml = replace_body_positions_in_xml(xml_text, scaled_bodies)
        scaled_xml = replace_fromto_in_xml(scaled_xml, scaled_fromtos)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the scaled XML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(scaled_xml)
        
        print(f"Successfully scaled XML file from {new_height/ORIGINAL_MODEL_HEIGHT_M:.2f}x scale")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error scaling XML file: {e}")
        return False


def copy_and_scale_xml(
    input_path: str,
    new_height: float,
    output_suffix: str = None,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Copy an XML file and scale it to a new height.
    
    Args:
        input_path: Path to the input XML file
        new_height: New height in meters
        output_suffix: Suffix to add to output filename (default: height value)
        output_dir: Directory where the scaled XML should be written (defaults to
            the directory of ``input_path``)
    
    Returns:
        Path to the output file if successful, None otherwise
    """
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        return None
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    extension = os.path.splitext(input_path)[1]
    
    if output_suffix is None:
        # Convert height to avoid decimal points in filename
        height_int = round(new_height * 100)  # Convert to centimeters
        output_suffix = f"_height_{height_int}cm"
    
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{base_name}{output_suffix}{extension}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Scale the XML file
    if scale_xml_file(input_path, output_path, new_height):
        return output_path
    return None


def batch_scale_xml(input_path: str, heights: List[float], output_dir: Optional[str] = None) -> List[str]:
    """
    Create multiple scaled versions of an XML file.
    
    Args:
        input_path: Path to the input XML file
        heights: List of heights in meters to scale to
    
    Returns:
        List of successfully created output file paths
    """
    output_files = []
    
    for height in heights:
        output_path = copy_and_scale_xml(input_path, height, output_dir=output_dir)
        if output_path:
            output_files.append(output_path)
    
    return output_files


if __name__ == "__main__":
    # Example usage of the scaling functions
    file_path = "D:\\Isaac\\IsaacLab2.1.2\\ProtoMotions\\protomotions\\data\\assets\\mjcf\\smpl_humanoid_lower_body_base.xml"
    
    # Test with different heights
    test_heights = [1.5, 1.8, 2.0, 2.2]
    
    print("=== XML Scaling Script ===\n")
    
    # Show original analysis
    print("Original file analysis:")
    names = get_body_names_from_file(file_path)
    print(f"Found {len(names)} body names: {', '.join(names)}")
    
    bodies_pos = get_body_positions_from_file(file_path)
    print("\nOriginal body positions:")
    for name, pos in bodies_pos:
        if pos:
            print(f"  {name:15s} -> {pos}")
    
    fromtos = extract_ordered_fromtos(file_path)
    print("\nOriginal fromto values:")
    for name, vals in fromtos:
        print(f"  {name:8s} -> {vals}")
    
    print(f"\n{'='*50}")
    
    # Create scaled versions
    print("Creating scaled XML files...")
    for height in test_heights:
        print(f"\nScaling to {height}m height:")
        
        # Scale the data
        scaled_bodies = scale_z_values(bodies_pos.copy(), height)
        scaled_fromtos = scale_fromto(fromtos, scaled_bodies)
        
        print("Scaled body positions:")
        for name, pos in scaled_bodies:
            if pos and any("Hip" in name or "Knee" in name or "Ankle" in name for x in [name]):
                print(f"  {name:15s} -> {pos}")
        
        print("Scaled fromto values:")
        for name, vals in scaled_fromtos:
            print(f"  {name:8s} -> {vals}")
        
        # Create the scaled XML file
        output_file = copy_and_scale_xml(file_path, height)
        if output_file:
            print(f"✓ Created: {os.path.basename(output_file)}")
        else:
            print(f"✗ Failed to create scaled file for height {height}m")
    
    print(f"\n{'='*50}")
    print("Batch scaling example:")
    output_files = batch_scale_xml(file_path, [1.6, 1.9, 2.1])
    print(f"Created {len(output_files)} scaled files:")
    for output_file in output_files:
        print(f"  - {os.path.basename(output_file)}")
    
    print("\nDone! Check the output files in the same directory as the input file.")
