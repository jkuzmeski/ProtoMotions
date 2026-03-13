#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Subject-specific lower-body asset generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml

RESCALE_DIR = Path(__file__).resolve().parent.parent / "rescale"
if str(RESCALE_DIR) not in sys.path:
    sys.path.insert(0, str(RESCALE_DIR))

from scaling_usda import extract_body_data_from_xml, update_usda_file

try:
    from .subject_profiles import SubjectProfile
except ImportError:
    from subject_profiles import SubjectProfile


BASE_HEIGHT_CM = 170.0
BASE_ROOT_HEIGHT_M = 0.95
BASE_PELVIS_WIDTH_M = 0.19
BASE_THIGH_LENGTH_M = 0.3789
BASE_SHANK_LENGTH_M = 0.3982
BASE_FOOT_LENGTH_M = 0.1233
BASE_FOOT_WIDTH_M = 0.08

# Lower-body segment mass fractions from de Leva's adjusted Zatsiorsky-Seluyanov
# anthropometric parameters as reported in Karatsidis et al. 2017, Table 1.
#
# The lower-body-only model represents pelvis + both thighs + both shanks + both feet.
# Foot mass is split across the ankle box and toe box using the base model's geometry
# mass ratio so subject mass can be distributed onto the existing segment topology.
PELVIS_BODY_MASS_FRACTION = 0.1117
THIGH_BODY_MASS_FRACTION = 0.1416
SHANK_BODY_MASS_FRACTION = 0.0433
FOOT_BODY_MASS_FRACTION = 0.0137
BASE_ANKLE_GEOM_MASS_KG = 1.323104
BASE_TOE_GEOM_MASS_KG = 0.31744
TOE_FOOT_MASS_SHARE = BASE_TOE_GEOM_MASS_KG / (
    BASE_ANKLE_GEOM_MASS_KG + BASE_TOE_GEOM_MASS_KG
)
ANKLE_FOOT_MASS_SHARE = 1.0 - TOE_FOOT_MASS_SHARE
MODELED_BODY_MASS_FRACTION = (
    PELVIS_BODY_MASS_FRACTION
    + 2.0 * THIGH_BODY_MASS_FRACTION
    + 2.0 * SHANK_BODY_MASS_FRACTION
    + 2.0 * FOOT_BODY_MASS_FRACTION
)


def _parse_float_list(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(value) for value in raw.split()]


def _format_float_list(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"expected body {name!r} in lower-body XML")
    return body


def _find_geom(body: ET.Element) -> ET.Element:
    geom = body.find("geom")
    if geom is None:
        raise ValueError(f"expected geom under body {body.get('name')!r}")
    return geom


def _geom_volume(geom: ET.Element) -> float:
    geom_type = geom.get("type")
    size = _parse_float_list(geom.get("size"))
    if geom_type == "box" and len(size) == 3:
        return 8.0 * size[0] * size[1] * size[2]
    if geom_type == "capsule" and len(size) >= 1:
        fromto = _parse_float_list(geom.get("fromto"))
        if len(fromto) != 6:
            raise ValueError("capsule geom is missing a valid fromto attribute")
        radius = size[0]
        total_length = math.dist(fromto[:3], fromto[3:])
        cylinder_length = max(total_length - 2.0 * radius, 0.0)
        return (
            math.pi * radius * radius * cylinder_length
            + (4.0 / 3.0) * math.pi * radius**3
        )
    raise ValueError(
        f"Unsupported geom for subject mass scaling: {geom_type!r} on {geom.get('name')!r}"
    )


def _segment_mass_targets(total_body_mass_kg: float) -> dict[str, float]:
    foot_mass = FOOT_BODY_MASS_FRACTION * total_body_mass_kg
    ankle_mass = foot_mass * ANKLE_FOOT_MASS_SHARE
    toe_mass = foot_mass * TOE_FOOT_MASS_SHARE
    return {
        "Pelvis": PELVIS_BODY_MASS_FRACTION * total_body_mass_kg,
        "L_Hip": THIGH_BODY_MASS_FRACTION * total_body_mass_kg,
        "R_Hip": THIGH_BODY_MASS_FRACTION * total_body_mass_kg,
        "L_Knee": SHANK_BODY_MASS_FRACTION * total_body_mass_kg,
        "R_Knee": SHANK_BODY_MASS_FRACTION * total_body_mass_kg,
        "L_Ankle": ankle_mass,
        "R_Ankle": ankle_mass,
        "L_Toe": toe_mass,
        "R_Toe": toe_mass,
    }


def _scale_attr(
    element: ET.Element,
    attr_name: str,
    *,
    axes: tuple[int, ...],
    factor: float,
) -> None:
    values = _parse_float_list(element.get(attr_name))
    if not values:
        return
    for axis in axes:
        if axis < len(values):
            values[axis] *= factor
    element.set(attr_name, _format_float_list(values))


def _compute_lowest_relative_z(root_body: ET.Element) -> float:
    """Compute the minimum geom z offset relative to the freejoint root."""

    lowest = float("inf")

    def visit(body: ET.Element, parent_offset: list[float]) -> None:
        nonlocal lowest
        body_pos = _parse_float_list(body.get("pos"))
        offset = parent_offset[:]
        if len(body_pos) == 3:
            offset = [parent_offset[i] + body_pos[i] for i in range(3)]

        for geom in body.findall("geom"):
            size = _parse_float_list(geom.get("size"))
            if geom.get("type") == "box" and len(size) == 3:
                geom_pos = _parse_float_list(geom.get("pos"))
                geom_z = offset[2] + (geom_pos[2] if len(geom_pos) == 3 else 0.0)
                lowest = min(lowest, geom_z - size[2])
            elif geom.get("type") == "capsule" and len(size) >= 1:
                fromto = _parse_float_list(geom.get("fromto"))
                if len(fromto) == 6:
                    lowest = min(lowest, offset[2] + min(fromto[2], fromto[5]) - size[0])

        for child in body.findall("body"):
            visit(child, offset)

    visit(root_body, [0.0, 0.0, 0.0])
    return lowest


def _generate_lower_body_urdf(xml_root: ET.Element, subject_stem: str, output_path: Path) -> None:
    """Generate a kinematic URDF with intermediate links for 3-DOF joints."""

    root_body = xml_root.find("./worldbody/body[@name='Pelvis']")
    if root_body is None:
        raise ValueError("expected Pelvis root body in lower-body XML")

    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{subject_stem}">',
        '  <link name="Pelvis"/>',
    ]

    def add_chain(
        parent_link: str,
        body_name: str,
        joint_prefix: str,
        origin_xyz: list[float],
    ) -> str:
        x_link = f"{joint_prefix}_x_link"
        y_link = f"{joint_prefix}_y_link"
        z_link = body_name
        axes = (
            ("x", parent_link, x_link, "1 0 0", origin_xyz),
            ("y", x_link, y_link, "0 1 0", [0.0, 0.0, 0.0]),
            ("z", y_link, z_link, "0 0 1", [0.0, 0.0, 0.0]),
        )
        for axis_name, parent, child, axis_vec, xyz in axes:
            lines.append(f'  <link name="{child}"/>')
            xyz_str = " ".join(f"{value:.6f}" for value in xyz)
            lines.append(
                f'  <joint name="{joint_prefix}_{axis_name}" type="revolute">'
            )
            lines.append(f'    <parent link="{parent}"/>')
            lines.append(f'    <child link="{child}"/>')
            lines.append(f'    <origin xyz="{xyz_str}" rpy="0 0 0"/>')
            lines.append(f'    <axis xyz="{axis_vec}"/>')
            lines.append('    <limit lower="-6.283185" upper="6.283185" effort="500" velocity="100"/>')
            lines.append("  </joint>")
        return z_link

    left_hip = _find_body(xml_root, "L_Hip")
    left_knee = _find_body(xml_root, "L_Knee")
    left_ankle = _find_body(xml_root, "L_Ankle")
    left_toe = _find_body(xml_root, "L_Toe")
    right_hip = _find_body(xml_root, "R_Hip")
    right_knee = _find_body(xml_root, "R_Knee")
    right_ankle = _find_body(xml_root, "R_Ankle")
    right_toe = _find_body(xml_root, "R_Toe")

    left_chain = add_chain("Pelvis", "L_Hip", "L_Hip", _parse_float_list(left_hip.get("pos")))
    left_chain = add_chain(left_chain, "L_Knee", "L_Knee", _parse_float_list(left_knee.get("pos")))
    left_chain = add_chain(left_chain, "L_Ankle", "L_Ankle", _parse_float_list(left_ankle.get("pos")))
    add_chain(left_chain, "L_Toe", "L_Toe", _parse_float_list(left_toe.get("pos")))

    right_chain = add_chain("Pelvis", "R_Hip", "R_Hip", _parse_float_list(right_hip.get("pos")))
    right_chain = add_chain(right_chain, "R_Knee", "R_Knee", _parse_float_list(right_knee.get("pos")))
    right_chain = add_chain(right_chain, "R_Ankle", "R_Ankle", _parse_float_list(right_ankle.get("pos")))
    add_chain(right_chain, "R_Toe", "R_Toe", _parse_float_list(right_toe.get("pos")))

    lines.append("</robot>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(slots=True)
class SubjectAssets:
    """Resolved asset paths for a generated subject."""

    subject_id: str
    subject_stem: str
    mjcf_path: Path
    usda_path: Path
    urdf_path: Path
    metadata_path: Path
    default_root_height: float
    asset_root: Path


class SubjectAssetBuilder:
    """Generate deterministic lower-body subject assets."""

    def __init__(
        self,
        *,
        profile: SubjectProfile,
        rescale_dir: Path,
        assets_root: Path,
    ) -> None:
        self.profile = profile
        self.rescale_dir = rescale_dir.resolve()
        self.assets_root = assets_root.resolve()
        self.subject_stem = f"smpl_humanoid_lower_body_subject_{profile.subject_id}"

    @property
    def base_xml_path(self) -> Path:
        return self.rescale_dir / f"smpl_humanoid_lower_body_{self.profile.model_variant}.xml"

    @property
    def base_usda_path(self) -> Path:
        return self.rescale_dir / f"smpl_humanoid_lower_body_{self.profile.model_variant}.usda"

    @property
    def mjcf_path(self) -> Path:
        return self.assets_root / "mjcf" / f"{self.subject_stem}.xml"

    @property
    def usda_path(self) -> Path:
        return self.assets_root / "usd" / f"{self.subject_stem}.usda"

    @property
    def urdf_path(self) -> Path:
        return self.assets_root / "urdf" / "for_retargeting" / f"{self.subject_stem}.urdf"

    @property
    def metadata_path(self) -> Path:
        return self.assets_root / "subjects" / f"{self.subject_stem}.yaml"

    def build(self, force: bool = False) -> SubjectAssets:
        """Create or reuse subject-specific assets."""
        requested_profile = self.profile.as_metadata()
        if self.profile.contact_pads:
            raise ValueError(
                "Subject-specific contact-pad lower-body assets are not implemented yet. "
                "Use the standard subject profile flow without contact pads."
            )
        if (
            not force
            and self.mjcf_path.exists()
            and self.usda_path.exists()
            and self.urdf_path.exists()
            and self.metadata_path.exists()
        ):
            metadata = yaml.safe_load(self.metadata_path.read_text(encoding="utf-8")) or {}
            has_requested_mass_model = (
                self.profile.mass_kg is None
                or (
                    isinstance(metadata.get("mass_properties"), dict)
                    and metadata["mass_properties"].get("body_mass_kg")
                    == round(self.profile.mass_kg, 6)
                )
            )
            if has_requested_mass_model and all(
                metadata.get(key) == value for key, value in requested_profile.items()
            ):
                return SubjectAssets(
                    subject_id=self.profile.subject_id,
                    subject_stem=self.subject_stem,
                    mjcf_path=self.mjcf_path,
                    usda_path=self.usda_path,
                    urdf_path=self.urdf_path,
                    metadata_path=self.metadata_path,
                    default_root_height=float(metadata["default_root_height"]),
                    asset_root=self.assets_root,
                )

        self.mjcf_path.parent.mkdir(parents=True, exist_ok=True)
        self.usda_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        xml_tree = ET.parse(self.base_xml_path)
        xml_root = xml_tree.getroot()

        base_tree = ET.parse(self.base_xml_path)
        base_root = base_tree.getroot()
        base_pelvis = _find_body(base_root, "Pelvis")
        base_lowest = _compute_lowest_relative_z(base_pelvis)
        clearance = BASE_ROOT_HEIGHT_M + base_lowest

        self._apply_subject_scaling(xml_root)
        mass_properties = self._apply_subject_mass_distribution(xml_root)
        pelvis = _find_body(xml_root, "Pelvis")
        lowest = _compute_lowest_relative_z(pelvis)
        default_root_height = -lowest + clearance

        xml_tree.write(self.mjcf_path, encoding="utf-8", xml_declaration=False)

        body_data, hierarchy = extract_body_data_from_xml(str(self.mjcf_path))
        if body_data is None or hierarchy is None:
            raise ValueError("failed to derive USDA data from generated XML")
        update_usda_file(str(self.base_usda_path), body_data, hierarchy, str(self.usda_path))
        _generate_lower_body_urdf(xml_root, self.subject_stem, self.urdf_path)

        metadata = {
            **self.profile.as_metadata(),
            "asset_stem": self.subject_stem,
            "asset_root": str(self.assets_root),
            "mjcf_path": str(self.mjcf_path),
            "usda_path": str(self.usda_path),
            "urdf_path": str(self.urdf_path),
            "default_root_height": round(default_root_height, 6),
            "contact_bodies": ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"],
            "non_termination_contact_bodies": ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"],
            "mass_properties": mass_properties,
        }
        self.metadata_path.write_text(
            yaml.safe_dump(metadata, sort_keys=False),
            encoding="utf-8",
        )

        return SubjectAssets(
            subject_id=self.profile.subject_id,
            subject_stem=self.subject_stem,
            mjcf_path=self.mjcf_path,
            usda_path=self.usda_path,
            urdf_path=self.urdf_path,
            metadata_path=self.metadata_path,
            default_root_height=default_root_height,
            asset_root=self.assets_root,
        )

    def _apply_subject_scaling(self, xml_root: ET.Element) -> None:
        global_ratio = self.profile.height_cm / BASE_HEIGHT_CM
        pelvis_ratio = self.profile.pelvis_width_m / BASE_PELVIS_WIDTH_M
        left_thigh_ratio = self.profile.thigh_lengths_m[0] / BASE_THIGH_LENGTH_M
        right_thigh_ratio = self.profile.thigh_lengths_m[1] / BASE_THIGH_LENGTH_M
        left_shank_ratio = self.profile.shank_lengths_m[0] / BASE_SHANK_LENGTH_M
        right_shank_ratio = self.profile.shank_lengths_m[1] / BASE_SHANK_LENGTH_M
        left_foot_ratio = self.profile.foot_lengths_m[0] / BASE_FOOT_LENGTH_M
        right_foot_ratio = self.profile.foot_lengths_m[1] / BASE_FOOT_LENGTH_M
        left_foot_width_ratio = (
            (self.profile.foot_widths_m[0] / BASE_FOOT_WIDTH_M)
            if self.profile.foot_widths_m[0] is not None
            else global_ratio
        )
        right_foot_width_ratio = (
            (self.profile.foot_widths_m[1] / BASE_FOOT_WIDTH_M)
            if self.profile.foot_widths_m[1] is not None
            else global_ratio
        )

        for element in xml_root.iter():
            if element.get("pos"):
                _scale_attr(element, "pos", axes=(0, 1, 2), factor=global_ratio)
            if element.get("size"):
                values = _parse_float_list(element.get("size"))
                values = [value * global_ratio for value in values]
                element.set("size", _format_float_list(values))
            if element.get("fromto"):
                values = _parse_float_list(element.get("fromto"))
                values = [value * global_ratio for value in values]
                element.set("fromto", _format_float_list(values))

        pelvis = _find_body(xml_root, "Pelvis")
        pelvis_geom = _find_geom(pelvis)
        _scale_attr(pelvis_geom, "size", axes=(1,), factor=pelvis_ratio / global_ratio)

        for body_name, sign in (("L_Hip", 1.0), ("R_Hip", -1.0)):
            hip = _find_body(xml_root, body_name)
            pos = _parse_float_list(hip.get("pos"))
            if len(pos) == 3:
                pos[1] = sign * (self.profile.pelvis_width_m / 2.0)
                hip.set("pos", _format_float_list(pos))

        for side, thigh_ratio, shank_ratio in (
            ("L", left_thigh_ratio, left_shank_ratio),
            ("R", right_thigh_ratio, right_shank_ratio),
        ):
            knee = _find_body(xml_root, f"{side}_Knee")
            ankle = _find_body(xml_root, f"{side}_Ankle")
            thigh_geom = _find_geom(_find_body(xml_root, f"{side}_Hip"))
            shank_geom = _find_geom(knee)

            _scale_attr(knee, "pos", axes=(2,), factor=thigh_ratio / global_ratio)
            _scale_attr(thigh_geom, "fromto", axes=(2, 5), factor=thigh_ratio / global_ratio)

            _scale_attr(ankle, "pos", axes=(2,), factor=shank_ratio / global_ratio)
            _scale_attr(shank_geom, "fromto", axes=(2, 5), factor=shank_ratio / global_ratio)

        for side, foot_ratio, foot_width_ratio in (
            ("L", left_foot_ratio, left_foot_width_ratio),
            ("R", right_foot_ratio, right_foot_width_ratio),
        ):
            ankle = _find_body(xml_root, f"{side}_Ankle")
            toe = _find_body(xml_root, f"{side}_Toe")
            ankle_geom = _find_geom(ankle)
            toe_geom = _find_geom(toe)
            _scale_attr(ankle_geom, "pos", axes=(0,), factor=foot_ratio / global_ratio)
            _scale_attr(ankle_geom, "size", axes=(0,), factor=foot_ratio / global_ratio)
            _scale_attr(ankle_geom, "size", axes=(1,), factor=foot_width_ratio / global_ratio)
            _scale_attr(toe, "pos", axes=(0,), factor=foot_ratio / global_ratio)
            _scale_attr(toe_geom, "pos", axes=(0,), factor=foot_ratio / global_ratio)
            _scale_attr(toe_geom, "size", axes=(0,), factor=foot_ratio / global_ratio)
            _scale_attr(toe_geom, "size", axes=(1,), factor=foot_width_ratio / global_ratio)

    def _apply_subject_mass_distribution(self, xml_root: ET.Element) -> dict[str, object]:
        if self.profile.mass_kg is None:
            return {
                "body_mass_kg": None,
                "modeled_mass_kg": None,
                "modeled_body_mass_fraction": MODELED_BODY_MASS_FRACTION,
                "source": "base_geom_densities",
                "segment_masses_kg": {},
                "segment_densities_kg_m3": {},
                "note": (
                    "No subject mass_kg provided. MJCF mass and inertia remain derived "
                    "from the base model geom densities."
                ),
            }

        segment_masses = _segment_mass_targets(self.profile.mass_kg)
        segment_densities: dict[str, float] = {}
        for body_name, target_mass in segment_masses.items():
            body = _find_body(xml_root, body_name)
            geom = _find_geom(body)
            volume = _geom_volume(geom)
            if volume <= 0.0:
                raise ValueError(f"non-positive geom volume for body {body_name}")
            density = target_mass / volume
            geom.set("density", f"{density:.6f}")
            segment_densities[body_name] = density

        modeled_mass = sum(segment_masses.values())
        return {
            "body_mass_kg": round(self.profile.mass_kg, 6),
            "modeled_mass_kg": round(modeled_mass, 6),
            "modeled_body_mass_fraction": MODELED_BODY_MASS_FRACTION,
            "source": "de_leva_1996_adjusted_zatsiorsky_lower_body_mass_fractions",
            "inertia_scaling_rule": (
                "Segment masses are assigned from body-mass fractions and MuJoCo "
                "derives inertias from the scaled segment geometry plus per-geom density."
            ),
            "segment_masses_kg": {
                name: round(value, 6) for name, value in segment_masses.items()
            },
            "segment_densities_kg_m3": {
                name: round(value, 6) for name, value in segment_densities.items()
            },
            "foot_split": {
                "ankle_share_of_foot_segment": round(ANKLE_FOOT_MASS_SHARE, 6),
                "toe_share_of_foot_segment": round(TOE_FOOT_MASS_SHARE, 6),
            },
        }
