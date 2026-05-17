"""MuJoCo adapter utilities for the Spot locomanipulation mjviser example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .spot_locomanip_policy import (
    DEFAULT_ARM_COMMAND,
    MJCF_ARM_JOINT_NAMES,
    MJCF_GRIPPER_JOINT_NAME,
    DEFAULT_STAND_BASE_HEIGHT,
    DEFAULT_STAND_LEG_JOINTS,
    SPOT_LEG_JOINT_NAMES,
    policy_action_to_mujoco_ctrl,
)

LEG_GAIN_SCALE_MIN = 1.0
LEG_GAIN_SCALE_MAX = 2.0
DEFAULT_LEG_GAIN_SCALE = 1.25
REFERENCE_LEG_ACTUATOR_GAINS: dict[str, tuple[float, float]] = {
    name: (60.0, 1.5) for name in SPOT_LEG_JOINT_NAMES
}
DEFAULT_ACTUATOR_GAINS: dict[str, tuple[float, float]] = {
    **{
        name: (kp * DEFAULT_LEG_GAIN_SCALE, kd * DEFAULT_LEG_GAIN_SCALE)
        for name, (kp, kd) in REFERENCE_LEG_ACTUATOR_GAINS.items()
    },
    "arm_sh0": (320.0, 8.0),
    "arm_sh1": (150.0, 15.3),
    "arm_el0": (120.0, 5.2),
    "arm_el1": (102.0, 2.04),
    "arm_wr0": (102.0, 2.04),
    "arm_wr1": (102.0, 2.04),
    MJCF_GRIPPER_JOINT_NAME: (16.0, 1.0),
}

MANIPULATION_OBJECT_HIDDEN_Z = 10.0
_CHAIR_LEG_HEIGHT = 0.35 * 2.0 / 3.0
_CHAIR_SEAT_HEIGHT = _CHAIR_LEG_HEIGHT + 0.03 / 2.0
MANIPULATION_OBJECT_BODY_POSES: dict[str, tuple[float, float, float]] = {
    # Furniture and objects arranged as free bodies so contacts are governed by
    # MuJoCo physics instead of static world geoms.  The centers are separated
    # by at least their box extents so enabling the scene does not start from a
    # deep-penetration state.
    "manip_cube": (1.0, -0.6, 0.25),
    "manip_table": (1.5, 0.6, (0.04 + 2.0 * 0.35) / 2.0),
    "manip_small_cube_0": (1.32, 0.45, 0.49),
    "manip_small_cube_1": (1.56, 0.45, 0.49),
    "manip_small_cube_2": (1.32, 0.70, 0.49),
    "manip_small_cube_3": (1.56, 0.70, 0.49),
    "manip_small_cube_4": (1.44, 0.45, 0.69),
    "manip_chair": (2.05, -0.35, _CHAIR_SEAT_HEIGHT),
}
MANIPULATION_OBJECT_GEOM_NAMES: tuple[str, ...] = (
    "manip_cube_geom",
    "manip_small_cube_0_geom",
    "manip_small_cube_1_geom",
    "manip_small_cube_2_geom",
    "manip_small_cube_3_geom",
    "manip_small_cube_4_geom",
    "manip_table_top",
    "manip_table_leg_0",
    "manip_table_leg_1",
    "manip_table_leg_2",
    "manip_table_leg_3",
    "manip_chair_seat",
    "manip_chair_back",
    "manip_chair_leg_0",
    "manip_chair_leg_1",
    "manip_chair_leg_2",
    "manip_chair_leg_3",
)
MANIPULATION_OBJECT_VISUAL_SPECS: dict[
    str, dict[str, str | tuple[float, ...] | tuple[int, int, int]]
] = {
    "manip_cube": {
        "geom_name": "manip_cube_geom",
        "dimensions": (0.5, 0.5, 0.5),
        "color": (204, 51, 51),
    },
    **{
        f"manip_small_cube_{idx}": {
            "geom_name": f"manip_small_cube_{idx}_geom",
            "dimensions": (0.2, 0.2, 0.2),
            "color": (51, 102, 204),
        }
        for idx in range(5)
    },
    "manip_table_top": {
        "geom_name": "manip_table_top",
        "dimensions": (0.8, 0.6, 0.04),
        "color": (140, 122, 97),
    },
    **{
        f"manip_table_leg_{idx}": {
            "geom_name": f"manip_table_leg_{idx}",
            "dimensions": (0.04, 0.04, 0.35),
            "color": (140, 122, 97),
        }
        for idx in range(4)
    },
    "manip_chair_seat": {
        "geom_name": "manip_chair_seat",
        "dimensions": (0.35, 0.35, 0.03),
        "color": (102, 51, 26),
    },
    "manip_chair_back": {
        "geom_name": "manip_chair_back",
        "dimensions": (0.35, 0.03, 0.35),
        "color": (102, 51, 26),
    },
    **{
        f"manip_chair_leg_{idx}": {
            "geom_name": f"manip_chair_leg_{idx}",
            "dimensions": (0.04, 0.04, _CHAIR_LEG_HEIGHT),
            "color": (102, 51, 26),
        }
        for idx in range(4)
    },
}
_MANIPULATION_OBJECT_RGBA: dict[str, tuple[float, float, float, float]] = {
    "manip_cube_geom": (0.8, 0.2, 0.2, 0.0),
    **{f"manip_small_cube_{idx}_geom": (0.2, 0.4, 0.8, 0.0) for idx in range(5)},
    "manip_table_top": (0.55, 0.48, 0.38, 0.0),
    "manip_table_leg_0": (0.55, 0.48, 0.38, 0.0),
    "manip_table_leg_1": (0.55, 0.48, 0.38, 0.0),
    "manip_table_leg_2": (0.55, 0.48, 0.38, 0.0),
    "manip_table_leg_3": (0.55, 0.48, 0.38, 0.0),
    "manip_chair_seat": (0.4, 0.2, 0.1, 0.0),
    "manip_chair_back": (0.4, 0.2, 0.1, 0.0),
    "manip_chair_leg_0": (0.4, 0.2, 0.1, 0.0),
    "manip_chair_leg_1": (0.4, 0.2, 0.1, 0.0),
    "manip_chair_leg_2": (0.4, 0.2, 0.1, 0.0),
    "manip_chair_leg_3": (0.4, 0.2, 0.1, 0.0),
}


def resolve_spot_scene_arm_xml() -> Path:
    """Return the public MuJoCo Menagerie Spot-with-arm scene from robot_descriptions."""

    try:
        from robot_descriptions import spot_mj_description
    except ImportError as exc:  # pragma: no cover - exercised by dependency-missing users.
        raise RuntimeError(
            "robot_descriptions with spot_mj_description is required for this example"
        ) from exc
    package_path = Path(spot_mj_description.PACKAGE_PATH)
    scene_path = package_path / "scene_arm.xml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"Spot arm MJCF scene not found at {scene_path}")
    return scene_path


def _actuator_id(mujoco, model, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise KeyError(f"MuJoCo model does not contain actuator {name!r}")
    return int(actuator_id)


def apply_default_actuator_gains(model, *, leg_gain_scale: float = DEFAULT_LEG_GAIN_SCALE) -> None:
    """Apply the example's joint PD gains on Menagerie actuators."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised by dependency-missing users.
        raise RuntimeError("mujoco is required for this example") from exc

    clamped_leg_gain_scale = float(
        np.clip(leg_gain_scale, LEG_GAIN_SCALE_MIN, LEG_GAIN_SCALE_MAX)
    )
    for name, (kp, kd) in DEFAULT_ACTUATOR_GAINS.items():
        if name in REFERENCE_LEG_ACTUATOR_GAINS:
            reference_kp, reference_kd = REFERENCE_LEG_ACTUATOR_GAINS[name]
            kp = reference_kp * clamped_leg_gain_scale
            kd = reference_kd * clamped_leg_gain_scale
        actuator_id = _actuator_id(mujoco, model, name)
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd


def _add_manipulation_objects_to_scene_xml(scene_path: Path) -> Path:
    """Create a temporary scene XML with optional dynamic manipulation objects."""

    tree = ET.parse(scene_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo scene {scene_path} does not contain a worldbody")

    def add_free_box_body(
        body_name: str,
        *,
        mass: float,
        inertia: tuple[float, float, float],
        geoms: tuple[dict[str, str], ...],
    ) -> None:
        if root.find(f".//body[@name='{body_name}']") is not None:
            return
        body = ET.SubElement(
            worldbody,
            "body",
            name=body_name,
            pos=f"0 0 {MANIPULATION_OBJECT_HIDDEN_Z:g}",
            gravcomp="1",
        )
        ET.SubElement(body, "freejoint", name=f"{body_name}_freejoint")
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=f"{mass:g}",
            diaginertia=f"{inertia[0]:g} {inertia[1]:g} {inertia[2]:g}",
        )
        for geom in geoms:
            geom_name = geom["name"]
            rgba = _MANIPULATION_OBJECT_RGBA[geom_name]
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="box",
                size=geom["size"],
                pos=geom.get("pos", "0 0 0"),
                rgba=f"{rgba[0]:g} {rgba[1]:g} {rgba[2]:g} {rgba[3]:g}",
                contype="1",
                conaffinity="1",
                friction="0.8 0.02 0.01",
            )

    add_free_box_body(
        "manip_cube",
        mass=5.0,
        inertia=(0.21, 0.21, 0.21),
        geoms=({"name": "manip_cube_geom", "size": "0.25 0.25 0.25"},),
    )
    for idx in range(5):
        add_free_box_body(
            f"manip_small_cube_{idx}",
            mass=1.0,
            inertia=(0.007, 0.007, 0.007),
            geoms=(
                {"name": f"manip_small_cube_{idx}_geom", "size": "0.1 0.1 0.1"},
            ),
        )

    table_width = 0.8
    table_depth = 0.6
    table_thickness = 0.04
    table_leg_width = 0.04
    table_leg_height = 0.35
    table_leg_z = -(table_thickness + table_leg_height) / 2.0
    table_inset = 0.05
    table_leg_positions = (
        (-table_width / 2 + table_inset, -table_depth / 2 + table_inset, table_leg_z),
        (table_width / 2 - table_inset, -table_depth / 2 + table_inset, table_leg_z),
        (-table_width / 2 + table_inset, table_depth / 2 - table_inset, table_leg_z),
        (table_width / 2 - table_inset, table_depth / 2 - table_inset, table_leg_z),
    )
    add_free_box_body(
        "manip_table",
        mass=12.0,
        inertia=(0.45, 0.65, 0.85),
        geoms=(
            {"name": "manip_table_top", "size": "0.4 0.3 0.02"},
            *(
                {
                    "name": f"manip_table_leg_{idx}",
                    "size": f"{table_leg_width / 2:g} {table_leg_width / 2:g} {table_leg_height / 2:g}",
                    "pos": f"{pos[0]:g} {pos[1]:g} {pos[2]:g}",
                }
                for idx, pos in enumerate(table_leg_positions)
            ),
        ),
    )

    chair_width = 0.35
    chair_depth = 0.35
    chair_thickness = 0.03
    chair_back_height = 0.35
    chair_back_thickness = 0.03
    chair_leg_width = 0.04
    chair_leg_height = _CHAIR_LEG_HEIGHT
    chair_leg_z = -(chair_thickness + chair_leg_height) / 2.0
    chair_inset = 0.05
    chair_leg_positions = (
        (-chair_width / 2 + chair_inset, -chair_depth / 2 + chair_inset, chair_leg_z),
        (chair_width / 2 - chair_inset, -chair_depth / 2 + chair_inset, chair_leg_z),
        (-chair_width / 2 + chair_inset, chair_depth / 2 - chair_inset, chair_leg_z),
        (chair_width / 2 - chair_inset, chair_depth / 2 - chair_inset, chair_leg_z),
    )
    add_free_box_body(
        "manip_chair",
        mass=8.0,
        inertia=(0.16, 0.22, 0.24),
        geoms=(
            {"name": "manip_chair_seat", "size": "0.175 0.175 0.015"},
            {
                "name": "manip_chair_back",
                "size": "0.175 0.015 0.175",
                "pos": f"0 {chair_depth / 2 - chair_back_thickness / 2:g} {chair_back_height / 2:g}",
            },
            *(
                {
                    "name": f"manip_chair_leg_{idx}",
                    "size": f"{chair_leg_width / 2:g} {chair_leg_width / 2:g} {chair_leg_height / 2:g}",
                    "pos": f"{pos[0]:g} {pos[1]:g} {pos[2]:g}",
                }
                for idx, pos in enumerate(chair_leg_positions)
            ),
        ),
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    pair_targets = ("floor", "FL", "FR", "HL", "HR")
    for geom_name in MANIPULATION_OBJECT_GEOM_NAMES:
        for target_name in pair_targets:
            if (
                root.find(
                    f".//pair[@geom1='{target_name}'][@geom2='{geom_name}']"
                )
                is None
                and root.find(
                    f".//pair[@geom1='{geom_name}'][@geom2='{target_name}']"
                )
                is None
            ):
                ET.SubElement(contact, "pair", geom1=target_name, geom2=geom_name)

    ET.indent(tree)
    temp = tempfile.NamedTemporaryFile(
        "w",
        suffix="_embodik_objects.xml",
        prefix="scene_arm_",
        dir=scene_path.parent,
        delete=False,
    )
    temp_path = Path(temp.name)
    try:
        tree.write(temp, encoding="unicode", xml_declaration=True)
    finally:
        temp.close()
    return temp_path


def load_spot_mujoco_model(
    scene_path: Path | None = None,
    *,
    apply_default_gains: bool = True,
    include_manipulation_objects: bool = True,
):
    """Load the public Menagerie Spot arm MuJoCo model."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised by dependency-missing users.
        raise RuntimeError("mujoco is required for this example") from exc
    resolved_scene_path = scene_path or resolve_spot_scene_arm_xml()
    temp_scene_path: Path | None = None
    if include_manipulation_objects:
        temp_scene_path = _add_manipulation_objects_to_scene_xml(resolved_scene_path)
        resolved_scene_path = temp_scene_path
    try:
        model = mujoco.MjModel.from_xml_path(str(resolved_scene_path))
    finally:
        if temp_scene_path is not None:
            temp_scene_path.unlink(missing_ok=True)
    if apply_default_gains:
        apply_default_actuator_gains(model)
    return model


@dataclass(frozen=True)
class JointScalarAddress:
    qpos: int
    qvel: int


@dataclass(frozen=True)
class FreeJointAddress:
    qpos: int
    qvel: int


class SpotMujocoAdapter:
    """Name-based state/control mapper for Menagerie Spot arm MJCF."""

    body_name = "body"

    def __init__(self, model):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by dependency-missing users.
            raise RuntimeError("mujoco is required for this adapter") from exc
        self._mujoco = mujoco
        self.model = model
        self.control_names = (
            *SPOT_LEG_JOINT_NAMES,
            *MJCF_ARM_JOINT_NAMES,
            MJCF_GRIPPER_JOINT_NAME,
        )
        self._actuator_ids = {
            name: self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.control_names
        }
        self._joint_addresses = {
            name: self._joint_scalar_address(name) for name in self.control_names
        }
        self._body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        self._manipulation_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in MANIPULATION_OBJECT_BODY_POSES
        }
        self._manipulation_geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in MANIPULATION_OBJECT_GEOM_NAMES
        }
        self._manipulation_freejoint_addresses = {
            name: self._freejoint_address(f"{name}_freejoint")
            for name in MANIPULATION_OBJECT_BODY_POSES
        }
        self._manipulation_objects_enabled = False
        # Keep the geoms collision-enabled in the compiled XML so MuJoCo builds
        # broad-phase metadata for object-object and full robot-object contacts.
        # The default hidden state is disabled here; set_manipulation_objects_enabled()
        # toggles runtime bits after the model is compiled.
        for geom_id in self._manipulation_geom_ids.values():
            if geom_id >= 0:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0

    def _name_to_id(self, obj_type, name: str) -> int:
        idx = self._mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise KeyError(f"MuJoCo model does not contain {obj_type.name} named {name!r}")
        return int(idx)

    def _joint_scalar_address(self, name: str) -> JointScalarAddress:
        joint_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_JOINT, name)
        return JointScalarAddress(
            qpos=int(self.model.jnt_qposadr[joint_id]),
            qvel=int(self.model.jnt_dofadr[joint_id]),
        )

    def _freejoint_address(self, name: str) -> FreeJointAddress:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return FreeJointAddress(qpos=-1, qvel=-1)
        return FreeJointAddress(
            qpos=int(self.model.jnt_qposadr[joint_id]),
            qvel=int(self.model.jnt_dofadr[joint_id]),
        )

    def _initialize_home_data(self, data) -> None:
        """Write the deterministic locomanipulation home state into ``data``."""

        self._mujoco.mj_resetData(self.model, data)
        data.time = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        if hasattr(data, "xfrc_applied"):
            data.xfrc_applied[:] = 0.0
        if hasattr(data, "qacc_warmstart"):
            data.qacc_warmstart[:] = 0.0
        if getattr(data, "act", None) is not None and data.act.size:
            data.act[:] = 0.0
        data.qpos[:7] = [0.0, 0.0, DEFAULT_STAND_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0]
        for name, value in zip(SPOT_LEG_JOINT_NAMES, DEFAULT_STAND_LEG_JOINTS):
            data.qpos[self._joint_addresses[name].qpos] = float(value)
        for name, value in zip(
            (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME),
            DEFAULT_ARM_COMMAND,
        ):
            data.qpos[self._joint_addresses[name].qpos] = float(value)
        self.clear_applied_forces(data)
        self.apply_ctrl_targets(data, policy_action_to_mujoco_ctrl(np.zeros(12), DEFAULT_ARM_COMMAND))
        self._mujoco.mj_forward(self.model, data)

    def reset_home(self, data) -> None:
        """Reset to the locomanipulation initial stand/stowed configuration.

        Build the home state in a fresh ``MjData`` and copy it into the live
        data object so the result does not depend on contacts, velocities,
        warm-starts, or other runtime state from the moment reset was pressed.
        """

        fresh = self._mujoco.MjData(self.model)
        self._initialize_home_data(fresh)
        data.time = fresh.time
        data.qpos[:] = fresh.qpos
        data.qvel[:] = fresh.qvel
        data.ctrl[:] = fresh.ctrl
        data.qfrc_applied[:] = fresh.qfrc_applied
        if hasattr(data, "xfrc_applied"):
            data.xfrc_applied[:] = fresh.xfrc_applied
        if hasattr(data, "qacc_warmstart"):
            data.qacc_warmstart[:] = fresh.qacc_warmstart
        if getattr(data, "act", None) is not None and data.act.size:
            data.act[:] = fresh.act
        self._mujoco.mj_forward(self.model, data)

    def read_joint_positions(self, data, names: tuple[str, ...]) -> np.ndarray:
        return np.array(
            [data.qpos[self._joint_addresses[name].qpos] for name in names], dtype=float
        )

    def read_joint_velocities(self, data, names: tuple[str, ...]) -> np.ndarray:
        return np.array(
            [data.qvel[self._joint_addresses[name].qvel] for name in names], dtype=float
        )

    def read_measured_arm_command(self, data) -> np.ndarray:
        """Return current MuJoCo arm/gripper joint positions in policy command order."""

        return self.read_joint_positions(
            data,
            (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME),
        )

    def read_policy_observation(self, data) -> dict[str, np.ndarray]:
        """Read the MuJoCo state in the structure expected by the policy helper."""

        arm_pos = self.read_joint_positions(data, MJCF_ARM_JOINT_NAMES)
        arm_vel = self.read_joint_velocities(data, MJCF_ARM_JOINT_NAMES)
        gripper_pos = self.read_joint_positions(data, (MJCF_GRIPPER_JOINT_NAME,))
        gripper_vel = self.read_joint_velocities(data, (MJCF_GRIPPER_JOINT_NAME,))
        body_xmat = np.asarray(data.xmat[self._body_id], dtype=float).reshape(3, 3)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=float)
        return {
            "base_lin_vel": np.asarray(data.qvel[:3], dtype=float).copy(),
            "base_ang_vel": np.asarray(data.qvel[3:6], dtype=float).copy(),
            "projected_gravity": body_xmat.T @ world_gravity,
            "joint_pos": self.read_joint_positions(data, SPOT_LEG_JOINT_NAMES),
            "joint_vel": self.read_joint_velocities(data, SPOT_LEG_JOINT_NAMES),
            "arm_state": np.concatenate([arm_pos, arm_vel]),
            "gripper_state": np.concatenate([gripper_pos, gripper_vel]),
            "base_pose": np.asarray(data.qpos[:7], dtype=float).copy(),
        }

    def body_pose_wxyz(self, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a MuJoCo body pose as ``(position, wxyz)`` in world coordinates."""

        body_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = np.asarray(data.xpos[body_id], dtype=float).copy()
        quat = np.empty(4, dtype=float)
        self._mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[body_id], dtype=float))
        return pos, quat

    def geom_pose_wxyz(self, data, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a MuJoCo geom pose as ``(position, wxyz)`` in world coordinates."""

        geom_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = np.asarray(data.geom_xpos[geom_id], dtype=float).copy()
        quat = np.empty(4, dtype=float)
        self._mujoco.mju_mat2Quat(quat, np.asarray(data.geom_xmat[geom_id], dtype=float))
        return pos, quat

    def apply_ctrl_targets(self, data, targets: Mapping[str, float]) -> None:
        for name, value in targets.items():
            actuator_id = self._actuator_ids.get(name)
            if actuator_id is None:
                raise KeyError(f"No actuator named {name!r}")
            data.ctrl[actuator_id] = float(value)

    def clear_applied_forces(self, data) -> None:
        data.qfrc_applied[:] = 0.0
        if hasattr(data, "xfrc_applied"):
            data.xfrc_applied[:] = 0.0

    def apply_arm_joint_torques(self, data, torques: np.ndarray | None) -> None:
        if torques is None:
            return
        values = np.asarray(torques, dtype=float)
        if values.shape != (len(MJCF_ARM_JOINT_NAMES),):
            raise ValueError(
                f"Expected {len(MJCF_ARM_JOINT_NAMES)} arm torques, got {values.shape}"
            )
        for name, value in zip(MJCF_ARM_JOINT_NAMES, values):
            data.qfrc_applied[self._joint_addresses[name].qvel] += float(value)

    def has_manipulation_objects(self) -> bool:
        return all(idx >= 0 for idx in self._manipulation_body_ids.values()) and all(
            idx >= 0 for idx in self._manipulation_geom_ids.values()
        ) and all(
            address.qpos >= 0 and address.qvel >= 0
            for address in self._manipulation_freejoint_addresses.values()
        )

    def manipulation_objects_enabled(self) -> bool:
        return bool(self._manipulation_objects_enabled and self.has_manipulation_objects())

    def set_manipulation_objects_enabled(self, data, enabled: bool) -> None:
        """Show/hide optional manipulation objects while preserving compiled contacts."""

        if not self.has_manipulation_objects():
            return
        for name, pose in MANIPULATION_OBJECT_BODY_POSES.items():
            self.model.body_gravcomp[self._manipulation_body_ids[name]] = 0.0 if enabled else 1.0
            address = self._manipulation_freejoint_addresses[name]
            if enabled:
                data.qpos[address.qpos : address.qpos + 3] = np.asarray(pose, dtype=float)
            else:
                data.qpos[address.qpos : address.qpos + 3] = [0.0, 0.0, MANIPULATION_OBJECT_HIDDEN_Z]
            data.qpos[address.qpos + 3 : address.qpos + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[address.qvel : address.qvel + 6] = 0.0
        for name in MANIPULATION_OBJECT_GEOM_NAMES:
            geom_id = self._manipulation_geom_ids[name]
            if enabled:
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
            else:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
            rgba = np.asarray(_MANIPULATION_OBJECT_RGBA[name], dtype=float).copy()
            rgba[3] = 1.0 if enabled else 0.0
            self.model.geom_rgba[geom_id] = rgba
        self._manipulation_objects_enabled = bool(enabled)
        self._mujoco.mj_forward(self.model, data)

    def manipulation_object_contact_count(self, data) -> int:
        if not self.has_manipulation_objects():
            return 0
        object_geom_ids = set(self._manipulation_geom_ids.values())
        count = 0
        for idx in range(data.ncon):
            contact = data.contact[idx]
            if int(contact.geom1) in object_geom_ids or int(contact.geom2) in object_geom_ids:
                count += 1
        return count

    def current_ctrl_targets(self, data) -> dict[str, float]:
        return {
            name: float(data.ctrl[actuator_id]) for name, actuator_id in self._actuator_ids.items()
        }
