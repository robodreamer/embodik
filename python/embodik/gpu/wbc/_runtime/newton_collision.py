"""Exact, device-resident Newton collision queries for replicated robot worlds."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import warp as wp

from ..collision_model import (
    panda_collision_geometry_name,
    replicated_shape_pairs,
    retained_collision_pairs,
)
from ..contracts import RobotSolveSpec
from .newton_model import _quat_xyzw_to_matrix, _span, _unique_suffix_index

_COLLISION_LINK_PREFIX = "__embodik_collision__"


def _write_grouped_collision_urdf(
    source_path: Path, cache_dir: Path
) -> tuple[Path, dict[str, str]]:
    """Put each URDF collision element on a named fixed child body.

    Newton can decompose one mesh into multiple shapes. The child body keeps
    those pieces grouped under the single geometry name exposed by EmbodiK.
    """

    tree = ET.parse(source_path)
    root = tree.getroot()
    generated: dict[str, str] = {}
    new_links: list[ET.Element] = []
    new_joints: list[ET.Element] = []
    counter = 0
    for link in list(root.findall("link")):
        link_name = str(link.attrib["name"])
        collisions = list(link.findall("collision"))
        for ordinal, collision in enumerate(collisions):
            for mesh in collision.findall(".//mesh"):
                filename = mesh.attrib.get("filename")
                if filename and not filename.startswith("package://"):
                    mesh.attrib["filename"] = str(
                        (source_path.parent / filename).resolve()
                    )
            origin = collision.find("origin")
            if origin is not None:
                collision.remove(origin)
            link.remove(collision)
            generated_name = f"{_COLLISION_LINK_PREFIX}{counter}"
            counter += 1
            generated[generated_name] = panda_collision_geometry_name(
                link_name, ordinal
            )
            child = ET.Element("link", {"name": generated_name})
            child.append(collision)
            joint = ET.Element(
                "joint",
                {"name": f"{generated_name}_joint", "type": "fixed"},
            )
            ET.SubElement(joint, "parent", {"link": link_name})
            ET.SubElement(joint, "child", {"link": generated_name})
            joint.append(
                copy.deepcopy(origin)
                if origin is not None
                else ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            )
            new_links.append(child)
            new_joints.append(joint)
        for visual in list(link.findall("visual")):
            link.remove(visual)
    root.extend(new_links)
    root.extend(new_joints)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{source_path.stem}-collision-groups.urdf"
    tree.write(output, encoding="unicode")
    return output, generated


@wp.kernel
def _reset_query_outputs(
    query_distance: float,
    minimum_distance: wp.array(dtype=wp.float32),
    winner: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    minimum_distance[world] = query_distance
    winner[world] = 2147483647
    active[world] = 0
    overflow[world] = 0


@wp.kernel
def _reset_pair_distances(
    query_distance: float,
    pair_distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    pair_distances[index] = query_distance


@wp.kernel
def _reduce_minimum_distance(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    shape0: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    minimum_distance: wp.array(dtype=wp.float32),
):
    contact = wp.tid()
    if contact < wp.min(contact_count[0], contact_capacity):
        world = shape_world[shape0[contact]]
        if world >= 0:
            wp.atomic_min(minimum_distance, world, distance[contact])


@wp.kernel
def _reduce_pair_distances(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    source_shape_count: int,
    pair_count: int,
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    pair_lookup: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    pair_distances: wp.array(dtype=wp.float32),
):
    contact = wp.tid()
    if contact < wp.min(contact_count[0], contact_capacity):
        shape_a = shape0[contact]
        shape_b = shape1[contact]
        world = shape_world[shape_a]
        source_a = shape_a % source_shape_count
        source_b = shape_b % source_shape_count
        pair = pair_lookup[source_a * source_shape_count + source_b]
        if world >= 0 and pair >= 0:
            wp.atomic_min(pair_distances, world * pair_count + pair, distance[contact])


@wp.kernel
def _select_winner(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    shape0: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    minimum_distance: wp.array(dtype=wp.float32),
    winner: wp.array(dtype=wp.int32),
):
    contact = wp.tid()
    if contact < wp.min(contact_count[0], contact_capacity):
        world = shape_world[shape0[contact]]
        if world >= 0 and wp.abs(distance[contact] - minimum_distance[world]) <= 1.0e-7:
            wp.atomic_min(winner, world, contact)


@wp.kernel
def _select_pair_winners(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    source_shape_count: int,
    pair_count: int,
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    shape_world: wp.array(dtype=wp.int32),
    pair_lookup: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    pair_distances: wp.array(dtype=wp.float32),
    pair_winner: wp.array(dtype=wp.int32),
):
    contact = wp.tid()
    if contact < wp.min(contact_count[0], contact_capacity):
        shape_a = shape0[contact]
        shape_b = shape1[contact]
        world = shape_world[shape_a]
        source_a = shape_a % source_shape_count
        source_b = shape_b % source_shape_count
        pair = pair_lookup[source_a * source_shape_count + source_b]
        if world >= 0 and pair >= 0:
            output = world * pair_count + pair
            if wp.abs(distance[contact] - pair_distances[output]) <= 1.0e-7:
                wp.atomic_min(pair_winner, output, contact)


@wp.kernel
def _reset_pair_winners(pair_winner: wp.array(dtype=wp.int32)):
    pair_winner[wp.tid()] = 2147483647


@wp.kernel
def _reset_probe_pair_outputs(
    query_distance: float,
    pair_distance: wp.array(dtype=wp.float32),
    pair_winner: wp.array(dtype=wp.int32),
    pair_shape: wp.array(dtype=wp.vec2i),
    pair_normal: wp.array(dtype=wp.vec3),
    pair_point0: wp.array(dtype=wp.vec3),
    pair_point1: wp.array(dtype=wp.vec3),
    pair_active: wp.array(dtype=wp.int32),
):
    pair = wp.tid()
    pair_distance[pair] = query_distance
    pair_winner[pair] = 2147483647
    pair_shape[pair] = wp.vec2i(-1, -1)
    pair_normal[pair] = wp.vec3()
    pair_point0[pair] = wp.vec3()
    pair_point1[pair] = wp.vec3()
    pair_active[pair] = 0


@wp.kernel
def _gather_pair_witnesses(
    pair_winner: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    pair_count: int,
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    normal: wp.array(dtype=wp.vec3),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    out_shape_pair: wp.array(dtype=wp.vec2i),
    out_normal: wp.array(dtype=wp.vec3),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_active: wp.array(dtype=wp.int32),
):
    output = wp.tid()
    world = output // pair_count
    contact = pair_winner[output]
    if overflow[world] == 0 and contact != 2147483647:
        out_shape_pair[output] = wp.vec2i(shape0[contact], shape1[contact])
        out_normal[output] = normal[contact]
        out_point0[output] = point0[contact]
        out_point1[output] = point1[contact]
        out_active[output] = 1
    else:
        out_shape_pair[output] = wp.vec2i(-1, -1)
        out_normal[output] = wp.vec3()
        out_point0[output] = wp.vec3()
        out_point1[output] = wp.vec3()
        out_active[output] = 0


@wp.kernel
def _mark_overflow(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    broad_count: wp.array(dtype=wp.int32),
    broad_capacity: int,
    split_query_count: wp.array(dtype=wp.int32),
    split_query_capacity: int,
    gjk_count: wp.array(dtype=wp.int32),
    gjk_capacity: int,
    split_gjk_count: wp.array(dtype=wp.int32),
    split_gjk_capacity: int,
    split_manifold_count: wp.array(dtype=wp.int32),
    split_manifold_capacity: int,
    mesh_count: wp.array(dtype=wp.int32),
    mesh_capacity: int,
    triangle_count: wp.array(dtype=wp.int32),
    triangle_capacity: int,
    mesh_mesh_count: wp.array(dtype=wp.int32),
    mesh_mesh_capacity: int,
    mesh_plane_count: wp.array(dtype=wp.int32),
    mesh_plane_capacity: int,
    sdf_sdf_count: wp.array(dtype=wp.int32),
    sdf_sdf_capacity: int,
    overflow: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    if (
        contact_count[0] > contact_capacity
        or broad_count[0] > broad_capacity
        or (split_query_capacity >= 0 and split_query_count[0] > split_query_capacity)
        or (gjk_capacity >= 0 and gjk_count[0] > gjk_capacity)
        or (split_gjk_capacity >= 0 and split_gjk_count[0] > split_gjk_capacity)
        or (
            split_manifold_capacity >= 0
            and split_manifold_count[0] > split_manifold_capacity
        )
        or (mesh_capacity >= 0 and mesh_count[0] > mesh_capacity)
        or (triangle_capacity >= 0 and triangle_count[0] > triangle_capacity)
        or (mesh_mesh_capacity >= 0 and mesh_mesh_count[0] > mesh_mesh_capacity)
        or (mesh_plane_capacity >= 0 and mesh_plane_count[0] > mesh_plane_capacity)
        or (sdf_sdf_capacity >= 0 and sdf_sdf_count[0] > sdf_sdf_capacity)
    ):
        overflow[world] = 1


@wp.kernel
def _gather_winner(
    query_distance: float,
    winner: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    normal: wp.array(dtype=wp.vec3),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_shape_pair: wp.array(dtype=wp.vec2i),
    out_normal: wp.array(dtype=wp.vec3),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    contact = winner[world]
    if overflow[world] != 0:
        # A conservative negative sentinel cannot be mistaken for safe clearance.
        out_distance[world] = -1.0e30
        out_shape_pair[world] = wp.vec2i(-1, -1)
        out_normal[world] = wp.vec3()
        out_point0[world] = wp.vec3()
        out_point1[world] = wp.vec3()
        active[world] = 0
    elif contact != 2147483647:
        out_distance[world] = distance[contact]
        out_shape_pair[world] = wp.vec2i(shape0[contact], shape1[contact])
        out_normal[world] = normal[contact]
        out_point0[world] = point0[contact]
        out_point1[world] = point1[contact]
        active[world] = 1
    else:
        # No emitted contact proves only that all retained pairs exceed the query radius.
        out_distance[world] = query_distance
        out_shape_pair[world] = wp.vec2i(-1, -1)
        out_normal[world] = wp.vec3()
        out_point0[world] = wp.vec3()
        out_point1[world] = wp.vec3()
        active[world] = 0


@wp.kernel
def _finalize_probe(
    query_distance: float,
    winner: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    minimum_distance: wp.array(dtype=wp.float32),
    shape_pair: wp.array(dtype=wp.vec2i),
    normal: wp.array(dtype=wp.vec3),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    shape_pair[world] = wp.vec2i(-1, -1)
    normal[world] = wp.vec3()
    point0[world] = wp.vec3()
    point1[world] = wp.vec3()
    if overflow[world] != 0:
        minimum_distance[world] = -1.0e30
        active[world] = 0
    elif winner[world] != 2147483647:
        active[world] = 1
    else:
        minimum_distance[world] = query_distance
        active[world] = 0


@wp.kernel
def _set_int_scalar(value: int, output: wp.array(dtype=wp.int32)):
    output[0] = value


@wp.kernel
def _fill_int(value: int, output: wp.array(dtype=wp.int32)):
    output[wp.tid()] = value


@wp.kernel
def _publish_convex_envelope_certificate(
    query_distance: float,
    envelope_active: wp.array(dtype=wp.int32),
    envelope_overflow: wp.array(dtype=wp.int32),
    minimum_distance: wp.array(dtype=wp.float32),
    winner: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    shape_pair: wp.array(dtype=wp.vec2i),
    normal: wp.array(dtype=wp.vec3),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    certified_output: wp.array(dtype=wp.int32),
):
    """Publish an exact empty result or a fail-closed speculative rejection."""

    world = wp.tid()
    certified = envelope_active[world] == 0 and envelope_overflow[world] == 0
    minimum_distance[world] = query_distance if certified else -1.0e30
    winner[world] = 2147483647
    active[world] = 0
    overflow[world] = 0 if certified else 1
    shape_pair[world] = wp.vec2i(-1, -1)
    normal[world] = wp.vec3()
    point0[world] = wp.vec3()
    point1[world] = wp.vec3()
    certified_output[world] = 1 if certified else 0


@dataclass(frozen=True)
class NewtonCollisionResult:
    """CUDA tensor views; no bulk host transfer occurs in ``query``."""

    distance_m: Any
    pair_distance_m: Any
    pair_gradient: Any
    pair_active: Any
    active: Any
    gradient: Any
    normal: Any
    point0_world_m: Any
    point1_world_m: Any
    shape_pair: Any
    overflow: Any
    raw_contact_count: Any
    query_envelope_clear_certified: Any = None


class NewtonModelCollisionQuery:
    """Query exact URDF meshes for the closest configured pair in each world.

    Newton's mesh contact reduction is deliberately disabled: it selects a
    representative manifold for simulation and can discard the true minimum
    required by a WBC velocity damper.
    """

    def __init__(
        self,
        batch_size: int,
        urdf_path: Path,
        *,
        robot_spec: RobotSolveSpec | None = None,
        collision_pairs: tuple[tuple[str, str], ...] = (),
        default_configuration: tuple[float, ...] | None = None,
        cache_dir: Path | None = None,
        device: str = "cuda:0",
        query_distance_m: float = 0.10,
        contacts_per_world: int = 1024,
        triangle_pairs_per_world: int = 512,
        deterministic: bool = True,
        sort_contacts: bool | None = None,
        overflow_replay_bucket_size: int = 64,
        _enable_overflow_isolation: bool = True,
        _convex_hull_mode: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < query_distance_m <= 0.5:
            raise ValueError("query_distance_m must be in (0, 0.5]")
        if contacts_per_world <= 0 or triangle_pairs_per_world <= 0:
            raise ValueError("collision capacities must be positive")
        if overflow_replay_bucket_size <= 0:
            raise ValueError("overflow_replay_bucket_size must be positive")

        import newton
        import torch

        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("NewtonPandaCollisionQuery requires CUDA")
        source_path = Path(urdf_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        self.batch_size = batch_size
        self.query_distance_m = float(query_distance_m)
        self.contacts_per_world = contacts_per_world
        self.triangle_pairs_per_world = triangle_pairs_per_world
        self.deterministic = deterministic
        self.sort_contacts = deterministic if sort_contacts is None else sort_contacts
        if self.sort_contacts and not deterministic:
            raise ValueError("contact sorting requires deterministic contact keys")
        self.overflow_replay_bucket_size = overflow_replay_bucket_size
        self._enable_overflow_isolation = _enable_overflow_isolation
        self._convex_hull_mode = _convex_hull_mode
        self.newton = newton
        self.torch = torch
        self._source_path = source_path
        self._cache_dir = (
            None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        )
        self._collision_pairs = tuple(collision_pairs)
        self._default_configuration = default_configuration
        self._device_label = device
        self._overflow_replay_queries: dict[int, NewtonModelCollisionQuery] = {}
        self.last_replay_query_count = 0
        self.last_replay_rejected_world_count = 0

        self.robot_spec = robot_spec
        self._legacy_panda = robot_spec is None
        generated_collision_names: dict[str, str] = {}
        model_path = source_path
        if not self._legacy_panda:
            if cache_dir is None:
                raise ValueError("model-derived collision query requires cache_dir")
            model_path, generated_collision_names = _write_grouped_collision_urdf(
                source_path, Path(cache_dir).expanduser().resolve()
            )
        source = newton.ModelBuilder()
        source.add_urdf(
            str(model_path),
            floating=False if robot_spec is None else robot_spec.floating_base,
            hide_visuals=True,
            enable_self_collisions=True,
            collapse_fixed_joints=False,
        )
        geometry_records: list[tuple[str, int, int]] = []
        geometry_shape_groups: dict[str, list[int]] = {}
        for body, label in enumerate(source.body_label):
            collision_shapes = [
                shape
                for shape in source.body_shapes.get(body, ())
                if int(source.shape_flags[shape]) != 0
            ]
            if self._legacy_panda:
                named_shapes = [
                    (panda_collision_geometry_name(label, ordinal), shape)
                    for ordinal, shape in enumerate(collision_shapes)
                ]
            else:
                body_name = label.rsplit("/", 1)[-1]
                geometry_name = generated_collision_names.get(body_name)
                named_shapes = (
                    []
                    if geometry_name is None
                    else [(geometry_name, shape) for shape in collision_shapes]
                )
            for geometry_name, shape in named_shapes:
                geometry_records.append((geometry_name, shape, body))
                geometry_shape_groups.setdefault(geometry_name, []).append(shape)
                # Newton combines the gaps from both shapes.
                source.shape_gap[shape] = self.query_distance_m * 0.5

        self.geometry_records = tuple(geometry_records)
        self.geometry_shape_groups = {
            name: tuple(shapes) for name, shapes in geometry_shape_groups.items()
        }
        self._geometry_name_by_source_shape = {
            int(shape): name for name, shape, _body in self.geometry_records
        }
        self.source_shape_count = source.shape_count
        self.source_body_count = source.body_count
        if self._legacy_panda:
            self.source_pairs = retained_collision_pairs(self.geometry_records)
            self.semantic_pair_count = len(self.source_pairs)
            self._source_pair_semantic_indices = tuple(range(self.semantic_pair_count))
        else:
            geometry_by_name = self.geometry_shape_groups
            missing = sorted(
                {
                    name
                    for pair in collision_pairs
                    for name in pair
                    if name not in geometry_by_name
                }
            )
            if missing:
                raise RuntimeError(
                    "configured collision geometries are absent from Newton: "
                    + ", ".join(missing)
                )
            if not collision_pairs:
                raise ValueError(
                    "model-derived collision query requires collision_pairs"
                )
            unordered = [frozenset(pair) for pair in collision_pairs]
            if any(len(pair) != 2 for pair in unordered) or len(set(unordered)) != len(
                unordered
            ):
                raise ValueError("collision_pairs must be unique two-geometry pairs")
            source_pairs: list[tuple[int, int]] = []
            source_pair_semantic_indices: list[int] = []
            for semantic_index, (name_a, name_b) in enumerate(collision_pairs):
                for shape_a in geometry_by_name[name_a]:
                    for shape_b in geometry_by_name[name_b]:
                        source_pairs.append((shape_a, shape_b))
                        source_pair_semantic_indices.append(semantic_index)
            self.source_pairs = tuple(source_pairs)
            self.semantic_pair_count = len(collision_pairs)
            self._source_pair_semantic_indices = tuple(source_pair_semantic_indices)
        if self._legacy_panda and (
            len(self.geometry_records) != 17 or len(self.source_pairs) != 60
        ):
            raise RuntimeError(
                "Panda collision topology drift: expected 17 geometries and 60 pairs, "
                f"got {len(self.geometry_records)} and {len(self.source_pairs)}"
            )

        if self._legacy_panda:
            self.input_configuration_dim = 7
            self.active_dim = 7
            self._mapped_newton_q_indices = tuple(range(7))
            self._source_q_indices = tuple(range(7))
            self._active_newton_qd_indices = tuple(range(7))
            defaults = [0.0] * source.joint_coord_count
            defaults[7:] = [0.04] * (source.joint_coord_count - 7)
        else:
            assert robot_spec is not None
            self.input_configuration_dim = robot_spec.configuration_dim
            self.active_dim = len(robot_spec.active_velocity_indices)
            defaults = self._configure_model_mappings(
                source, robot_spec, default_configuration
            )

        if self._convex_hull_mode:
            source.shape_type = [
                (
                    newton.GeoType.CONVEX_MESH
                    if shape_type == newton.GeoType.MESH
                    else shape_type
                )
                for shape_type in source.shape_type
            ]

        builder = newton.ModelBuilder()
        builder.replicate(source, world_count=batch_size)
        self.model = builder.finalize(device=device, requires_grad=False)
        self._streams: dict[int, Any] = {}
        self.source_configuration_dim = source.joint_coord_count
        self.source_velocity_dim = source.joint_dof_count
        if self.model.joint_coord_count != batch_size * self.source_configuration_dim:
            raise RuntimeError("unexpected replicated collision coordinate layout")
        if self.model.joint_dof_count != batch_size * self.source_velocity_dim:
            raise RuntimeError("unexpected replicated collision velocity layout")

        explicit_pairs = replicated_shape_pairs(
            self.source_pairs,
            shape_count=self.source_shape_count,
            world_count=batch_size,
        )
        self._shape_pairs = wp.array(
            explicit_pairs, dtype=wp.vec2i, device=self.model.device
        )
        pair_lookup = [-1] * (self.source_shape_count * self.source_shape_count)
        for (shape_a, shape_b), semantic_index in zip(
            self.source_pairs,
            self._source_pair_semantic_indices,
            strict=True,
        ):
            pair_lookup[shape_a * self.source_shape_count + shape_b] = semantic_index
            pair_lookup[shape_b * self.source_shape_count + shape_a] = semantic_index
        self._pair_lookup = wp.array(
            pair_lookup, dtype=wp.int32, device=self.model.device
        )
        self._contact_capacity = contacts_per_world * batch_size
        self._triangle_capacity = triangle_pairs_per_world * batch_size
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self._shape_pairs,
            rigid_contact_max=self._contact_capacity,
            max_triangle_pairs=self._triangle_capacity,
            reduce_contacts=False,
            deterministic=deterministic and self.sort_contacts,
            requires_grad=False,
            verify_buffers=False,
        )
        self.contacts = self.pipeline.contacts()
        self.state = self.model.state(requires_grad=False)

        self._q_full = (
            torch.tensor(defaults, dtype=torch.float32, device=device)
            .expand(batch_size, -1)
            .clone()
        )
        self._mapped_newton_q_indices_tensor = torch.tensor(
            self._mapped_newton_q_indices, dtype=torch.long, device=device
        )
        self._source_q_indices_tensor = torch.tensor(
            self._source_q_indices, dtype=torch.long, device=device
        )
        self._q_warp = wp.from_torch(self._q_full.reshape(-1))
        self._qd_warp = wp.zeros(
            self.model.joint_dof_count,
            dtype=wp.float32,
            device=self.model.device,
        )
        self._distance = wp.empty(
            self._contact_capacity, dtype=wp.float32, device=self.model.device
        )
        self._point0 = wp.empty(
            self._contact_capacity, dtype=wp.vec3, device=self.model.device
        )
        self._point1 = wp.empty_like(self._point0)
        self._minimum_distance = wp.empty(
            batch_size, dtype=wp.float32, device=self.model.device
        )
        self._pair_distance = wp.empty(
            batch_size * self.semantic_pair_count,
            dtype=wp.float32,
            device=self.model.device,
        )
        pair_slots = batch_size * self.semantic_pair_count
        self._pair_winner = wp.empty(
            pair_slots, dtype=wp.int32, device=self.model.device
        )
        self._pair_shape_pair = wp.empty(
            pair_slots, dtype=wp.vec2i, device=self.model.device
        )
        self._pair_normal = wp.empty(
            pair_slots, dtype=wp.vec3, device=self.model.device
        )
        self._pair_point0 = wp.empty_like(self._pair_normal)
        self._pair_point1 = wp.empty_like(self._pair_normal)
        self._pair_active = wp.empty(
            pair_slots, dtype=wp.int32, device=self.model.device
        )
        self._winner = wp.empty(batch_size, dtype=wp.int32, device=self.model.device)
        self._active = wp.empty(batch_size, dtype=wp.int32, device=self.model.device)
        self._overflow = wp.empty(batch_size, dtype=wp.int32, device=self.model.device)
        self._overflow_zero_counter = wp.zeros(
            1, dtype=wp.int32, device=self.model.device
        )
        self._query_envelope_clear_certified = wp.zeros(
            batch_size, dtype=wp.int32, device=self.model.device
        )
        self._shape_pair = wp.empty(
            batch_size, dtype=wp.vec2i, device=self.model.device
        )
        self._normal = wp.empty(batch_size, dtype=wp.vec3, device=self.model.device)
        self._selected_point0 = wp.empty(
            batch_size, dtype=wp.vec3, device=self.model.device
        )
        self._selected_point1 = wp.empty_like(self._selected_point0)

        self._body_jacobian = wp.empty(
            (
                self.model.articulation_count,
                self.model.max_joints_per_articulation * 6,
                self.model.max_dofs_per_articulation,
            ),
            dtype=wp.float32,
            device=self.model.device,
        )
        self._joint_motion_subspace = wp.zeros(
            self.model.joint_dof_count,
            dtype=wp.spatial_vector,
            device=self.model.device,
        )
        body_row = [-1] * source.body_count
        for joint_index, body_index in enumerate(source.joint_child):
            body_row[int(body_index)] = joint_index
        if any(index < 0 for index in body_row):
            raise RuntimeError("collision body is missing an articulation Jacobian row")
        source_shape_rows = [body_row[int(body)] for body in source.shape_body]
        self._source_shape_rows = torch.tensor(
            source_shape_rows, dtype=torch.long, device=device
        )
        self._active_newton_qd_indices_tensor = torch.tensor(
            self._active_newton_qd_indices, dtype=torch.long, device=device
        )
        self._shape_body_torch = wp.to_torch(self.model.shape_body).long()
        self._body_q_torch = wp.to_torch(self.state.body_q)
        self._body_com_torch = wp.to_torch(self.model.body_com)
        self._body_jacobian_torch = wp.to_torch(self._body_jacobian)

        self._query_graphs: dict[tuple[bool, bool], Any] = {}
        self._zero_gradient = torch.zeros(
            (batch_size, self.active_dim), dtype=torch.float32, device=device
        )

    def _configure_model_mappings(
        self,
        source: Any,
        spec: RobotSolveSpec,
        default_configuration: tuple[float, ...] | None,
    ) -> list[float]:
        """Map a full EmbodiK configuration and active tangent into Newton."""

        if not all(
            (
                spec.joint_configuration_indices,
                spec.joint_configuration_sizes,
                spec.joint_velocity_indices,
                spec.joint_velocity_sizes,
            )
        ):
            raise RuntimeError("collision query requires model-derived joint spans")
        default = (
            (0.0,) * spec.configuration_dim
            if default_configuration is None
            else tuple(float(value) for value in default_configuration)
        )
        if len(default) != spec.configuration_dim:
            raise ValueError("default_configuration must match configuration_dim")

        q_starts = source.joint_q_start
        qd_starts = source.joint_qd_start
        active_lookup = {
            source_index: active_index
            for active_index, source_index in enumerate(spec.active_velocity_indices)
        }
        newton_velocity_by_source: dict[int, int] = {}
        mapped_newton_q: list[int] = []
        mapped_source_q: list[int] = []
        free_source = [
            index
            for index, (q_size, v_size) in enumerate(
                zip(
                    spec.joint_configuration_sizes,
                    spec.joint_velocity_sizes,
                    strict=True,
                )
            )
            if q_size == 7 and v_size == 6
        ]
        free_newton = [
            index
            for index in range(source.joint_count)
            if _span(q_starts, index, source.joint_coord_count) == 7
            and _span(qd_starts, index, source.joint_dof_count) == 6
        ]
        if spec.floating_base and (len(free_source) != 1 or len(free_newton) != 1):
            raise RuntimeError(
                "floating collision model requires exactly one free joint"
            )
        if not spec.floating_base and (free_source or free_newton):
            raise RuntimeError(
                "fixed collision model unexpectedly contains a free joint"
            )

        free_source_index = free_source[0] if free_source else None
        if free_source_index is not None:
            free_newton_index = free_newton[0]
            source_q_start = spec.joint_configuration_indices[free_source_index]
            source_v_start = spec.joint_velocity_indices[free_source_index]
            newton_q_start = int(q_starts[free_newton_index])
            base_velocities = tuple(range(source_v_start, source_v_start + 6))
            if not set(base_velocities).issubset(active_lookup):
                raise RuntimeError("all free-root tangent velocities must be active")
            mapped_newton_q.extend(range(newton_q_start, newton_q_start + 7))
            mapped_source_q.extend(range(source_q_start, source_q_start + 7))
            newton_qd_start = int(qd_starts[free_newton_index])
            newton_velocity_by_source.update(
                {
                    source_v_start + offset: newton_qd_start + offset
                    for offset in range(6)
                }
            )
        for joint_index, joint_name in enumerate(spec.joint_names):
            q_size = spec.joint_configuration_sizes[joint_index]
            v_size = spec.joint_velocity_sizes[joint_index]
            if joint_index == free_source_index or (q_size == 0 and v_size == 0):
                continue
            if q_size != 1 or v_size != 1:
                raise RuntimeError(
                    f"collision joint {joint_name!r} has unsupported nq/nv={q_size}/{v_size}"
                )
            newton_joint = _unique_suffix_index(
                source.joint_label, joint_name, kind="joint"
            )
            if (
                _span(q_starts, newton_joint, source.joint_coord_count) != 1
                or _span(qd_starts, newton_joint, source.joint_dof_count) != 1
            ):
                raise RuntimeError(
                    f"Newton collision joint {joint_name!r} is not scalar"
                )
            newton_q = int(q_starts[newton_joint])
            source_q = spec.joint_configuration_indices[joint_index]
            source_velocity = spec.joint_velocity_indices[joint_index]
            newton_velocity_by_source[source_velocity] = int(qd_starts[newton_joint])
            mapped_newton_q.append(newton_q)
            mapped_source_q.append(source_q)
        if len(mapped_source_q) != spec.configuration_dim:
            raise RuntimeError("collision query does not map every model configuration")
        self._mapped_newton_q_indices = tuple(mapped_newton_q)
        self._source_q_indices = tuple(mapped_source_q)
        self._active_newton_qd_indices = tuple(
            newton_velocity_by_source[index] for index in spec.active_velocity_indices
        )
        defaults = [0.0] * source.joint_coord_count
        for newton_index, source_index in zip(
            mapped_newton_q, mapped_source_q, strict=True
        ):
            defaults[newton_index] = default[source_index]
        return defaults

    def _current_warp_stream(self) -> Any:
        """Mirror PyTorch's active stream, including parent graph capture."""

        torch_stream = self.torch.cuda.current_stream(self._q_full.device)
        pointer = int(torch_stream.cuda_stream)
        stream = self._streams.get(pointer)
        if stream is None:
            stream = wp.Stream(self.model.device, cuda_stream=pointer)
            self._streams[pointer] = stream
        return stream

    def geometry_name(self, replicated_shape_index: int) -> str:
        """Return the URDF collision-geometry label for a replicated shape."""

        if replicated_shape_index < 0:
            return "none"
        source_shape = replicated_shape_index % self.source_shape_count
        return self._geometry_name_by_source_shape.get(
            source_shape, f"shape_{replicated_shape_index}"
        )

    def _launch_overflow_check(self) -> None:
        narrow = self.pipeline.narrow_phase
        def counter_capacity(counter: Any, entries: Any) -> tuple[Any, int]:
            return (
                counter if counter is not None else self._overflow_zero_counter,
                entries.shape[0] if entries is not None else -1,
            )

        split_enabled = bool(narrow.split_gjk_mpr)
        gjk_count, gjk_capacity = counter_capacity(
            narrow.gjk_candidate_pairs_count, narrow.gjk_candidate_pairs
        )
        split_query_count = (
            self.pipeline.broad_phase_pair_count
            if split_enabled and narrow.sparse_gjk_pairs
            else gjk_count
        )
        split_query_capacity = (
            narrow.split_query_results.shape[0]
            if split_enabled and narrow.split_query_results is not None
            else -1
        )
        split_gjk_count = (
            narrow.split_gjk_work_count
            if narrow.split_gjk_work_count is not None
            else self._overflow_zero_counter
        )
        split_gjk_capacity = (
            narrow.split_gjk_work_items.shape[0]
            if narrow.split_gjk_work_items is not None
            else -1
        )
        split_manifold_count = (
            narrow.split_manifold_work_count
            if narrow.split_manifold_work_count is not None
            else self._overflow_zero_counter
        )
        split_manifold_capacity = (
            narrow.split_manifold_work_items.shape[0]
            if narrow.split_manifold_work_items is not None
            else -1
        )
        mesh_plane_count = (
            narrow.shape_pairs_mesh_plane_count
            if narrow.shape_pairs_mesh_plane_count is not None
            else self._overflow_zero_counter
        )
        mesh_plane_capacity = (
            narrow.shape_pairs_mesh_plane.shape[0]
            if narrow.shape_pairs_mesh_plane is not None
            else -1
        )
        sdf_sdf_count = (
            narrow.shape_pairs_sdf_sdf_count
            if narrow.shape_pairs_sdf_sdf_count is not None
            else self._overflow_zero_counter
        )
        sdf_sdf_capacity = (
            narrow.shape_pairs_sdf_sdf.shape[0]
            if narrow.shape_pairs_sdf_sdf is not None
            else -1
        )
        mesh_count, mesh_capacity = counter_capacity(
            narrow.shape_pairs_mesh_count, narrow.shape_pairs_mesh
        )
        triangle_count, triangle_capacity = counter_capacity(
            narrow.triangle_pairs_count, narrow.triangle_pairs
        )
        mesh_mesh_count, mesh_mesh_capacity = counter_capacity(
            narrow.shape_pairs_mesh_mesh_count, narrow.shape_pairs_mesh_mesh
        )
        wp.launch(
            _mark_overflow,
            dim=self.batch_size,
            inputs=[
                self.contacts.rigid_contact_count,
                self._contact_capacity,
                self.pipeline.broad_phase_pair_count,
                self.pipeline.broad_phase_shape_pairs.shape[0],
                split_query_count,
                split_query_capacity,
                gjk_count,
                gjk_capacity,
                split_gjk_count,
                split_gjk_capacity,
                split_manifold_count,
                split_manifold_capacity,
                mesh_count,
                mesh_capacity,
                triangle_count,
                triangle_capacity,
                mesh_mesh_count,
                mesh_mesh_capacity,
                mesh_plane_count,
                mesh_plane_capacity,
                sdf_sdf_count,
                sdf_sdf_capacity,
                self._overflow,
            ],
            device=self.model.device,
        )

    def _validate(self, q_configuration: Any) -> None:
        torch = self.torch
        if not isinstance(q_configuration, torch.Tensor):
            raise TypeError("q_configuration must be a torch.Tensor")
        if q_configuration.shape != (
            self.batch_size,
            self.input_configuration_dim,
        ):
            raise ValueError(
                "q_configuration must have shape "
                f"{(self.batch_size, self.input_configuration_dim)}, got "
                f"{tuple(q_configuration.shape)}"
            )
        if (
            q_configuration.device != self._q_full.device
            or q_configuration.dtype is not torch.float32
        ):
            raise ValueError(
                "q_configuration must be CUDA float32 on the configured device"
            )
        if not bool(torch.isfinite(q_configuration).all().item()):
            raise ValueError("q_configuration contains non-finite values")

    def query(
        self, q_configuration: Any, *, compute_gradient: bool = True
    ) -> NewtonCollisionResult:
        """Return closest proximity data and tangent gradients on CUDA."""

        self._validate(q_configuration)
        return self._query_trusted(q_configuration, compute_gradient=compute_gradient)

    def _query_trusted(
        self,
        q_configuration: Any,
        *,
        compute_gradient: bool = True,
        probe_only: bool = False,
    ) -> NewtonCollisionResult:
        """Query validated tensors and isolate global overflow by replay buckets."""

        if compute_gradient and probe_only:
            raise ValueError("probe-only collision queries cannot compute gradients")
        result = self._query_once(
            q_configuration,
            compute_gradient=compute_gradient,
            probe_only=probe_only,
        )
        self.last_replay_query_count = 0
        self.last_replay_rejected_world_count = 0
        if (
            not self._enable_overflow_isolation
            or self.batch_size == 1
            or self.torch.cuda.is_current_stream_capturing()
        ):
            return result
        if not bool(result.overflow.any().item()):
            return result
        replayed = self._replay_overflow_isolated(
            q_configuration,
            compute_gradient=compute_gradient,
            probe_only=probe_only,
        )
        self.last_replay_rejected_world_count = int(replayed.overflow.sum().item())
        return replayed

    def _query_once(
        self,
        q_configuration: Any,
        *,
        compute_gradient: bool,
        probe_only: bool = False,
    ) -> NewtonCollisionResult:
        """Run one fixed-shape pipeline query without overflow replay."""

        mapped_q = q_configuration.index_select(1, self._source_q_indices_tensor)
        self._q_full.index_copy_(1, self._mapped_newton_q_indices_tensor, mapped_q)
        graph_key = (compute_gradient, probe_only)
        graph = self._query_graphs.get(graph_key)
        if graph is None:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._execute_query_graph(
                    compute_gradient,
                    probe_only=probe_only,
                )
            graph = capture.graph
            self._query_graphs[graph_key] = graph
        stream = self._current_warp_stream()
        if self.torch.cuda.is_current_stream_capturing():
            # Flatten the exact query into a parent solve capture; CUDA does
            # not permit launching this child graph on its capture stream.
            with wp.ScopedStream(stream, sync_enter=False):
                self._execute_query_graph(
                    compute_gradient,
                    probe_only=probe_only,
                )
        else:
            wp.capture_launch(graph, stream=stream)

        return self._result_views(compute_gradient)

    def _result_views(self, compute_gradient: bool) -> NewtonCollisionResult:
        """Return stable tensor views populated by the most recent query graph."""

        pair_gradient = self._analytic_pair_gradients() if compute_gradient else None
        if pair_gradient is None:
            q_gradient = self._zero_gradient
            pair_gradient = self._zero_gradient[:, None, :].expand(
                -1, self.semantic_pair_count, -1
            )
        else:
            closest_pair = self.torch.argmin(
                wp.to_torch(self._pair_distance).reshape(
                    self.batch_size, self.semantic_pair_count
                ),
                dim=-1,
            )
            q_gradient = pair_gradient[
                self.torch.arange(self.batch_size, device=self._q_full.device),
                closest_pair,
            ]

        return NewtonCollisionResult(
            distance_m=wp.to_torch(self._minimum_distance),
            pair_distance_m=wp.to_torch(self._pair_distance).reshape(
                self.batch_size, self.semantic_pair_count
            ),
            pair_gradient=pair_gradient,
            pair_active=wp.to_torch(self._pair_active)
            .reshape(self.batch_size, self.semantic_pair_count)
            .bool(),
            active=wp.to_torch(self._active).bool(),
            gradient=q_gradient,
            normal=wp.to_torch(self._normal),
            point0_world_m=wp.to_torch(self._selected_point0),
            point1_world_m=wp.to_torch(self._selected_point1),
            shape_pair=wp.to_torch(self._shape_pair),
            overflow=wp.to_torch(self._overflow).bool(),
            raw_contact_count=wp.to_torch(self.contacts.rigid_contact_count),
            query_envelope_clear_certified=wp.to_torch(
                self._query_envelope_clear_certified
            ).bool(),
        )

    def publish_convex_envelope_certificate(
        self, envelope: NewtonModelCollisionQuery
    ) -> NewtonCollisionResult:
        """Convert a convex-superset query into an exact-empty or rejection result."""

        if (
            envelope.batch_size != self.batch_size
            or envelope.query_distance_m != self.query_distance_m
            or not envelope._convex_hull_mode
        ):
            raise ValueError("incompatible convex collision envelope")
        wp.launch(
            _set_int_scalar,
            dim=1,
            inputs=[0],
            outputs=[self.contacts.rigid_contact_count],
            device=self.model.device,
        )
        wp.launch(
            _reset_probe_pair_outputs,
            dim=self.batch_size * self.semantic_pair_count,
            inputs=[self.query_distance_m],
            outputs=[
                self._pair_distance,
                self._pair_winner,
                self._pair_shape_pair,
                self._pair_normal,
                self._pair_point0,
                self._pair_point1,
                self._pair_active,
            ],
            device=self.model.device,
        )
        wp.launch(
            _publish_convex_envelope_certificate,
            dim=self.batch_size,
            inputs=[
                self.query_distance_m,
                envelope._active,
                envelope._overflow,
            ],
            outputs=[
                self._minimum_distance,
                self._winner,
                self._active,
                self._overflow,
                self._shape_pair,
                self._normal,
                self._selected_point0,
                self._selected_point1,
                self._query_envelope_clear_certified,
            ],
            device=self.model.device,
        )
        return self._result_views(compute_gradient=False)

    def _overflow_replay_query(self, batch_size: int) -> NewtonModelCollisionQuery:
        query = self._overflow_replay_queries.get(batch_size)
        if query is None:
            cache_dir = (
                None
                if self._cache_dir is None
                else self._cache_dir / f"overflow-replay-b{batch_size}"
            )
            query = NewtonModelCollisionQuery(
                batch_size,
                self._source_path,
                robot_spec=self.robot_spec,
                collision_pairs=self._collision_pairs,
                default_configuration=self._default_configuration,
                cache_dir=cache_dir,
                device=self._device_label,
                query_distance_m=self.query_distance_m,
                contacts_per_world=self.contacts_per_world,
                triangle_pairs_per_world=self.triangle_pairs_per_world,
                deterministic=self.deterministic,
                sort_contacts=self.sort_contacts,
                overflow_replay_bucket_size=self.overflow_replay_bucket_size,
                _enable_overflow_isolation=False,
                _convex_hull_mode=self._convex_hull_mode,
            )
            self._overflow_replay_queries[batch_size] = query
        return query

    def _clone_result(self, result: NewtonCollisionResult) -> NewtonCollisionResult:
        return NewtonCollisionResult(
            distance_m=result.distance_m.clone(),
            pair_distance_m=result.pair_distance_m.clone(),
            pair_gradient=result.pair_gradient.clone(),
            pair_active=result.pair_active.clone(),
            active=result.active.clone(),
            gradient=result.gradient.clone(),
            normal=result.normal.clone(),
            point0_world_m=result.point0_world_m.clone(),
            point1_world_m=result.point1_world_m.clone(),
            shape_pair=result.shape_pair.clone(),
            overflow=result.overflow.clone(),
            raw_contact_count=result.raw_contact_count.clone(),
            query_envelope_clear_certified=(
                result.query_envelope_clear_certified.clone()
            ),
        )

    def _replay_segment(
        self,
        q_configuration: Any,
        *,
        compute_gradient: bool,
        probe_only: bool,
    ) -> list[NewtonCollisionResult]:
        batch_size = q_configuration.shape[0]
        query = self._overflow_replay_query(batch_size)
        result = query._query_once(
            q_configuration,
            compute_gradient=compute_gradient,
            probe_only=probe_only,
        )
        self.last_replay_query_count += 1
        if not bool(result.overflow.any().item()) or batch_size == 1:
            return [self._clone_result(result)]
        midpoint = batch_size // 2
        return self._replay_segment(
            q_configuration[:midpoint],
            compute_gradient=compute_gradient,
            probe_only=probe_only,
        ) + self._replay_segment(
            q_configuration[midpoint:],
            compute_gradient=compute_gradient,
            probe_only=probe_only,
        )

    def _replay_overflow_isolated(
        self,
        q_configuration: Any,
        *,
        compute_gradient: bool,
        probe_only: bool,
    ) -> NewtonCollisionResult:
        leaves: list[NewtonCollisionResult] = []
        bucket = self.overflow_replay_bucket_size
        for start in range(0, self.batch_size, bucket):
            leaves.extend(
                self._replay_segment(
                    q_configuration[start : start + bucket],
                    compute_gradient=compute_gradient,
                    probe_only=probe_only,
                )
            )
        torch = self.torch
        return NewtonCollisionResult(
            distance_m=torch.cat([item.distance_m for item in leaves]),
            pair_distance_m=torch.cat([item.pair_distance_m for item in leaves]),
            pair_gradient=torch.cat([item.pair_gradient for item in leaves]),
            pair_active=torch.cat([item.pair_active for item in leaves]),
            active=torch.cat([item.active for item in leaves]),
            gradient=torch.cat([item.gradient for item in leaves]),
            normal=torch.cat([item.normal for item in leaves]),
            point0_world_m=torch.cat([item.point0_world_m for item in leaves]),
            point1_world_m=torch.cat([item.point1_world_m for item in leaves]),
            shape_pair=torch.cat([item.shape_pair for item in leaves]),
            overflow=torch.cat([item.overflow for item in leaves]),
            raw_contact_count=torch.stack(
                [item.raw_contact_count[0] for item in leaves]
            ).sum()[None],
            query_envelope_clear_certified=torch.cat(
                [item.query_envelope_clear_certified for item in leaves]
            ),
        )

    def _analytic_pair_gradients(self) -> Any:
        """Return one active-tangent witness gradient per semantic pair."""

        torch = self.torch
        pair_shape = wp.to_torch(self._pair_shape_pair).reshape(
            self.batch_size, self.semantic_pair_count, 2
        )
        pair_active = (
            wp.to_torch(self._pair_active)
            .reshape(self.batch_size, self.semantic_pair_count)
            .bool()
        )
        source_shape = torch.remainder(pair_shape, self.source_shape_count)
        rows = self._source_shape_rows[source_shape]
        jacobian = self._body_jacobian_torch.reshape(
            self.batch_size,
            self.model.max_joints_per_articulation,
            6,
            self.model.max_dofs_per_articulation,
        )
        batch = torch.arange(self.batch_size, device=self._q_full.device)[:, None]
        batch = batch.expand(-1, self.semantic_pair_count)

        def point_jacobian(side: int, point: Any) -> Any:
            selected = jacobian[batch, rows[:, :, side]].index_select(
                -1, self._active_newton_qd_indices_tensor
            )
            shape = pair_shape[:, :, side].clamp_min(0)
            body = self._shape_body_torch[shape]
            body_pose = self._body_q_torch[body]
            rotation = _quat_xyzw_to_matrix(torch, body_pose[:, :, 3:])
            center = body_pose[:, :, :3] + (
                rotation @ self._body_com_torch[body].unsqueeze(-1)
            ).squeeze(-1)
            offset = point - center
            angular_at_point = torch.cross(
                selected[:, :, 3:, :].transpose(-2, -1),
                offset[:, :, None, :],
                dim=-1,
            ).transpose(-2, -1)
            return selected[:, :, :3, :] + angular_at_point

        point0 = wp.to_torch(self._pair_point0).reshape(
            self.batch_size, self.semantic_pair_count, 3
        )
        point1 = wp.to_torch(self._pair_point1).reshape(
            self.batch_size, self.semantic_pair_count, 3
        )
        normal = wp.to_torch(self._pair_normal).reshape(
            self.batch_size, self.semantic_pair_count, 3
        )
        relative = point_jacobian(1, point1) - point_jacobian(0, point0)
        gradient = torch.sum(normal[:, :, :, None] * relative, dim=2)
        return torch.where(
            pair_active[:, :, None], gradient, torch.zeros_like(gradient)
        )

    def _execute_query_graph(
        self,
        compute_gradient: bool,
        *,
        probe_only: bool = False,
    ) -> None:
        """Record one fixed-shape exact query."""

        self.newton.eval_fk(self.model, self._q_warp, self._qd_warp, self.state)
        wp.launch(
            _fill_int,
            dim=self.batch_size,
            inputs=[0],
            outputs=[self._query_envelope_clear_certified],
            device=self.model.device,
        )
        self._execute_exact_query_after_fk(
            compute_gradient=compute_gradient, probe_only=probe_only
        )

    def _execute_exact_query_after_fk(
        self, *, compute_gradient: bool, probe_only: bool
    ) -> None:
        """Run Newton's unchanged exact mesh query after FK has been evaluated."""

        self.pipeline.collide(self.state, self.contacts)
        self.newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.contacts,
            out_distance=self._distance,
            out_point0_world=None if probe_only else self._point0,
            out_point1_world=None if probe_only else self._point1,
        )
        contact_distance = self._distance
        wp.launch(
            _reset_query_outputs,
            dim=self.batch_size,
            inputs=[self.query_distance_m],
            outputs=[
                self._minimum_distance,
                self._winner,
                self._active,
                self._overflow,
            ],
            device=self.model.device,
        )
        if probe_only:
            wp.launch(
                _reset_probe_pair_outputs,
                dim=self.batch_size * self.semantic_pair_count,
                inputs=[self.query_distance_m],
                outputs=[
                    self._pair_distance,
                    self._pair_winner,
                    self._pair_shape_pair,
                    self._pair_normal,
                    self._pair_point0,
                    self._pair_point1,
                    self._pair_active,
                ],
                device=self.model.device,
            )
        else:
            wp.launch(
                _reset_pair_distances,
                dim=self.batch_size * self.semantic_pair_count,
                inputs=[self.query_distance_m],
                outputs=[self._pair_distance],
                device=self.model.device,
            )
        wp.launch(
            _reduce_minimum_distance,
            dim=self._contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self._contact_capacity,
                self.contacts.rigid_contact_shape0,
                self.model.shape_world,
                contact_distance,
                self._minimum_distance,
            ],
            device=self.model.device,
        )
        if not probe_only:
            wp.launch(
                _reduce_pair_distances,
                dim=self._contact_capacity,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self._contact_capacity,
                    self.source_shape_count,
                    self.semantic_pair_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.model.shape_world,
                    self._pair_lookup,
                    contact_distance,
                    self._pair_distance,
                ],
                device=self.model.device,
            )
            wp.launch(
                _reset_pair_winners,
                dim=self.batch_size * self.semantic_pair_count,
                outputs=[self._pair_winner],
                device=self.model.device,
            )
            wp.launch(
                _select_pair_winners,
                dim=self._contact_capacity,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self._contact_capacity,
                    self.source_shape_count,
                    self.semantic_pair_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.model.shape_world,
                    self._pair_lookup,
                    contact_distance,
                    self._pair_distance,
                    self._pair_winner,
                ],
                device=self.model.device,
            )
        wp.launch(
            _select_winner,
            dim=self._contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self._contact_capacity,
                self.contacts.rigid_contact_shape0,
                self.model.shape_world,
                contact_distance,
                self._minimum_distance,
                self._winner,
            ],
            device=self.model.device,
        )
        self._launch_overflow_check()
        if probe_only:
            wp.launch(
                _finalize_probe,
                dim=self.batch_size,
                inputs=[
                    self.query_distance_m,
                    self._winner,
                    self._overflow,
                    self._minimum_distance,
                ],
                outputs=[
                    self._shape_pair,
                    self._normal,
                    self._selected_point0,
                    self._selected_point1,
                    self._active,
                ],
                device=self.model.device,
            )
        else:
            wp.launch(
                _gather_winner,
                dim=self.batch_size,
                inputs=[
                    self.query_distance_m,
                    self._winner,
                    self._overflow,
                    contact_distance,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.rigid_contact_normal,
                    self._point0,
                    self._point1,
                ],
                outputs=[
                    self._minimum_distance,
                    self._shape_pair,
                    self._normal,
                    self._selected_point0,
                    self._selected_point1,
                    self._active,
                ],
                device=self.model.device,
            )
            wp.launch(
                _gather_pair_witnesses,
                dim=self.batch_size * self.semantic_pair_count,
                inputs=[
                    self._pair_winner,
                    self._overflow,
                    self.semantic_pair_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.rigid_contact_normal,
                    self._point0,
                    self._point1,
                ],
                outputs=[
                    self._pair_shape_pair,
                    self._pair_normal,
                    self._pair_point0,
                    self._pair_point1,
                    self._pair_active,
                ],
                device=self.model.device,
            )
        if compute_gradient:
            self.newton.eval_jacobian(
                self.model,
                self.state,
                J=self._body_jacobian,
                joint_S_s=self._joint_motion_subspace,
            )

    def decode_shape_pair(self, shape_a: int, shape_b: int) -> tuple[str, str]:
        """Decode one returned pair for diagnostics outside the solve path."""

        if shape_a < 0 or shape_b < 0:
            raise ValueError("inactive collision query has no shape pair")
        try:
            return (
                self._geometry_name_by_source_shape[shape_a % self.source_shape_count],
                self._geometry_name_by_source_shape[shape_b % self.source_shape_count],
            )
        except KeyError as error:
            raise ValueError(
                "shape pair does not belong to configured collision geometry"
            ) from error

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": (
                "newton_warp_convex_envelope"
                if self._convex_hull_mode
                else "newton_warp_exact_mesh"
            ),
            "newton_version": self.newton.__version__,
            "warp_version": wp.__version__,
            "device": str(self.model.device),
            "precision": "float32",
            "batch_size": self.batch_size,
            "collision_geometry_count_per_world": len(self.geometry_shape_groups),
            "collision_shape_count_per_world": len(self.geometry_records),
            "retained_pair_count_per_world": len(self.source_pairs),
            "semantic_pair_count_per_world": self.semantic_pair_count,
            "query_distance_m": self.query_distance_m,
            "contact_reduction": False,
            "contacts_per_world": self.contacts_per_world,
            "triangle_pairs_per_world": self.triangle_pairs_per_world,
            "deterministic": self.deterministic,
            "contact_sort_enabled": self.sort_contacts,
            "bulk_host_roundtrip_in_query": False,
            "scalar_input_validation_sync": True,
            "gradient_semantics": "analytic_witness_normal_tangent",
            "overflow_policy": "global_fail_closed_with_recursive_bucket_replay",
            "overflow_replay_bucket_size": self.overflow_replay_bucket_size,
            "last_replay_query_count": self.last_replay_query_count,
            "last_replay_rejected_world_count": (self.last_replay_rejected_world_count),
            "convex_envelope_certificate_supported": True,
            "model_derived": not self._legacy_panda,
            "convex_hull_mode": self._convex_hull_mode,
        }


class NewtonPandaCollisionQuery(NewtonModelCollisionQuery):
    """Backward-compatible wrapper for the validated Panda specialization."""

    def __init__(self, batch_size: int, urdf_path: Path, **kwargs: Any) -> None:
        super().__init__(batch_size, urdf_path, **kwargs)
