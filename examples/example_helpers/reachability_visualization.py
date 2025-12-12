#!/usr/bin/env python3
"""Viser helper for visualizing sampled reachability maps."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import viser

from . import reachability_constants as reach_consts


@dataclasses.dataclass
class ReachabilityMapData:
    points: np.ndarray
    metrics: Dict[str, np.ndarray]
    visitation: np.ndarray
    orientations_rpy: np.ndarray
    orientations_quat: np.ndarray
    visitation_lookup: Dict[Tuple[int, int, int], float]
    sample_xyz: np.ndarray
    sample_indices: np.ndarray
    sample_configs: np.ndarray
    sample_orientations: np.ndarray
    sample_quaternions: np.ndarray
    bucket_configs: Dict[Tuple[int, int, int], np.ndarray]
    bucket_orientations: Dict[Tuple[int, int, int], np.ndarray]
    bucket_quaternions: Dict[Tuple[int, int, int], np.ndarray]
    bucket_sample_indices: Dict[Tuple[int, int, int], np.ndarray]
    quantized_points: np.ndarray
    resolution: float
    lower_bounds: np.ndarray
    joint_names: Tuple[str, ...]
    path: Path
    metric_ranges: Dict[str, Tuple[float, float]]
    metric_order: Tuple[str, ...]
    metadata: Dict[str, float]
    robot_metadata: Dict[str, Any]


def _list_map_files(directory: Path) -> List[Path]:
    candidates: List[Path] = []
    if directory.exists():
        candidates.extend(sorted(directory.glob("3D_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(directory.glob("filt_3D_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True))
    return candidates


def _interpolate_colormap(weights: np.ndarray) -> np.ndarray:
    """Red (low score) -> Yellow -> Green (high score) gradient."""

    weights = np.clip(weights, 0.0, 1.0)
    colors = np.zeros((weights.shape[0], 3), dtype=np.float32)

    midpoint = reach_consts.GRADIENT_MIDPOINT
    lower_scale = 1.0 / max(midpoint, 1e-6)
    upper_scale = 1.0 / max(1.0 - midpoint, 1e-6)

    mid_mask = weights < midpoint
    if np.any(mid_mask):
        t = weights[mid_mask] * lower_scale
        colors[mid_mask, 0] = 1.0  # red remains max
        colors[mid_mask, 1] = t    # ramp up green channel
        colors[mid_mask, 2] = 0.0

    high_mask = ~mid_mask
    if np.any(high_mask):
        t = (weights[high_mask] - midpoint) * upper_scale
        colors[high_mask, 0] = 1.0 - t        # red fades out
        colors[high_mask, 1] = 1.0            # stay fully green
        colors[high_mask, 2] = 0.0
    return colors


def _rpy_to_matrix_batch(rpy_array: np.ndarray) -> np.ndarray:
    if rpy_array.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    rpy_values = np.asarray(rpy_array, dtype=np.float64)
    roll = rpy_values[:, 0]
    pitch = rpy_values[:, 1]
    yaw = rpy_values[:, 2]

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    rx = np.zeros((rpy_values.shape[0], 3, 3), dtype=np.float64)
    ry = np.zeros_like(rx)
    rz = np.zeros_like(rx)

    rx[:, 0, 0] = 1.0
    rx[:, 1, 1] = cr
    rx[:, 1, 2] = -sr
    rx[:, 2, 1] = sr
    rx[:, 2, 2] = cr

    ry[:, 0, 0] = cp
    ry[:, 0, 2] = sp
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0] = -sp
    ry[:, 2, 2] = cp

    rz[:, 0, 0] = cy
    rz[:, 0, 1] = -sy
    rz[:, 1, 0] = sy
    rz[:, 1, 1] = cy
    rz[:, 2, 2] = 1.0

    ry_rx = np.einsum("nij,njk->nik", ry, rx)
    rot = np.einsum("nij,njk->nik", rz, ry_rx)
    return rot.astype(np.float32, copy=False)


def _orientation_difference_angles(target_matrix: np.ndarray, batch_matrices: np.ndarray) -> np.ndarray:
    if batch_matrices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    rel = np.einsum("ij,njk->nik", target_matrix.T, batch_matrices)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_angle = 0.5 * (trace - 1.0)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle).astype(np.float32)


def _rpy_to_quaternion_batch(rpy_array: np.ndarray) -> np.ndarray:
    if rpy_array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rpy = np.asarray(rpy_array, dtype=np.float64)
    if rpy.ndim == 1:
        rpy = rpy.reshape(1, 3)

    half_roll = 0.5 * rpy[:, 0]
    half_pitch = 0.5 * rpy[:, 1]
    half_yaw = 0.5 * rpy[:, 2]

    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quats = np.stack((w, x, y, z), axis=1)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    quats = quats.astype(np.float32, copy=False)
    if np.any(valid):
        quats[valid] /= norms[valid]
    if np.any(~valid):
        quats[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quats


def _quaternion_difference_angles(target_quat: np.ndarray, batch_quats: np.ndarray) -> np.ndarray:
    if batch_quats.size == 0:
        return np.zeros((0,), dtype=np.float32)

    target = np.asarray(target_quat, dtype=np.float64).reshape(4)
    target_norm = np.linalg.norm(target)
    if target_norm < 1e-8:
        return np.full(batch_quats.shape[0], np.pi, dtype=np.float32)
    target /= target_norm

    batch = np.asarray(batch_quats, dtype=np.float64)
    if batch.ndim == 1:
        batch = batch.reshape(1, 4)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    normalized = np.zeros_like(batch)
    valid = norms[:, 0] > 1e-8
    if np.any(valid):
        normalized[valid] = batch[valid] / norms[valid]
    if np.any(~valid):
        normalized[~valid] = target
    dots = np.abs(np.einsum("ij,j->i", normalized, target))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return angles.astype(np.float32, copy=False)


LOGGER = logging.getLogger("reachability.visualization")


class ReachabilityMapViewer:
    """Utility that adds GUI controls to display 3D reachability maps in Viser."""

    def __init__(
        self,
        server: viser.ViserServer,
        base_directory: Path,
        *,
        gui_folder_name: str = "Reachability Visualization",
        scene_path: str = "/reachability_map",
        config_callback: Optional[Callable[[Tuple[str, ...], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], None]] = None,
        orientation_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> None:
        self._server = server
        self._base_directory = base_directory
        self._scene_path = scene_path.rstrip("/")
        self._config_callback = config_callback
        self._rng = np.random.default_rng()
        self._reference_orientation_provider = orientation_provider

        self._map_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._status_text: Optional[viser.GuiTextHandle] = None
        self._show_checkbox: Optional[viser.GuiCheckboxHandle] = None
        self._point_size_slider: Optional[viser.GuiSliderHandle] = None
        self._color_checkbox: Optional[viser.GuiCheckboxHandle] = None
        self._base_color_picker: Optional[viser.GuiRgbaHandle] = None
        self._min_score_slider: Optional[viser.GuiSliderHandle] = None
        self._metric_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._visitation_scale_slider: Optional[viser.GuiSliderHandle] = None
        self._z_slice_slider: Optional[viser.GuiSliderHandle] = None
        self._z_min_slider: Optional[viser.GuiSliderHandle] = None
        self._x_min_slider: Optional[viser.GuiSliderHandle] = None
        self._x_max_slider: Optional[viser.GuiSliderHandle] = None
        self._y_min_slider: Optional[viser.GuiSliderHandle] = None
        self._y_max_slider: Optional[viser.GuiSliderHandle] = None
        self._slice_rotation_slider: Optional[viser.GuiSliderHandle] = None
        self._slice_rotation_x_slider: Optional[viser.GuiSliderHandle] = None
        self._slice_rotation_y_slider: Optional[viser.GuiSliderHandle] = None
        self._slice_reset_button: Optional[viser.GuiButtonHandle] = None
        self._global_scores_text: Optional[viser.GuiTextHandle] = None
        self._global_scores_coverage_text: Optional[viser.GuiTextHandle] = None
        self._global_scores_metric_texts: Dict[str, viser.GuiTextHandle] = {}
        self._refresh_button: Optional[viser.GuiButtonHandle] = None
        self._load_button: Optional[viser.GuiButtonHandle] = None
        self._active_map_text: Optional[viser.GuiTextHandle] = None
        self._metadata_profile_text: Optional[viser.GuiTextHandle] = None
        self._metadata_profile_source_text: Optional[viser.GuiTextHandle] = None
        self._metadata_overrides_text: Optional[viser.GuiTextHandle] = None
        self._pick_checkbox: Optional[viser.GuiCheckboxHandle] = None
        self._validation_index_number: Optional[viser.GuiNumberHandle] = None
        self._validation_config_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._validation_visitation_text: Optional[viser.GuiTextHandle] = None
        self._validation_status: Optional[viser.GuiTextHandle] = None
        self._validation_update_guard = False
        self._validation_config_mapping: Dict[str, int] = {}
        self._validation_selected_index: Optional[int] = None
        self._validation_current_index: Optional[int] = None
        self._validation_engaged: bool = False
        self._last_bucket_key: Optional[Tuple[int, int, int]] = None
        self._orientation_filter_checkbox: Optional[viser.GuiCheckboxHandle] = None
        self._orientation_tolerance_slider: Optional[viser.GuiSliderHandle] = None
        self._reference_orientation_quat: Optional[np.ndarray] = None
        self._orientation_reference_locked: bool = False
        self._metric_label_to_key: Dict[str, str] = {"Manipulability": "Manipulability", "Visitation": "Visitation"}
        self._orientation_debounce_timer: Optional[threading.Timer] = None
        self._orientation_filtered_counts: Dict[Tuple[int, int, int], int] = {}

        self._point_cloud_handle: Optional[viser.PointCloudHandle] = None
        self._picker_handle: Optional[Any] = None
        self._current_picker_indices: Optional[np.ndarray] = None
        self._picker_positions_cache: Optional[np.ndarray] = None
        self._picker_keys_cache: Optional[np.ndarray] = None
        self._validation_point_handle: Optional[viser.PointCloudHandle] = None
        self._debug_target_handle: Optional[viser.PointCloudHandle] = None
        self._map_cache: Dict[str, ReachabilityMapData] = {}
        self._current_map_id: Optional[str] = None

        self._create_gui(gui_folder_name)
        self.refresh_options()

    # ------------------------------------------------------------------ GUI setup

    def _create_gui(self, folder_name: str) -> None:
        base_path = self._base_directory
        base_str = str(base_path)
        with self._server.gui.add_folder(folder_name, expand_by_default=False):
            self._status_text = self._server.gui.add_text("Viewer status", initial_value="Status: idle")
            self._active_map_text = self._server.gui.add_text("Active map", initial_value="Map: --")
            self._server.gui.add_text("Map directory", initial_value=base_str)

            controls_folder = self._server.gui.add_folder("Map Selection", expand_by_default=True)
            with controls_folder:
                self._map_dropdown = self._server.gui.add_dropdown("Available maps", options=("-- no maps --",), initial_value="-- no maps --")
                self._refresh_button = self._server.gui.add_button("Refresh map list")
                self._load_button = self._server.gui.add_button("Load selected map")

                metadata_folder = self._server.gui.add_folder("Robot Metadata", expand_by_default=False)
                with metadata_folder:
                    self._metadata_profile_text = self._server.gui.add_text(
                        "Limit profile",
                        initial_value="Limit profile: --",
                    )
                    self._metadata_profile_source_text = self._server.gui.add_text(
                        "Profile source",
                        initial_value="Profile source: --",
                    )
                    self._metadata_overrides_text = self._server.gui.add_text(
                        "Overrides",
                        initial_value="Overrides: --",
                    )

            display_folder = self._server.gui.add_folder("Display Controls", expand_by_default=True)
            with display_folder:
                self._show_checkbox = self._server.gui.add_checkbox("Show map", initial_value=False)
                self._metric_dropdown = self._server.gui.add_dropdown(
                    "Score metric", options=("Manipulability", "Visitation"), initial_value="Manipulability"
                )
                self._point_size_slider = self._server.gui.add_slider(
                    "Point size (cm)",
                    min=reach_consts.POINT_SIZE_SLIDER_MIN_CM,
                    max=reach_consts.POINT_SIZE_SLIDER_MAX_CM,
                    step=reach_consts.POINT_SIZE_SLIDER_STEP_CM,
                    initial_value=reach_consts.POINT_SIZE_DEFAULT_CM,
                )
                self._color_checkbox = self._server.gui.add_checkbox("Color by score", initial_value=True)
                self._base_color_picker = self._server.gui.add_rgba(
                    "Base color", initial_value=reach_consts.DEFAULT_BASE_COLOR
                )
                self._min_score_slider = self._server.gui.add_slider(
                    "Min normalized score",
                    min=reach_consts.MIN_SCORE_SLIDER_MIN,
                    max=reach_consts.MIN_SCORE_SLIDER_MAX,
                    step=reach_consts.MIN_SCORE_SLIDER_STEP,
                    initial_value=reach_consts.MIN_SCORE_DEFAULT,
                )
                self._visitation_scale_slider = self._server.gui.add_slider(
                    "Visitation scale (counts)",
                    min=reach_consts.VISITATION_SCALE_MIN,
                    max=reach_consts.VISITATION_SCALE_MAX,
                    step=reach_consts.VISITATION_SCALE_STEP,
                    initial_value=reach_consts.VISITATION_SCALE_DEFAULT,
                )
                self._visitation_scale_slider.disabled = True
                self._pick_checkbox = self._server.gui.add_checkbox(
                    "Enable sample picking", initial_value=False
                )
                self._pick_checkbox.disabled = True

                # Slice view controls - organized by axis
                # Z-axis (height)
                self._z_min_slider = self._server.gui.add_slider(
                    "Z: Min height (m)",
                    min=reach_consts.Z_SLICE_MIN_M,
                    max=reach_consts.Z_SLICE_MAX_M,
                    step=reach_consts.Z_SLICE_STEP_M,
                    initial_value=reach_consts.Z_SLICE_MIN_M,
                )
                self._z_slice_slider = self._server.gui.add_slider(
                    "Z: Max height (m)",
                    min=reach_consts.Z_SLICE_MIN_M,
                    max=reach_consts.Z_SLICE_MAX_M,
                    step=reach_consts.Z_SLICE_STEP_M,
                    initial_value=reach_consts.Z_SLICE_DEFAULT_M,
                )
                self._slice_rotation_slider = self._server.gui.add_slider(
                    "Z: Rotation (deg)",
                    min=-90.0,
                    max=90.0,
                    step=1.0,
                    initial_value=0.0,
                )

                # X-axis (distance)
                self._x_min_slider = self._server.gui.add_slider(
                    "X: Min distance (m)",
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    initial_value=-2.0,
                )
                self._x_max_slider = self._server.gui.add_slider(
                    "X: Max distance (m)",
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    initial_value=2.0,
                )
                self._slice_rotation_x_slider = self._server.gui.add_slider(
                    "X: Rotation (deg)",
                    min=-90.0,
                    max=90.0,
                    step=1.0,
                    initial_value=0.0,
                )

                # Y-axis (distance)
                self._y_min_slider = self._server.gui.add_slider(
                    "Y: Min distance (m)",
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    initial_value=-2.0,
                )
                self._y_max_slider = self._server.gui.add_slider(
                    "Y: Max distance (m)",
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    initial_value=2.0,
                )
                self._slice_rotation_y_slider = self._server.gui.add_slider(
                    "Y: Rotation (deg)",
                    min=-90.0,
                    max=90.0,
                    step=1.0,
                    initial_value=0.0,
                )

                # Reset button for slice view controls
                self._slice_reset_button = self._server.gui.add_button("Reset slice view")

            global_scores_folder = self._server.gui.add_folder("Global Scores", expand_by_default=True)
            with global_scores_folder:
                # Coverage text (always shown for IK maps)
                self._global_scores_coverage_text = self._server.gui.add_text(
                    "Coverage",
                    initial_value="--"
                )
                # Create text widgets for common metrics upfront
                self._global_scores_metric_texts = {
                    "Manipulability": self._server.gui.add_text("Manipulability", initial_value="--"),
                    "RangeOfMotion": self._server.gui.add_text("RangeOfMotion", initial_value="--"),
                    "SingularityAvoidance": self._server.gui.add_text("SingularityAvoidance", initial_value="--"),
                }

            orientation_folder = self._server.gui.add_folder("Orientation Filter", expand_by_default=False)
            with orientation_folder:
                self._orientation_filter_checkbox = self._server.gui.add_checkbox(
                    "Enable orientation filter", initial_value=False
                )
                self._orientation_tolerance_slider = self._server.gui.add_slider(
                    "Tolerance (deg)",
                    min=reach_consts.ORIENTATION_TOLERANCE_MIN_DEG,
                    max=reach_consts.ORIENTATION_TOLERANCE_MAX_DEG,
                    step=reach_consts.ORIENTATION_TOLERANCE_STEP_DEG,
                    initial_value=reach_consts.ORIENTATION_TOLERANCE_DEFAULT_DEG,
                )
                if self._orientation_tolerance_slider:
                    self._orientation_tolerance_slider.disabled = True

            validation_folder = self._server.gui.add_folder("Sample Validation", expand_by_default=False)
            with validation_folder:
                self._validation_index_number = self._server.gui.add_number(
                    "Sample index",
                    reach_consts.VALIDATION_SAMPLE_INDEX_DEFAULT,
                    step=reach_consts.VALIDATION_SAMPLE_INDEX_STEP,
                    disabled=True,
                )
                self._validation_config_dropdown = self._server.gui.add_dropdown(
                    "Config for bucket",
                    options=("-- none --",),
                    initial_value="-- none --",
                    disabled=True,
                )
                self._validation_visitation_text = self._server.gui.add_text(
                    "Bucket visitation", initial_value="Visitation: --"
                )
                self._validation_status = self._server.gui.add_text(
                    "Validation status", initial_value="No sample selected"
                )

            if self._refresh_button:
                @self._refresh_button.on_click
                def _refresh(_event) -> None:
                    self.refresh_options()

            if self._load_button:
                @self._load_button.on_click
                def _load(_event) -> None:
                    self.load_selected_map()

            if self._show_checkbox:
                @self._show_checkbox.on_update
                def _update_show(_event) -> None:
                    # Uncheck "Enable sample picking" when "Show map" is unchecked
                    if not self._show_checkbox.value and self._pick_checkbox:
                        self._pick_checkbox.value = False
                    self._request_scene_refresh()

            if self._point_size_slider:
                @self._point_size_slider.on_update
                def _update_size(_event) -> None:
                    self._request_scene_refresh()

            if self._color_checkbox:
                @self._color_checkbox.on_update
                def _update_color_mode(_event) -> None:
                    self._request_scene_refresh()

            if self._base_color_picker:
                @self._base_color_picker.on_update
                def _update_base_color(_event) -> None:
                    self._request_scene_refresh()

        if self._min_score_slider:
            @self._min_score_slider.on_update
            def _update_min_score(_event) -> None:
                self._request_scene_refresh()

            if self._orientation_filter_checkbox:
                @self._orientation_filter_checkbox.on_update
                def _update_orientation_filter(_event) -> None:
                    enabled = bool(self._orientation_filter_checkbox.value)
                    if self._orientation_tolerance_slider is not None:
                        self._orientation_tolerance_slider.disabled = not enabled
                    if enabled:
                        if not self._update_reference_orientation(force_refresh=True):
                            if self._orientation_tolerance_slider is not None:
                                self._orientation_tolerance_slider.disabled = True
                            self._orientation_filter_checkbox.value = False
                            self._set_status("Status: orientation filter requires pose data; disabled")
                            return
                        self._orientation_reference_locked = True
                        self._schedule_orientation_update(delay=0.05)
                    else:
                        self._orientation_reference_locked = False
                        self._reference_orientation_quat = None
                        self._cancel_orientation_timer()
                        self._update_scene()

        if self._orientation_tolerance_slider:
            @self._orientation_tolerance_slider.on_update
            def _update_orientation_tolerance(_event) -> None:
                if self._orientation_filter_checkbox and self._orientation_filter_checkbox.value:
                    self._schedule_orientation_update()

        if self._metric_dropdown:
            @self._metric_dropdown.on_update
            def _update_metric(_event) -> None:
                if self._visitation_scale_slider:
                    current_label = self._metric_dropdown.value
                    current_key = self._metric_label_to_key.get(current_label, current_label)
                    self._visitation_scale_slider.disabled = current_key != "Visitation"
                self._request_scene_refresh()

            if self._visitation_scale_slider:
                @self._visitation_scale_slider.on_update
                def _update_vis_scale(_event) -> None:
                    self._request_scene_refresh()

            if self._pick_checkbox:
                @self._pick_checkbox.on_update
                def _update_pick_enabled(_event) -> None:
                    pick_enabled = bool(self._pick_checkbox.value)
                    if not pick_enabled and self._picker_handle is not None:
                        self._picker_handle.remove()
                        self._picker_handle = None
                        self._current_picker_indices = None
                        self._picker_positions_cache = None
                        self._picker_keys_cache = None
                    if not pick_enabled:
                        self._validation_engaged = False
                    if self._validation_index_number is not None:
                        self._validation_index_number.disabled = True
                    if self._validation_visitation_text is not None:
                        self._validation_visitation_text.value = "Visitation: --"
                    if self._validation_config_dropdown:
                        disable_dropdown = (not pick_enabled and not self._orientation_filter_active()) or not self._validation_config_mapping
                        if not disable_dropdown and len(self._validation_config_mapping) <= 1 and not self._orientation_filter_active():
                            disable_dropdown = True
                        self._validation_config_dropdown.disabled = disable_dropdown
                        if disable_dropdown:
                            self._validation_config_dropdown.options = ("-- none --",)
                            self._validation_config_dropdown.value = "-- none --"
                            self._validation_selected_index = None
                    if not pick_enabled and self._validation_point_handle is not None:
                        self._validation_point_handle.remove()
                        self._validation_point_handle = None
                    self._request_scene_refresh()

            if self._z_slice_slider:
                @self._z_slice_slider.on_update
                def _update_z_max(_event) -> None:
                    self._request_scene_refresh()
            if self._z_min_slider:
                @self._z_min_slider.on_update
                def _update_z_min(_event) -> None:
                    self._request_scene_refresh()
            if self._x_min_slider:
                @self._x_min_slider.on_update
                def _update_x_min(_event) -> None:
                    self._request_scene_refresh()
            if self._x_max_slider:
                @self._x_max_slider.on_update
                def _update_x_max(_event) -> None:
                    self._request_scene_refresh()
            if self._y_min_slider:
                @self._y_min_slider.on_update
                def _update_y_min(_event) -> None:
                    self._request_scene_refresh()
            if self._y_max_slider:
                @self._y_max_slider.on_update
                def _update_y_max(_event) -> None:
                    self._request_scene_refresh()
            if self._slice_rotation_slider:
                @self._slice_rotation_slider.on_update
                def _update_slice_rotation(_event) -> None:
                    self._request_scene_refresh()

            if self._slice_rotation_x_slider:
                @self._slice_rotation_x_slider.on_update
                def _update_slice_rotation_x(_event) -> None:
                    self._request_scene_refresh()

            if self._slice_rotation_y_slider:
                @self._slice_rotation_y_slider.on_update
                def _update_slice_rotation_y(_event) -> None:
                    self._request_scene_refresh()

            if self._slice_reset_button:
                @self._slice_reset_button.on_click
                def _reset_slice_view(_event) -> None:
                    """Reset all slice view sliders to default values."""
                    if self._z_min_slider:
                        self._z_min_slider.value = reach_consts.Z_SLICE_MIN_M
                    if self._z_slice_slider:
                        self._z_slice_slider.value = reach_consts.Z_SLICE_DEFAULT_M
                    if self._slice_rotation_slider:
                        self._slice_rotation_slider.value = 0.0
                    if self._x_min_slider:
                        self._x_min_slider.value = -2.0
                    if self._x_max_slider:
                        self._x_max_slider.value = 2.0
                    if self._slice_rotation_x_slider:
                        self._slice_rotation_x_slider.value = 0.0
                    if self._y_min_slider:
                        self._y_min_slider.value = -2.0
                    if self._y_max_slider:
                        self._y_max_slider.value = 2.0
                    if self._slice_rotation_y_slider:
                        self._slice_rotation_y_slider.value = 0.0
                    self._request_scene_refresh()

            if self._validation_index_number:
                @self._validation_index_number.on_update
                def _update_validation_index(_event) -> None:
                    if self._validation_update_guard:
                        return
                    self._request_scene_refresh()

        if self._validation_config_dropdown:
            @self._validation_config_dropdown.on_update
            def _on_validation_config_select(_event) -> None:
                if self._validation_update_guard:
                    return
                if self._validation_config_dropdown is None:
                    return
                data = self._current_map()
                if data is None or not data.sample_configs.size or not self._validation_active():
                    return
                label = str(self._validation_config_dropdown.value)
                idx = self._validation_config_mapping.get(label)
                if idx is None:
                    self._validation_selected_index = None
                    return
                self._validation_selected_index = idx
                base_idx = self._validation_current_index
                if base_idx is None:
                    base_idx = self._get_validation_index(data)
                applied_idx = self._apply_validation_configuration(data, base_idx)
                if base_idx is not None and 0 <= base_idx < data.sample_xyz.shape[0]:
                    point = data.sample_xyz[base_idx].reshape(1, 3)
                else:
                    point = np.zeros((1, 3), dtype=np.float32)
                configs_available = len(self._validation_config_mapping) or 1
                bucket_visitation = None
                bucket_key_tuple: Optional[Tuple[int, int, int]] = None
                if data.sample_indices.size:
                    bucket_key_tuple = tuple(int(v) for v in data.sample_indices[base_idx])
                    if data.visitation_lookup:
                        value = data.visitation_lookup.get(bucket_key_tuple)
                        if value is not None:
                            bucket_visitation = int(round(value))
                self._set_validation_status_message(
                    base_idx, point, configs_available, applied_idx, bucket_visitation
                )
                if self._validation_visitation_text is not None:
                    if bucket_visitation is None:
                        self._validation_visitation_text.value = "Visitation: --"
                    else:
                        self._validation_visitation_text.value = (
                            f"Visitation: {bucket_visitation}"
                        )
                self._last_bucket_key = bucket_key_tuple

    # ------------------------------------------------------------------ Public API

    def _update_reference_orientation(
        self, *, force_refresh: bool = False, allow_fallback: bool = True
    ) -> bool:
        if self._orientation_reference_locked and not force_refresh:
            return self._reference_orientation_quat is not None

        provider_quat: Optional[np.ndarray] = None
        if self._reference_orientation_provider is not None and (force_refresh or not self._orientation_reference_locked):
            try:
                provider_quat = self._reference_orientation_provider()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("reachviz: orientation provider error: %s", exc)
                provider_quat = None
        if provider_quat is not None:
            quat = np.asarray(provider_quat, dtype=np.float32)
            if quat.size == 4:
                norm = float(np.linalg.norm(quat))
                if norm > 1e-8:
                    self._reference_orientation_quat = quat / norm
                    return True

        if not allow_fallback and self._reference_orientation_quat is not None:
            return True

        data = self._current_map()
        if data is None:
            return self._reference_orientation_quat is not None

        quat_source: Optional[np.ndarray] = None
        if getattr(data, "orientations_quat", np.zeros((0, 4))).size:
            quat_source = data.orientations_quat[:1]
        elif getattr(data, "orientations_rpy", np.zeros((0, 3))).size:
            quat_source = _rpy_to_quaternion_batch(data.orientations_rpy[:1])
        if quat_source is None or quat_source.size == 0:
            return self._reference_orientation_quat is not None
        quat = np.asarray(quat_source[0], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return self._reference_orientation_quat is not None
        self._reference_orientation_quat = quat / norm
        return True

    def refresh_options(self) -> None:
        options = _list_map_files(self._base_directory)
        option_labels: List[str] = []
        lookup: Dict[str, ReachabilityMapData] = {}
        for path in options:
            label = path.name
            option_labels.append(label)
            # keep cache entries to avoid re-loading
            if label in self._map_cache:
                lookup[label] = self._map_cache[label]

        if self._map_dropdown:
            if option_labels:
                self._map_dropdown.options = option_labels
                current = self._map_dropdown.value
                if current not in option_labels:
                    self._map_dropdown.value = option_labels[0]
            else:
                self._map_dropdown.options = ("-- no maps --",)
                self._map_dropdown.value = "-- no maps --"

        # purge cache entries that are no longer available
        for key in list(self._map_cache.keys()):
            if key not in option_labels:
                self._map_cache.pop(key)

        if self._status_text:
            if option_labels:
                self._status_text.value = f"Status: {len(option_labels)} map(s) found"
            else:
                self._status_text.value = "Status: no maps found"

    def load_selected_map(self) -> None:
        if not self._map_dropdown:
            return
        label = self._map_dropdown.value
        LOGGER.info("reachviz/load: requested label=%s", label)
        if label is None or label == "-- no maps --":
            self._set_status("Select a map to load.")
            self._update_metadata_display(None)
            return

        path = self._base_directory / label
        if not path.exists():
            self._set_status(f"File not found: {path}")
            self.refresh_options()
            self._update_metadata_display(None)
            return

        data = self._map_cache.get(label)
        if data is None:
            try:
                data = self._load_map(path)
                self._map_cache[label] = data
            except Exception as exc:  # pragma: no cover - user-facing failure
                self._set_status(f"Failed to load map: {exc}")
                self._update_metadata_display(None)
                return

        self._current_map_id = label
        LOGGER.debug(
            "reachviz/load: loaded map '%s' with %d points", label, data.points.shape[0]
        )
        # Update global scores display after loading
        self._update_global_scores_display()
        self._update_metadata_display(data)

        # Get available metrics from map data
        available_metrics = list(data.metric_order) if data.metric_order else list(data.metrics.keys())
        if not available_metrics:
            available_metrics = ["Manipulability"]

        LOGGER.info(
            "Reachability map '%s' loaded with metrics: %s (from metric_order: %s, metrics.keys: %s)",
            label,
            available_metrics,
            data.metric_order,
            list(data.metrics.keys()) if data.metrics else []
        )
        if self._metric_dropdown:
            label_mapping: Dict[str, str] = {}
            metric_labels: List[str] = []
            for metric_key in available_metrics:
                if metric_key == "Visitation":
                    continue
                label = metric_key
                if label and label.lower() not in {"manipulability", "visitation"}:
                    label = "".join((" " + c if c.isupper() else c for c in metric_key)).strip().title()
                else:
                    label = metric_key.title()
                label_mapping[label] = metric_key
                metric_labels.append(label)
            visitation_label = "Visitation"
            label_mapping[visitation_label] = "Visitation"
            metric_labels.append(visitation_label)

            self._metric_label_to_key = label_mapping
            self._metric_dropdown.options = tuple(metric_labels)
            if self._metric_dropdown.value not in metric_labels:
                self._metric_dropdown.value = metric_labels[0]
        if self._visitation_scale_slider:
            current_label = self._metric_dropdown.value if self._metric_dropdown else None
            current_key = self._metric_label_to_key.get(current_label, current_label)
            self._visitation_scale_slider.disabled = current_key != "Visitation"
        if self._min_score_slider:
            self._min_score_slider.value = reach_consts.MIN_SCORE_DEFAULT
        if self._z_slice_slider:
            self._z_slice_slider.value = float(np.max(data.points[:, 2]))
            self._z_slice_slider.min = float(np.min(data.points[:, 2]))
            self._z_slice_slider.max = float(np.max(data.points[:, 2]))
            if self._z_min_slider:
                self._z_min_slider.max = float(np.max(data.points[:, 2]))
                self._z_min_slider.min = float(np.min(data.points[:, 2]))
            # Update x sliders based on data bounds
            x_min_data = float(np.min(data.points[:, 0]))
            x_max_data = float(np.max(data.points[:, 0]))
            if self._x_min_slider:
                self._x_min_slider.min = x_min_data - 0.5
                self._x_min_slider.max = x_max_data + 0.5
            if self._x_max_slider:
                self._x_max_slider.min = x_min_data - 0.5
                self._x_max_slider.max = x_max_data + 0.5
            # Update y sliders based on data bounds
            y_min_data = float(np.min(data.points[:, 1]))
            y_max_data = float(np.max(data.points[:, 1]))
            if self._y_min_slider:
                self._y_min_slider.min = y_min_data - 0.5
                self._y_min_slider.max = y_max_data + 0.5
            if self._y_max_slider:
                self._y_max_slider.min = y_min_data - 0.5
                self._y_max_slider.max = y_max_data + 0.5
        if self._visitation_scale_slider:
            max_vis = float(np.max(data.visitation)) if data.visitation.size else 1.0
            self._visitation_scale_slider.max = max(reach_consts.VISITATION_SCALE_MIN, max_vis)
            self._visitation_scale_slider.value = min(
                self._visitation_scale_slider.value, self._visitation_scale_slider.max
            )
            self._visitation_scale_slider.disabled = not (
                self._metric_dropdown and self._metric_dropdown.value == "Visitation"
            )
        if self._pick_checkbox:
            has_samples = data.sample_configs.size > 0 and len(data.bucket_configs) > 0
            self._pick_checkbox.disabled = not has_samples
            if not has_samples:
                self._pick_checkbox.value = False
        if self._validation_index_number:
            has_validation_samples = data.sample_configs.size > 0
            validation_active = self._validation_active() and has_validation_samples
            if not has_validation_samples:
                self._validation_index_number.disabled = True
                if self._validation_config_dropdown:
                    self._validation_config_dropdown.options = ("-- none --",)
                    self._validation_config_dropdown.value = "-- none --"
                    self._validation_config_dropdown.disabled = True
                    self._validation_config_mapping = {}
                    self._validation_selected_index = None
                if self._validation_status:
                    self._validation_status.value = "No sample selected"
                if self._validation_point_handle is not None:
                    self._validation_point_handle.remove()
                    self._validation_point_handle = None
            else:
                max_index = max(0, data.sample_configs.shape[0] - 1)
                if hasattr(self._validation_index_number, "min"):
                    self._validation_index_number.min = reach_consts.VALIDATION_SAMPLE_INDEX_DEFAULT
                if hasattr(self._validation_index_number, "max"):
                    self._validation_index_number.max = float(max_index)
                current_val = float(self._validation_index_number.value)
                current_idx = int(round(current_val))
                current_idx = max(0, min(current_idx, max_index))
                self._validation_index_number.value = float(current_idx)
                self._validation_index_number.disabled = not validation_active
                if self._validation_config_dropdown:
                    self._update_validation_config_dropdown(data, current_idx)
                    dropdown_disable = not validation_active
                    if (
                        not dropdown_disable
                        and len(self._validation_config_mapping) <= 1
                        and not self._orientation_filter_active()
                    ):
                        dropdown_disable = True
                    self._validation_config_dropdown.disabled = dropdown_disable
                if self._validation_status:
                    self._validation_status.value = "Select a sample index to preview"
        if self._active_map_text:
            self._active_map_text.value = f"Map: {label}"
        self._last_bucket_key = None
        if self._orientation_filter_checkbox is not None:
            self._orientation_filter_checkbox.value = False
        if self._orientation_tolerance_slider is not None:
            self._orientation_tolerance_slider.disabled = True
        self._reference_orientation_quat = None
        if self._validation_visitation_text is not None:
            self._validation_visitation_text.value = "Visitation: --"
        if self._show_checkbox and not self._show_checkbox.value:
            self._set_status("Status: map loaded (hidden)")
        else:
            self._set_status("Status: map loaded")
        LOGGER.info("reachviz: invoking _update_scene after load")
        self._update_scene()

    def _update_global_scores_display(self) -> None:
        """Update global scores display - simple text format."""
        data = self._current_map()
        if data is None:
            if self._global_scores_coverage_text:
                self._global_scores_coverage_text.value = "No map loaded"
            for text_handle in self._global_scores_metric_texts.values():
                text_handle.value = "--"
            return

        # Get global scores from metadata
        global_scores = data.metadata.get("_global_scores")
        if global_scores is None:
            if self._global_scores_coverage_text:
                self._global_scores_coverage_text.value = "No global scores available"
            for text_handle in self._global_scores_metric_texts.values():
                text_handle.value = "--"
            return

        # Handle IK format (has "coverage" and "metrics")
        # Note: The saved format flattens metrics to top level, so check both structures
        if "coverage" in global_scores:
            cov = global_scores["coverage"]
            if self._global_scores_coverage_text:
                self._global_scores_coverage_text.value = (
                    f"{cov.get('coverage_ratio', 0.0):.1%} "
                    f"({cov.get('valid_count', 0)}/{cov.get('total_count', 0)})"
                )

            # Check if metrics are nested under "metrics" key or at top level
            metrics_dict = global_scores.get("metrics", {})
            if not metrics_dict:
                # Metrics might be at top level (flattened structure from saving)
                # Filter out "coverage" and any non-dict values
                metrics_dict = {
                    k: v for k, v in global_scores.items()
                    if k != "coverage" and isinstance(v, dict)
                }

            # Create text widgets for metrics that don't exist yet
            for metric_name in metrics_dict.keys():
                if metric_name not in self._global_scores_metric_texts:
                    # Create new text widget for this metric
                    self._global_scores_metric_texts[metric_name] = (
                        self._server.gui.add_text(
                            metric_name,
                            initial_value="--"
                        )
                    )

            # Update existing metric text widgets
            for metric_name, scores in metrics_dict.items():
                avg = scores.get("average", 0.0)
                if metric_name in self._global_scores_metric_texts:
                    self._global_scores_metric_texts[metric_name].value = f"{avg:.6f}"
                else:
                    # Metric not in our predefined widgets, skip it
                    LOGGER.debug("Metric '%s' not in predefined widgets, skipping", metric_name)

            # Hide/remove text widgets for metrics that no longer exist
            metrics_to_remove = [
                name for name in self._global_scores_metric_texts.keys()
                if name not in metrics_dict
            ]
            for metric_name in metrics_to_remove:
                # Note: Viser doesn't have a remove method, so we'll just hide by setting to empty
                self._global_scores_metric_texts[metric_name].value = "--"
        else:
            # Handle FK format (direct metric dictionary)
            if self._global_scores_coverage_text:
                self._global_scores_coverage_text.value = "--"

            # Create text widgets for metrics that don't exist yet
            for metric_name in global_scores.keys():
                if isinstance(global_scores[metric_name], dict):
                    if metric_name not in self._global_scores_metric_texts:
                        # Create new text widget for this metric
                        self._global_scores_metric_texts[metric_name] = (
                            self._server.gui.add_text(
                                metric_name,
                                initial_value="--"
                            )
                        )

            # Update existing metric text widgets
            for metric_name, scores in global_scores.items():
                if isinstance(scores, dict):
                    avg = scores.get("average", 0.0)
                    if metric_name in self._global_scores_metric_texts:
                        self._global_scores_metric_texts[metric_name].value = f"{avg:.6f}"
                    else:
                        # Metric not in our predefined widgets, skip it
                        LOGGER.debug("Metric '%s' not in predefined widgets, skipping", metric_name)

            # Hide metrics that no longer exist
            metrics_to_remove = [
                name for name in self._global_scores_metric_texts.keys()
                if name not in global_scores or not isinstance(global_scores.get(name), dict)
            ]
            for metric_name in metrics_to_remove:
                self._global_scores_metric_texts[metric_name].value = "--"

    def _update_metadata_display(self, data: Optional[ReachabilityMapData] = None) -> None:
        if self._metadata_profile_text is None:
            return

        profile_line = "Limit profile: --"
        source_line = "Profile source: --"
        overrides_line = "Overrides: --"

        if data is None:
            data = self._current_map()

        if data and isinstance(data.robot_metadata, dict):
            metadata = data.robot_metadata
            limit_info = metadata.get("limit_profile")
            selection = metadata.get("limit_profile_selection")

            if isinstance(limit_info, dict):
                summary = limit_info.get("summary")
                profile_name = str(limit_info.get("profile", "baseline"))
                overrides_dict = limit_info.get("overrides")
                enabled = bool(limit_info.get("enabled", bool(overrides_dict)))

                if not summary:
                    override_count = len(overrides_dict) if isinstance(overrides_dict, dict) else 0
                    if enabled:
                        summary = f"{profile_name} (enabled, {override_count} joint{'s' if override_count != 1 else ''})"
                    else:
                        summary = f"{profile_name} (baseline)"

                if selection and isinstance(selection, str) and selection != profile_name:
                    summary = f"{summary} (selected: {selection})"
                profile_line = f"Limit profile: {summary}"

                source = limit_info.get("source")
                source_line = f"Profile source: {source}" if source else "Profile source: --"

                if isinstance(overrides_dict, dict) and overrides_dict:
                    formatted_entries: List[str] = []
                    for joint, bounds in sorted(overrides_dict.items()):
                        min_val: Optional[float]
                        max_val: Optional[float]
                        if isinstance(bounds, dict):
                            min_val = bounds.get("min")
                            max_val = bounds.get("max")
                        elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                            min_val, max_val = bounds[0], bounds[1]
                        else:
                            min_val = max_val = None
                        if min_val is None or max_val is None:
                            formatted_entries.append(f"{joint}: {bounds}")
                        else:
                            formatted_entries.append(f"{joint}: [{float(min_val):.3f}, {float(max_val):.3f}]")
                    max_entries = 3
                    if len(formatted_entries) > max_entries:
                        extra = len(formatted_entries) - max_entries
                        visible = ", ".join(formatted_entries[:max_entries])
                        overrides_line = f"Overrides: {visible}, +{extra} more"
                    else:
                        overrides_line = "Overrides: " + ", ".join(formatted_entries)
                else:
                    overrides_line = "Overrides: (none)"
            else:
                summary = metadata.get("limit_profile_summary")
                if summary:
                    profile_line = f"Limit profile: {summary}"
                elif selection:
                    profile_line = f"Limit profile: selection={selection}"
                limit_overrides = metadata.get("limit_overrides")
                if isinstance(limit_overrides, dict) and limit_overrides:
                    formatted_entries = [
                        f"{joint}: [{float(bounds.get('min')):.3f}, {float(bounds.get('max')):.3f}]"
                        for joint, bounds in limit_overrides.items()
                        if isinstance(bounds, dict) and "min" in bounds and "max" in bounds
                    ]
                    if formatted_entries:
                        overrides_line = "Overrides: " + ", ".join(formatted_entries[:3])
                        if len(formatted_entries) > 3:
                            overrides_line += f", +{len(formatted_entries) - 3} more"

        self._metadata_profile_text.value = profile_line
        if self._metadata_profile_source_text:
            self._metadata_profile_source_text.value = source_line
        if self._metadata_overrides_text:
            self._metadata_overrides_text.value = overrides_line

    def on_new_map(self, map_path: Path) -> None:
        """Notify the viewer that a new map was generated."""

        map_path = map_path.resolve()
        if map_path.parent != self._base_directory.resolve():
            return

        self.refresh_options()
        label = map_path.name
        if self._map_dropdown:
            self._map_dropdown.value = label
        self.load_selected_map()

    # ------------------------------------------------------------------ Internal helpers

    def report_generation_status(
        self, stage: str, *, progress: Optional[float] = None, detail: Optional[str] = None
    ) -> None:
        label = stage.replace("_", " ").strip()
        if not label:
            label = "idle"
        label = label.title()
        parts = [label]
        if progress is not None and np.isfinite(progress):
            parts.append(f"{progress * 100.0:.1f}%")
        if detail:
            parts.append(detail)
        message = "Status: " + " | ".join(parts)
        self._set_status(message)

    def _set_status(self, text: str) -> None:
        if self._status_text:
            self._status_text.value = text

    def _update_validation_config_dropdown(self, data: ReachabilityMapData, base_index: int) -> None:
        if self._orientation_filter_checkbox and self._orientation_filter_checkbox.value:
            self._update_reference_orientation()
        if self._validation_config_dropdown is None:
            return
        if not self._validation_engaged:
            self._validation_config_dropdown.options = ("-- none --",)
            self._validation_config_dropdown.value = "-- none --"
            self._validation_config_dropdown.disabled = True
            self._validation_config_mapping = {}
            self._validation_selected_index = None
            return
        if base_index < 0 or base_index >= data.sample_configs.shape[0]:
            self._validation_config_dropdown.options = ("-- none --",)
            self._validation_config_dropdown.value = "-- none --"
            self._validation_config_dropdown.disabled = True
            self._validation_config_mapping = {}
            self._validation_selected_index = None
            return

        bucket_key = tuple(int(v) for v in data.sample_indices[base_index])
        bucket_indices = self._get_bucket_sample_indices(data, bucket_key)
        if bucket_indices is None or not bucket_indices.size:
            self._validation_config_dropdown.options = ("-- none --",)
            self._validation_config_dropdown.value = "-- none --"
            self._validation_config_dropdown.disabled = True
            self._validation_config_mapping = {}
            self._validation_selected_index = None
            return

        matches = bucket_indices.astype(int)
        if matches.size:
            preferred_idx = int(base_index) if base_index in matches else int(matches[0])
        else:
            preferred_idx = -1

        reference_quat, tolerance_rad = self._reference_quat_and_tolerance()
        filtered_matches: Optional[np.ndarray] = None
        if reference_quat is not None and tolerance_rad is not None:
            filtered_matches = self._bucket_filtered_sample_indices(
                data, bucket_key, reference_quat, tolerance_rad, allow_bucket_fallback=False
            )
            if filtered_matches is not None and filtered_matches.size:
                matches = filtered_matches.astype(int)
                if preferred_idx not in matches and matches.size:
                    preferred_idx = int(matches[0])
            else:
                LOGGER.info(
                    "reachviz: validation bucket %s has no configs within orientation tolerance; displaying all.",
                    bucket_key,
                )

        labels = [f"Config #{int(idx)}" for idx in matches]
        mapping = {label: int(idx) for label, idx in zip(labels, matches.tolist())}

        if matches.size:
            if preferred_idx not in matches:
                preferred_idx = int(matches[0])
            preferred_label = f"Config #{preferred_idx}"
        else:
            preferred_idx = -1
            preferred_label = "-- none --"

        self._validation_update_guard = True
        try:
            if labels:
                if preferred_label not in labels:
                    preferred_label = labels[0]
                self._validation_config_dropdown.options = labels
                self._validation_config_dropdown.value = preferred_label
                should_disable = len(labels) <= 1 and not self._orientation_filter_active()
                self._validation_config_dropdown.disabled = should_disable
            else:
                self._validation_config_dropdown.options = ("-- none --",)
                self._validation_config_dropdown.value = "-- none --"
                self._validation_config_dropdown.disabled = True
        finally:
            self._validation_update_guard = False

        self._validation_config_mapping = mapping
        self._validation_selected_index = mapping.get(preferred_label)
        if self._validation_selected_index is None and mapping:
            self._validation_selected_index = next(iter(mapping.values()))
        if self._validation_config_dropdown.disabled and mapping:
            self._validation_selected_index = next(iter(mapping.values()))

    def _apply_validation_configuration(
        self, data: ReachabilityMapData, default_idx: int
    ) -> Optional[int]:
        if self._config_callback is None or not self._validation_active() or not data.sample_configs.size:
            return None
        target_idx = self._validation_selected_index
        if target_idx is None:
            target_idx = default_idx

        if self._orientation_filter_active() and self._reference_orientation_quat is not None:
            bucket_key = tuple(int(v) for v in data.sample_indices[default_idx])
            bucket_samples = self._get_bucket_sample_indices(data, bucket_key)
            if bucket_samples is not None and bucket_samples.size:
                tolerance_rad = self._orientation_tolerance_rad()
                bucket_quats: Optional[np.ndarray]
                if (
                    data.sample_quaternions.ndim == 2
                    and data.sample_quaternions.shape[0] >= data.sample_configs.shape[0]
                ):
                    bucket_quats = data.sample_quaternions[bucket_samples]
                elif (
                    data.sample_orientations.ndim == 2
                    and data.sample_orientations.shape[0] >= data.sample_configs.shape[0]
                ):
                    bucket_quats = _rpy_to_quaternion_batch(data.sample_orientations[bucket_samples])
                else:
                    bucket_quats = self._get_bucket_quaternions(data, bucket_key)
                if bucket_quats is not None and bucket_quats.size:
                    angles = _quaternion_difference_angles(self._reference_orientation_quat, bucket_quats)
                    eligible = np.nonzero(angles <= tolerance_rad)[0]
                    if eligible.size:
                        target_idx = int(bucket_samples[eligible[0]])
                    else:
                        LOGGER.info(
                            "reachviz: validation skipped; no configs for bucket %s satisfied orientation tolerance %.1f deg",
                            bucket_key,
                            float(np.degrees(tolerance_rad)),
                        )
                        return None

        target_idx = max(0, min(int(target_idx), data.sample_configs.shape[0] - 1))
        self._validation_selected_index = target_idx
        config = data.sample_configs[target_idx]
        self._config_callback(data.joint_names, config.copy())
        return target_idx

    def _set_validation_status_message(
        self,
        sample_idx: int,
        point: np.ndarray,
        configs_available: int,
        applied_idx: Optional[int],
        bucket_visitation: Optional[int] = None,
    ) -> None:
        if self._validation_status is None:
            return
        xyz = point.reshape(-1)
        xyz_str = ", ".join(f"{coord:.3f}" for coord in xyz[:3])
        message = f"Sample #{sample_idx} at [{xyz_str}] | {configs_available} config(s)"
        if applied_idx is not None:
            message += f" | showing config #{applied_idx}"
        if bucket_visitation is not None:
            message += f" | visitation {bucket_visitation}"
        self._validation_status.value = message

    def _get_sample_quaternion(self, data: ReachabilityMapData, sample_idx: int) -> Optional[np.ndarray]:
        if data.sample_quaternions.size and 0 <= sample_idx < data.sample_quaternions.shape[0]:
            quat = data.sample_quaternions[sample_idx]
            if quat.size:
                return np.asarray(quat, dtype=np.float32)
        if data.sample_orientations.size and 0 <= sample_idx < data.sample_orientations.shape[0]:
            return _rpy_to_quaternion_batch(data.sample_orientations[sample_idx : sample_idx + 1])[0]
        return None

    def _get_bucket_quaternions(
        self, data: ReachabilityMapData, bucket_key: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        bucket_quats = data.bucket_quaternions.get(bucket_key)
        if bucket_quats is not None and bucket_quats.size:
            return bucket_quats
        bucket_rpy = data.bucket_orientations.get(bucket_key)
        if bucket_rpy is not None and bucket_rpy.size:
            return _rpy_to_quaternion_batch(bucket_rpy)
        return None

    def _get_bucket_sample_indices(
        self, data: ReachabilityMapData, bucket_key: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        indices = data.bucket_sample_indices.get(bucket_key)
        if indices is not None and indices.size:
            return indices
        matches = np.nonzero(np.all(data.sample_indices == np.asarray(bucket_key, dtype=int), axis=1))[0]
        return matches if matches.size else None

    def _orientation_filter_active(self) -> bool:
        return bool(
            self._orientation_filter_checkbox
            and self._orientation_filter_checkbox.value
            and self._reference_orientation_quat is not None
        )

    def _orientation_tolerance_rad(self) -> float:
        tolerance_deg = reach_consts.ORIENTATION_TOLERANCE_DEFAULT_DEG
        if self._orientation_tolerance_slider is not None:
            tolerance_deg = float(self._orientation_tolerance_slider.value)
        tolerance_deg = float(
            np.clip(
                tolerance_deg,
                reach_consts.ORIENTATION_TOLERANCE_MIN_DEG,
                reach_consts.ORIENTATION_TOLERANCE_MAX_DEG,
            )
        )
        return float(np.radians(tolerance_deg))

    def _validation_active(self) -> bool:
        return bool(self._pick_checkbox and self._pick_checkbox.value)

    def _cancel_orientation_timer(self) -> None:
        timer = self._orientation_debounce_timer
        if timer is not None:
            timer.cancel()
            self._orientation_debounce_timer = None

    def _schedule_orientation_update(self, delay: float = 0.35) -> None:
        self._cancel_orientation_timer()

        def _invoke() -> None:
            self._orientation_debounce_timer = None
            self._update_scene()

        timer = threading.Timer(delay, _invoke)
        timer.daemon = True
        self._orientation_debounce_timer = timer
        timer.start()

    @staticmethod
    def _quaternions_differ(
        previous: Optional[np.ndarray], current: Optional[np.ndarray], atol: float = 1e-4
    ) -> bool:
        if previous is None or current is None:
            return previous is not current
        if previous.shape != (4,) or current.shape != (4,):
            return True
        dot = float(abs(np.dot(previous, current)))
        dot = np.clip(dot, -1.0, 1.0)
        # dot close to 1 -> identical; dot close to -1 -> opposite but same orientation
        return (1.0 - dot) > atol

    def _request_scene_refresh(self) -> None:
        if self._orientation_filter_active():
            self._schedule_orientation_update()
        else:
            self._update_scene()

    def _reference_quat_and_tolerance(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if not self._orientation_filter_active():
            return None, None
        quat = self._reference_orientation_quat
        if quat is None:
            return None, None
        return quat, self._orientation_tolerance_rad()

    def _bucket_filtered_sample_indices(
        self,
        data: ReachabilityMapData,
        bucket_key: Tuple[int, int, int],
        reference_quat: np.ndarray,
        tolerance_rad: float,
        *,
        allow_bucket_fallback: bool = True,
    ) -> Optional[np.ndarray]:
        sample_indices = self._get_bucket_sample_indices(data, bucket_key)
        if sample_indices is None or not sample_indices.size:
            return None

        if (
            data.sample_quaternions.ndim == 2
            and data.sample_quaternions.shape[0] >= data.sample_configs.shape[0]
        ):
            bucket_quats = data.sample_quaternions[sample_indices]
        elif (
            data.sample_orientations.ndim == 2
            and data.sample_orientations.shape[0] >= data.sample_configs.shape[0]
        ):
            bucket_quats = _rpy_to_quaternion_batch(data.sample_orientations[sample_indices])
        else:
            if not allow_bucket_fallback:
                return None
            bucket_quats = self._get_bucket_quaternions(data, bucket_key)

        if bucket_quats is None or not np.size(bucket_quats):
            return None

        angles = _quaternion_difference_angles(reference_quat, bucket_quats)
        eligible = np.nonzero(angles <= tolerance_rad)[0]
        if eligible.size == 0:
            return None
        return sample_indices[eligible]

    def _compute_indices(self, points: np.ndarray, resolution: float, lower_bounds: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.int32)
        if resolution <= 0.0:
            return np.round(points, decimals=6).astype(np.int32)
        lb = lower_bounds.reshape(1, 3)
        return np.floor((points - lb) / resolution + 1e-9).astype(np.int32)

    def _load_samples(
        self, map_path: Path, resolution: float, lower_bounds: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[Tuple[int, int, int], np.ndarray],
        Dict[Tuple[int, int, int], np.ndarray],
        Dict[Tuple[int, int, int], np.ndarray],
        Dict[Tuple[int, int, int], np.ndarray],
        Tuple[str, ...],
    ]:
        job_name = map_path.stem
        if job_name.startswith("filt_"):
            job_name = job_name[len("filt_") :]
        if job_name.startswith("3D_"):
            job_name = job_name[len("3D_") :]
        samples_path = map_path.parent / f"samples_{job_name}.npz"
        if not samples_path.exists():
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_cfg = np.zeros((0, 0), dtype=np.float32)
            empty_idx = np.zeros((0, 3), dtype=np.int32)
            empty_orient = np.zeros((0, 3), dtype=np.float32)
            empty_quat = np.zeros((0, 4), dtype=np.float32)
            return (empty_xyz, empty_idx, empty_cfg, empty_orient, empty_quat, {}, {}, {}, {}, tuple())

        try:
            data = np.load(samples_path)
        except ValueError as exc:
            if "allow_pickle" in str(exc):
                data = np.load(samples_path, allow_pickle=True)
            else:
                raise
        sample_xyz = data.get("xyz", np.zeros((0, 3), dtype=np.float32)).astype(np.float32, copy=False)
        sample_ijk = data.get("ijk", np.zeros((sample_xyz.shape[0], 3), dtype=np.int32)).astype(np.int32, copy=False)
        sample_configs = data.get("q", np.zeros((0, 0), dtype=np.float32)).astype(np.float32, copy=False)
        raw_rpy = data.get("rpy")
        sample_rpy: Optional[np.ndarray]
        if raw_rpy is None:
            sample_rpy = None
        else:
            sample_rpy = np.asarray(raw_rpy, dtype=np.float32)
            if sample_rpy.shape[0] != sample_configs.shape[0]:
                sample_rpy = None
        raw_quat = data.get("quat")
        sample_quat: Optional[np.ndarray]
        if raw_quat is None:
            sample_quat = None
        else:
            sample_quat = np.asarray(raw_quat, dtype=np.float32)
            if sample_quat.shape[0] != sample_configs.shape[0] or sample_quat.shape[1] != 4:
                sample_quat = None
        if sample_quat is None and sample_rpy is not None:
            sample_quat = _rpy_to_quaternion_batch(sample_rpy)
        raw_joint_names = data.get("joint_names", np.array([], dtype="U64"))
        if raw_joint_names.dtype.kind == "O":
            joint_names = tuple(str(name) for name in raw_joint_names.tolist())
        else:
            joint_names = tuple(np.asarray(raw_joint_names, dtype=str))
        bucket_configs: Dict[Tuple[int, int, int], np.ndarray] = {}
        bucket_orientations: Dict[Tuple[int, int, int], np.ndarray] = {}
        bucket_quaternions: Dict[Tuple[int, int, int], np.ndarray] = {}
        bucket_index_lists: Dict[Tuple[int, int, int], List[int]] = {}
        bucket_sample_indices: Dict[Tuple[int, int, int], np.ndarray] = {}
        if sample_xyz.size and sample_configs.size:
            if sample_ijk.size:
                LOGGER.debug("_load_samples: Using saved ijk indices (shape=%s), first few keys: %s",
                           sample_ijk.shape, [tuple(row.tolist()) for row in sample_ijk[:5]])
                key_arrays = [tuple(row.tolist()) for row in sample_ijk]
            else:
                LOGGER.debug("_load_samples: Recomputing bucket indices using resolution=%.4f, lower_bounds=%s",
                           resolution, lower_bounds)
                quantized = self._compute_indices(sample_xyz, resolution, lower_bounds)
                key_arrays = [tuple(row.tolist()) for row in quantized]
            tmp_cfg: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
            tmp_orient: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
            tmp_quat: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
            for idx, key in enumerate(key_arrays):
                tmp_cfg.setdefault(key, []).append(sample_configs[idx])
                bucket_index_lists.setdefault(key, []).append(idx)
                if sample_rpy is not None:
                    tmp_orient.setdefault(key, []).append(sample_rpy[idx])
                if sample_quat is not None:
                    tmp_quat.setdefault(key, []).append(sample_quat[idx])
            bucket_configs = {key: np.vstack(cfg_list) for key, cfg_list in tmp_cfg.items()}
            LOGGER.debug("_load_samples: Created bucket_configs with %d unique keys: %s",
                       len(bucket_configs), list(bucket_configs.keys())[:10])
            if tmp_orient:
                bucket_orientations = {
                    key: np.vstack(orient_list).astype(np.float32, copy=False)
                    for key, orient_list in tmp_orient.items()
                }
            if tmp_quat:
                bucket_quaternions = {
                    key: np.vstack(quat_list).astype(np.float32, copy=False)
                    for key, quat_list in tmp_quat.items()
                }
        if bucket_index_lists:
            bucket_sample_indices = {
                key: np.asarray(indices, dtype=np.int32)
                for key, indices in bucket_index_lists.items()
            }
        sample_rpy_array = (
            sample_rpy if sample_rpy is not None else np.zeros((0, 3), dtype=np.float32)
        )
        sample_quat_array = (
            sample_quat if sample_quat is not None else np.zeros((0, 4), dtype=np.float32)
        )
        return (
            sample_xyz,
            sample_ijk,
            sample_configs,
            sample_rpy_array,
            sample_quat_array,
            bucket_configs,
            bucket_orientations,
            bucket_quaternions,
            bucket_sample_indices,
            joint_names,
        )

    def _load_map(self, path: Path) -> ReachabilityMapData:
        import h5py  # defer import until needed

        poses_dataset_array: Optional[np.ndarray] = None
        robot_metadata: Dict[str, Any] = {}
        with h5py.File(path, "r") as h5_file:
            if "Spheres" not in h5_file or "sphere_dataset" not in h5_file["Spheres"]:
                raise ValueError("HDF5 file missing /Spheres/sphere_dataset dataset.")
            sphere_group = h5_file["Spheres"]
            sphere_dataset = sphere_group["sphere_dataset"]
            dataset = np.asarray(sphere_dataset, dtype=np.float32)
            resolution = float(sphere_dataset.attrs.get("Resolution", 0.1))
            lower_bounds = np.asarray(
                sphere_dataset.attrs.get("LowerBounds", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            )
            if not lower_bounds.any():
                # Fallback for legacy files storing center offsets
                center_offset = np.asarray(
                    sphere_dataset.attrs.get("CenterOffset", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                )
                lower_bounds = center_offset - 0.5 * float(resolution)
            metric_names_attr = sphere_dataset.attrs.get("MetricNames")
            metric_cols_attr = sphere_dataset.attrs.get("MetricColumns")
            visitation_col_attr = sphere_dataset.attrs.get("VisitationColumn")
            metadata_attrs = {
                "RangeOfMotionRawMin": sphere_dataset.attrs.get("RangeOfMotionRawMin"),
                "RangeOfMotionRawMax": sphere_dataset.attrs.get("RangeOfMotionRawMax"),
                "SingularityWeightedMax": sphere_dataset.attrs.get("SingularityWeightedMax"),
                "SingularityRawMax": sphere_dataset.attrs.get("SingularityRawMax"),
                "ManipulabilityScaling": sphere_dataset.attrs.get("ManipulabilityScaling"),
                "ManipulabilityRawMax": sphere_dataset.attrs.get("ManipulabilityRawMax"),
            }
            # Load GlobalScores if available
            global_scores_json = sphere_dataset.attrs.get("GlobalScores")
            global_scores: Optional[Dict[str, Any]] = None
            if global_scores_json is not None:
                try:
                    if isinstance(global_scores_json, bytes):
                        global_scores_json = global_scores_json.decode("utf-8")
                    global_scores = json.loads(global_scores_json)
                except (json.JSONDecodeError, AttributeError) as exc:
                    LOGGER.warning("Failed to parse GlobalScores: %s", exc)
                    global_scores = None
            robot_metadata_json = sphere_dataset.attrs.get("RobotMetadata")
            if robot_metadata_json is not None:
                try:
                    if isinstance(robot_metadata_json, bytes):
                        robot_metadata_json = robot_metadata_json.decode("utf-8")
                    metadata_obj = json.loads(robot_metadata_json)
                    if isinstance(metadata_obj, dict):
                        robot_metadata = metadata_obj
                    else:
                        LOGGER.warning("RobotMetadata attribute is not a dictionary; ignoring.")
                except (json.JSONDecodeError, AttributeError) as exc:
                    LOGGER.warning("Failed to parse RobotMetadata: %s", exc)
            poses_group = h5_file.get("Poses")
            if poses_group is not None and "poses_dataset" in poses_group:
                poses_dataset_array = np.asarray(poses_group["poses_dataset"], dtype=np.float32)
        if dataset.shape[1] < 4:
            raise ValueError("Sphere dataset must contain xyz + metric columns.")

        points = dataset[:, :3]

        if visitation_col_attr is not None:
            visitation_column = int(visitation_col_attr)
        else:
            visitation_column = 4 if dataset.shape[1] > 4 else -1

        metric_names: List[str] = []
        metric_columns: List[int] = []

        if metric_names_attr is not None and metric_cols_attr is not None:
            # Decode metric names from bytes if needed
            metric_names_raw = np.asarray(metric_names_attr)
            if metric_names_raw.dtype.kind == 'S':  # String dtype (bytes)
                metric_names = [name.decode("utf-8") for name in metric_names_raw.tolist()]
            else:
                metric_names = [str(name) for name in metric_names_raw.tolist()]
            metric_columns = [int(col) for col in np.asarray(metric_cols_attr).tolist()]
            LOGGER.debug("Loaded metric names from HDF5: %s (columns: %s)", metric_names, metric_columns)
        else:
            # Fallback: try to infer metrics from dataset shape
            # IK maps should have: xyz(3) + Manipulability(1) + Visitation(1) + RangeOfMotion(1) + SingularityAvoidance(1) = 7 columns
            # FK maps might have different structure
            if dataset.shape[1] >= reach_consts.SPHERE_DATASET_MIN_COLS_WITH_METRICS:
                # Likely IK map with all metrics
                metric_names = ["Manipulability", "RangeOfMotion", "SingularityAvoidance"]
                metric_columns = [reach_consts.SPHERE_DATASET_MANIP_COL, reach_consts.SPHERE_DATASET_ROM_COL, reach_consts.SPHERE_DATASET_SINGULARITY_COL]
                if visitation_column < 0:
                    visitation_column = reach_consts.SPHERE_DATASET_VISITATION_COL
                LOGGER.debug("Inferred IK metrics from dataset shape: %s (columns: %s)", metric_names, metric_columns)
            else:
                # Fallback to single metric
                metric_names = ["Manipulability"]
                metric_columns = [3 if dataset.shape[1] > 3 else dataset.shape[1] - 1]
                if visitation_column < 0 and dataset.shape[1] > metric_columns[0] + 1:
                    visitation_column = metric_columns[0] + 1
                LOGGER.debug("Using fallback single metric: %s (column: %s)", metric_names, metric_columns)

        metrics: Dict[str, np.ndarray] = {}
        metric_ranges: Dict[str, Tuple[float, float]] = {}
        for name, col in zip(metric_names, metric_columns):
            if col < 0 or col >= dataset.shape[1]:
                continue
            values = dataset[:, col].astype(np.float32, copy=False)
            metrics[name] = values
            if values.size:
                min_val = float(np.min(values))
                max_val = float(np.max(values))
            else:
                min_val = 0.0
                max_val = 0.0
            if math.isclose(min_val, max_val):
                max_val = min_val + 1e-6
            metric_ranges[name] = (min_val, max_val)

        for name in list(metric_ranges.keys()):
            if name in {"RangeOfMotion", "SingularityAvoidance"}:
                metric_ranges[name] = (0.0, 1.0)
            elif name == "Manipulability":
                # Manipulability is stored raw (0 to ManipulabilityRawMax)
                # Normalize it to (0, 1) range using ManipulabilityRawMax
                manip_max = metadata_attrs.get("ManipulabilityRawMax")
                if manip_max is not None and manip_max > 0.0:
                    metric_ranges[name] = (0.0, float(manip_max))
                # Otherwise fallback to min/max from data

        if not metrics:
            fallback_col = 3 if dataset.shape[1] > 3 else dataset.shape[1] - 1
            fallback_values = dataset[:, fallback_col].astype(np.float32, copy=False)
            metrics["Manipulability"] = fallback_values
            if fallback_values.size:
                min_val = float(np.min(fallback_values))
                max_val = float(np.max(fallback_values))
            else:
                min_val = 0.0
                max_val = 0.0
            if math.isclose(min_val, max_val):
                max_val = min_val + 1e-6
            metric_ranges["Manipulability"] = (min_val, max_val)
            metric_names = ["Manipulability"]
            metric_columns = [fallback_col]
            if visitation_column < 0 and dataset.shape[1] > fallback_col + 1:
                visitation_column = fallback_col + 1

        if 0 <= visitation_column < dataset.shape[1]:
            visitation = dataset[:, visitation_column].astype(np.float32, copy=False)
        else:
            visitation = np.ones(points.shape[0], dtype=np.float32)

        if "Visitation" not in metrics:
            metrics["Visitation"] = visitation
            if visitation.size:
                vis_min = float(np.min(visitation))
                vis_max = float(np.max(visitation))
            else:
                vis_min = 0.0
                vis_max = 0.0
            if math.isclose(vis_min, vis_max):
                vis_max = vis_min + 1e-6
            metric_ranges.setdefault("Visitation", (vis_min, vis_max))
            if "Visitation" not in metric_names:
                metric_names.append("Visitation")
                metric_columns.append(visitation_column if visitation_column >= 0 else -1)

        metadata: Dict[str, float] = {}
        for key, value in metadata_attrs.items():
            if value is not None:
                metadata[key] = float(value)

        # Store global_scores in metadata for later use
        if global_scores is not None:
            metadata["_global_scores"] = global_scores  # Store as special key

        (
            sample_xyz,
            sample_indices,
            sample_configs,
            sample_orientations,
            sample_quaternions,
            bucket_configs,
            bucket_orientations,
            bucket_quaternions,
            bucket_sample_indices,
            joint_names,
        ) = self._load_samples(path, resolution, lower_bounds)
        quantized_points = self._compute_indices(points, resolution, lower_bounds)
        visitation_lookup: Dict[Tuple[int, int, int], float] = {}
        if quantized_points.size and visitation.size:
            visitation_lookup = {
                tuple(int(v) for v in quantized_points[i]): float(visitation[i])
                for i in range(quantized_points.shape[0])
            }

        if (
            poses_dataset_array is not None
            and poses_dataset_array.ndim == 2
            and poses_dataset_array.shape[0] == points.shape[0]
            and poses_dataset_array.shape[1] >= 6
        ):
            orientations_rpy = poses_dataset_array[:, 3:6].astype(np.float32, copy=False)
            if poses_dataset_array.shape[1] >= 10:
                orientations_quat = poses_dataset_array[:, 6:10].astype(np.float32, copy=False)
            else:
                orientations_quat = _rpy_to_quaternion_batch(orientations_rpy)
        else:
            orientations_rpy = np.zeros((points.shape[0], 3), dtype=np.float32)
            orientations_quat = np.zeros((0, 4), dtype=np.float32)
        if not orientations_quat.size and orientations_rpy.size:
            orientations_quat = _rpy_to_quaternion_batch(orientations_rpy)

        return ReachabilityMapData(
            points=points,
            metrics=metrics,
            visitation=visitation,
            orientations_rpy=orientations_rpy,
            orientations_quat=orientations_quat,
            visitation_lookup=visitation_lookup,
            sample_xyz=sample_xyz,
            sample_indices=sample_indices,
            sample_configs=sample_configs,
            sample_orientations=sample_orientations,
            sample_quaternions=sample_quaternions,
            bucket_configs=bucket_configs,
            bucket_orientations=bucket_orientations,
            bucket_quaternions=bucket_quaternions,
            bucket_sample_indices=bucket_sample_indices,
            quantized_points=quantized_points,
            resolution=resolution,
            lower_bounds=lower_bounds,
            joint_names=joint_names,
            path=path,
            metric_ranges=metric_ranges,
            metric_order=tuple(metric_names),
            metadata=metadata,
            robot_metadata=robot_metadata,
        )

    def _current_map(self) -> Optional[ReachabilityMapData]:
        if not self._current_map_id:
            return None
        return self._map_cache.get(self._current_map_id)

    def _compute_metric(
        self, data: ReachabilityMapData, orientation_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        metric_label = "Manipulability"
        if self._metric_dropdown:
            metric_label = self._metric_dropdown.value

        metric_key = self._metric_label_to_key.get(metric_label, metric_label)
        active_mask: Optional[np.ndarray] = None
        if orientation_mask is not None:
            if orientation_mask.dtype != bool:
                active_mask = np.asarray(orientation_mask, dtype=bool)
            else:
                active_mask = orientation_mask
            if active_mask.size != data.points.shape[0]:
                active_mask = None

        if metric_key == "Visitation":
            scale = 1.0
            if self._visitation_scale_slider:
                scale = max(
                    reach_consts.VISITATION_SCALE_MIN, float(self._visitation_scale_slider.value)
                )
            raw = data.visitation.astype(np.float32)
            if active_mask is not None and self._orientation_filtered_counts:
                filtered_values = np.zeros_like(raw, dtype=np.float32)
                quantized = data.quantized_points
                if quantized.shape[0] == raw.shape[0]:
                    for idx, key_vals in enumerate(quantized):
                        count = self._orientation_filtered_counts.get(
                            (int(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
                        )
                        if count:
                            filtered_values[idx] = float(count)
                raw = filtered_values
            normalized = np.zeros_like(raw, dtype=np.float32)

            if active_mask is not None and np.any(active_mask):
                filtered_values = raw[active_mask]
                if filtered_values.size:
                    filtered_max = float(np.max(filtered_values))
                    effective_scale = float(scale)
                    if filtered_max > 0.0 and effective_scale > filtered_max:
                        effective_scale = max(filtered_max, reach_consts.VISITATION_SCALE_MIN)
                    elif effective_scale <= 0.0:
                        effective_scale = reach_consts.VISITATION_SCALE_MIN
                    normalized_subset = np.clip(filtered_values / effective_scale, 0.0, 1.0)
                    normalized[active_mask] = normalized_subset.astype(np.float32, copy=False)
            else:
                if scale <= 0.0:
                    scale = reach_consts.VISITATION_SCALE_MIN
                normalized = np.clip(raw / scale, 0.0, 1.0)

            if active_mask is not None and not np.any(active_mask):
                normalized.fill(0.0)
        else:
            values = data.metrics.get(metric_key)
            if values is None or not values.size:
                return np.zeros(data.points.shape[0], dtype=np.float32)

            # Special handling for Manipulability: normalize using ManipulabilityRawMax
            if metric_key == "Manipulability":
                manip_max = data.metadata.get("ManipulabilityRawMax")
                if manip_max is not None and manip_max > 0.0:
                    # Normalize by dividing by ManipulabilityRawMax
                    normalized = (values.astype(np.float32) / float(manip_max))
                    normalized = np.clip(normalized, 0.0, 1.0)
                else:
                    # Fallback to old ManipulabilityScaling for backward compatibility
                    manip_scaling = data.metadata.get("ManipulabilityScaling")
                    if manip_scaling is not None and manip_scaling > 0.0:
                        normalized = (values.astype(np.float32) / float(manip_scaling))
                        normalized = np.clip(normalized, 0.0, 1.0)
                    else:
                        # Final fallback: use data min/max
                        min_val, max_val = data.metric_ranges.get(
                            metric_key, (float(np.min(values)), float(np.max(values)))
                        )
                        denom = max(max_val - min_val, 1e-6)
                        normalized = (values - min_val) / denom
                        normalized = np.clip(normalized.astype(np.float32), 0.0, 1.0)
            else:
                # Standard min/max normalization for other metrics
                min_val, max_val = data.metric_ranges.get(
                    metric_key, (float(np.min(values)), float(np.max(values)))
                )
                denom = max(max_val - min_val, 1e-6)
                normalized = (values - min_val) / denom
                normalized = np.clip(normalized.astype(np.float32), 0.0, 1.0)

            if active_mask is not None:
                normalized = normalized.astype(np.float32, copy=False)
                normalized[~active_mask] = 0.0
        normalized = np.nan_to_num(normalized.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        return normalized

    def _make_colors(self, normalized_scores: np.ndarray) -> np.ndarray:
        if self._color_checkbox and self._color_checkbox.value:
            colors = _interpolate_colormap(normalized_scores)
        else:
            rgba = reach_consts.DEFAULT_BASE_COLOR
            if self._base_color_picker:
                rgba = self._base_color_picker.value
            base_rgb = np.array(rgba[:3], dtype=np.float32)
            colors = np.full((normalized_scores.shape[0], 3), base_rgb, dtype=np.float32)
        return np.asarray(colors, dtype=np.float32, order="C")

    def _compute_orientation_mask(
        self, data: ReachabilityMapData, tolerance_rad: float
    ) -> Optional[np.ndarray]:
        reference_quat = self._reference_orientation_quat
        if reference_quat is None:
            return None

        total_points = data.points.shape[0]
        if total_points == 0:
            return None

        start_time = time.perf_counter()
        mask = np.zeros(total_points, dtype=bool)
        voxel_matches = 0
        skipped_voxels = 0
        matched_points = 0
        counts_map: Dict[Tuple[int, int, int], int] = {}

        quantized = data.quantized_points.astype(np.int32, copy=False)
        voxel_index_map: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, key_vals in enumerate(quantized):
            voxel_index_map.setdefault(tuple(int(v) for v in key_vals), []).append(idx)

        for bucket_key, index_list in voxel_index_map.items():
            filtered = self._bucket_filtered_sample_indices(
                data,
                bucket_key,
                reference_quat,
                tolerance_rad,
                allow_bucket_fallback=False,
            )
            if filtered is not None:
                indices_array = np.asarray(index_list, dtype=int)
                mask[indices_array] = True
                voxel_matches += 1
                matched_points += int(indices_array.size)
                counts_map[bucket_key] = int(filtered.size)
            else:
                skipped_voxels += 1

        elapsed = time.perf_counter() - start_time
        self._orientation_filtered_counts = counts_map
        if elapsed >= 0.1:
            LOGGER.info(
                "reachviz: orientation mask computed in %.3fs | matched_voxels=%d | skipped_voxels=%d | matched_points=%d",
                elapsed,
                voxel_matches,
                skipped_voxels,
                matched_points,
            )
        return mask

    def _update_scene(self) -> None:
        LOGGER.debug("reachviz: _update_scene invoked")
        if self._point_cloud_handle is not None:
            self._point_cloud_handle.remove()
            self._point_cloud_handle = None

        data = self._current_map()
        if data is None:
            if self._validation_active():
                self._validation_current_index = None
                if self._validation_point_handle is not None:
                    self._validation_point_handle.remove()
                    self._validation_point_handle = None
            LOGGER.info("reachviz: no map loaded or map is empty")
            return

        if not len(data.points):
            if self._validation_active():
                self._validation_current_index = None
                if self._validation_point_handle is not None:
                    self._validation_point_handle.remove()
                    self._validation_point_handle = None
            self._set_status("Status: map contains no points")
            LOGGER.info("reachviz: no map loaded or map is empty")
            return

        show_checked = self._show_checkbox.value if self._show_checkbox else None
        LOGGER.debug(
            "reachviz: updating scene | show=%s | total_points=%d",
            show_checked,
            data.points.shape[0],
        )

        if self._validation_index_number is not None:
            self._validation_index_number.disabled = not (self._validation_active() and self._validation_engaged)

        if self._show_checkbox and not self._show_checkbox.value:
            self._update_validation_point(data)
            return

        orientation_mask: Optional[np.ndarray] = None
        self._orientation_filtered_counts = {}
        if self._orientation_filter_checkbox and self._orientation_filter_checkbox.value:
            if self._update_reference_orientation():
                tolerance_deg = reach_consts.ORIENTATION_TOLERANCE_DEFAULT_DEG
                if self._orientation_tolerance_slider is not None:
                    tolerance_deg = float(self._orientation_tolerance_slider.value)
                tolerance_deg = float(
                    np.clip(
                        tolerance_deg,
                        reach_consts.ORIENTATION_TOLERANCE_MIN_DEG,
                        reach_consts.ORIENTATION_TOLERANCE_MAX_DEG,
                    )
                )
                tolerance_rad = np.radians(tolerance_deg)
                orientation_mask = self._compute_orientation_mask(data, tolerance_rad)
                LOGGER.info(
                    "reachviz: orientation filter active | tolerance=%.1fdeg | matches=%d",
                    tolerance_deg,
                    int(np.count_nonzero(orientation_mask))
                    if orientation_mask is not None
                    else 0,
                )
                if not data.bucket_quaternions:
                    LOGGER.info(
                        "reachviz: orientation filter is using aggregated voxel orientations (no per-sample quaternion data); consider regenerating the map"
                    )
                elif orientation_mask is not None and int(np.count_nonzero(orientation_mask)) == orientation_mask.size:
                    LOGGER.info(
                        "reachviz: orientation filter matched every voxel; orientation bins may be too coarse for the selected tolerance"
                    )
            else:
                # Disable the filter gracefully if data unavailable.
                if self._orientation_filter_checkbox is not None:
                    self._orientation_filter_checkbox.value = False
                if self._orientation_tolerance_slider is not None:
                    self._orientation_tolerance_slider.disabled = True
                self._reference_orientation_quat = None
                self._set_status("Status: orientation filter disabled (no orientation data)")
                LOGGER.info("reachviz: orientation filter disabled (no orientation data)")

        normalized_scores = self._compute_metric(data, orientation_mask)

        min_norm = 0.0
        if self._min_score_slider:
            min_norm = float(self._min_score_slider.value)

        # Apply slice filters with rotation
        mask = np.isfinite(normalized_scores) & (normalized_scores >= min_norm)

        # Height filters (z-axis)
        z_min = -np.inf
        z_max = np.inf
        if self._z_min_slider:
            z_min = float(self._z_min_slider.value)
        if self._z_slice_slider:
            z_max = float(self._z_slice_slider.value)
        mask &= (data.points[:, 2] >= z_min) & (data.points[:, 2] <= z_max)

        # Distance filters along x-axis and y-axis with rotation
        x_min = -np.inf
        x_max = np.inf
        y_min = -np.inf
        y_max = np.inf
        rotation_deg = 0.0
        rotation_x_deg = 0.0
        rotation_y_deg = 0.0
        if self._x_min_slider:
            x_min = float(self._x_min_slider.value)
        if self._x_max_slider:
            x_max = float(self._x_max_slider.value)
        if self._y_min_slider:
            y_min = float(self._y_min_slider.value)
        if self._y_max_slider:
            y_max = float(self._y_max_slider.value)
        if self._slice_rotation_slider:
            rotation_deg = float(self._slice_rotation_slider.value)
        if self._slice_rotation_x_slider:
            rotation_x_deg = float(self._slice_rotation_x_slider.value)
        if self._slice_rotation_y_slider:
            rotation_y_deg = float(self._slice_rotation_y_slider.value)

        if np.isfinite(x_min) or np.isfinite(x_max) or np.isfinite(y_min) or np.isfinite(y_max) or rotation_deg != 0.0 or rotation_x_deg != 0.0 or rotation_y_deg != 0.0:
            # Apply rotations: first rotate around z-axis, then x-axis, then y-axis
            points_rotated = data.points.copy()

            # Rotation around z-axis (in xy plane)
            if rotation_deg != 0.0:
                rotation_rad = np.radians(rotation_deg)
                cos_rot = np.cos(rotation_rad)
                sin_rot = np.sin(rotation_rad)
                # Rotation matrix around z-axis: [cos -sin 0; sin cos 0; 0 0 1]
                x_new = points_rotated[:, 0] * cos_rot - points_rotated[:, 1] * sin_rot
                y_new = points_rotated[:, 0] * sin_rot + points_rotated[:, 1] * cos_rot
                points_rotated[:, 0] = x_new
                points_rotated[:, 1] = y_new

            # Rotation around x-axis (in yz plane)
            if rotation_x_deg != 0.0:
                rotation_x_rad = np.radians(rotation_x_deg)
                cos_rot_x = np.cos(rotation_x_rad)
                sin_rot_x = np.sin(rotation_x_rad)
                # Rotation matrix around x-axis: [1 0 0; 0 cos -sin; 0 sin cos]
                y_new = points_rotated[:, 1] * cos_rot_x - points_rotated[:, 2] * sin_rot_x
                z_new = points_rotated[:, 1] * sin_rot_x + points_rotated[:, 2] * cos_rot_x
                points_rotated[:, 1] = y_new
                points_rotated[:, 2] = z_new

            # Rotation around y-axis (in xz plane)
            if rotation_y_deg != 0.0:
                rotation_y_rad = np.radians(rotation_y_deg)
                cos_rot_y = np.cos(rotation_y_rad)
                sin_rot_y = np.sin(rotation_y_rad)
                # Rotation matrix around y-axis: [cos 0 sin; 0 1 0; -sin 0 cos]
                x_new = points_rotated[:, 0] * cos_rot_y + points_rotated[:, 2] * sin_rot_y
                z_new = -points_rotated[:, 0] * sin_rot_y + points_rotated[:, 2] * cos_rot_y
                points_rotated[:, 0] = x_new
                points_rotated[:, 2] = z_new

            # Apply x-distance filters in rotated coordinate system
            if np.isfinite(x_min):
                mask &= points_rotated[:, 0] >= x_min
            if np.isfinite(x_max):
                mask &= points_rotated[:, 0] <= x_max
            # Apply y-distance filters in rotated coordinate system
            if np.isfinite(y_min):
                mask &= points_rotated[:, 1] >= y_min
            if np.isfinite(y_max):
                mask &= points_rotated[:, 1] <= y_max

        LOGGER.debug("reachviz: score/z/x mask count=%d", int(np.count_nonzero(mask)))
        if orientation_mask is not None:
            masked = mask & orientation_mask
            if np.any(masked):
                mask = masked
            else:
                self._set_status("Status: orientation filter removed all points; filter disabled")
                if self._orientation_filter_checkbox is not None:
                    self._orientation_filter_checkbox.value = False
                if self._orientation_tolerance_slider is not None:
                    self._orientation_tolerance_slider.disabled = True
                self._reference_orientation_quat = None
                LOGGER.info("reachviz: orientation filter removed all points; filter disabled")

        if not np.any(mask):
            self._set_status("Status: no points after filtering")
            LOGGER.info("reachviz: final mask empty after filters")
            return

        LOGGER.debug("reachviz: final mask count=%d", int(np.count_nonzero(mask)))

        points = data.points[mask]
        metric_subset = normalized_scores[mask]
        colors = self._make_colors(metric_subset)
        if colors.dtype != np.float32:
            colors = colors.astype(np.float32, copy=False)

        point_size = reach_consts.DEFAULT_POINT_SIZE_M
        if self._point_size_slider:
            point_size = max(
                reach_consts.MIN_POINT_SIZE_M, self._point_size_slider.value / 100.0
            )

        name = f"{self._scene_path}/cloud"
        self._point_cloud_handle = self._server.scene.add_point_cloud(
            name, points=points, colors=colors, point_size=point_size, point_shape="circle"
        )
        self._set_status(f"Status: displaying {points.shape[0]} points")
        mask_indices = np.nonzero(mask)[0]
        self._update_picker(data, points, mask_indices)
        self._update_validation_point(data)

    def _update_picker(self, data: ReachabilityMapData, points: np.ndarray, mask_indices: np.ndarray) -> None:
        if self._pick_checkbox is None or not self._pick_checkbox.value:
            if self._picker_handle is not None:
                self._picker_handle.remove()
                self._picker_handle = None
            self._current_picker_indices = None
            self._picker_positions_cache = None
            self._picker_keys_cache = None
            return
        if points.size == 0:
            self._current_picker_indices = None
            self._picker_positions_cache = None
            self._picker_keys_cache = None
            if self._picker_handle is not None:
                self._picker_handle.remove()
                self._picker_handle = None
            return
        positions = points.astype(np.float32, copy=False)
        if self._picker_positions_cache is not None:
            if (
                positions.shape == self._picker_positions_cache.shape
                and np.allclose(positions, self._picker_positions_cache)
            ):
                self._current_picker_indices = mask_indices
                self._picker_keys_cache = data.quantized_points[mask_indices].astype(np.int32, copy=False)
                return
        if self._picker_handle is not None:
                self._picker_handle.remove()
                self._picker_handle = None
        orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (positions.shape[0], 1))
        self._picker_handle = self._server.scene.add_batched_axes(
            f"{self._scene_path}/picker",
            batched_positions=positions,
            batched_wxyzs=orientations,
            axes_length=0.0125,
            axes_radius=0.001,
            visible=True,
        )
        self._picker_handle.on_click(self._handle_pick_event)
        self._current_picker_indices = mask_indices
        self._picker_positions_cache = positions.copy()
        self._picker_keys_cache = data.quantized_points[mask_indices].astype(np.int32, copy=False)

    def _handle_pick_event(self, event: viser.SceneNodePointerEvent[Any]) -> None:
        LOGGER.debug("_handle_pick_event called: instance_index=%s", event.instance_index)
        data = self._current_map()
        if data is None:
            LOGGER.warning("_handle_pick_event: data is None")
            return
        if self._current_picker_indices is None:
            LOGGER.warning("_handle_pick_event: _current_picker_indices is None")
            return
        if event.instance_index is None:
            LOGGER.warning("_handle_pick_event: event.instance_index is None")
            return
        idx = int(event.instance_index)
        if idx < 0 or idx >= len(self._current_picker_indices):
            LOGGER.warning("_handle_pick_event: idx %d out of range [0, %d)", idx, len(self._current_picker_indices))
            return
        global_idx = int(self._current_picker_indices[idx])
        if self._picker_keys_cache is None:
            LOGGER.warning("_handle_pick_event: _picker_keys_cache is None")
            return
        quant_key_array = np.asarray(self._picker_keys_cache[idx], dtype=int)
        quant_key = tuple(int(v) for v in quant_key_array.tolist())
        LOGGER.debug("_handle_pick_event: quant_key=%s", quant_key)
        LOGGER.debug("_handle_pick_event: clicked point position=%s (from picker cache)",
                   self._picker_positions_cache[idx] if self._picker_positions_cache is not None else "N/A")
        LOGGER.debug("_handle_pick_event: data.points[global_idx]=%s (from sphere dataset)",
                   data.points[global_idx] if global_idx is not None and 0 <= global_idx < data.points.shape[0] else "N/A")
        LOGGER.debug("_handle_pick_event: resolution=%.6f, lower_bounds=%s",
                   data.resolution, data.lower_bounds)
        self._last_bucket_key = quant_key
        visitation_total: Optional[int] = None
        vis_value = data.visitation_lookup.get(quant_key)
        if vis_value is not None and np.isfinite(vis_value):
            visitation_total = int(round(float(vis_value)))
        configs = data.bucket_configs.get(quant_key)
        LOGGER.debug("_handle_pick_event: bucket_configs has %d keys, configs for quant_key=%s: %s",
                   len(data.bucket_configs), quant_key, "found" if configs is not None else "NOT FOUND")
        if configs is None or configs.size == 0:
            LOGGER.warning("_handle_pick_event: no configs for quant_key %s (bucket_configs keys: %s)",
                          quant_key, list(data.bucket_configs.keys())[:5] if data.bucket_configs else "empty")
            self._set_status("Status: no samples for selected point")
            return

        orientation_changed = False
        if self._orientation_filter_checkbox and self._orientation_filter_checkbox.value:
            prev_quat = None
            if self._reference_orientation_quat is not None:
                prev_quat = np.asarray(self._reference_orientation_quat, dtype=np.float32).copy()
            self._update_reference_orientation()
            current_quat = self._reference_orientation_quat
            if self._quaternions_differ(prev_quat, current_quat):
                orientation_changed = True
        if orientation_changed:
            self._set_status("Status: orientation changed; recomputing filter")
            self._request_scene_refresh()
            return

        filtered_orientation_count: Optional[int] = None
        reference_quat, tolerance_rad = self._reference_quat_and_tolerance()
        if (
            visitation_total is not None
            and reference_quat is not None
            and tolerance_rad is not None
        ):
            filtered_indices = self._bucket_filtered_sample_indices(
                data, quant_key, reference_quat, tolerance_rad, allow_bucket_fallback=False
            )
            filtered_orientation_count = int(filtered_indices.size) if filtered_indices is not None else 0

        if self._validation_visitation_text is not None:
            if visitation_total is None:
                self._validation_visitation_text.value = "Visitation: --"
            elif filtered_orientation_count is not None:
                self._validation_visitation_text.value = (
                    f"Raw: {visitation_total} | Filtered: {filtered_orientation_count}"
                )
            else:
                self._validation_visitation_text.value = f"Visitation: {visitation_total}"

        active_orientation_filter = self._orientation_filter_active()
        bucket_sample_indices = self._get_bucket_sample_indices(data, quant_key)
        selected_index: Optional[int] = None

        if active_orientation_filter and self._reference_orientation_quat is not None:
            tolerance_rad = self._orientation_tolerance_rad()
            reference_quat = self._reference_orientation_quat
            local_candidates: Optional[np.ndarray] = None
            angles: Optional[np.ndarray] = None

            if (
                data.sample_quaternions.ndim == 2
                and data.sample_quaternions.shape[0] >= data.sample_configs.shape[0]
                and bucket_sample_indices is not None
                and bucket_sample_indices.size
            ):
                bucket_quats = data.sample_quaternions[bucket_sample_indices]
                angles = _quaternion_difference_angles(reference_quat, bucket_quats)
                local_candidates = np.nonzero(angles <= tolerance_rad)[0]
            elif (
                data.sample_orientations.ndim == 2
                and data.sample_orientations.shape[0] >= data.sample_configs.shape[0]
                and bucket_sample_indices is not None
                and bucket_sample_indices.size
            ):
                bucket_quats = _rpy_to_quaternion_batch(data.sample_orientations[bucket_sample_indices])
                angles = _quaternion_difference_angles(reference_quat, bucket_quats)
                local_candidates = np.nonzero(angles <= tolerance_rad)[0]
            else:
                bucket_quats = self._get_bucket_quaternions(data, quant_key)
                if bucket_quats is not None and bucket_quats.size:
                    angles = _quaternion_difference_angles(reference_quat, bucket_quats)
                    local_candidates = np.nonzero(angles <= tolerance_rad)[0]

            if local_candidates is not None and local_candidates.size:
                selected_index = int(local_candidates[self._rng.integers(local_candidates.size)])
            else:
                LOGGER.info(
                    "reachviz: sample picking skipped; no samples for bucket %s satisfied orientation tolerance %.1f deg",
                    quant_key,
                    float(np.degrees(tolerance_rad)),
                )
                self._set_status("Status: no configs within orientation tolerance")
                return

        if selected_index is None:
            selected_index = int(self._rng.integers(configs.shape[0])) if configs.size else 0

        selected_global_idx: Optional[int] = None
        if bucket_sample_indices is not None and bucket_sample_indices.size and selected_index is not None:
            selected_index = int(selected_index)
            if 0 <= selected_index < len(bucket_sample_indices):
                selected_global_idx = int(bucket_sample_indices[selected_index])

        config = configs[selected_index] if configs.size else np.zeros((0,), dtype=np.float32)
        LOGGER.debug("_handle_pick_event: selected config shape=%s, joint_names length=%d",
                   config.shape, len(data.joint_names))

        # Get the bucket center pose from the sphere dataset
        bucket_center_position = None
        bucket_center_quaternion = None
        if global_idx is not None and 0 <= global_idx < data.points.shape[0]:
            bucket_center_position = data.points[global_idx].copy()
            if data.orientations_quat.size > 0 and global_idx < data.orientations_quat.shape[0]:
                bucket_center_quaternion = data.orientations_quat[global_idx].copy()
            elif data.orientations_rpy.size > 0 and global_idx < data.orientations_rpy.shape[0]:
                # Convert RPY to quaternion if needed
                from embodik.examples.reachability_visualization import _rpy_to_quaternion_batch
                bucket_center_quaternion = _rpy_to_quaternion_batch(data.orientations_rpy[global_idx:global_idx+1])[0]

        if self._config_callback:
            LOGGER.debug("_handle_pick_event: calling config_callback with %d joint names", len(data.joint_names))
            # Pass pose information if available
            if bucket_center_position is not None and bucket_center_quaternion is not None:
                self._config_callback(data.joint_names, config.copy(), bucket_center_position, bucket_center_quaternion)
            else:
                self._config_callback(data.joint_names, config.copy())
        else:
            LOGGER.warning("_handle_pick_event: _config_callback is None!")
        self._set_status("Status: applied reachability sample")

        if self._validation_active():
            self._validation_engaged = True
            if selected_global_idx is not None:
                self._select_validation_sample(data, selected_global_idx)

    def _select_validation_sample(self, data: ReachabilityMapData, sample_idx: int) -> None:
        total_samples = data.sample_configs.shape[0]
        if total_samples == 0:
            self._validation_current_index = None
            return
        self._validation_engaged = True
        clamped_idx = max(0, min(sample_idx, total_samples - 1))
        self._validation_current_index = clamped_idx
        self._validation_selected_index = clamped_idx
        if not self._validation_active():
            if self._validation_index_number is not None and not self._validation_update_guard:
                self._validation_update_guard = True
                try:
                    self._validation_index_number.value = float(clamped_idx)
                finally:
                    self._validation_update_guard = False
            return
        self._validation_update_guard = True
        try:
            if self._validation_index_number is not None:
                self._validation_index_number.disabled = False
                self._validation_index_number.value = float(clamped_idx)
        finally:
            self._validation_update_guard = False
        self._update_validation_point(data)

    def _get_validation_index(self, data: ReachabilityMapData) -> int:
        if not data.sample_configs.size:
            return 0
        idx = 0
        if self._validation_index_number is not None:
            idx = int(round(self._validation_index_number.value))
        idx = max(0, min(idx, data.sample_configs.shape[0] - 1))
        if self._validation_index_number is not None:
            if not self._validation_update_guard:
                self._validation_update_guard = True
                try:
                    if self._validation_index_number.value != float(idx):
                        self._validation_index_number.value = float(idx)
                finally:
                    self._validation_update_guard = False
        return idx

    def _update_validation_point(self, data: ReachabilityMapData) -> None:
        if self._orientation_filter_checkbox and self._orientation_filter_checkbox.value:
            self._update_reference_orientation()
        enabled = self._validation_active()
        if not self._validation_engaged:
            if self._validation_point_handle is not None:
                self._validation_point_handle.remove()
                self._validation_point_handle = None
            if self._validation_status:
                self._validation_status.value = "Select a sample to validate"
            if self._validation_index_number is not None:
                self._validation_index_number.disabled = True
            if self._validation_config_dropdown is not None:
                self._validation_config_dropdown.options = ("-- none --",)
                self._validation_config_dropdown.value = "-- none --"
                self._validation_config_dropdown.disabled = True
                self._validation_config_mapping = {}
                self._validation_selected_index = None
            if self._validation_visitation_text is not None:
                self._validation_visitation_text.value = "Visitation: --"
            return
        if not enabled or not data.sample_configs.size:
            if self._validation_point_handle is not None:
                self._validation_point_handle.remove()
                self._validation_point_handle = None
            if not enabled and self._validation_status:
                self._validation_status.value = "Validation disabled"
            if self._validation_visitation_text is not None:
                self._validation_visitation_text.value = "Visitation: --"
            return

        idx = self._get_validation_index(data)
        point = data.sample_xyz[idx].reshape(1, 3)
        colors = np.array([[1.0, 0.2, 0.2]], dtype=np.float32)
        if self._validation_point_handle is None:
            self._validation_point_handle = self._server.scene.add_point_cloud(
                f"{self._scene_path}/validation_point",
                points=point,
                colors=colors,
                point_size=0.05,
                point_shape="circle",
            )
        else:
            self._validation_point_handle.points = point
            self._validation_point_handle.colors = colors
        self._validation_current_index = idx
        self._update_validation_config_dropdown(data, idx)
        configs_available = len(self._validation_config_mapping) if self._validation_config_mapping else 1
        applied_idx = self._apply_validation_configuration(data, idx)
        bucket_visitation = None
        bucket_key_tuple: Optional[Tuple[int, int, int]] = None
        if data.sample_indices.size:
            bucket_key_tuple = tuple(int(v) for v in data.sample_indices[idx])
            if data.visitation_lookup:
                value = data.visitation_lookup.get(bucket_key_tuple)
                if value is not None:
                    bucket_visitation = int(round(value))
        self._last_bucket_key = bucket_key_tuple
        self._set_validation_status_message(idx, point, configs_available, applied_idx, bucket_visitation)
        filtered_orientation_count: Optional[int] = None
        reference_quat, tolerance_rad = self._reference_quat_and_tolerance()
        if (
            bucket_key_tuple is not None
            and bucket_visitation is not None
            and reference_quat is not None
            and tolerance_rad is not None
        ):
            filtered_indices = self._bucket_filtered_sample_indices(
                data, bucket_key_tuple, reference_quat, tolerance_rad, allow_bucket_fallback=False
            )
            filtered_orientation_count = int(filtered_indices.size) if filtered_indices is not None else 0
        if self._validation_visitation_text is not None:
            if bucket_visitation is None:
                self._validation_visitation_text.value = " --"
            elif filtered_orientation_count is not None:
                self._validation_visitation_text.value = (
                    f"Raw: {bucket_visitation} | Filtered: {filtered_orientation_count}"
                )
            else:
                self._validation_visitation_text.value = f": {bucket_visitation}"

    def show_debug_target(self, position: Optional[np.ndarray]) -> None:
        if self._server is None:
            return
        if position is None:
            if self._debug_target_handle is not None:
                self._debug_target_handle.remove()
                self._debug_target_handle = None
            return
        point = np.asarray(position, dtype=np.float32).reshape(1, 3)
        if self._point_size_slider:
            base_size = max(
                reach_consts.MIN_POINT_SIZE_M, self._point_size_slider.value / 100.0
            )
        else:
            base_size = reach_consts.DEFAULT_POINT_SIZE_M
        point_size = base_size * reach_consts.IK_DEBUG_TARGET_POINT_SCALE
        rgba = np.array(reach_consts.IK_DEBUG_TARGET_COLOR, dtype=np.float32)
        colors = np.array([[rgba[0], rgba[1], rgba[2]]], dtype=np.float32)
        if self._debug_target_handle is None:
            self._debug_target_handle = self._server.scene.add_point_cloud(
                f"{self._scene_path}/ik_debug_target",
                points=point,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
            )
        else:
            self._debug_target_handle.points = point
            self._debug_target_handle.colors = colors
            self._debug_target_handle.point_size = point_size
