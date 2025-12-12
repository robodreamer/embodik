"""Helper class for common reachability example functionality.

This module provides a reusable helper class that encapsulates common patterns
used across reachability analysis examples (03, 04, 04_1), reducing code duplication
while preserving example-specific behaviors.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import viser

from .reachability_constants import (
    SPHERE_DATASET_MANIP_COL,
    SPHERE_DATASET_VISITATION_COL,
    SPHERE_DATASET_ROM_COL,
    SPHERE_DATASET_SINGULARITY_COL,
    SPHERE_DATASET_MIN_COLS,
    SPHERE_DATASET_MIN_COLS_WITH_METRICS,
    SPHERE_DATASET_OLD_NUM_COLS,
    IK_REACH_MAP_MANIP_COL,
    IK_REACH_MAP_VISITATION_COL,
    IK_REACH_MAP_ROM_COL,
    IK_REACH_MAP_SINGULARITY_COL,
)
from .reachability_metrics import (
    ReachabilityMetricHelper,
    normalization_cache_path,
    load_normalization_cache,
    save_normalization_cache,
    RANGE_OF_MOTION_NORMALIZED_LIMIT,
)
from .reachability_scoring import (
    compute_average_metric_score,
    compute_weighted_metric_score,
    compute_coverage_score,
)
from .reachability_ik import (
    IKReachabilityConfig,
    generate_cartesian_grid,
    create_grid_preview_file,
)
from .reachability_gui import ReachabilityGuiHandles, ReachabilityJobState
from .reachability_visualization import ReachabilityMapViewer

LOGGER = logging.getLogger("embodik.reachability_example_helper")

__all__ = ["ReachabilityExampleHelper"]


class ReachabilityExampleHelper:
    """Helper class for common reachability example functionality.

    This class encapsulates shared patterns for:
    - Metric display and normalization
    - Grid preview generation
    - Reachability job launching
    - Post-processing and metadata management
    """

    def __init__(
        self,
        server: viser.ViserServer,
        viewer: ReachabilityMapViewer,
        reachability_gui: ReachabilityGuiHandles,
        reachability_state: ReachabilityJobState,
        reachability_lock: threading.Lock,
        metric_helpers: Dict[str, Optional[ReachabilityMetricHelper]],
        metric_display_handles: Dict[str, viser.GuiTextHandle],
        normalization_status_handle: Optional[viser.GuiTextHandle],
        normalization_sample_slider: Optional[viser.GuiSliderHandle],
        normalization_button: Optional[viser.GuiButtonHandle],
        current_metric_ranges: Dict[str, Tuple[float, float]],
        metric_metadata: Dict[str, float],
        normalization_cache_file: Path,
        output_dir: Path,
        *,
        get_q_current: Callable[[], np.ndarray],
        update_metric_display_callback: Optional[Callable[[], None]] = None,
        get_selected_arm: Optional[Callable[[], str]] = None,
        get_reference_position: Optional[Callable[[], np.ndarray]] = None,
        manip_scaling: float = 500.0,
    ):
        """Initialize the helper.

        Args:
            server: Viser server instance
            viewer: ReachabilityMapViewer instance
            reachability_gui: ReachabilityGuiHandles instance
            reachability_state: Shared reachability job state
            reachability_lock: Lock for thread-safe access to state
            metric_helpers: Dictionary mapping arm names to metric helpers
            metric_display_handles: Dictionary of GUI text handles for metrics
            normalization_status_handle: GUI text handle for normalization status
            normalization_sample_slider: GUI slider for normalization samples
            normalization_button: GUI button for normalization estimation
            current_metric_ranges: Current metric normalization ranges
            metric_metadata: Metric metadata dictionary
            normalization_cache_file: Path to normalization cache file
            output_dir: Output directory for reachability maps
            get_q_current: Callable that returns current joint configuration
            update_metric_display_callback: Optional callback to update metric display
            get_selected_arm: Optional callable that returns selected arm name
            get_reference_position: Optional callable that returns reference position for grid preview
            manip_scaling: Manipulability scaling factor
        """
        self.server = server
        self.viewer = viewer
        self.reachability_gui = reachability_gui
        self.reachability_state = reachability_state
        self.reachability_lock = reachability_lock
        self.metric_helpers = metric_helpers
        self.metric_display_handles = metric_display_handles
        self.normalization_status_handle = normalization_status_handle
        self.normalization_sample_slider = normalization_sample_slider
        self.normalization_button = normalization_button
        self.current_metric_ranges = current_metric_ranges
        self.metric_metadata = metric_metadata
        self.normalization_cache_file = normalization_cache_file
        self.output_dir = output_dir
        self.get_q_current = get_q_current
        self.update_metric_display_callback = update_metric_display_callback
        self.get_selected_arm = get_selected_arm
        self.get_reference_position = get_reference_position
        self.manip_scaling = manip_scaling

        # Register normalization button callback
        if self.normalization_button:
            self.normalization_button.on_click(self._on_normalization_click)

    def update_metric_display(self) -> None:
        """Update metric display GUI with current values."""
        if self.update_metric_display_callback:
            self.update_metric_display_callback()
        elif self.metric_display_handles:
            # Fallback: use metric helpers directly
            selected_arm = "left"
            if self.get_selected_arm:
                try:
                    selected_arm = self.get_selected_arm()
                except Exception:
                    pass

            metric_helper = self.metric_helpers.get(selected_arm)
            if metric_helper is None:
                for handle in self.metric_display_handles.values():
                    if handle:
                        handle.value = "norm: -- | raw: --"
                return

            try:
                values = metric_helper.compute_metrics(
                    q_current=self.get_q_current(),
                    current_metric_ranges=self.current_metric_ranges,
                    viewer=self.viewer,
                )

                handle = self.metric_display_handles.get("Manipulability")
                if handle:
                    handle.value = f"norm: {values['manip_norm']:.3f} | raw: {values['manip_raw']:.4f}"
                handle = self.metric_display_handles.get("RangeOfMotion")
                if handle:
                    handle.value = f"norm: {values['rom_norm']:.3f} | raw: {values['rom_raw']:.4f}"
                handle = self.metric_display_handles.get("SingularityAvoidance")
                if handle:
                    handle.value = f"norm: {values['sing_norm']:.3f} | raw: {values['sing_raw']:.4f}"
                handle = self.metric_display_handles.get("JointLimitDistance")
                if handle:
                    handle.value = f"score: {values['limit_norm']:.3f} | raw: {values['limit_distance']:.4f}"
                handle = self.metric_display_handles.get("Visitation")
                if handle:
                    if values["vis_raw"] is None or values["vis_norm"] is None:
                        handle.value = "norm: -- | raw: --"
                    else:
                        handle.value = f"norm: {values['vis_norm']:.3f} | raw: {values['vis_raw']:.1f}"
            except Exception as e:
                LOGGER.debug("Failed to update metric display: %s", e)

    def _on_normalization_click(self, _event) -> None:
        """Handle normalization button click."""
        if not self.normalization_sample_slider:
            return

        sample_count = max(0, int(self.normalization_sample_slider.value))
        if self.normalization_status_handle:
            self.normalization_status_handle.value = f"Status: running ({sample_count} samples)..."

        selected_arm = "left"
        if self.get_selected_arm:
            try:
                selected_arm = self.get_selected_arm()
            except Exception:
                pass

        metric_helper = self.metric_helpers.get(selected_arm)
        if metric_helper is None:
            if self.normalization_status_handle:
                self.normalization_status_handle.value = "Status: error (metric helper not available)"
            return

        try:
            manip_max, rom_max, sing_max = metric_helper.estimate_metric_ranges(sample_count)
        except Exception as exc:
            if self.normalization_status_handle:
                self.normalization_status_handle.value = f"Status: error ({exc})"
            return

        self.metric_metadata["ManipulabilityRawMax"] = float(manip_max)
        self.metric_metadata["RangeOfMotionRawMax"] = float(rom_max)
        self.metric_metadata["SingularityWeightedMax"] = float(sing_max)
        self.metric_metadata.setdefault("SingularityRawMax", float(sing_max))

        # Update all helpers' metadata
        for helper in self.metric_helpers.values():
            if helper is not None:
                helper.metric_metadata.update(self.metric_metadata)

        save_normalization_cache(self.normalization_cache_file, self.metric_metadata)
        if self.normalization_status_handle:
            self.normalization_status_handle.value = (
                f"Status: Manip max {manip_max:.4f}, ROM max {rom_max:.2f}, Sing max {sing_max:.3f}"
            )
        self.update_metric_display()

    def sync_metric_metadata(self) -> None:
        """Sync metric metadata from loaded map."""
        data = self.viewer._current_map()
        if data is None:
            return

        self.current_metric_ranges.update(data.metric_ranges)
        self.current_metric_ranges.setdefault("Manipulability", (0.0, 1.0))
        self.current_metric_ranges.setdefault("RangeOfMotion", (0.0, 1.0))
        self.current_metric_ranges.setdefault("SingularityAvoidance", (0.0, 1.0))
        self.current_metric_ranges.setdefault("Visitation", (0.0, 1.0))

        new_metadata = self.metric_metadata.copy()
        new_metadata.update(data.metadata)

        if new_metadata.get("RangeOfMotionRawMax", 0.0) <= 0.0:
            new_metadata["RangeOfMotionRawMax"] = RANGE_OF_MOTION_NORMALIZED_LIMIT
        if new_metadata.get("SingularityWeightedMax", 0.0) <= 0.0:
            new_metadata["SingularityWeightedMax"] = 1.0
        if new_metadata.get("ManipulabilityRawMax", 0.0) <= 0.0:
            manip_scaling = new_metadata.get("ManipulabilityScaling", self.manip_scaling)
            new_metadata["ManipulabilityRawMax"] = float(manip_scaling) * 0.1

        self.metric_metadata.update(new_metadata)

        # Update all helpers' metadata
        for helper in self.metric_helpers.values():
            if helper is not None:
                helper.metric_metadata.update(self.metric_metadata)

        self.update_metric_display()

    def generate_grid_preview(
        self,
        *,
        cart_res: float,
        grid_size_scale: float,
        grid_center_offset: Tuple[float, float, float],
        base_x_limits: Tuple[float, float],
        base_y_limits: Tuple[float, float],
        base_z_limits: Tuple[float, float],
        reference_position: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """Generate grid preview file.

        Args:
            cart_res: Cartesian resolution (m)
            grid_size_scale: Grid size scaling factor
            grid_center_offset: Grid center offset (x, y, z) in meters
            base_x_limits: Base X limits (min, max)
            base_y_limits: Base Y limits (min, max)
            base_z_limits: Base Z limits (min, max)
            reference_position: Optional reference position (defaults to get_reference_position)

        Returns:
            Path to generated preview file, or None if generation failed
        """
        if reference_position is None:
            if self.get_reference_position:
                try:
                    reference_position = self.get_reference_position()
                except Exception:
                    LOGGER.warning("Failed to get reference position for grid preview")
                    return None
            else:
                reference_position = np.array([0.0, 0.0, 0.0])

        effective_grid_step = cart_res

        def scale_limits(limits: Tuple[float, float], scale: float) -> Tuple[float, float]:
            """Scale limits around their center."""
            center = (limits[0] + limits[1]) / 2.0
            half_range = (limits[1] - limits[0]) / 2.0
            scaled_half_range = half_range * scale
            return (center - scaled_half_range, center + scaled_half_range)

        scaled_x_limits = scale_limits(base_x_limits, grid_size_scale)
        scaled_y_limits = scale_limits(base_y_limits, grid_size_scale)
        scaled_z_limits = scale_limits(base_z_limits, grid_size_scale)

        # Compute grid center: reference position + offset
        grid_center = np.array([
            reference_position[0] + grid_center_offset[0],
            reference_position[1] + grid_center_offset[1],
            reference_position[2] + grid_center_offset[2]
        ], dtype=np.float32)

        # Center the scaled limits around grid_center
        x_half_range = (scaled_x_limits[1] - scaled_x_limits[0]) / 2.0
        y_half_range = (scaled_y_limits[1] - scaled_y_limits[0]) / 2.0
        z_half_range = (scaled_z_limits[1] - scaled_z_limits[0]) / 2.0

        final_x_limits = (grid_center[0] - x_half_range, grid_center[0] + x_half_range)
        final_y_limits = (grid_center[1] - y_half_range, grid_center[1] + y_half_range)
        final_z_limits = (grid_center[2] - z_half_range, grid_center[2] + z_half_range)

        preview_config = IKReachabilityConfig(
            grid_step_size=effective_grid_step,
            x_limits=final_x_limits,
            y_limits=final_y_limits,
            z_limits=final_z_limits,
            cartesian_resolution=cart_res,
            manip_scaling=self.manip_scaling,
        )

        # Generate grid points
        grid_points = generate_cartesian_grid(preview_config, reference_position, reach_radius=None)

        # Create preview file
        grid_preview_path = create_grid_preview_file(
            preview_config, grid_points, self.output_dir, reference_position
        )

        if grid_preview_path is not None and grid_preview_path.exists():
            # Clear cache entry for preview file to force reload of updated data
            preview_name = grid_preview_path.name
            if hasattr(self.viewer, '_map_cache') and preview_name in self.viewer._map_cache:
                del self.viewer._map_cache[preview_name]

            # Refresh map list first
            self.viewer.refresh_options()

            # Enable "Show map" BEFORE loading
            if self.viewer._show_checkbox is not None:
                self.viewer._show_checkbox.value = True

            # Load the preview map
            LOGGER.info("Loading grid preview map...")
            self.viewer.on_new_map(grid_preview_path)

        return grid_preview_path

    def post_process_reachability_map(
        self,
        map_path: Path,
        *,
        is_ik_map: bool = False,
        cache_key: Optional[str] = None,
    ) -> None:
        """Post-process a generated reachability map.

        Computes global scores, stores metadata, and updates normalization cache.

        Args:
            map_path: Path to the generated HDF5 map file
            is_ik_map: Whether this is an IK-generated map (affects column layout)
            cache_key: Optional cache key for normalization (defaults to map filename)
        """
        try:
            LOGGER.info("Post-processing reachability results...")
            with h5py.File(map_path, "r+") as f:
                if "/Spheres/sphere_dataset" not in f:
                    LOGGER.warning("Map file missing sphere dataset")
                    return

                dataset = f["/Spheres/sphere_dataset"]
                reach_map_np = dataset[:]

                if is_ik_map:
                    # IK map sphere_dataset format: xyz(3) + manipulability(1) + visitation(1) + rom(1) + sing(1) = 7 columns
                    if reach_map_np.shape[1] >= SPHERE_DATASET_MIN_COLS_WITH_METRICS:
                        manip_values = reach_map_np[:, SPHERE_DATASET_MANIP_COL]
                        visitation = reach_map_np[:, SPHERE_DATASET_VISITATION_COL]
                        rom_values = reach_map_np[:, SPHERE_DATASET_ROM_COL]
                        sing_values = reach_map_np[:, SPHERE_DATASET_SINGULARITY_COL]
                    else:
                        LOGGER.warning("IK map has insufficient columns: %d (expected at least %d)", reach_map_np.shape[1], SPHERE_DATASET_MIN_COLS_WITH_METRICS)
                        return
                else:
                    # FK map format:
                    # Old format (5 cols): xyz(3) + manip(1) + visitation(1) = 5 columns
                    # New format (7 cols): xyz(3) + manip(1) + visitation(1) + rom(1) + sing(1) = 7 columns
                    if reach_map_np.shape[1] >= SPHERE_DATASET_MIN_COLS_WITH_METRICS:
                        # New format with ROM and Singularity
                        manip_values = reach_map_np[:, SPHERE_DATASET_MANIP_COL]
                        visitation = reach_map_np[:, SPHERE_DATASET_VISITATION_COL]
                        rom_values = reach_map_np[:, SPHERE_DATASET_ROM_COL]
                        sing_values = reach_map_np[:, SPHERE_DATASET_SINGULARITY_COL]
                    elif reach_map_np.shape[1] >= SPHERE_DATASET_OLD_NUM_COLS:
                        # Old format without ROM and Singularity (backward compatibility)
                        LOGGER.info("FK map has old format (%d columns), ROM and Singularity metrics not available", SPHERE_DATASET_OLD_NUM_COLS)
                        manip_values = reach_map_np[:, SPHERE_DATASET_MANIP_COL]
                        visitation = reach_map_np[:, SPHERE_DATASET_VISITATION_COL]
                        # Create dummy ROM and Singularity arrays (all zeros) for compatibility
                        rom_values = np.zeros_like(manip_values)
                        sing_values = np.zeros_like(manip_values)
                    else:
                        LOGGER.warning("FK map has insufficient columns: %d (expected at least %d)", reach_map_np.shape[1], SPHERE_DATASET_MIN_COLS)
                        return

                # Only consider visited voxels
                existing_coverage: Optional[Dict[str, float]] = None
                existing_scores_raw = dataset.attrs.get("GlobalScores")
                if existing_scores_raw is not None:
                    try:
                        if isinstance(existing_scores_raw, bytes):
                            existing_scores_raw = existing_scores_raw.decode("utf-8")
                        existing_scores = json.loads(existing_scores_raw)
                        coverage_candidate = existing_scores.get("coverage")
                        if isinstance(coverage_candidate, dict):
                            existing_coverage = coverage_candidate
                    except (json.JSONDecodeError, AttributeError):
                        existing_coverage = None

                visited_mask = visitation > 0
                if not np.any(visited_mask):
                    LOGGER.warning("No visited voxels in map")
                    return

                visited_manip = manip_values[visited_mask]
                visited_rom = rom_values[visited_mask]
                visited_sing = sing_values[visited_mask]
                visited_weights = visitation[visited_mask]

                coverage_stats = existing_coverage
                if coverage_stats is None:
                    visited_count = int(np.count_nonzero(visited_mask))
                    total_count = int(visitation.shape[0])
                    coverage_stats = compute_coverage_score(visited_count, total_count)

                metric_scores = {
                    "Manipulability": {
                        **compute_average_metric_score(visited_manip, "Manipulability"),
                        **compute_weighted_metric_score(visited_manip, visited_weights, "Manipulability"),
                    },
                    "RangeOfMotion": {
                        **compute_average_metric_score(visited_rom, "RangeOfMotion"),
                        **compute_weighted_metric_score(visited_rom, visited_weights, "RangeOfMotion"),
                    },
                    "SingularityAvoidance": {
                        **compute_average_metric_score(visited_sing, "SingularityAvoidance"),
                        **compute_weighted_metric_score(visited_sing, visited_weights, "SingularityAvoidance"),
                    },
                }

                # Store global scores in HDF5 attributes
                coverage_payload = {}
                if coverage_stats:
                    coverage_payload = {
                        k: float(v)
                        if isinstance(v, (np.integer, np.floating))
                        else int(v)
                        if isinstance(v, np.integer)
                        else v
                        for k, v in coverage_stats.items()
                    }

                scores_json = {"coverage": coverage_payload, "metrics": {}}
                for metric_name, scores in metric_scores.items():
                    scores_json["metrics"][metric_name] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v
                        for k, v in scores.items()
                    }
                json_str = json.dumps(scores_json)

                str_dtype = h5py.string_dtype(encoding='utf-8')
                if "GlobalScores" in dataset.attrs:
                    del dataset.attrs["GlobalScores"]
                dataset.attrs.create("GlobalScores", json_str, dtype=str_dtype)

                if coverage_stats:
                    LOGGER.info(
                        "Computed global scores: coverage=%.1f%% (%d/%d), metrics=%s",
                        coverage_stats.get("coverage_ratio", 0.0) * 100.0,
                        coverage_stats.get("valid_count", 0),
                        coverage_stats.get("total_count", 0),
                        ", ".join(
                            f"{name}: {metrics.get('average', 0.0):.3f}"
                            for name, metrics in metric_scores.items()
                        ),
                    )
                else:
                    LOGGER.info(
                        "Computed global scores (coverage unavailable): metrics=%s",
                        ", ".join(
                            f"{name}: {metrics.get('average', 0.0):.3f}"
                            for name, metrics in metric_scores.items()
                        ),
                    )

                # Store metric ranges in metadata
                manip_max = float(np.max(visited_manip)) if len(visited_manip) > 0 else 1.0
                rom_max = float(np.max(visited_rom)) if len(visited_rom) > 0 else 1.0
                sing_max = float(np.max(visited_sing)) if len(visited_sing) > 0 else 1.0

                # Update normalization cache
                if cache_key is None:
                    cache_key = map_path.stem

                normalization_cache_file = normalization_cache_path(self.output_dir, cache_key)
                cache_metadata = {
                    "ManipulabilityRawMax": max(manip_max, 1e-6),
                    "RangeOfMotionRawMax": max(rom_max, 1e-6),
                    "SingularityWeightedMax": max(sing_max, 1e-6),
                    "ManipulabilityScaling": self.manip_scaling,
                }
                save_normalization_cache(normalization_cache_file, cache_metadata)
                LOGGER.info("Saved normalization cache: Manip max=%.4f, ROM max=%.4f, Sing max=%.4f",
                           manip_max, rom_max, sing_max)
        except Exception as e:
            LOGGER.warning("Failed to post-process reachability map: %s", e)

    def _extract_coverage_info(self, map_path: Path) -> Optional[Dict[str, float]]:
        """Extract reachability coverage (valid vs total) from a reachability map if present."""
        if not map_path.exists():
            return None
        try:
            with h5py.File(map_path, "r") as handle:
                sphere_group = handle.get("Spheres")
                if sphere_group is None or "sphere_dataset" not in sphere_group:
                    return None
                dataset = sphere_group["sphere_dataset"]
                scores_json = dataset.attrs.get("GlobalScores")
                if scores_json is None:
                    return None
                if isinstance(scores_json, bytes):
                    scores_json = scores_json.decode("utf-8")
                global_scores = json.loads(scores_json)
                coverage = global_scores.get("coverage")
                if isinstance(coverage, dict):
                    return coverage
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            LOGGER.debug("Failed to read coverage info from %s: %s", map_path, exc)
        return None

    def create_progress_callback(self) -> Callable[[float, str], None]:
        """Create a progress callback function.

        Returns:
            Callable that updates reachability_state with progress
        """
        def _progress_callback(value: float, stage: str) -> None:
            with self.reachability_lock:
                self.reachability_state.progress = max(0.0, min(float(value), 1.0))
                self.reachability_state.stage = stage

        return _progress_callback

    def update_controls_for_mode(
        self,
        *,
        grid_center_offset_sliders: Optional[Tuple[Optional[viser.GuiSliderHandle], Optional[viser.GuiSliderHandle], Optional[viser.GuiSliderHandle]]] = None,
    ) -> None:
        """Update GUI controls based on analysis mode.

        Enables/disables FK and IK-specific controls based on the current mode.

        Args:
            grid_center_offset_sliders: Optional tuple of (x, y, z) grid center offset sliders
        """
        mode = self.reachability_gui.analysis_mode_dropdown.value
        is_ik_mode = mode == "IK Grid"

        # Enable/disable FK-specific controls
        self.reachability_gui.samples_slider.disabled = is_ik_mode
        self.reachability_gui.batch_slider.disabled = is_ik_mode
        self.reachability_gui.post_process_checkbox.disabled = is_ik_mode

        # Enable/disable IK-specific controls
        if self.reachability_gui.ik_orientation_mode_dropdown:
            self.reachability_gui.ik_orientation_mode_dropdown.disabled = not is_ik_mode
        if self.reachability_gui.ik_nullspace_bias_checkbox:
            self.reachability_gui.ik_nullspace_bias_checkbox.disabled = not is_ik_mode
        if self.reachability_gui.ik_show_process_checkbox:
            self.reachability_gui.ik_show_process_checkbox.disabled = not is_ik_mode
        if self.reachability_gui.ik_grid_size_slider:
            self.reachability_gui.ik_grid_size_slider.disabled = not is_ik_mode
        if self.reachability_gui.ik_show_grid_preview_checkbox:
            self.reachability_gui.ik_show_grid_preview_checkbox.disabled = not is_ik_mode
        if self.reachability_gui.ik_num_workers_slider:
            self.reachability_gui.ik_num_workers_slider.disabled = not is_ik_mode
        interactive_solver_checkbox = getattr(self.reachability_gui, "ik_use_interactive_solver_checkbox", None)
        if interactive_solver_checkbox:
            interactive_solver_checkbox.disabled = not is_ik_mode
            if interactive_solver_checkbox.disabled:
                interactive_solver_checkbox.value = False
        pos_tol_slider = getattr(self.reachability_gui, "ik_position_tol_slider", None)
        if pos_tol_slider:
            pos_tol_slider.disabled = not is_ik_mode
        ori_tol_slider = getattr(self.reachability_gui, "ik_orientation_tol_slider", None)
        if ori_tol_slider:
            ori_tol_slider.disabled = not is_ik_mode
        traversal_dropdown = getattr(self.reachability_gui, "ik_traversal_mode_dropdown", None)
        if traversal_dropdown:
            traversal_dropdown.disabled = not is_ik_mode
        if getattr(self.reachability_gui, "ik_interactive_debug_checkbox", None):
            checkbox = self.reachability_gui.ik_interactive_debug_checkbox
            base_enabled = (
                is_ik_mode
                and self.reachability_gui.ik_show_process_checkbox
                and self.reachability_gui.ik_show_process_checkbox.value
            )
            checkbox.disabled = not base_enabled
            if checkbox.disabled:
                checkbox.value = False
        if getattr(self.reachability_gui, "ik_debug_next_button", None):
            self.reachability_gui.ik_debug_next_button.disabled = True

        # Enable/disable grid center offset controls (if provided)
        if grid_center_offset_sliders:
            x_slider, y_slider, z_slider = grid_center_offset_sliders
            if x_slider:
                x_slider.disabled = not is_ik_mode
            if y_slider:
                y_slider.disabled = not is_ik_mode
            if z_slider:
                z_slider.disabled = not is_ik_mode

    def create_grid_center_offset_sliders(
        self,
        *,
        default_z_offset: float = 0.0,
        expand_by_default: bool = False,
    ) -> Tuple[viser.GuiSliderHandle, viser.GuiSliderHandle, viser.GuiSliderHandle]:
        """Create grid center offset sliders.

        Args:
            default_z_offset: Default Z offset value (default: 0.0 for base origin, 0.3 for end-effector reference)
            expand_by_default: Whether to expand the folder by default (unused; kept for compatibility)

        Returns:
            Tuple of (x_slider, y_slider, z_slider) handles
        """
        # These sliders now live in the Reachability Mapping panel (04_hmnd_alpha_example sets up the folder).
        x_slider = self.server.gui.add_slider(
            "Grid center X offset (m)",
            min=-1.0,
            max=1.0,
            initial_value=0.0,
            step=0.05,
            disabled=True,
        )
        y_slider = self.server.gui.add_slider(
            "Grid center Y offset (m)",
            min=-1.0,
            max=1.0,
            initial_value=0.0,
            step=0.05,
            disabled=True,
        )
        z_slider = self.server.gui.add_slider(
            "Grid center Z offset (m)",
            min=-0.5,
            max=1.0,
            initial_value=default_z_offset,
            step=0.05,
            disabled=True,
        )
        return (x_slider, y_slider, z_slider)

    def compute_final_grid_limits(
        self,
        *,
        base_x_limits: Tuple[float, float],
        base_y_limits: Tuple[float, float],
        base_z_limits: Tuple[float, float],
        grid_size_scale: float,
        grid_center_offset: Tuple[float, float, float],
        reference_position: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Compute final grid limits from base limits, scale, offset, and reference position.

        Args:
            base_x_limits: Base X limits (min, max)
            base_y_limits: Base Y limits (min, max)
            base_z_limits: Base Z limits (min, max)
            grid_size_scale: Grid size scaling factor
            grid_center_offset: Grid center offset (x, y, z) in meters
            reference_position: Reference position (e.g., base origin or end-effector position)

        Returns:
            Tuple of (final_x_limits, final_y_limits, final_z_limits)
        """
        def scale_limits(limits: Tuple[float, float], scale: float) -> Tuple[float, float]:
            """Scale limits around their center."""
            center = (limits[0] + limits[1]) / 2.0
            half_range = (limits[1] - limits[0]) / 2.0
            scaled_half_range = half_range * scale
            return (center - scaled_half_range, center + scaled_half_range)

        # Scale limits first
        scaled_x_limits = scale_limits(base_x_limits, grid_size_scale)
        scaled_y_limits = scale_limits(base_y_limits, grid_size_scale)
        scaled_z_limits = scale_limits(base_z_limits, grid_size_scale)

        # Compute grid center: reference position + offset
        grid_center = np.array([
            reference_position[0] + grid_center_offset[0],
            reference_position[1] + grid_center_offset[1],
            reference_position[2] + grid_center_offset[2]
        ], dtype=np.float32)

        # Center the scaled limits around grid_center
        x_half_range = (scaled_x_limits[1] - scaled_x_limits[0]) / 2.0
        y_half_range = (scaled_y_limits[1] - scaled_y_limits[0]) / 2.0
        z_half_range = (scaled_z_limits[1] - scaled_z_limits[0]) / 2.0

        final_x_limits = (grid_center[0] - x_half_range, grid_center[0] + x_half_range)
        final_y_limits = (grid_center[1] - y_half_range, grid_center[1] + y_half_range)
        final_z_limits = (grid_center[2] - z_half_range, grid_center[2] + z_half_range)

        return (final_x_limits, final_y_limits, final_z_limits)

    def setup_grid_preview_live_updates(
        self,
        *,
        grid_preview_callback: Callable[[], None],
        grid_size_slider: Optional[viser.GuiSliderHandle] = None,
        grid_center_x_slider: Optional[viser.GuiSliderHandle] = None,
        grid_center_y_slider: Optional[viser.GuiSliderHandle] = None,
        grid_center_z_slider: Optional[viser.GuiSliderHandle] = None,
        cart_res_slider: Optional[viser.GuiSliderHandle] = None,
    ) -> None:
        """Set up live-update callbacks for grid preview.

        When any of the specified sliders change, the grid preview will be automatically
        regenerated if the preview checkbox is enabled.

        Args:
            grid_preview_callback: Callback function to regenerate grid preview
            grid_size_slider: Optional grid size scale slider
            grid_center_x_slider: Optional grid center X offset slider
            grid_center_y_slider: Optional grid center Y offset slider
            grid_center_z_slider: Optional grid center Z offset slider
            cart_res_slider: Optional cartesian resolution slider
        """
        def _on_slider_changed(_event) -> None:
            """Handle slider changes - regenerate preview if shown."""
            if self.reachability_gui.ik_show_grid_preview_checkbox and self.reachability_gui.ik_show_grid_preview_checkbox.value:
                grid_preview_callback()

        # Register callbacks for all provided sliders
        if grid_size_slider:
            grid_size_slider.on_update(_on_slider_changed)
        if grid_center_x_slider:
            grid_center_x_slider.on_update(_on_slider_changed)
        if grid_center_y_slider:
            grid_center_y_slider.on_update(_on_slider_changed)
        if grid_center_z_slider:
            grid_center_z_slider.on_update(_on_slider_changed)
        if cart_res_slider:
            cart_res_slider.on_update(_on_slider_changed)

    def cancel_reachability_job(self) -> None:
        """Cancel the current reachability job if running."""
        with self.reachability_lock:
            cancel_event = self.reachability_state.cancel_event
            running = bool(self.reachability_state.running)
        if running and isinstance(cancel_event, threading.Event):
            cancel_event.set()
            self.reachability_gui.status_text.value = "Status: Cancelling"

    def setup_reachability_job(
        self,
        cancel_event: threading.Event,
        status_message: str = "Status: Starting",
    ) -> None:
        """Set up state and GUI for a new reachability job.

        Args:
            cancel_event: Event for cancelling the job
            status_message: Initial status message
        """
        with self.reachability_lock:
            self.reachability_state.running = True
            self.reachability_state.progress = 0.0
            self.reachability_state.stage = "starting"
            self.reachability_state.outputs = None
            self.reachability_state.error = None
            self.reachability_state.cancel_event = cancel_event

        self.reachability_gui.generate_button.disabled = True
        self.reachability_gui.cancel_button.disabled = False
        self.reachability_gui.progress_value.value = 0.0
        self.reachability_gui.last_output_text.value = "Map: --"
        self.reachability_gui.status_text.value = status_message

    def start_reachability_worker(
        self,
        worker_func: Callable[[], None],
        thread_name: str,
    ) -> None:
        """Start a worker thread for reachability generation.

        Args:
            worker_func: Worker function to run in thread
            thread_name: Name for the worker thread
        """
        worker_thread = threading.Thread(target=worker_func, daemon=True, name=thread_name)
        with self.reachability_lock:
            self.reachability_state.thread = worker_thread
        worker_thread.start()

    def finalize_reachability_job(
        self,
        outputs: Optional[Dict[str, Path]],
        error_message: Optional[str],
        cancel_event: threading.Event,
    ) -> None:
        """Finalize reachability job state after completion.

        Args:
            outputs: Job outputs dictionary
            error_message: Error message if job failed
            cancel_event: Cancellation event
        """
        with self.reachability_lock:
            self.reachability_state.outputs = outputs
            self.reachability_state.error = error_message
            self.reachability_state.running = False
            self.reachability_state.cancel_event = None
            self.reachability_state.thread = None
            if error_message:
                self.reachability_state.stage = "error"
            elif cancel_event.is_set():
                self.reachability_state.stage = "cancelled"
            else:
                self.reachability_state.stage = "completed"
                self.reachability_state.progress = 1.0

        coverage_info: Optional[Dict[str, float]] = None
        job_metadata: Optional[Dict[str, object]] = None
        map_path: Optional[Path] = None
        if outputs:
            map_path = outputs.get("map_3d") or outputs.get("map")
            if map_path is not None and map_path.exists():
                coverage_info = self._extract_coverage_info(map_path)
                if coverage_info:
                    self.reachability_state.metadata["coverage"] = coverage_info
            metadata_path = outputs.get("metadata")
            if metadata_path is not None and metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as meta_file:
                        job_metadata = json.load(meta_file)
                    self.reachability_state.metadata["job_metadata"] = job_metadata
                except Exception as exc:
                    LOGGER.warning("Failed to read reachability metadata file %s: %s", metadata_path, exc)

        if error_message:
            self.reachability_gui.status_text.value = f"Status: Error ({error_message})"
            self.reachability_gui.last_output_text.value = "Map: --"
        elif cancel_event.is_set():
            self.reachability_gui.status_text.value = "Status: Cancelled"
            self.reachability_gui.last_output_text.value = "Map: --"
        else:
            self.reachability_gui.status_text.value = "Status: Completed"
            if map_path is not None:
                if coverage_info:
                    coverage_ratio = coverage_info.get("coverage_ratio", 0.0)
                    valid_count = int(coverage_info.get("valid_count", 0))
                    total_count = int(coverage_info.get("total_count", 0))
                    coverage_text = f"{valid_count}/{total_count} ({coverage_ratio:.1%})"
                    self.reachability_gui.last_output_text.value = (
                        f"Map: {map_path.name} | Reachable: {coverage_text}"
                    )
                else:
                    self.reachability_gui.last_output_text.value = f"Map: {map_path.name}"
            else:
                self.reachability_gui.last_output_text.value = "Map: --"

        self.reachability_gui.generate_button.disabled = False
        self.reachability_gui.cancel_button.disabled = True

        if map_path is not None and self.viewer is not None:
            try:
                self.viewer.on_new_map(map_path)
            except Exception as exc:  # pragma: no cover - diagnostics only
                LOGGER.warning("Failed to load map '%s' in viewer: %s", map_path, exc)

        if job_metadata:
            fk_meta = job_metadata.get("fk_envelope")
            if isinstance(fk_meta, dict):
                enabled = fk_meta.get("enabled")
                removed = fk_meta.get("removed_points", 0)
                initial = fk_meta.get("initial_grid_points", 0)
                remaining = fk_meta.get("remaining_grid_points", 0)
                samples = fk_meta.get("samples", 0)
                margin = fk_meta.get("margin_m", 0.0)
                if enabled:
                    LOGGER.info(
                        "FK envelope pruning removed %d of %d grid points (remaining=%d, samples=%d, margin=%.3fm).",
                        removed,
                        initial,
                        remaining,
                        samples,
                        margin,
                    )
                else:
                    LOGGER.info(
                        "FK envelope pruning disabled for this job (initial grid points=%d).",
                        initial,
                    )
