"""Shared helper for reachability GUI panel creation.

This module provides reusable components for creating reachability analysis GUI panels
that can be used across different robot examples.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import viser

LOGGER = logging.getLogger("embodik.reachability_gui")

__all__ = [
    "ReachabilityJobState",
    "ReachabilityGuiHandles",
    "create_reachability_gui_panel",
]


@dataclass
class ReachabilityJobState:
    """State container for reachability generation jobs."""

    thread: Optional[threading.Thread] = None
    cancel_event: Optional[threading.Event] = None
    running: bool = False
    progress: float = 0.0
    stage: str = "idle"
    outputs: Optional[Dict[str, Path]] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class IKControlHandles:
    """IK-specific control handles."""

    orientation_mode_dropdown: viser.GuiDropdownHandle
    nullspace_bias_checkbox: viser.GuiCheckboxHandle
    show_process_checkbox: viser.GuiCheckboxHandle
    grid_size_slider: viser.GuiSliderHandle
    show_grid_preview_checkbox: viser.GuiCheckboxHandle
    traversal_mode_dropdown: viser.GuiDropdownHandle
    position_tol_slider: viser.GuiSliderHandle
    orientation_tol_slider: viser.GuiSliderHandle
    use_interactive_solver_checkbox: viser.GuiCheckboxHandle
    num_workers_slider: viser.GuiSliderHandle
    interactive_debug_checkbox: viser.GuiCheckboxHandle
    debug_next_button: viser.GuiButtonHandle


@dataclass
class ReachabilityGuiHandles:
    """Container for reachability GUI handles."""

    analysis_mode_dropdown: viser.GuiDropdownHandle
    samples_slider: viser.GuiSliderHandle
    batch_slider: viser.GuiSliderHandle
    cart_res_slider: viser.GuiSliderHandle
    ang_res_slider: viser.GuiSliderHandle
    post_process_checkbox: viser.GuiCheckboxHandle
    # IK-specific controls
    ik_orientation_mode_dropdown: Optional[viser.GuiDropdownHandle]
    ik_nullspace_bias_checkbox: Optional[viser.GuiCheckboxHandle]
    ik_show_process_checkbox: Optional[viser.GuiCheckboxHandle]
    ik_grid_size_slider: Optional[viser.GuiSliderHandle]
    ik_show_grid_preview_checkbox: Optional[viser.GuiCheckboxHandle]
    ik_num_workers_slider: Optional[viser.GuiSliderHandle]
    ik_use_interactive_solver_checkbox: Optional[viser.GuiCheckboxHandle]
    ik_position_tol_slider: Optional[viser.GuiSliderHandle]
    ik_orientation_tol_slider: Optional[viser.GuiSliderHandle]
    ik_traversal_mode_dropdown: Optional[viser.GuiDropdownHandle]
    ik_interactive_debug_checkbox: Optional[viser.GuiCheckboxHandle]
    ik_debug_next_button: Optional[viser.GuiButtonHandle]
    generate_button: viser.GuiButtonHandle
    cancel_button: viser.GuiButtonHandle
    status_text: viser.GuiTextHandle
    progress_value: viser.GuiNumberHandle
    last_output_text: viser.GuiTextHandle


def create_shared_ik_controls(
    gui: viser._gui_handles.GuiApi,
    *,
    grid_size_initial: float,
    num_workers_initial: float,
    traversal_initial: str = "Distance",
    position_tol_initial_mm: float = 50.0,
    orientation_tol_initial_deg: float = 5.0,
) -> IKControlHandles:
    """Create IK-specific controls used by multiple examples."""

    orientation_mode_dropdown = gui.add_dropdown(
        "Orientation mode",
        options=["Fixed", "Reference", "Multiple"],
        initial_value="Reference",
        disabled=True,
    )
    nullspace_bias_checkbox = gui.add_checkbox(
        "Use nullspace bias",
        initial_value=False,
        disabled=True,
    )
    show_process_checkbox = gui.add_checkbox(
        "Show IK process",
        initial_value=False,
        disabled=True,
    )
    grid_size_slider = gui.add_slider(
        "Grid size scale",
        min=0.1,
        max=1.0,
        initial_value=grid_size_initial,
        step=0.05,
        disabled=True,
    )
    show_grid_preview_checkbox = gui.add_checkbox(
        "Show grid preview",
        initial_value=False,
        disabled=True,
    )
    traversal_mode_dropdown = gui.add_dropdown(
        "Grid traversal order",
        options=["Raw", "Distance", "Axis sweep"],
        initial_value=traversal_initial,
        disabled=True,
    )
    position_tol_slider = gui.add_slider(
        "Position tolerance (mm)",
        min=1.0,
        max=200.0,
        initial_value=position_tol_initial_mm,
        step=5.0,
        disabled=True,
    )
    orientation_tol_slider = gui.add_slider(
        "Orientation tolerance (deg)",
        min=1.0,
        max=180.0,
        initial_value=orientation_tol_initial_deg,
        step=1.0,
        disabled=True,
    )
    use_interactive_solver_checkbox = gui.add_checkbox(
        "Use interactive IK solver",
        initial_value=False,
        disabled=True,
    )
    num_workers_slider = gui.add_slider(
        "Parallel workers",
        min=1.0,
        max=16.0,
        initial_value=num_workers_initial,
        step=1.0,
        disabled=True,
    )
    interactive_debug_checkbox = gui.add_checkbox(
        "Interactive IK debug",
        initial_value=False,
        disabled=True,
    )
    debug_next_button = gui.add_button("IK Debug: Next target")
    debug_next_button.disabled = True

    return IKControlHandles(
        orientation_mode_dropdown=orientation_mode_dropdown,
        nullspace_bias_checkbox=nullspace_bias_checkbox,
        show_process_checkbox=show_process_checkbox,
        grid_size_slider=grid_size_slider,
        show_grid_preview_checkbox=show_grid_preview_checkbox,
        traversal_mode_dropdown=traversal_mode_dropdown,
        position_tol_slider=position_tol_slider,
        orientation_tol_slider=orientation_tol_slider,
        use_interactive_solver_checkbox=use_interactive_solver_checkbox,
        num_workers_slider=num_workers_slider,
        interactive_debug_checkbox=interactive_debug_checkbox,
        debug_next_button=debug_next_button,
    )


def create_reachability_gui_panel(
    server: viser.ViserServer,
    folder_name: str = "Reachability Mapping",
    expand_by_default: bool = False,
    *,
    default_samples_k: float = 50.0,
    max_samples_k: float = 200.0,
    default_batch_size: float = 5000.0,
    max_batch_size: float = 50000.0,
    default_cart_res_cm: float = 20.0,
    max_cart_res_cm: float = 20.0,
    default_ang_res_deg: float = 30.0,
    max_ang_res_deg: float = 45.0,
    default_num_workers: int = 1,
) -> ReachabilityGuiHandles:
    """Create reachability GUI panel.

    Args:
        server: Viser server instance
        folder_name: Name of the GUI folder
        expand_by_default: Whether folder should be expanded by default
        default_samples_k: Default samples in thousands
        max_samples_k: Maximum samples in thousands
        default_batch_size: Default batch size
        max_batch_size: Maximum batch size
        default_cart_res_cm: Default cartesian resolution in cm
        max_cart_res_cm: Maximum cartesian resolution in cm
        default_ang_res_deg: Default angular resolution in degrees
        max_ang_res_deg: Maximum angular resolution in degrees

    Returns:
        ReachabilityGuiHandles container with all GUI handles
    """
    with server.gui.add_folder(folder_name, expand_by_default=expand_by_default):
        # Analysis mode selection
        analysis_mode_dropdown = server.gui.add_dropdown(
            "Analysis Mode",
            options=["FK Sampling", "IK Grid"],
            initial_value="FK Sampling"
        )

        ik_controls = create_shared_ik_controls(
            server.gui,
            grid_size_initial=0.5,
            num_workers_initial=float(default_num_workers),
            traversal_initial="Distance",
        )

        # FK-specific controls
        samples_slider = server.gui.add_slider(
            "Samples (k)",
            min=5.0,
            max=max_samples_k,
            step=5.0,
            initial_value=min(max_samples_k, max(5.0, default_samples_k)),
        )
        batch_slider = server.gui.add_slider(
            "Batch size",
            min=500.0,
            max=max_batch_size,
            step=500.0,
            initial_value=float(min(max_batch_size, max(500.0, default_batch_size))),
        )
        cart_res_slider = server.gui.add_slider(
            "Cartesian res (cm)",
            min=1.0,  # Minimum 1cm resolution
            max=max_cart_res_cm,
            step=1.0,
            initial_value=float(min(max_cart_res_cm, max(1.0, default_cart_res_cm))),
        )
        ang_res_slider = server.gui.add_slider(
            "Angular res (deg)",
            min=10.0,
            max=max_ang_res_deg,
            step=5.0,
            initial_value=float(min(max_ang_res_deg, max(10.0, default_ang_res_deg))),
        )
        post_process_checkbox = server.gui.add_checkbox(
            "Post-process filter", initial_value=False
        )
        generate_button = server.gui.add_button("Generate reachability map")
        cancel_button = server.gui.add_button("Cancel generation")
        status_text = server.gui.add_text("Status", initial_value="Status: idle")
        progress_value = server.gui.add_number("Progress (%)", 0.0, disabled=True)
        last_output_text = server.gui.add_text("Last result", initial_value="Map: --")

    cancel_button.disabled = True

    return ReachabilityGuiHandles(
        analysis_mode_dropdown=analysis_mode_dropdown,
        samples_slider=samples_slider,
        batch_slider=batch_slider,
        cart_res_slider=cart_res_slider,
        ang_res_slider=ang_res_slider,
        post_process_checkbox=post_process_checkbox,
        ik_orientation_mode_dropdown=ik_controls.orientation_mode_dropdown,
        ik_nullspace_bias_checkbox=ik_controls.nullspace_bias_checkbox,
        ik_show_process_checkbox=ik_controls.show_process_checkbox,
        ik_grid_size_slider=ik_controls.grid_size_slider,
        ik_show_grid_preview_checkbox=ik_controls.show_grid_preview_checkbox,
        ik_num_workers_slider=ik_controls.num_workers_slider,
        ik_use_interactive_solver_checkbox=ik_controls.use_interactive_solver_checkbox,
        ik_position_tol_slider=ik_controls.position_tol_slider,
        ik_orientation_tol_slider=ik_controls.orientation_tol_slider,
        ik_traversal_mode_dropdown=ik_controls.traversal_mode_dropdown,
        ik_interactive_debug_checkbox=ik_controls.interactive_debug_checkbox,
        ik_debug_next_button=ik_controls.debug_next_button,
        generate_button=generate_button,
        cancel_button=cancel_button,
        status_text=status_text,
        progress_value=progress_value,
        last_output_text=last_output_text,
    )
