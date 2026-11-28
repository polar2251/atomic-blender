"""UI-agnostic core business logic for Atomic Blender Studio."""

from atomic_blender.core.clipboard import copy_pil_image_to_clipboard, open_folder
from atomic_blender.core.config import (
    JOURNAL_PRESETS,
    MATERIAL_FINISH_PRESETS,
    GuiRenderConfig,
    build_render_cli_args,
    compute_preview_resolution,
    embed_png_dpi,
    extract_input_path,
    format_input_display,
    inspect_structure_summary,
    resolve_batch_output_paths,
    select_path_strip_indices,
    validate_gui_config,
)
from atomic_blender.core.history import UndoHistory
from atomic_blender.core.measurements import (
    auto_detect_adsorbate_indices,
    compute_adsorbate_spotlight_alphas,
    compute_atom_alphas,
    compute_atom_radius_scales,
    compute_depth_fade_alphas,
    compute_selection_measurement,
    find_neighbor_atoms,
)
from atomic_blender.core.preview_worker import BlenderPreviewWorker

__all__ = [
    "BlenderPreviewWorker",
    "GuiRenderConfig",
    "JOURNAL_PRESETS",
    "MATERIAL_FINISH_PRESETS",
    "UndoHistory",
    "auto_detect_adsorbate_indices",
    "build_render_cli_args",
    "compute_adsorbate_spotlight_alphas",
    "compute_atom_alphas",
    "compute_atom_radius_scales",
    "compute_depth_fade_alphas",
    "compute_preview_resolution",
    "compute_selection_measurement",
    "copy_pil_image_to_clipboard",
    "embed_png_dpi",
    "extract_input_path",
    "find_neighbor_atoms",
    "format_input_display",
    "inspect_structure_summary",
    "open_folder",
    "resolve_batch_output_paths",
    "select_path_strip_indices",
    "validate_gui_config",
]
