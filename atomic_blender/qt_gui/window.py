"""Main PySide6 desktop window for Atomic Blender Studio."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from atomic_blender.cli import (
    _collect_style_files,
    find_blender_executable,
    package_dir,
    resolve_style_path,
)
from atomic_blender.core import (
    JOURNAL_PRESETS,
    MATERIAL_FINISH_PRESETS,
    BlenderPreviewWorker,
    GuiRenderConfig,
    UndoHistory,
    auto_detect_adsorbate_indices,
    build_render_cli_args,
    compute_adsorbate_spotlight_alphas,
    compute_atom_alphas,
    compute_preview_resolution,
    compute_selection_measurement,
    copy_pil_image_to_clipboard,
    embed_png_dpi,
    extract_input_path,
    format_input_display,
    inspect_structure_summary,
    open_folder as _open_folder,
    resolve_batch_output_paths,
    validate_gui_config,
)
from atomic_blender.elements import classify_dropped_file
from atomic_blender.qt_gui.gl_viewport import AtomicGLViewport, pil_to_qpixmap
from atomic_blender.viewport3d import (
    ViewportScene,
    atom_label,
    compose_dual_view_panel,
    find_coordination_polyhedra,
    find_dashed_contacts,
    preset_to_quat,
    quat_to_axes,
    render_atom_labels_and_selection_overlay,
    render_reference_axes_overlay,
    render_scale_bar_overlay,
)


class _WorkerBridge(QObject):
    """Thread-safe Qt signal relay for background inspect, preview, and render workers."""

    inspect_done = Signal(int, str, str, object, str)
    preview_done = Signal(int, str)
    preview_err = Signal(int, str)
    preview_status = Signal(int, str)
    render_log = Signal(str)
    render_done = Signal(str, int, str)
    panel_done = Signal(int, str, str)
    batch_done = Signal(int, int, int, str)


class _WelcomeScreen(QWidget):
    """Branded landing screen with animated emblem, drop prompt, and 3 quick sample pills."""

    sample_requested = Signal(str)
    open_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dark_backdrop: bool = True
        self._phase: float = 0.0
        self._frames: list[QPixmap] = []
        self._frame_idx: int = 0
        self._load_frames()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addStretch(2)

        self.lbl_brand = QLabel("Atomic Blender Studio")
        self.lbl_brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_brand.setStyleSheet("font-size: 13pt; font-weight: 300; color: #94A3B8;")
        layout.addWidget(self.lbl_brand)

        self.lbl_title = QLabel("Welcome")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 38pt; font-weight: 300; color: #E6EDF3; margin-bottom: 6px;")
        layout.addWidget(self.lbl_title)

        self.emblem_box = _EmblemCanvas(self)
        self.emblem_box.setFixedSize(220, 190)
        layout.addWidget(self.emblem_box, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_prompt = QLabel("Drag & drop a structure file anywhere  ·  or click Open")
        self.lbl_prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_prompt.setStyleSheet("font-size: 10pt; color: #CBD5E1; margin-top: 4px;")
        layout.addWidget(self.lbl_prompt)

        self.lbl_formats = QLabel("CIF  ·  POSCAR  ·  XYZ  ·  LAMMPS  ·  XSF  ·  view / style / recipe JSON")
        self.lbl_formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_formats.setStyleSheet("font-size: 8.5pt; color: #64748B; margin-bottom: 10px;")
        layout.addWidget(self.lbl_formats)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(10)
        pills_row.addStretch(1)
        self.sample_buttons: dict[str, QPushButton] = {}
        for key, label in (("cu_bulk", "Cu Bulk"), ("water_box", "Water Box"), ("co_pt111", "CO / Pt(111)")):
            btn = QPushButton(label)
            btn.setProperty("pill", "true")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Load built-in showcase structure: {label}")
            btn.clicked.connect(lambda _checked=False, k=key: self.sample_requested.emit(k))
            pills_row.addWidget(btn)
            self.sample_buttons[key] = btn
        pills_row.addStretch(1)
        layout.addLayout(pills_row)

        layout.addStretch(3)

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _load_frames(self) -> None:
        try:
            from PIL import Image

            gif_path = package_dir() / "assets" / "welcome_loop.gif"
            if gif_path.is_file():
                with Image.open(gif_path) as im:
                    n = max(1, int(getattr(im, "n_frames", 1)))
                    for i in range(n):
                        im.seek(i)
                        pm = pil_to_qpixmap(im.convert("RGBA")).scaled(
                            190,
                            190,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self._frames.append(pm)
        except Exception:
            self._frames = []

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self._phase += 0.08
        if self._frames:
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)
        self.emblem_box.update()

    def set_dark_backdrop(self, dark: bool) -> None:
        self.dark_backdrop = bool(dark)
        if dark:
            self.lbl_brand.setStyleSheet("font-size: 13pt; font-weight: 300; color: #94A3B8;")
            self.lbl_title.setStyleSheet("font-size: 38pt; font-weight: 300; color: #E6EDF3; margin-bottom: 6px;")
            self.lbl_prompt.setStyleSheet("font-size: 10pt; color: #CBD5E1; margin-top: 4px;")
        else:
            self.lbl_brand.setStyleSheet("font-size: 13pt; font-weight: 300; color: #475569;")
            self.lbl_title.setStyleSheet("font-size: 38pt; font-weight: 300; color: #0F172A; margin-bottom: 6px;")
            self.lbl_prompt.setStyleSheet("font-size: 10pt; color: #334155; margin-top: 4px;")
        self.update()
        self.emblem_box.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        bg = QColor("#0F1115") if self.dark_backdrop else QColor("#ECEFF4")
        painter.fillRect(self.rect(), bg)


class _EmblemCanvas(QWidget):
    """Paints either the pre-rendered molecule loop or a smooth orbiting molecular emblem."""

    def __init__(self, welcome: _WelcomeScreen) -> None:
        super().__init__(welcome)
        self._welcome = welcome

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._welcome._frames:
            pm = self._welcome._frames[self._welcome._frame_idx]
            x = (self.width() - pm.width()) * 0.5
            y = (self.height() - pm.height()) * 0.5
            painter.drawPixmap(QPointF(x, y), pm)
            return

        cx, cy = self.width() * 0.5, self.height() * 0.5
        ang = self._welcome._phase
        orbitals = (
            ("#FB923C", 54.0, 17.0, 0.0),
            ("#38BDF8", 54.0, 16.0, 2.094),
            ("#E2E8F0", 54.0, 14.0, 4.188),
        )
        painter.setPen(QPen(QColor("#64748B"), 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        pts = []
        for col, r_orb, rad, off in orbitals:
            px = cx + r_orb * math.cos(ang + off)
            py = cy + r_orb * math.sin(ang + off) * 0.62
            pts.append((px, py, rad, col))
            painter.drawLine(QPointF(cx, cy), QPointF(px, py))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#14B8A6"))
        painter.drawEllipse(QPointF(cx, cy), 22.0, 22.0)
        for px, py, rad, col in pts:
            painter.setBrush(QColor(col))
            painter.drawEllipse(QPointF(px, py), rad, rad)


class AtomicStudioQtWindow(QMainWindow):
    """PySide6 + OpenGL 3.3 commercial desktop studio for Atomic Blender."""

    def __init__(
        self,
        initial_input: str | None = None,
        prompt_missing_blender: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Atomic Blender Studio — Desktop Client")
        self.setAcceptDrops(True)

        screen = QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            w = min(1480, max(940, ag.width() - 32))
            h = min(920, max(640, ag.height() - 48))
        else:
            w, h = 1360, 860
        self.resize(w, h)
        self.setMinimumSize(880, 580)

        self._preview_tmp_dir = tempfile.TemporaryDirectory(prefix="atomic_qt_prev_")
        self._temp_view_files: list[Path] = []
        self._recent_files: list[str] = []

        self._structure_cache: dict[str, Any] | None = None
        self._base_structure_cache: dict[str, Any] | None = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._trajectory_index: int = 0
        self._structure_version: int = 0
        self._written_struct_version: int = -1
        self._structure_summary: str = ""

        self._vp_scene: ViewportScene | None = None
        self._use_interactive_cam: bool = False
        self._selected_atoms: set[int] = set()
        self._selection_order: list[int] = []
        self._custom_element_colors: dict[str, str] = {}
        self._show_atoms: bool = True

        self._history = UndoHistory(max_depth=50)
        self._restoring_history: bool = False

        self._inspect_seq: int = 0
        self._preview_seq: int = 0
        self._busy: bool = False
        self._render_cancelled: bool = False
        self._active_render_proc: subprocess.Popen[str] | None = None
        self._task_start: float = 0.0
        self._target_dpi: int = 300
        self._res_w: int = 1400
        self._res_h: int = 1400
        self._style_preset_value: str = "default"
        self._view_preset_value: str = "iso"
        self._view_file_value: str = ""
        self._crop_z_mode: str = "none"
        self._crop_z_custom: str = "2.0,4.0"
        self._pending_blender_exe: str = ""
        self._prompt_missing_blender = prompt_missing_blender

        self._bridge = _WorkerBridge(self)
        self._bridge.inspect_done.connect(self._on_inspect_done)
        self._bridge.preview_done.connect(self._on_preview_done)
        self._bridge.preview_err.connect(self._on_preview_err)
        self._bridge.preview_status.connect(self._on_preview_status)
        self._bridge.render_log.connect(self._append_log)
        self._bridge.render_done.connect(self._on_render_done)
        self._bridge.panel_done.connect(self._on_panel_done)
        self._bridge.batch_done.connect(self._on_batch_done)

        self._preview_worker = BlenderPreviewWorker(
            on_done=lambda seq, out_p: self._bridge.preview_done.emit(int(seq), str(out_p)),
            on_error=lambda seq, msg: self._bridge.preview_err.emit(int(seq), str(msg)),
            on_status=lambda seq, msg: self._bridge.preview_status.emit(int(seq), str(msg)),
        )

        self._build_ui()
        self._bind_shortcuts()
        self._detect_blender_initial()

        if initial_input:
            self.open_path(initial_input)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(self.splitter)

        # ---- Left Sidebar ----
        self.sidebar = QFrame(self.splitter)
        self.sidebar.setObjectName("SidebarPanel")
        self.sidebar.setMinimumWidth(308)
        self.sidebar.setMaximumWidth(372)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(10, 10, 10, 10)
        sb_layout.setSpacing(8)

        # FILE header
        file_hdr = QLabel("FILE")
        file_hdr.setObjectName("SectionHeader")
        sb_layout.addWidget(file_hdr)

        file_row = QHBoxLayout()
        file_row.setSpacing(5)
        self.entry_input = QLineEdit()
        self.entry_input.setPlaceholderText("Select or drop CIF / POSCAR / XYZ...")
        self.entry_input.returnPressed.connect(self._on_input_entered)
        self.btn_open = QPushButton("Open")
        self.btn_open.setToolTip("Open structure file (Ctrl+O)")
        self.btn_open.clicked.connect(self._browse_input)
        self.btn_recent = QToolButton()
        self.btn_recent.setText("▾")
        self.btn_recent.setToolTip("Recent structure files")
        self.btn_recent.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._recent_menu = QMenu(self.btn_recent)
        self.btn_recent.setMenu(self._recent_menu)
        file_row.addWidget(self.entry_input, 1)
        file_row.addWidget(self.btn_open)
        file_row.addWidget(self.btn_recent)
        sb_layout.addLayout(file_row)

        # Tabs: View / Select / Scene
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_view_tab(), "View")
        self.tabs.addTab(self._build_select_tab(), "Select")
        self.tabs.addTab(self._build_scene_tab(), "Scene")
        sb_layout.addWidget(self.tabs, 1)

        # Bottom Action Dock
        self.btn_render = QPushButton("Render High-Res PNG")
        self.btn_render.setProperty("primary", "true")
        self.btn_render.setToolTip("Render publication-grade PNG in Blender Cycles (Ctrl+R)")
        self.btn_render.clicked.connect(self._on_render)
        sb_layout.addWidget(self.btn_render)

        sec_row = QHBoxLayout()
        sec_row.setSpacing(5)
        self.btn_top_side = QPushButton("Top+Side")
        self.btn_top_side.setToolTip("Render Top and Side views stitched into one panel")
        self.btn_top_side.clicked.connect(self._on_render_top_side_panel)
        self.btn_batch = QPushButton("Batch...")
        self.btn_batch.setToolTip("Batch render multiple structure files")
        self.btn_batch.clicked.connect(self._on_batch_render)
        self.btn_blender = QPushButton("Blender")
        self.btn_blender.setToolTip("Open generated .blend scene in Blender")
        self.btn_blender.clicked.connect(self._on_open_in_blender)
        self.btn_folder = QPushButton("Folder")
        self.btn_folder.setToolTip("Open output folder")
        self.btn_folder.clicked.connect(self._on_open_folder)
        for b in (self.btn_top_side, self.btn_batch, self.btn_blender, self.btn_folder):
            sec_row.addWidget(b, 1)
        sb_layout.addLayout(sec_row)

        # ---- Right Workspace ----
        workspace = QWidget(self.splitter)
        ws_layout = QVBoxLayout(workspace)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        ws_layout.setSpacing(0)

        # Top Toolbar
        tb = QFrame()
        tb.setObjectName("ToolbarFrame")
        tb_layout = QHBoxLayout(tb)
        tb_layout.setContentsMargins(8, 5, 8, 5)
        tb_layout.setSpacing(6)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit camera to structure (R or Home)")
        self.btn_fit.clicked.connect(self._on_fit_view)
        self.btn_save_view = QPushButton("Save View")
        self.btn_save_view.setToolTip("Export camera framing as view.json")
        self.btn_save_view.clicked.connect(self._on_save_view_json)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setToolTip("Copy current viewport or render to clipboard (Ctrl+C)")
        self.btn_copy.clicked.connect(self._on_copy_image)
        self.chk_dark = QCheckBox("Dark")
        self.chk_dark.setChecked(True)
        self.chk_dark.setToolTip("Toggle dark / light studio backdrop")
        self.chk_dark.toggled.connect(self._on_dark_toggled)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo visual edit (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo visual edit (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.redo)
        self.btn_shortcuts = QPushButton("?")
        self.btn_shortcuts.setMaximumWidth(28)
        self.btn_shortcuts.setToolTip("Show keyboard & mouse shortcuts (F1)")
        self.btn_shortcuts.clicked.connect(self._show_shortcuts_dialog)
        self.btn_log = QPushButton("Log")
        self.btn_log.setCheckable(True)
        self.btn_log.setToolTip("Toggle Blender Cycles output log")
        self.btn_log.toggled.connect(self._toggle_log_panel)

        for w_item in (self.btn_fit, self.btn_save_view, self.btn_copy, self.chk_dark):
            tb_layout.addWidget(w_item)
        tb_layout.addStretch(1)
        for w_item in (self.btn_undo, self.btn_redo, self.btn_shortcuts, self.btn_log):
            tb_layout.addWidget(w_item)
        ws_layout.addWidget(tb)

        # Center stack: Welcome vs OpenGL 3D Viewport
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        ws_layout.addWidget(self.v_splitter, 1)

        center_holder = QWidget()
        center_v = QVBoxLayout(center_holder)
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.setSpacing(0)

        self.canvas_stack = QStackedWidget()
        self.welcome_widget = _WelcomeScreen()
        self.welcome_widget.sample_requested.connect(self.load_builtin_sample)
        self.welcome_widget.open_requested.connect(self._browse_input)
        self.viewport = AtomicGLViewport()
        self.viewport.camera_changed.connect(self._on_gl_camera_changed)
        self.viewport.camera_settled.connect(self._on_gl_camera_settled)
        self.viewport.atom_clicked.connect(self._on_gl_atom_clicked)
        self.viewport.interaction_started.connect(self._on_gl_interaction_started)

        self.canvas_stack.addWidget(self.welcome_widget)
        self.canvas_stack.addWidget(self.viewport)
        center_v.addWidget(self.canvas_stack, 1)

        # Multi-frame trajectory strip (visible only when len(frames) > 1)
        self.traj_bar = QFrame()
        self.traj_bar.setObjectName("ToolbarFrame")
        traj_l = QHBoxLayout(self.traj_bar)
        traj_l.setContentsMargins(10, 4, 10, 4)
        traj_l.setSpacing(6)
        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setMaximumWidth(28)
        self.btn_prev_frame.clicked.connect(lambda: self._step_frame(-1))
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.valueChanged.connect(self._on_frame_slider_changed)
        self.slider_frame.sliderReleased.connect(self._request_blender_preview)
        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setMaximumWidth(28)
        self.btn_next_frame.clicked.connect(lambda: self._step_frame(1))
        self.lbl_frame = QLabel("Frame 1/1")
        self.lbl_frame.setObjectName("MutedLabel")
        traj_l.addWidget(self.btn_prev_frame)
        traj_l.addWidget(self.slider_frame, 1)
        traj_l.addWidget(self.btn_next_frame)
        traj_l.addWidget(self.lbl_frame)
        self.traj_bar.hide()
        center_v.addWidget(self.traj_bar)

        self.v_splitter.addWidget(center_holder)

        # Collapsible Log panel
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(90)
        self.log_box.hide()
        self.v_splitter.addWidget(self.log_box)
        self.v_splitter.setStretchFactor(0, 5)
        self.v_splitter.setStretchFactor(1, 1)

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(workspace)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # ---- Status Bar ----
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.lbl_status = QLabel("Ready — OpenGL 3D + Auto Cycles Preview ready")
        sb.addWidget(self.lbl_status, 1)

        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 0)
        self.render_progress.setFixedWidth(135)
        self.render_progress.hide()
        sb.addPermanentWidget(self.render_progress)

        self.btn_cancel_render = QPushButton("Cancel")
        self.btn_cancel_render.setProperty("danger", "true")
        self.btn_cancel_render.clicked.connect(self._on_cancel_render)
        self.btn_cancel_render.hide()
        sb.addPermanentWidget(self.btn_cancel_render)

    def _wrap_scroll(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    def _build_view_tab(self) -> QScrollArea:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Display Card
        card = QFrame()
        card.setObjectName("CardGroup")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)

        hdr = QLabel("DISPLAY")
        hdr.setObjectName("SectionHeader")
        cl.addWidget(hdr)

        row_style = QHBoxLayout()
        self.chk_show_atoms = QCheckBox("Atoms")
        self.chk_show_atoms.setChecked(True)
        self.chk_show_atoms.toggled.connect(self._on_display_control_changed)
        self.combo_atom_style = QComboBox()
        self.combo_atom_style.addItems(["Van der Waals", "Ball and Stick", "Licorice"])
        self.combo_atom_style.setCurrentText("Van der Waals")
        self.combo_atom_style.currentTextChanged.connect(self._on_atom_style_changed)
        row_style.addWidget(self.chk_show_atoms)
        row_style.addWidget(self.combo_atom_style, 1)
        cl.addLayout(row_style)

        self.chk_show_cell = QCheckBox("Crystal Lattice")
        self.chk_show_cell.setChecked(True)
        self.chk_show_cell.toggled.connect(self._on_display_control_changed)
        cl.addWidget(self.chk_show_cell)

        ov_grid = QGridLayout()
        ov_grid.setHorizontalSpacing(8)
        ov_grid.setVerticalSpacing(5)
        self.chk_show_axes = QCheckBox("Ref Axes")
        self.chk_show_axes.setChecked(True)
        self.chk_cell_vectors = QCheckBox("Lattice (a,b,c)")
        self.chk_polyhedra = QCheckBox("Polyhedra")
        self.chk_scale_bar = QCheckBox("Scale Bar (Å)")
        self.chk_scale_bar.setChecked(True)
        self.chk_hbonds = QCheckBox("H-Bonds")
        self.chk_labels = QCheckBox("Atom Labels")
        for chk in (
            self.chk_show_axes,
            self.chk_cell_vectors,
            self.chk_polyhedra,
            self.chk_scale_bar,
            self.chk_hbonds,
            self.chk_labels,
        ):
            chk.toggled.connect(self._on_display_control_changed)
        ov_grid.addWidget(self.chk_show_axes, 0, 0)
        ov_grid.addWidget(self.chk_cell_vectors, 0, 1)
        ov_grid.addWidget(self.chk_polyhedra, 1, 0)
        ov_grid.addWidget(self.chk_scale_bar, 1, 1)
        ov_grid.addWidget(self.chk_hbonds, 2, 0)
        ov_grid.addWidget(self.chk_labels, 2, 1)
        cl.addLayout(ov_grid)
        lay.addWidget(card)

        # Appearance Card
        app_card = QFrame()
        app_card.setObjectName("CardGroup")
        al = QGridLayout(app_card)
        al.setContentsMargins(8, 8, 8, 8)
        al.setSpacing(6)

        al.addWidget(QLabel("Texture"), 0, 0)
        self.combo_texture = QComboBox()
        self.combo_texture.addItems(["Studio", "Chalk", "Flat", "Metallic"])
        self.combo_texture.setToolTip("Surface material finish for Cycles render & GL preview")
        self.combo_texture.currentTextChanged.connect(self._on_display_control_changed)
        al.addWidget(self.combo_texture, 0, 1, 1, 2)

        al.addWidget(QLabel("Scale"), 1, 0)
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(30, 200)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self._on_scale_slider_changed)
        self.slider_scale.sliderReleased.connect(self._on_display_control_changed)
        self.lbl_scale = QLabel("1.00×")
        self.lbl_scale.setMinimumWidth(38)
        al.addWidget(self.slider_scale, 1, 1)
        al.addWidget(self.lbl_scale, 1, 2)

        al.addWidget(QLabel("Bond"), 2, 0)
        self.combo_bonds = QComboBox()
        self.combo_bonds.addItems(["none", "style", "table", "auto"])
        self.combo_bonds.setCurrentText("none")
        self.combo_bonds.currentTextChanged.connect(self._on_display_control_changed)
        al.addWidget(self.combo_bonds, 2, 1, 1, 2)

        al.addWidget(QLabel("Color"), 3, 0)
        self.combo_bond_color = QComboBox()
        self.combo_bond_color.addItems(["style", "bicolor", "single"])
        self.combo_bond_color.currentTextChanged.connect(self._on_display_control_changed)
        al.addWidget(self.combo_bond_color, 3, 1, 1, 2)

        self.chk_auto_preview = QCheckBox("Auto Blender Preview on Drop")
        self.chk_auto_preview.setChecked(True)
        self.chk_auto_preview.setToolTip("Automatically render fast low-res Cycles preview when camera drag stops")
        al.addWidget(self.chk_auto_preview, 4, 0, 1, 3)

        lay.addWidget(app_card)
        lay.addStretch(1)
        return self._wrap_scroll(inner)

    def _build_select_tab(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_sel_all = QPushButton("All")
        self.btn_sel_inv = QPushButton("Invert")
        self.btn_sel_clr = QPushButton("Clear")
        self.btn_sel_spot = QPushButton("Spotlight")
        self.btn_sel_all.clicked.connect(self._select_all_atoms)
        self.btn_sel_inv.clicked.connect(self._invert_selection)
        self.btn_sel_clr.clicked.connect(self._clear_selection)
        self.btn_sel_spot.clicked.connect(self._apply_spotlight)
        for b in (self.btn_sel_all, self.btn_sel_inv, self.btn_sel_clr, self.btn_sel_spot):
            btn_row.addWidget(b, 1)
        lay.addLayout(btn_row)

        self.atom_list = QListWidget()
        self.atom_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.atom_list.itemSelectionChanged.connect(self._on_atom_list_selection_changed)
        lay.addWidget(self.atom_list, 1)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(5, 100)
        self.slider_opacity.setValue(30)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        self.slider_opacity.sliderReleased.connect(self._request_blender_preview)
        self.lbl_opacity = QLabel("30%")
        self.lbl_opacity.setMinimumWidth(34)
        op_row.addWidget(self.slider_opacity, 1)
        op_row.addWidget(self.lbl_opacity)
        lay.addLayout(op_row)

        self.chk_fade_unselected = QCheckBox("Fade Unselected Atoms")
        self.chk_fade_unselected.toggled.connect(lambda: self._sync_selection_and_alphas(trigger_blender=True))
        lay.addWidget(self.chk_fade_unselected)

        self.lbl_measurement = QLabel("Click 1–4 atoms to measure distance / angle / dihedral")
        self.lbl_measurement.setObjectName("MeasureLabel")
        self.lbl_measurement.setWordWrap(True)
        lay.addWidget(self.lbl_measurement)
        return panel

    def _build_scene_tab(self) -> QScrollArea:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Camera Presets
        cam_card = QFrame()
        cam_card.setObjectName("CardGroup")
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)
        hdr = QLabel("CAMERA PRESETS")
        hdr.setObjectName("SectionHeader")
        cl.addWidget(hdr)

        p_grid = QGridLayout()
        p_grid.setSpacing(4)
        presets = [
            ("ISO", "iso", 0, 0),
            ("Side", "side-oblique", 0, 1),
            ("a", "a", 0, 2),
            ("b", "b", 0, 3),
            ("c", "c", 1, 0),
            ("a*", "a*", 1, 1),
            ("b*", "b*", 1, 2),
            ("c*", "c*", 1, 3),
        ]
        for lbl, key, r, c in presets:
            b = QPushButton(lbl)
            b.setProperty("preset", "true")
            b.clicked.connect(lambda _ch=False, k=key: self._pick_view_preset(k))
            p_grid.addWidget(b, r, c)
        cl.addLayout(p_grid)
        lay.addWidget(cam_card)

        # Supercell + Crop Z
        cell_card = QFrame()
        cell_card.setObjectName("CardGroup")
        cg = QGridLayout(cell_card)
        cg.setContentsMargins(8, 8, 8, 8)
        cg.setSpacing(5)
        cg.addWidget(QLabel("Replicate"), 0, 0)
        self.spin_rep_x = QSpinBox()
        self.spin_rep_y = QSpinBox()
        self.spin_rep_z = QSpinBox()
        for idx_s, sp in enumerate((self.spin_rep_x, self.spin_rep_y, self.spin_rep_z), start=1):
            sp.setRange(1, 20)
            sp.setValue(1)
            sp.valueChanged.connect(self._schedule_inspect)
            cg.addWidget(sp, 0, idx_s)

        self.chk_wrap = QCheckBox("Wrap PBC")
        self.chk_wrap.setChecked(True)
        self.chk_wrap.toggled.connect(self._schedule_inspect)
        cg.addWidget(self.chk_wrap, 1, 0, 1, 2)

        cg.addWidget(QLabel("Crop Z"), 2, 0)
        self.combo_crop_z = QComboBox()
        self.combo_crop_z.addItems(["none", "auto", "custom"])
        self.combo_crop_z.currentTextChanged.connect(self._on_crop_z_changed)
        cg.addWidget(self.combo_crop_z, 2, 1, 1, 2)

        self.entry_crop_z = QLineEdit("2.0,4.0")
        self.entry_crop_z.setEnabled(False)
        self.entry_crop_z.editingFinished.connect(lambda: self._rebuild_viewport(reset_camera=False, trigger_preview=True))
        cg.addWidget(self.entry_crop_z, 2, 3)
        lay.addWidget(cell_card)

        # Render Settings Card
        ren_card = QFrame()
        ren_card.setObjectName("CardGroup")
        rg = QGridLayout(ren_card)
        rg.setContentsMargins(8, 8, 8, 8)
        rg.setSpacing(6)

        rg.addWidget(QLabel("Style"), 0, 0)
        self.combo_style = QComboBox()
        style_names = sorted(set(_collect_style_files().keys()) | {"default", "studio", "matte", "ball_and_stick"})
        self.combo_style.addItems(style_names)
        self.combo_style.setCurrentText("default")
        self.combo_style.currentTextChanged.connect(lambda: self._rebuild_viewport(reset_camera=False, trigger_preview=True))
        rg.addWidget(self.combo_style, 0, 1, 1, 2)

        rg.addWidget(QLabel("Resolution"), 1, 0)
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(list(JOURNAL_PRESETS.keys()))
        self.combo_resolution.setToolTip("Select output pixels and journal publication DPI metadata")
        self.combo_resolution.currentTextChanged.connect(self._on_res_preset_changed)
        rg.addWidget(self.combo_resolution, 1, 1, 1, 2)

        rg.addWidget(QLabel("Samples"), 2, 0)
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1, 512)
        self.spin_samples.setValue(64)
        rg.addWidget(self.spin_samples, 2, 1, 1, 2)

        rg.addWidget(QLabel("Output"), 3, 0)
        self.entry_output = QLineEdit()
        self.entry_output.setPlaceholderText("Auto (<input>.png)")
        self.btn_browse_out = QPushButton("...")
        self.btn_browse_out.setMaximumWidth(28)
        self.btn_browse_out.clicked.connect(self._browse_output)
        rg.addWidget(self.entry_output, 3, 1)
        rg.addWidget(self.btn_browse_out, 3, 2)

        rg.addWidget(QLabel("Blender"), 4, 0)
        self.entry_blender = QLineEdit()
        self.entry_blender.setPlaceholderText("Auto-detect")
        self.btn_browse_blender = QPushButton("...")
        self.btn_browse_blender.setMaximumWidth(28)
        self.btn_browse_blender.clicked.connect(self._browse_blender)
        rg.addWidget(self.entry_blender, 4, 1)
        rg.addWidget(self.btn_browse_blender, 4, 2)

        lay.addWidget(ren_card)
        lay.addStretch(1)
        return self._wrap_scroll(inner)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence.StandardKey.Open, self, self._browse_input)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_render)
        QShortcut(QKeySequence("Ctrl+C"), self, self._on_copy_image)
        QShortcut(QKeySequence(Qt.Key.Key_F1), self, self._show_shortcuts_dialog)
        QShortcut(QKeySequence("?"), self, self._show_shortcuts_dialog)
        QShortcut(QKeySequence("R"), self, self._on_fit_view)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._clear_selection)

    def _detect_blender_initial(self) -> None:
        try:
            exe = str(find_blender_executable())
            self.entry_blender.setText(exe)
        except FileNotFoundError:
            if self._prompt_missing_blender:
                self.lbl_status.setText("Blender not auto-detected — OpenGL 3D works; set Blender path in Scene tab for Cycles.")

    # ---- Tab & Navigation Helpers ----
    def switch_tab(self, name: str) -> None:
        mapping = {"view": 0, "select": 1, "scene": 2}
        idx = mapping.get(name.strip().lower(), 0)
        self.tabs.setCurrentIndex(idx)

    # ---- File Opening & Inspection ----
    def open_path(self, path_str: str) -> None:
        p = Path(extract_input_path(path_str)).expanduser()
        self.entry_input.setText(format_input_display(str(p)))
        if not self.entry_output.text().strip():
            self.entry_output.setText(str(p.with_suffix(".png")))
        self._schedule_inspect()

    def load_builtin_sample(self, sample_key: str) -> None:
        """Load one of the built-in showcase structures cleanly from Welcome pills."""
        rel_map = {
            "cu_bulk": "fcc_cu_bulk.cif",
            "water_box": "water_box.xyz",
            "co_pt111": "co_on_pt111.vasp",
        }
        rel = rel_map.get(sample_key, "fcc_cu_bulk.cif")
        cand_paths = [
            package_dir().parent / "examples" / rel,
            Path.cwd() / "examples" / rel,
        ]
        target = next((p for p in cand_paths if p.is_file()), None)
        if target is None:
            fallback = {
                "cu_bulk": {
                    "symbols": ["Cu"] * 4,
                    "positions": [[0.0, 0.0, 0.0], [0.0, 1.8075, 1.8075], [1.8075, 0.0, 1.8075], [1.8075, 1.8075, 0.0]],
                    "cell": [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]],
                    "pbc": [True, True, True],
                    "has_cell": True,
                    "origin": [0.0, 0.0, 0.0],
                },
                "water_box": {
                    "symbols": ["O", "H", "H"],
                    "positions": [[3.0, 3.0, 3.0], [3.757, 3.586, 3.0], [2.243, 3.586, 3.0]],
                    "cell": [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
                    "pbc": [True, True, True],
                    "has_cell": True,
                    "origin": [0.0, 0.0, 0.0],
                },
                "co_pt111": {
                    "symbols": ["Pt", "Pt", "Pt", "C", "O"],
                    "positions": [[0.0, 0.0, 0.0], [2.77, 0.0, 0.0], [1.385, 2.399, 0.0], [1.385, 0.8, 1.85], [1.385, 0.8, 3.00]],
                    "cell": [[5.54, 0.0, 0.0], [0.0, 4.80, 0.0], [0.0, 0.0, 14.0]],
                    "pbc": [True, True, False],
                    "has_cell": True,
                    "origin": [0.0, 0.0, 0.0],
                },
            }
            out_json = Path(self._preview_tmp_dir.name) / f"sample_{sample_key}.json"
            out_json.write_text(json.dumps(fallback.get(sample_key, fallback["cu_bulk"])), encoding="utf-8")
            target = out_json

        self.entry_output.setText(str(target.with_suffix(".png")))
        self.open_path(str(target))
        self.lbl_status.setText(f"Loading built-in sample: {target.name}")

    def _on_input_entered(self) -> None:
        raw = extract_input_path(self.entry_input.text())
        if raw:
            self.open_path(raw)

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Atomic Structure",
            "",
            "Structure files (*.cif *.vasp *.xyz *.extxyz *.data *.lmp *.json POSCAR CONTCAR);;All files (*.*)",
        )
        if path:
            self.entry_output.setText(str(Path(path).with_suffix(".png")))
            self.open_path(path)

    def _browse_output(self) -> None:
        initial = self.entry_output.text().strip() or "render.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save Output PNG", initial, "PNG image (*.png)")
        if path:
            self.entry_output.setText(path)

    def _browse_blender(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Blender Executable", "", "Executable (*.exe *);;All files (*.*)")
        if path:
            self.entry_blender.setText(path)

    def _add_recent_file(self, path_str: str) -> None:
        norm = str(Path(path_str).expanduser())
        if norm in self._recent_files:
            self._recent_files.remove(norm)
        self._recent_files.insert(0, norm)
        self._recent_files = self._recent_files[:8]
        self._recent_menu.clear()
        for item_p in self._recent_files:
            act = QAction(Path(item_p).name, self)
            act.setToolTip(item_p)
            act.triggered.connect(lambda _ch=False, p=item_p: self.open_path(p))
            self._recent_menu.addAction(act)

    def _schedule_inspect(self) -> None:
        path = extract_input_path(self.entry_input.text())
        if not path:
            return
        self._inspect_seq += 1
        req_seq = self._inspect_seq
        rep = (int(self.spin_rep_x.value()), int(self.spin_rep_y.value()), int(self.spin_rep_z.value()))
        wrap = bool(self.chk_wrap.isChecked())

        def worker() -> None:
            try:
                p = Path(path).expanduser()
                if not p.is_file():
                    self._bridge.inspect_done.emit(req_seq, path, f"File not found: {p.name}", None, "missing")
                    return
                from atomic_blender.convert import read_structure_frames

                frames = read_structure_frames(p, replicate=rep, wrap=wrap)
                eff_rep = rep if (frames and frames[0].get("has_cell", False)) else (1, 1, 1)
                summary = inspect_structure_summary(path, replicate=eff_rep)
                if len(frames) > 1:
                    summary = f"{summary} | {len(frames)} frames"
                self._bridge.inspect_done.emit(req_seq, path, summary, frames, "")
            except Exception as exc:
                self._bridge.inspect_done.emit(req_seq, path, f"Inspect error: {exc}", None, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_inspect_done(self, req_seq: int, req_path: str, summary: str, frames_obj: Any, err: str) -> None:
        cur_path = extract_input_path(self.entry_input.text())
        if req_seq != self._inspect_seq or (cur_path and cur_path != req_path):
            return
        if err or not frames_obj:
            self.lbl_status.setText(summary)
            return

        frames = [dict(f) for f in frames_obj]
        self._structure_summary = summary
        self._selected_atoms.clear()
        self._selection_order.clear()
        self._trajectory_frames = frames
        self._trajectory_index = 0
        self._base_structure_cache = dict(frames[0])
        self._structure_cache = dict(frames[0])
        self._structure_version += 1
        self._written_struct_version = -1
        self._use_interactive_cam = False
        self._history.clear()
        self._add_recent_file(req_path)

        if len(frames) > 1:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setRange(0, len(frames) - 1)
            self.slider_frame.setValue(0)
            self.slider_frame.blockSignals(False)
            self.lbl_frame.setText(f"Frame 1/{len(frames)}")
            self.traj_bar.show()
        else:
            self.traj_bar.hide()

        self.canvas_stack.setCurrentWidget(self.viewport)
        self._rebuild_viewport(reset_camera=True, trigger_preview=True)
        self.lbl_status.setText(f"Loaded {Path(req_path).name} (GL 3D ready)")

    # ---- Scene Construction & Camera ----
    def _resolve_style_dict(self) -> dict[str, Any]:
        try:
            sp = resolve_style_path(self.combo_style.currentText().strip() or "default")
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if self._custom_element_colors:
            elems = data.setdefault("elements", {})
            cols = elems.setdefault("colors", {})
            cols.update(self._custom_element_colors)
        fin = self.combo_texture.currentText().strip().lower()
        fin_overrides = MATERIAL_FINISH_PRESETS.get(fin)
        if fin_overrides:
            mat = data.setdefault("material", {})
            mat.update(fin_overrides)
            for sub_key in ("metal", "nonmetal"):
                if isinstance(mat.get(sub_key), dict):
                    for k, v in fin_overrides.items():
                        if k != "ao_factor":
                            mat[sub_key][k] = v
        return data

    def _effective_bonds_and_scale(self) -> tuple[float, str, str]:
        return (
            self.slider_scale.value() / 100.0,
            self.combo_bonds.currentText().strip() or "style",
            self.combo_bond_color.currentText().strip() or "style",
        )

    def _parse_crop_z_tuple(self) -> tuple[float, float] | None:
        if self._crop_z_mode == "auto":
            return (2.0, 4.0)
        if self._crop_z_mode == "custom":
            parts = [float(x.strip()) for x in self._crop_z_custom.split(",") if x.strip()]
            if len(parts) == 1:
                return (parts[0], parts[0])
            if len(parts) == 2:
                return (parts[0], parts[1])
        return None

    def _apply_view_to_viewport(self, vp: ViewportScene) -> None:
        aspect = self._res_w / max(1, self._res_h)
        crop_z = self._parse_crop_z_tuple()
        if self._view_file_value and Path(self._view_file_value).is_file():
            try:
                vd = json.loads(Path(self._view_file_value).read_text(encoding="utf-8"))
                q = tuple(float(x) for x in vd["rotation_quaternion"])
                vp.fit_camera(quat=q, aspect=aspect, crop_z=crop_z)  # type: ignore[arg-type]
                return
            except Exception:
                pass
        cell = vp.cell_vecs if vp.has_cell else None
        q = preset_to_quat(self._view_preset_value, cell)
        vp.fit_camera(quat=q, aspect=aspect, crop_z=crop_z)

    def _refresh_structure_annotations(self, style: dict[str, Any]) -> None:
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        n = len(syms)
        dashed: list[list[int]] = []
        if self.chk_hbonds.isChecked() and n > 1 and len(pos) == n:
            from atomic_blender.viewport3d import _find_bonds

            bond_cfg = dict(style.get("bonds", {}))
            b_mode = self.combo_bonds.currentText()
            if b_mode != "style":
                bond_cfg["mode"] = b_mode
                bond_cfg["enabled"] = b_mode != "none"
            pos_arr = np.asarray(pos, dtype=np.float64)
            bi, bj = _find_bonds(syms, pos_arr, bond_cfg)
            for ia, ib in find_dashed_contacts(syms, pos_arr, bond_i=bi, bond_j=bj, mode="hbond", d_min=1.55, d_max=2.55):
                dashed.append([ia, ib])
        if self._structure_cache.get("dashed_bonds", []) != dashed:
            self._structure_cache["dashed_bonds"] = dashed
            self._written_struct_version = -1

        if self.chk_polyhedra.isChecked() and n >= 5 and len(pos) == n:
            polys = find_coordination_polyhedra(
                syms,
                pos,
                style=style,
                cell_vecs=self._structure_cache.get("cell") if self._structure_cache.get("has_cell") else None,
                pbc=self._structure_cache.get("pbc"),
            )
            if self._structure_cache.get("polyhedra") != polys:
                self._structure_cache["polyhedra"] = polys
                self._written_struct_version = -1
        elif "polyhedra" in self._structure_cache:
            self._structure_cache.pop("polyhedra", None)
            self._written_struct_version = -1

        show_cvec = bool(self.chk_cell_vectors.isChecked())
        if bool(self._structure_cache.get("show_cell_vectors", False)) != show_cvec:
            if show_cvec:
                self._structure_cache["show_cell_vectors"] = True
            else:
                self._structure_cache.pop("show_cell_vectors", None)
            self._written_struct_version = -1

    def _rebuild_viewport(self, reset_camera: bool = False, trigger_preview: bool = False) -> None:
        if self._structure_cache is None:
            return
        prev_cam = None
        if not reset_camera and self._vp_scene is not None and self._use_interactive_cam:
            prev_cam = (self._vp_scene.quat, self._vp_scene.target.copy(), self._vp_scene.ortho_scale)

        style = self._resolve_style_dict()
        self._refresh_structure_annotations(style)
        r_scale, b_mode, b_col = self._effective_bonds_and_scale()

        vp = ViewportScene.from_neutral_dict(
            self._structure_cache,
            style=style,
            radius_scale=r_scale,
            bonds_mode=b_mode,
            bond_color=b_col,
            show_cell=bool(self.chk_show_cell.isChecked()),
            show_atoms=bool(self.chk_show_atoms.isChecked()),
            show_labels=bool(self.chk_labels.isChecked()),
            show_axes=bool(self.chk_show_axes.isChecked()),
            axes_labels=True,
            dark_backdrop=bool(self.chk_dark.isChecked()),
            selected_atoms=set(self._selected_atoms),
            show_scale_bar=bool(self.chk_scale_bar.isChecked()),
            show_measure_overlay=True,
            selection_order=list(self._selection_order),
            show_cell_vectors=bool(self.chk_cell_vectors.isChecked()),
        )
        if prev_cam is not None:
            vp.quat, vp.target, vp.ortho_scale = prev_cam
        else:
            self._apply_view_to_viewport(vp)
        self._vp_scene = vp

        self._refresh_atom_list()
        self._update_measurement_label()
        self.viewport.clear_blender_preview()
        self.viewport.set_scene(
            vp,
            res_w=self._res_w,
            res_h=self._res_h,
            dark_backdrop=bool(self.chk_dark.isChecked()),
            structure_summary=self._structure_summary,
            material_finish=self.combo_texture.currentText().strip().lower(),
        )
        self._record_history_snapshot()
        if trigger_preview:
            self._request_blender_preview()

    # ---- Control Callbacks ----
    def _on_atom_style_changed(self, style_text: str) -> None:
        self.chk_show_atoms.blockSignals(True)
        self.chk_show_atoms.setChecked(True)
        self.chk_show_atoms.blockSignals(False)

        self.slider_scale.blockSignals(True)
        self.combo_bonds.blockSignals(True)
        if style_text == "Van der Waals":
            self.slider_scale.setValue(135)
            self.lbl_scale.setText("1.35×")
            self.combo_bonds.setCurrentText("none")
        elif style_text == "Licorice":
            self.slider_scale.setValue(45)
            self.lbl_scale.setText("0.45×")
            self.combo_bonds.setCurrentText("table")
        else:
            self.slider_scale.setValue(100)
            self.lbl_scale.setText("1.00×")
            self.combo_bonds.setCurrentText("style")
        self.slider_scale.blockSignals(False)
        self.combo_bonds.blockSignals(False)
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_scale_slider_changed(self, val: int) -> None:
        self.lbl_scale.setText(f"{val / 100.0:.2f}×")
        self._rebuild_viewport(reset_camera=False, trigger_preview=False)

    def _on_display_control_changed(self, *_args: Any) -> None:
        self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_dark_toggled(self, dark: bool) -> None:
        self.welcome_widget.set_dark_backdrop(dark)
        if self._vp_scene is not None:
            self._rebuild_viewport(reset_camera=False, trigger_preview=True)

    def _on_crop_z_changed(self, mode: str) -> None:
        self._crop_z_mode = mode
        self.entry_crop_z.setEnabled(mode == "custom")
        self._crop_z_custom = self.entry_crop_z.text().strip() or "2.0,4.0"
        self._rebuild_viewport(reset_camera=not self._use_interactive_cam, trigger_preview=True)

    def _on_res_preset_changed(self, preset_name: str) -> None:
        w, h, dpi = JOURNAL_PRESETS.get(preset_name, (1400, 1400, 300))
        self._set_resolution(w, h, dpi=dpi, trigger_preview=True)

    def _set_resolution(self, w: int, h: int, dpi: int = 300, trigger_preview: bool = True) -> None:
        self._res_w = max(16, int(w))
        self._res_h = max(16, int(h))
        self._target_dpi = max(72, int(dpi))
        if self._vp_scene is not None and not self._use_interactive_cam:
            self._apply_view_to_viewport(self._vp_scene)
        if self._vp_scene is not None:
            self.viewport.set_scene(
                self._vp_scene,
                res_w=self._res_w,
                res_h=self._res_h,
                dark_backdrop=bool(self.chk_dark.isChecked()),
                structure_summary=self._structure_summary,
                material_finish=self.combo_texture.currentText().strip().lower(),
            )
            if trigger_preview:
                self._request_blender_preview()
        self.lbl_status.setText(f"Resolution {self._res_w}×{self._res_h} @ {self._target_dpi} DPI")

    def _pick_view_preset(self, preset: str) -> None:
        self._view_preset_value = preset
        self._view_file_value = ""
        self._use_interactive_cam = False
        if self._vp_scene is not None:
            self._apply_view_to_viewport(self._vp_scene)
            self.viewport.clear_blender_preview()
            self.viewport.mark_geometry_dirty()
            self.viewport.update()
            self._record_history_snapshot()
            self._request_blender_preview()

    def _on_fit_view(self) -> None:
        self._view_file_value = ""
        self._use_interactive_cam = False
        if self._vp_scene is not None:
            self._apply_view_to_viewport(self._vp_scene)
            self.viewport.clear_blender_preview()
            self.viewport.update()
            self._record_history_snapshot()
            self._request_blender_preview()

    def _on_save_view_json(self) -> None:
        if self._vp_scene is None:
            QMessageBox.information(self, "Save view.json", "Load or drop a structure first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save view.json", "view.json", "JSON files (*.json)")
        if not path:
            return
        payload = self._vp_scene.to_view_dict(self._res_w, self._res_h)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._view_file_value = path
        self.lbl_status.setText(f"Saved view -> {Path(path).name}")

    def _on_copy_image(self) -> None:
        img = self.viewport.render_snapshot_pil() if self._vp_scene is not None else None
        if img is None:
            return
        if copy_pil_image_to_clipboard(img):
            self.lbl_status.setText("Copied viewport image to clipboard")
        else:
            QMessageBox.warning(self, "Copy Image", "Clipboard copy failed on this system.")

    # ---- OpenGL Viewport Signals & Auto Blender Preview ----
    def _on_gl_interaction_started(self) -> None:
        self._preview_seq += 1
        self._use_interactive_cam = True

    def _on_gl_camera_changed(self) -> None:
        self._use_interactive_cam = True

    def _on_gl_camera_settled(self) -> None:
        self._use_interactive_cam = True
        self._record_history_snapshot()
        self._request_blender_preview()

    def _on_gl_atom_clicked(self, idx: int, additive: bool) -> None:
        if not additive:
            if idx in self._selected_atoms:
                self._selected_atoms.remove(idx)
                self._selection_order = [i for i in self._selection_order if i != idx]
            else:
                self._selected_atoms.add(idx)
                if idx not in self._selection_order:
                    self._selection_order.append(idx)
        else:
            self._selected_atoms.add(idx)
            if idx not in self._selection_order:
                self._selection_order.append(idx)
        self._sync_selection_and_alphas(trigger_blender=True)
        self._record_history_snapshot()

    # ---- Selection & Spotlight ----
    def _refresh_atom_list(self) -> None:
        self.atom_list.blockSignals(True)
        self.atom_list.clear()
        if self._structure_cache is not None:
            syms = self._structure_cache.get("symbols", [])
            pos = self._structure_cache.get("positions", [])
            for idx, (sym, xyz) in enumerate(zip(syms, pos)):
                item = QListWidgetItem(f"{atom_label(idx, sym):<5} ({xyz[0]:6.2f}, {xyz[1]:6.2f}, {xyz[2]:6.2f})")
                self.atom_list.addItem(item)
                if idx in self._selected_atoms:
                    item.setSelected(True)
        self.atom_list.blockSignals(False)

    def _on_atom_list_selection_changed(self) -> None:
        sel = {self.atom_list.row(it) for it in self.atom_list.selectedItems()}
        self._selected_atoms = sel
        self._selection_order = [i for i in self._selection_order if i in sel] + sorted(sel - set(self._selection_order))
        self._sync_selection_and_alphas(trigger_blender=True, sync_list=False)

    def _on_opacity_changed(self, val: int) -> None:
        self.lbl_opacity.setText(f"{val}%")
        self._sync_selection_and_alphas(trigger_blender=False, sync_list=False)

    def _update_measurement_label(self) -> None:
        if self._structure_cache is None:
            self.lbl_measurement.setText("Click 1–4 atoms to measure distance / angle / dihedral")
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        self.lbl_measurement.setText(compute_selection_measurement(syms, pos, self._selection_order))

    def _sync_selection_and_alphas(
        self,
        trigger_blender: bool = False,
        sync_list: bool = True,
        alphas_override: list[float] | None = None,
    ) -> None:
        if self._structure_cache is None:
            return
        n = len(self._structure_cache.get("symbols", []))
        if alphas_override is not None:
            alphas = list(alphas_override)
        elif self.chk_fade_unselected.isChecked() and self._selected_atoms:
            alphas = compute_atom_alphas(
                n,
                self._selected_atoms,
                self.slider_opacity.value() / 100.0,
                invert=True,
            )
        elif self._selected_atoms and self.slider_opacity.value() < 100:
            alphas = compute_atom_alphas(
                n,
                self._selected_atoms,
                self.slider_opacity.value() / 100.0,
                invert=False,
            )
        else:
            alphas = [1.0] * n

        if self._structure_cache.get("alphas") != alphas:
            self._structure_cache["alphas"] = alphas
            self._written_struct_version = -1

        if self._vp_scene is not None:
            self._vp_scene.selected_atoms = set(self._selected_atoms)
            self._vp_scene.selection_order = list(self._selection_order)
            self._vp_scene.atom_alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.02, 1.0)
            self.viewport.mark_geometry_dirty()
            self.viewport.update()

        if sync_list:
            self._refresh_atom_list()
        self._update_measurement_label()
        if trigger_blender:
            self._request_blender_preview()

    def _select_all_atoms(self) -> None:
        if self._structure_cache is None:
            return
        n = len(self._structure_cache.get("symbols", []))
        self._selected_atoms = set(range(n))
        self._selection_order = list(range(n))
        self._sync_selection_and_alphas(trigger_blender=True)

    def _invert_selection(self) -> None:
        if self._structure_cache is None:
            return
        n = len(self._structure_cache.get("symbols", []))
        self._selected_atoms = set(range(n)) - self._selected_atoms
        self._selection_order = sorted(self._selected_atoms)
        self._sync_selection_and_alphas(trigger_blender=True)

    def _clear_selection(self) -> None:
        self._selected_atoms.clear()
        self._selection_order.clear()
        self._sync_selection_and_alphas(trigger_blender=True)

    def _apply_spotlight(self) -> None:
        if self._structure_cache is None:
            return
        syms = self._structure_cache.get("symbols", [])
        pos = self._structure_cache.get("positions", [])
        focal, alphas = compute_adsorbate_spotlight_alphas(syms, pos, focal_indices=self._selected_atoms or None)
        self._selected_atoms = set(focal)
        self._selection_order = sorted(focal)
        self._sync_selection_and_alphas(trigger_blender=True, alphas_override=alphas)
        self.lbl_status.setText(f"Spotlight applied ({len(focal)} focal atoms)")

    # ---- Trajectory Stepping ----
    def _step_frame(self, delta: int) -> None:
        if not self._trajectory_frames:
            return
        idx = max(0, min(len(self._trajectory_frames) - 1, self._trajectory_index + delta))
        self.slider_frame.setValue(idx)

    def _on_frame_slider_changed(self, idx: int) -> None:
        if not (0 <= idx < len(self._trajectory_frames)):
            return
        self._trajectory_index = idx
        self._structure_cache = dict(self._trajectory_frames[idx])
        self._structure_version += 1
        self._written_struct_version = -1
        self.lbl_frame.setText(f"Frame {idx + 1}/{len(self._trajectory_frames)}")
        self._rebuild_viewport(reset_camera=False, trigger_preview=False)

    # ---- Fast Low-Spec Auto Blender Preview ----
    def _request_blender_preview(self) -> None:
        if not self.chk_auto_preview.isChecked():
            return
        if self._structure_cache is None or self._vp_scene is None:
            return
        blender_raw = self.entry_blender.text().strip()
        if not blender_raw:
            try:
                blender_raw = str(find_blender_executable())
                self.entry_blender.setText(blender_raw)
            except FileNotFoundError:
                return
        blender_path = Path(blender_raw).expanduser()
        if not blender_path.is_file():
            return

        struct_json = Path(self._preview_tmp_dir.name) / "preview_structure.json"
        if self._written_struct_version != self._structure_version or not struct_json.is_file():
            struct_json.write_text(json.dumps(self._structure_cache), encoding="utf-8")
            self._written_struct_version = self._structure_version

        style_name = self.combo_style.currentText().strip() or "default"
        style_path = resolve_style_path(style_name)
        style_mtime = style_path.stat().st_mtime if style_path.is_file() else 0.0
        eff_style_path = style_path
        fin_key = self.combo_texture.currentText().strip().lower()
        if self._custom_element_colors or fin_key != "studio":
            eff_style_path = Path(self._preview_tmp_dir.name) / "preview_style.json"
            eff_style_path.write_text(json.dumps(self._resolve_style_dict(), indent=2), encoding="utf-8")

        prev_w, prev_h = compute_preview_resolution(self._res_w, self._res_h, max_side=640)
        prev_samples = min(8, max(4, int(self.spin_samples.value())))
        view_arg: str | dict[str, Any] = self._vp_scene.to_view_dict(prev_w, prev_h)
        r_scale, b_mode, b_col = self._effective_bonds_and_scale()
        crop_z = "auto" if self._crop_z_mode == "auto" else (self._crop_z_custom if self._crop_z_mode == "custom" else None)

        geom_payload = [
            self._structure_version,
            str(style_path),
            style_mtime,
            sorted(self._custom_element_colors.items()),
            b_mode,
            b_col,
            round(r_scale, 4),
            bool(self.chk_show_cell.isChecked()),
            crop_z,
            [round(float(a), 3) for a in self._structure_cache.get("alphas", [])],
            list(self._structure_cache.get("dashed_bonds", [])),
            len(self._structure_cache.get("polyhedra", [])),
            bool(self._structure_cache.get("show_cell_vectors", False)),
        ]
        geom_key = json.dumps(geom_payload)
        scene_key = json.dumps(geom_payload + [fin_key])

        self._preview_seq += 1
        seq = self._preview_seq
        out_png = Path(self._preview_tmp_dir.name) / f"preview_{seq % 2}.png"
        if not self._busy:
            self.lbl_status.setText("Rendering fast Blender preview...")
        self._preview_worker.request(
            {
                "action": "render",
                "seq": seq,
                "blender_exe": str(blender_path),
                "geom_key": geom_key,
                "scene_key": scene_key,
                "structure": str(struct_json),
                "style": str(eff_style_path),
                "crop_z": crop_z,
                "resolution": [prev_w, prev_h],
                "samples": prev_samples,
                "bonds": None if b_mode == "style" else b_mode,
                "bond_color": None if b_col == "style" else b_col,
                "radius_scale": float(r_scale),
                "no_cell": not bool(self.chk_show_cell.isChecked()),
                "fit_view": False,
                "view": view_arg,
                "out": str(out_png),
            }
        )

    def _on_preview_status(self, seq: int, msg: str) -> None:
        if seq == self._preview_seq and not self._busy:
            self.lbl_status.setText(msg)

    def _on_preview_done(self, seq: int, out_png_str: str) -> None:
        if seq != self._preview_seq or self.viewport.is_interacting:
            return
        png_path = Path(out_png_str)
        if not png_path.is_file():
            return
        try:
            from PIL import Image

            pil_img = Image.open(png_path).convert("RGBA")
            self.viewport.set_blender_preview(pil_img)
            if not self._busy:
                self.lbl_status.setText("Blender preview ready (drag for 60fps GL 3D)")
        except Exception as exc:
            self._append_log(f"Preview decode warning: {exc}")

    def _on_preview_err(self, seq: int, msg: str) -> None:
        if seq == self._preview_seq:
            self._append_log(f"Preview error: {msg}")
            if not self._busy:
                self.lbl_status.setText("GL 3D active (Blender preview failed)")

    # ---- High-Res Render, Panel, Batch, Blender ----
    def _collect_config(self) -> GuiRenderConfig:
        r_scale, b_mode, b_col = self._effective_bonds_and_scale()
        rep = (int(self.spin_rep_x.value()), int(self.spin_rep_y.value()), int(self.spin_rep_z.value()))
        if self._structure_cache is not None and not self._structure_cache.get("has_cell", False):
            rep = (1, 1, 1)
        return GuiRenderConfig(
            input_path=extract_input_path(self.entry_input.text()),
            output_path=self.entry_output.text().strip(),
            style=self.combo_style.currentText().strip() or "default",
            view_mode="file" if self._view_file_value else "preset",
            view_preset=self._view_preset_value,
            view_file=self._view_file_value,
            fit_view=False,
            replicate=rep,
            wrap=bool(self.chk_wrap.isChecked()),
            crop_z_mode=self._crop_z_mode,
            crop_z_custom=self.entry_crop_z.text().strip() or "2.0,4.0",
            bonds=b_mode,
            bond_color=b_col,
            radius_scale=r_scale,
            show_cell=bool(self.chk_show_cell.isChecked()),
            resolution=(self._res_w, self._res_h),
            samples=int(self.spin_samples.value()),
            blender_exe=self.entry_blender.text().strip(),
        )

    def _effective_cfg_for_cli(self, cfg: GuiRenderConfig) -> GuiRenderConfig:
        eff_input = cfg.input_path
        eff_rep = cfg.replicate
        eff_style = cfg.style
        fin_key = self.combo_texture.currentText().strip().lower()
        if self._custom_element_colors or fin_key != "studio":
            tmp_fd, tmp_name = tempfile.mkstemp(prefix="atomic_qt_style_", suffix=".json")
            os.close(tmp_fd)
            sp = Path(tmp_name)
            sp.write_text(json.dumps(self._resolve_style_dict(), indent=2), encoding="utf-8")
            self._temp_view_files.append(sp)
            eff_style = str(sp)

        if self._structure_cache is not None:
            tmp_sfd, tmp_sname = tempfile.mkstemp(prefix="atomic_qt_struct_", suffix=".json")
            os.close(tmp_sfd)
            spath = Path(tmp_sname)
            spath.write_text(json.dumps(self._structure_cache), encoding="utf-8")
            self._temp_view_files.append(spath)
            eff_input = str(spath)
            eff_rep = (1, 1, 1)

        if self._vp_scene is not None:
            vd = self._vp_scene.to_view_dict(cfg.resolution[0], cfg.resolution[1])
            tmp_vfd, tmp_vname = tempfile.mkstemp(prefix="atomic_qt_view_", suffix=".json")
            os.close(tmp_vfd)
            vp_path = Path(tmp_vname)
            vp_path.write_text(json.dumps(vd, indent=2), encoding="utf-8")
            self._temp_view_files.append(vp_path)
            v_mode, v_file = "file", str(vp_path)
        else:
            v_mode, v_file = cfg.view_mode, cfg.view_file

        return GuiRenderConfig(
            input_path=eff_input,
            output_path=cfg.output_path,
            style=eff_style,
            view_mode=v_mode,
            view_preset=cfg.view_preset,
            view_file=v_file,
            fit_view=False,
            type_map=cfg.type_map,
            replicate=eff_rep,
            wrap=cfg.wrap,
            crop_z_mode=cfg.crop_z_mode,
            crop_z_custom=cfg.crop_z_custom,
            bonds=cfg.bonds,
            bond_color=cfg.bond_color,
            radius_scale=cfg.radius_scale,
            show_cell=cfg.show_cell,
            resolution=cfg.resolution,
            samples=cfg.samples,
            blender_exe=cfg.blender_exe,
            save_blend=cfg.save_blend,
        )

    def _popen_cli(self, cli_args: list[str]) -> subprocess.Popen[str]:
        cmd = [sys.executable, "-m", "atomic_blender.cli", *cli_args]
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            kwargs["creationflags"] = flags
        return subprocess.Popen(cmd, **kwargs)

    def _set_busy(self, busy: bool, status_text: str) -> None:
        self._busy = busy
        self.btn_render.setEnabled(not busy)
        self.btn_blender.setEnabled(not busy)
        self.btn_top_side.setEnabled(not busy)
        self.btn_batch.setEnabled(not busy)
        self.lbl_status.setText(status_text)
        self.render_progress.setVisible(busy)
        self.btn_cancel_render.setVisible(busy)

    def _on_cancel_render(self) -> None:
        self._render_cancelled = True
        proc = self._active_render_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        self._append_log("Render cancelled by user.")
        self.lbl_status.setText("Cancelling render...")

    def _ensure_blender_ready(self) -> str | None:
        raw = self.entry_blender.text().strip()
        if raw and Path(raw).expanduser().is_file():
            return str(Path(raw).expanduser())
        try:
            exe = str(find_blender_executable())
            self.entry_blender.setText(exe)
            return exe
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "Blender Not Found",
                "Could not locate Blender executable. Set the Blender path in the Scene tab first.",
            )
            return None

    def _on_render(self) -> None:
        if self._busy or self._ensure_blender_ready() is None:
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Configuration", str(exc))
            return

        out_png = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        self.entry_output.setText(str(out_png))
        args = build_render_cli_args(cli_cfg, out_png=out_png)
        self._task_start = time.monotonic()
        self._render_cancelled = False
        self._set_busy(True, "Running High-Res Cycles render...")
        self._append_log(f"$ atomic-blender {' '.join(args)}")

        def worker() -> None:
            proc = self._popen_cli(args)
            self._active_render_proc = proc
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    self._bridge.render_log.emit(raw_line.rstrip())
                rc = proc.wait()
            finally:
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                if self._active_render_proc is proc:
                    self._active_render_proc = None
            self._bridge.render_done.emit("render", int(rc), str(out_png))

        threading.Thread(target=worker, daemon=True).start()

    def _on_render_done(self, mode: str, rc: int, target_str: str) -> None:
        elapsed = time.monotonic() - self._task_start
        target_path = Path(target_str)
        if self._render_cancelled and rc != 0:
            self._set_busy(False, f"Cancelled after {elapsed:.1f}s")
            return
        if rc != 0:
            self._set_busy(False, f"Render failed (exit {rc}) after {elapsed:.1f}s")
            QMessageBox.critical(self, "Render Failed", f"Blender CLI exited with status {rc}. Open Log for details.")
            return
        if mode == "render":
            self._preview_seq += 1
            if target_path.is_file():
                try:
                    from PIL import Image

                    out_img = Image.open(target_path).convert("RGBA")
                    if self.chk_scale_bar.isChecked() and self._vp_scene is not None:
                        render_scale_bar_overlay(out_img, self._vp_scene.ortho_scale, self._res_w, self._res_h, dark_backdrop=bool(self.chk_dark.isChecked()))
                    if self.chk_show_axes.isChecked():
                        q = self._vp_scene.quat if self._vp_scene is not None else (1.0, 0.0, 0.0, 0.0)
                        cv = self._vp_scene.cell_vecs if (self._vp_scene is not None and self._vp_scene.has_cell) else None
                        render_reference_axes_overlay(out_img, q, show_labels=True, dark_backdrop=bool(self.chk_dark.isChecked()), cell_vecs=cv)
                    if self._vp_scene is not None and (self.chk_labels.isChecked() or self._selected_atoms):
                        render_atom_labels_and_selection_overlay(out_img, self._vp_scene, self._res_w, self._res_h, draw_labels=bool(self.chk_labels.isChecked()), draw_selection=bool(self._selected_atoms))
                    out_img.save(target_path)
                    embed_png_dpi(target_path, self._target_dpi)
                    self.viewport.set_blender_preview(out_img)
                except Exception as exc:
                    self._append_log(f"Post-process warning: {exc}")
            self._set_busy(False, f"Rendered {target_path.name} in {elapsed:.1f}s ({self._target_dpi} DPI)")
        elif mode == "open_blender":
            self._set_busy(False, f"Opening {target_path.name} in Blender...")
            subprocess.Popen([self._pending_blender_exe, str(target_path)])

    def _on_render_top_side_panel(self) -> None:
        if self._busy or self._ensure_blender_ready() is None:
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Configuration", str(exc))
            return

        base_out = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path.with_suffix(".png")
        panel_png = base_out.with_name(f"{base_out.stem}_panel.png")
        dark_bg = bool(self.chk_dark.isChecked())
        self._task_start = time.monotonic()
        self._set_busy(True, "Rendering Top + Side panel...")

        def worker() -> None:
            from PIL import Image

            with tempfile.TemporaryDirectory(prefix="atomic_qt_panel_") as tmp_dir:
                tmp_p = Path(tmp_dir)
                top_png, side_png = tmp_p / "top.png", tmp_p / "side.png"
                for preset, out_p in (("c", top_png), ("side-oblique", side_png)):
                    c = GuiRenderConfig(
                        input_path=cli_cfg.input_path,
                        output_path=str(out_p),
                        style=cli_cfg.style,
                        view_mode="preset",
                        view_preset=preset,
                        replicate=cli_cfg.replicate,
                        wrap=cli_cfg.wrap,
                        crop_z_mode=cli_cfg.crop_z_mode,
                        crop_z_custom=cli_cfg.crop_z_custom,
                        bonds=cli_cfg.bonds,
                        bond_color=cli_cfg.bond_color,
                        radius_scale=cli_cfg.radius_scale,
                        show_cell=cli_cfg.show_cell,
                        resolution=cli_cfg.resolution,
                        samples=cli_cfg.samples,
                        blender_exe=cli_cfg.blender_exe,
                    )
                    proc = self._popen_cli(build_render_cli_args(c, out_png=out_p))
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self._bridge.render_log.emit(line.rstrip())
                    rc = proc.wait()
                    if rc != 0:
                        self._bridge.panel_done.emit(int(rc), str(panel_png), f"Panel render exited with status {rc}")
                        return
                panel_img = compose_dual_view_panel(Image.open(top_png), Image.open(side_png), dark_backdrop=dark_bg)
                panel_img.save(panel_png)
                embed_png_dpi(panel_png, self._target_dpi)
                self._bridge.panel_done.emit(0, str(panel_png), "")

        threading.Thread(target=worker, daemon=True).start()

    def _on_panel_done(self, rc: int, panel_str: str, err: str) -> None:
        elapsed = time.monotonic() - self._task_start
        p = Path(panel_str)
        if rc != 0:
            self._set_busy(False, f"Panel failed after {elapsed:.1f}s")
            QMessageBox.critical(self, "Panel Render Failed", err)
            return
        self._set_busy(False, f"Saved panel {p.name} in {elapsed:.1f}s")
        try:
            from PIL import Image

            self.viewport.set_blender_preview(Image.open(p).convert("RGBA"))
        except Exception:
            pass

    def _on_batch_render(self) -> None:
        if self._busy or self._ensure_blender_ready() is None:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Structure Files for Batch Render",
            "",
            "Structure files (*.cif *.vasp *.xyz *.extxyz *.data *.lmp *.json);;All files (*.*)",
        )
        if not files:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Batch Output Folder")
        if not out_dir:
            return
        cfg = self._collect_config()
        in_paths = [Path(f) for f in files]
        out_paths = resolve_batch_output_paths(in_paths, Path(out_dir))
        self._task_start = time.monotonic()
        self._set_busy(True, f"Batch rendering 0/{len(in_paths)}...")

        def worker() -> None:
            done_cnt = 0
            last_png = ""
            for inp, outp in zip(in_paths, out_paths):
                c = GuiRenderConfig(
                    input_path=str(inp),
                    output_path=str(outp),
                    style=cfg.style,
                    view_mode=cfg.view_mode,
                    view_preset=cfg.view_preset,
                    view_file=cfg.view_file,
                    replicate=cfg.replicate,
                    wrap=cfg.wrap,
                    crop_z_mode=cfg.crop_z_mode,
                    crop_z_custom=cfg.crop_z_custom,
                    bonds=cfg.bonds,
                    bond_color=cfg.bond_color,
                    radius_scale=cfg.radius_scale,
                    show_cell=cfg.show_cell,
                    resolution=cfg.resolution,
                    samples=cfg.samples,
                    blender_exe=cfg.blender_exe,
                )
                proc = self._popen_cli(build_render_cli_args(c, out_png=outp))
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._bridge.render_log.emit(line.rstrip())
                rc = proc.wait()
                if rc != 0:
                    self._bridge.batch_done.emit(int(rc), done_cnt, len(in_paths), last_png)
                    return
                embed_png_dpi(outp, self._target_dpi)
                done_cnt += 1
                last_png = str(outp)
            self._bridge.batch_done.emit(0, done_cnt, len(in_paths), last_png)

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_done(self, rc: int, done_cnt: int, total_cnt: int, last_png: str) -> None:
        elapsed = time.monotonic() - self._task_start
        if rc != 0:
            self._set_busy(False, f"Batch failed at {done_cnt + 1}/{total_cnt} after {elapsed:.1f}s")
            QMessageBox.critical(self, "Batch Render Failed", f"Stopped after {done_cnt}/{total_cnt} files.")
            return
        self._set_busy(False, f"Batch rendered {done_cnt}/{total_cnt} PNGs in {elapsed:.1f}s")
        if last_png and Path(last_png).is_file():
            try:
                from PIL import Image

                self.viewport.set_blender_preview(Image.open(last_png).convert("RGBA"))
            except Exception:
                pass

    def _on_open_in_blender(self) -> None:
        if self._busy:
            return
        exe = self._ensure_blender_ready()
        if exe is None:
            return
        try:
            cfg = self._collect_config()
            in_path = validate_gui_config(cfg)
            cli_cfg = self._effective_cfg_for_cli(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Open in Blender", str(exc))
            return

        base = Path(cfg.output_path).expanduser() if cfg.output_path.strip() else in_path
        blend_path = base.with_suffix(".blend")
        args = build_render_cli_args(cli_cfg, out_png=None, save_blend=blend_path)
        self._pending_blender_exe = exe
        self._task_start = time.monotonic()
        self._set_busy(True, "Generating .blend scene...")

        def worker() -> None:
            proc = self._popen_cli(args)
            self._active_render_proc = proc
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._bridge.render_log.emit(line.rstrip())
                rc = proc.wait()
            finally:
                if self._active_render_proc is proc:
                    self._active_render_proc = None
            self._bridge.render_done.emit("open_blender", int(rc), str(blend_path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_open_folder(self) -> None:
        target = self.entry_output.text().strip() or extract_input_path(self.entry_input.text())
        if not target:
            QMessageBox.information(self, "Open Folder", "No input or output path selected yet.")
            return
        folder = Path(target).expanduser().resolve().parent
        if folder.is_dir():
            _open_folder(folder)

    # ---- Undo / Redo ----
    def _capture_undo_state(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "style": self.combo_style.currentText(),
            "texture": self.combo_texture.currentText(),
            "atom_style": self.combo_atom_style.currentText(),
            "bonds": self.combo_bonds.currentText(),
            "bond_color": self.combo_bond_color.currentText(),
            "scale": int(self.slider_scale.value()),
            "show_atoms": bool(self.chk_show_atoms.isChecked()),
            "show_cell": bool(self.chk_show_cell.isChecked()),
            "show_axes": bool(self.chk_show_axes.isChecked()),
            "show_cell_vectors": bool(self.chk_cell_vectors.isChecked()),
            "show_polyhedra": bool(self.chk_polyhedra.isChecked()),
            "show_scale_bar": bool(self.chk_scale_bar.isChecked()),
            "show_hbonds": bool(self.chk_hbonds.isChecked()),
            "show_labels": bool(self.chk_labels.isChecked()),
            "dark": bool(self.chk_dark.isChecked()),
            "selected_atoms": sorted(self._selected_atoms),
            "selection_order": list(self._selection_order),
            "opacity": int(self.slider_opacity.value()),
            "fade_unselected": bool(self.chk_fade_unselected.isChecked()),
        }
        if self._vp_scene is not None:
            snap["camera"] = {
                "quat": [round(float(v), 6) for v in self._vp_scene.quat],
                "ortho_scale": round(float(self._vp_scene.ortho_scale), 6),
                "target": [round(float(v), 6) for v in self._vp_scene.target],
            }
        return snap

    def _record_history_snapshot(self) -> None:
        if self._restoring_history or self._structure_cache is None:
            return
        self._history.push(self._capture_undo_state())

    def _apply_undo_state(self, snap: dict[str, Any]) -> None:
        self._restoring_history = True
        try:
            widgets = (
                self.combo_style,
                self.combo_texture,
                self.combo_atom_style,
                self.combo_bonds,
                self.combo_bond_color,
                self.slider_scale,
                self.chk_show_atoms,
                self.chk_show_cell,
                self.chk_show_axes,
                self.chk_cell_vectors,
                self.chk_polyhedra,
                self.chk_scale_bar,
                self.chk_hbonds,
                self.chk_labels,
                self.chk_dark,
                self.slider_opacity,
                self.chk_fade_unselected,
            )
            for w in widgets:
                w.blockSignals(True)
            self.combo_style.setCurrentText(str(snap.get("style", "default")))
            self.combo_texture.setCurrentText(str(snap.get("texture", "Studio")))
            self.combo_atom_style.setCurrentText(str(snap.get("atom_style", "Van der Waals")))
            self.combo_bonds.setCurrentText(str(snap.get("bonds", "none")))
            self.combo_bond_color.setCurrentText(str(snap.get("bond_color", "style")))
            self.slider_scale.setValue(int(snap.get("scale", 100)))
            self.lbl_scale.setText(f"{self.slider_scale.value() / 100.0:.2f}×")
            self.chk_show_atoms.setChecked(bool(snap.get("show_atoms", True)))
            self.chk_show_cell.setChecked(bool(snap.get("show_cell", True)))
            self.chk_show_axes.setChecked(bool(snap.get("show_axes", True)))
            self.chk_cell_vectors.setChecked(bool(snap.get("show_cell_vectors", False)))
            self.chk_polyhedra.setChecked(bool(snap.get("show_polyhedra", False)))
            self.chk_scale_bar.setChecked(bool(snap.get("show_scale_bar", True)))
            self.chk_hbonds.setChecked(bool(snap.get("show_hbonds", False)))
            self.chk_labels.setChecked(bool(snap.get("show_labels", False)))
            self.chk_dark.setChecked(bool(snap.get("dark", True)))
            self.slider_opacity.setValue(int(snap.get("opacity", 30)))
            self.lbl_opacity.setText(f"{self.slider_opacity.value()}%")
            self.chk_fade_unselected.setChecked(bool(snap.get("fade_unselected", False)))
            for w in widgets:
                w.blockSignals(False)

            self._selected_atoms = {int(i) for i in snap.get("selected_atoms", [])}
            self._selection_order = [int(i) for i in snap.get("selection_order", [])]
            self._rebuild_viewport(reset_camera=False, trigger_preview=False)
            cam = snap.get("camera")
            if isinstance(cam, dict) and self._vp_scene is not None:
                self._vp_scene.quat = tuple(float(v) for v in cam["quat"])  # type: ignore[assignment]
                self._vp_scene.ortho_scale = float(cam["ortho_scale"])
                self._vp_scene.target = np.asarray(cam["target"], dtype=np.float64)
                self._use_interactive_cam = True
                self.viewport.update()
            self._sync_selection_and_alphas(trigger_blender=True)
        finally:
            self._restoring_history = False

    def undo(self) -> None:
        if not self._history.can_undo:
            self.lbl_status.setText("Nothing to undo")
            return
        snap = self._history.undo()
        if snap is not None:
            self._apply_undo_state(snap)
            self.lbl_status.setText("Undo applied")

    def redo(self) -> None:
        if not self._history.can_redo:
            self.lbl_status.setText("Nothing to redo")
            return
        snap = self._history.redo()
        if snap is not None:
            self._apply_undo_state(snap)
            self.lbl_status.setText("Redo applied")

    # ---- Log, Shortcuts Dialog, Drag-and-Drop & Cleanup ----
    def _toggle_log_panel(self, visible: bool) -> None:
        self.log_box.setVisible(bool(visible))

    def _append_log(self, line: str) -> None:
        self.log_box.appendPlainText(line.rstrip())

    def _show_shortcuts_dialog(self) -> None:
        msg = (
            "Mouse Controls:\n"
            "  • Left-Drag: 60fps OpenGL orbit (triggers Cycles preview on release)\n"
            "  • Right-Drag / Middle-Drag / Shift+Left-Drag: Pan view\n"
            "  • Scroll Wheel: Zoom in / out\n"
            "  • Click Atom: Toggle select & measure geometry\n\n"
            "Keyboard Shortcuts:\n"
            "  • Ctrl+O: Open structure file\n"
            "  • Ctrl+R: Render High-Res PNG\n"
            "  • Ctrl+Z / Ctrl+Y: Undo / Redo\n"
            "  • Ctrl+C: Copy current image to clipboard\n"
            "  • R: Fit & reset camera\n"
            "  • Esc: Clear selection\n"
            "  • F1 or ?: Show this shortcuts dialog"
        )
        QMessageBox.information(self, "Keyboard & Mouse Shortcuts", msg)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        for raw in paths:
            p = Path(raw).expanduser()
            if not p.is_file():
                continue
            kind = classify_dropped_file(p)
            if kind == "view":
                self._view_file_value = str(p)
                self._use_interactive_cam = False
                if self._vp_scene is not None:
                    self._apply_view_to_viewport(self._vp_scene)
                    self.viewport.update()
                    self._request_blender_preview()
            elif kind == "style":
                self.combo_style.addItem(str(p))
                self.combo_style.setCurrentText(str(p))
            else:
                self.entry_output.setText(str(p.with_suffix(".png")))
                self.open_path(str(p))

    def closeEvent(self, event: Any) -> None:
        if hasattr(self, "welcome_widget") and hasattr(self.welcome_widget, "_timer"):
            self.welcome_widget._timer.stop()
        if self._active_render_proc is not None and self._active_render_proc.poll() is None:
            try:
                self._active_render_proc.kill()
            except Exception:
                pass
        try:
            self._preview_worker.stop()
        except Exception:
            pass
        try:
            self._preview_tmp_dir.cleanup()
        except Exception:
            pass
        for tmp_p in self._temp_view_files:
            try:
                tmp_p.unlink(missing_ok=True)
            except Exception:
                pass
        super().closeEvent(event)
