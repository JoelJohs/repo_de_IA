from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.data.load_external_test_folder import convert_external_folder_to_test
from src.inference.predict import predict_image

from .widgets.class_card import ClassListPanel
from .widgets.image_viewer import ImageViewer
from .widgets.prob_bar import ProbBarPanel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _list_images_in_dir(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            out.append(path)
    return out


def _list_dataset_images(dataset_dir: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(dataset_dir):
        return out
    for cls in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_path):
            out[cls] = _list_images_in_dir(cls_path)
    return out


def _extract_true_class(path: str, class_names: List[str]) -> Optional[str]:
    name = os.path.basename(path)
    for cls in class_names:
        if name.startswith(cls + "__"):
            return cls
    parent = os.path.basename(os.path.dirname(path))
    if parent in class_names:
        return parent
    return None


def _fmt_count(n: int) -> str:
    return f"{n:,}".replace(",", ".")


class PredictWorker(QThread):
    finished = pyqtSignal(str, list, float)
    error = pyqtSignal(str, str)

    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str],
        image_path: str,
        image_size: Tuple[int, int],
        top_k: int = 3,
    ) -> None:
        super().__init__()
        self._model = model
        self._class_names = class_names
        self._image_path = image_path
        self._image_size = image_size
        self._top_k = top_k

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            preds = predict_image(
                model=self._model,
                class_names=self._class_names,
                image_path=self._image_path,
                image_size=self._image_size,
                top_k=self._top_k,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            self.finished.emit(self._image_path, preds, dt)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(self._image_path, str(exc))


class LoadFolderWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        source_dir: str,
        test_dir: str,
        image_size: Tuple[int, int],
        jpeg_quality: int = 90,
    ) -> None:
        super().__init__()
        self._source_dir = source_dir
        self._test_dir = test_dir
        self._image_size = image_size
        self._jpeg_quality = jpeg_quality

    def run(self) -> None:
        try:
            summary = convert_external_folder_to_test(
                source_dir=self._source_dir,
                test_dir=self._test_dir,
                image_size=self._image_size,
                jpeg_quality=self._jpeg_quality,
            )
            self.finished.emit(
                {
                    "written": summary.written,
                    "skipped": summary.skipped,
                    "rejected": summary.rejected,
                    "per_subfolder": dict(summary.per_subfolder),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str],
        dataset_dir: str,
        test_dir: str,
        image_size: Tuple[int, int],
        model_path: str,
        metrics: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._class_names = list(class_names)
        self._dataset_dir = dataset_dir
        self._test_dir = test_dir
        self._image_size = image_size
        self._model_path = model_path
        self._metrics = metrics or {}
        self._logger = logger

        self._dataset_images: Dict[str, List[str]] = _list_dataset_images(dataset_dir)
        self._test_images: List[str] = _list_images_in_dir(test_dir)
        self._current_source = "test"
        self._current_worker: Optional[PredictWorker] = None

        self.setWindowTitle("🐾 CNN Clasificador de Animales")
        self.setMinimumSize(QSize(960, 600))
        self.resize(QSize(1100, 680))

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 520, 280])
        root.addWidget(splitter, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo · selecciona una imagen de la lista")

        self._populate_class_panel()
        self._populate_image_list()

        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open_file_dialog)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(60)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(10)

        title = QLabel("🐾 CNN Clasificador de Animales")
        title.setObjectName("title")

        subtitle_text = f"modelo: {os.path.basename(self._model_path)}  ·  input: {self._image_size[0]}×{self._image_size[1]}×3  ·  clases: {len(self._class_names)}"
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("subtitle")

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        layout.addLayout(title_box, stretch=1)

        acc = self._metrics.get("compile_metrics") or self._metrics.get("accuracy")
        if acc is not None:
            badge = QLabel(f"accuracy {float(acc) * 100:.2f}%")
            badge.setObjectName("badge")
            layout.addWidget(badge)
        else:
            badge = QLabel("sin métricas")
            badge.setObjectName("badge_warn")
            layout.addWidget(badge)

        return frame

    def _build_left_panel(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("panel")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 10, 10, 10)

        self._class_panel = ClassListPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._class_panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll)

        return wrapper

    def _build_center_panel(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("panel")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._image_viewer = ImageViewer()
        layout.addWidget(self._image_viewer, stretch=1)

        layout.addWidget(self._build_result_panel())
        return wrapper

    def _build_result_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("result")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        real_row = QHBoxLayout()
        real_row.setSpacing(8)
        real_lbl = QLabel("Real:")
        real_lbl.setObjectName("real_label")
        real_lbl.setMinimumWidth(50)
        self._real_value = QLabel("—")
        self._real_value.setObjectName("real_value")
        real_row.addWidget(real_lbl)
        real_row.addWidget(self._real_value, stretch=1)
        layout.addLayout(real_row)

        pred_row = QHBoxLayout()
        pred_row.setSpacing(8)
        pred_lbl = QLabel("Pred:")
        pred_lbl.setObjectName("pred_label")
        pred_lbl.setMinimumWidth(50)
        self._pred_value = QLabel("—")
        self._pred_value.setObjectName("pred_value")
        self._match_label = QLabel("")
        self._match_label.setMinimumWidth(110)
        pred_row.addWidget(pred_lbl)
        pred_row.addWidget(self._pred_value, stretch=1)
        pred_row.addWidget(self._match_label)
        layout.addLayout(pred_row)

        self._prob_panel = ProbBarPanel()
        layout.addWidget(self._prob_panel)

        return frame

    def _build_right_panel(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setObjectName("panel")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        section = QLabel("ORIGEN")
        section.setObjectName("section")
        layout.addWidget(section)

        radios = QHBoxLayout()
        radios.setSpacing(14)
        self._radio_test = QRadioButton("Test")
        self._radio_dataset = QRadioButton("Dataset")
        self._radio_test.setChecked(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self._radio_test)
        group.addButton(self._radio_dataset)
        self._radio_test.toggled.connect(self._on_source_changed)
        self._radio_dataset.toggled.connect(self._on_source_changed)
        radios.addWidget(self._radio_test)
        radios.addWidget(self._radio_dataset)
        radios.addStretch(1)
        layout.addLayout(radios)

        self._btn_load_folder = QPushButton("📁 Cargar carpeta test")
        self._btn_load_folder.setObjectName("btn_load_folder")
        self._btn_load_folder.setToolTip(
            "Tomar una carpeta externa (ej. prerender/) y mover sus imagenes "
            "redimensionadas a 32x32 a test/."
        )
        self._btn_load_folder.clicked.connect(self._open_load_folder_dialog)
        layout.addWidget(self._btn_load_folder)

        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list_widget, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_prev = QPushButton("◀ Ant")
        self._btn_random = QPushButton("🎲 Aleatoria")
        self._btn_next = QPushButton("Sig ▶")
        self._btn_prev.clicked.connect(self._select_prev)
        self._btn_random.clicked.connect(self._select_random)
        self._btn_next.clicked.connect(self._select_next)
        btn_row.addWidget(self._btn_prev)
        btn_row.addWidget(self._btn_random, stretch=1)
        btn_row.addWidget(self._btn_next)
        layout.addLayout(btn_row)

        self._count_label = QLabel("")
        self._count_label.setObjectName("class_count")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._count_label)

        return wrapper

    def _populate_class_panel(self) -> None:
        self._class_panel.set_classes(self._class_names)
        counts: Dict[str, int] = {}
        for cls in self._class_names:
            counts[cls] = len(self._dataset_images.get(cls, []))
        self._class_panel.set_counts(counts)

    def _populate_image_list(self) -> None:
        self._list_widget.clear()
        if self._current_source == "test":
            paths = list(self._test_images)
            self._count_label.setText(f"{len(paths)} imágenes en test/")
        else:
            paths = []
            for cls in self._class_names:
                paths.extend(self._dataset_images.get(cls, []))
            self._count_label.setText(f"{len(paths)} imágenes en dataset/")

        for path in paths:
            label = self._format_path_label(path)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self._list_widget.addItem(item)

        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)
        else:
            self._reset_result()

    def _format_path_label(self, path: str) -> str:
        if self._current_source == "test":
            return os.path.basename(path)
        parent = os.path.basename(os.path.dirname(path))
        return f"{parent}/{os.path.basename(path)}"

    def _on_source_changed(self, checked: bool) -> None:
        if not checked:
            return
        if self._radio_test.isChecked():
            self._current_source = "test"
        else:
            self._current_source = "dataset"
        self._populate_image_list()

    def _on_row_changed(self, row: int) -> None:
        if row < 0:
            self._reset_result()
            self._image_viewer.clear()
            return
        item = self._list_widget.item(row)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self._image_viewer.set_image(path)
        self._predict(path)

    def _reset_result(self) -> None:
        self._real_value.setText("—")
        self._pred_value.setText("—")
        self._match_label.setText("")
        self._match_label.setObjectName("")
        self._prob_panel.set_predictions([])

    def _predict(self, path: str) -> None:
        if self._current_worker is not None and self._current_worker.isRunning():
            self._current_worker.quit()
            self._current_worker.wait(200)
        self._current_worker = PredictWorker(
            model=self._model,
            class_names=self._class_names,
            image_path=path,
            image_size=self._image_size,
            top_k=3,
        )
        self._current_worker.finished.connect(self._on_predict_finished)
        self._current_worker.error.connect(self._on_predict_error)
        self.statusBar().showMessage(f"Prediciendo {os.path.basename(path)}…")
        self._current_worker.start()

    def _on_predict_finished(
        self, path: str, predictions: List[Tuple[str, float]], dt_ms: float
    ) -> None:
        current = self._list_widget.currentItem()
        if current is None or current.data(Qt.ItemDataRole.UserRole) != path:
            return
        self._prob_panel.set_predictions(predictions)
        if predictions:
            top_name, top_prob = predictions[0]
            self._pred_value.setText(f"{top_name.capitalize()}  ({top_prob:.4f})")
        else:
            self._pred_value.setText("—")

        true_cls = _extract_true_class(path, self._class_names)
        if true_cls is None:
            self._real_value.setText("(desconocida)")
            self._match_label.setText("")
            self._match_label.setObjectName("")
        else:
            self._real_value.setText(true_cls.capitalize())
            top_name = predictions[0][0] if predictions else ""
            if top_name == true_cls:
                self._match_label.setText("✅ Correcto")
                self._match_label.setObjectName("ok")
            else:
                self._match_label.setText("❌ Incorrecto")
                self._match_label.setObjectName("err")
            self._match_label.setStyleSheet("")
            self._match_label.setStyleSheet(
                "color: #9ece6a; font-size: 11pt; font-weight: 600;"
                if top_name == true_cls
                else "color: #f7768e; font-size: 11pt; font-weight: 600;"
            )

        size = self._image_size
        self.statusBar().showMessage(
            f"✓ {os.path.basename(path)} · {dt_ms:.0f} ms · input {size[0]}×{size[1]}"
        )
        if self._logger is not None:
            self._logger.info(
                "Predicted %s -> top=%s (%.4f) in %.1f ms",
                path,
                predictions[0][0] if predictions else None,
                predictions[0][1] if predictions else 0.0,
                dt_ms,
            )

    def _on_predict_error(self, path: str, message: str) -> None:
        current = self._list_widget.currentItem()
        if current is None or current.data(Qt.ItemDataRole.UserRole) != path:
            return
        self.statusBar().showMessage(f"⚠ Error: {message}")
        QMessageBox.warning(self, "Error de predicción", message)

    def _select_prev(self) -> None:
        row = self._list_widget.currentRow()
        if row > 0:
            self._list_widget.setCurrentRow(row - 1)

    def _select_next(self) -> None:
        row = self._list_widget.currentRow()
        if 0 <= row < self._list_widget.count() - 1:
            self._list_widget.setCurrentRow(row + 1)

    def _select_random(self) -> None:
        n = self._list_widget.count()
        if n <= 0:
            return
        cur = self._list_widget.currentRow()
        if n == 1:
            self._list_widget.setCurrentRow(0)
            return
        r = cur
        while r == cur:
            r = random.randrange(n)
        self._list_widget.setCurrentRow(r)

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if not path:
            return
        self._image_viewer.set_image(path)
        self._predict(path)
        self._real_value.setText("—")
        self._pred_value.setText("—")
        self._match_label.setText("(libre)")
        self._match_label.setObjectName("")
        self._match_label.setStyleSheet("color: #565f89; font-size: 10pt;")
        self._prob_panel.set_predictions([])

    def _open_load_folder_dialog(self) -> None:
        default_dir = os.path.join(os.path.dirname(self._test_dir), "prerender")
        if not os.path.isdir(default_dir):
            default_dir = self._test_dir
        path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con imagenes (ej. prerender/)",
            default_dir,
        )
        if not path:
            return
        self._btn_load_folder.setEnabled(False)
        self.statusBar().showMessage(
            f"Cargando y redimensionando desde {path}…"
        )
        self._load_worker = LoadFolderWorker(
            source_dir=path,
            test_dir=self._test_dir,
            image_size=self._image_size,
            jpeg_quality=90,
        )
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_finished(self, result: dict) -> None:
        self._btn_load_folder.setEnabled(True)
        self._test_images = _list_images_in_dir(self._test_dir)
        if self._current_source == "test":
            self._populate_image_list()
        else:
            self._radio_test.setChecked(True)
        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)
        sub_info = ""
        if result.get("per_subfolder"):
            parts = [
                f"{sub}: {cnt}"
                for sub, cnt in sorted(result["per_subfolder"].items())
            ]
            sub_info = "\nPor subcarpeta: " + ", ".join(parts)
        QMessageBox.information(
            self,
            "Carga completa",
            (
                f"Nuevas movidas: {result['written']}\n"
                f"Ya existian:    {result['skipped']}\n"
                f"Con error:      {result['rejected']}"
                f"{sub_info}"
            ),
        )
        if self._logger is not None:
            self._logger.info(
                "Carga carpeta externa: written=%s skipped=%s rejected=%s",
                result["written"],
                result["skipped"],
                result["rejected"],
            )
        self.statusBar().showMessage(
            f"Carga completa: {result['written']} nuevas en test/"
        )

    def _on_load_error(self, message: str) -> None:
        self._btn_load_folder.setEnabled(True)
        self.statusBar().showMessage(f"Error al cargar carpeta: {message}")
        QMessageBox.critical(self, "Error al cargar carpeta", message)
