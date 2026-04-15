from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import pyqtSignal, QRect, QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QPaintEvent, QKeyEvent, QGuiApplication, QPainter, QPen, QColor, QFont
import logging

from . import constants

logger = logging.getLogger(__name__)


class RegionSelector(QWidget):
    """Widget for selecting screen regions with enhanced UX."""

    region_selected = pyqtSignal(QRect, str)

    def __init__(self, parent=None, preset_name: str = ""):
        super().__init__(parent)
        logger.info("RegionSelector.__init__ start")
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100);")

        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.selecting = False
        self.preset_name = preset_name

        # Make fullscreen using parent's screen if available
        if parent and parent.window().windowHandle():
            screen = parent.window().windowHandle().screen().geometry()
        else:
            screen = QGuiApplication.primaryScreen().geometry()
        
        logger.info(f"RegionSelector geometry: {screen}")
        self.setGeometry(screen)
        
        # Show and raise to top
        self.show()
        self.raise_()
        self.activateWindow()

        # Instructions label
        self.instructions = QLabel("Click and drag to select a region. Press ESC to cancel.", self)
        self.instructions.setStyleSheet("""
            color: white;
            background-color: rgba(0, 0, 0, 180);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 14px;
        """)
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions.move(
            (screen.width() - self.instructions.sizeHint().width()) // 2,
            screen.height() - 100
        )
        self.instructions.show()
        logger.info("RegionSelector.__init__ done")

    def mousePressEvent(self, event: QMouseEvent):
        logger.info(f"RegionSelector.mousePressEvent: {event.button()}")
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.pos()
            self.selecting = True
            self.instructions.hide()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selecting:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        logger.info(f"RegionSelector.mouseReleaseEvent: {event.button()}, selecting={self.selecting}")
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False

            rect = QRect(self.start_pos, self.current_pos).normalized()
            logger.info(f"Region selected: {rect}")
            
            if rect.width() > constants.REGION_MIN_SIZE and rect.height() > constants.REGION_MIN_SIZE:
                name = self.preset_name or f"Region {rect.x()}x{rect.y()}"
                logger.info(f"Emitting region_selected: {rect}, {name}")
                self.region_selected.emit(rect, name)
            else:
                logger.info("Region too small")
            
            self.close()

    def paintEvent(self, event: QPaintEvent):
        if self.selecting:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = QRect(self.start_pos, self.current_pos).normalized()

            # Semi-transparent fill
            painter.fillRect(rect, QColor(255, 100, 100, 50))

            # Border
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            painter.drawRect(rect)

            # Draw dimensions label
            width = rect.width()
            height = rect.height()
            label = f"{width} x {height}"

            font = QFont("monospace", 10)
            painter.setFont(font)
            font_metrics = painter.fontMetrics()
            label_width = font_metrics.horizontalAdvance(label)
            label_height = font_metrics.height()

            label_x = rect.x() + 5
            label_y = rect.y() + 5

            # Draw label background
            label_rect = QRect(label_x - 4, label_y - 2, label_width + 8, label_height + 4)
            painter.fillRect(label_rect, QColor(0, 0, 0, 180))

            # Draw label text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_x, label_y + label_height - 2, label)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
