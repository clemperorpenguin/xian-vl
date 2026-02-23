from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal, QRect, QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QPaintEvent, QKeyEvent, QGuiApplication, QPainter, QPen, QColor

class RegionSelector(QWidget):
    """Widget for selecting screen regions"""

    region_selected = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100);")

        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.selecting = False

        # Make fullscreen
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.pos()
            self.selecting = True

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selecting:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False

            # Calculate selection rectangle
            rect = QRect(self.start_pos, self.current_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.region_selected.emit(rect)

            self.close()

    def paintEvent(self, event: QPaintEvent):
        if self.selecting:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(QRect(self.start_pos, self.current_pos))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
