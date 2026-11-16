"""Independent keyboard-accessible table help using the existing Material palette."""
import os

from PyQt6.QtCore import QEvent, QRect, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QStyle, QStyleOption, QStyleOptionToolButton, QToolButton, QVBoxLayout

from Application.statistics_table_explanations import EXPLANATION_ICON, table_explanation
from Application.theme.layout_metrics import COMPACT_SPACING, CONTROL_SPACING


class _ExplanationToggle(QToolButton):
    """Paint the bundled icon, an explicit 8 px gap, and a native disclosure arrow."""
    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(size.width() + 20)
        return size

    def minimumSizeHint(self):
        return self.sizeHint()

    def icon_rect(self):
        size = self.iconSize()
        return QRect(CONTROL_SPACING, (self.height() - size.height()) // 2, size.width(), size.height())

    def text_rect(self):
        left = self.icon_rect().right() + 1 + CONTROL_SPACING
        return QRect(left, 0, max(0, self.width() - left - 24), self.height())

    def paintEvent(self, event):
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        option.icon, option.text = QIcon(), ""
        painter = QPainter(self)
        self.style().drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option, painter, self)
        self.icon().paint(painter, self.icon_rect())
        painter.setFont(self.font())
        painter.setPen(self.palette().buttonText().color())
        painter.drawText(self.text_rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        arrow_option = QStyleOption()
        arrow_option.initFrom(self)
        left = min(self.width() - 18, self.text_rect().left() + self.fontMetrics().horizontalAdvance(self.text()) + 12)
        arrow_option.rect = QRect(left, (self.height() - 10) // 2, 10, 10)
        arrow = QStyle.PrimitiveElement.PE_IndicatorArrowUp if self.isChecked() else QStyle.PrimitiveElement.PE_IndicatorArrowDown
        self.style().drawPrimitive(arrow, arrow_option, painter, self)


class StatisticsTableExplanation(QFrame):
    def __init__(self, table_id, parent=None):
        super().__init__(parent)
        explanation = table_explanation(table_id)
        if explanation is None:
            raise ValueError("Unknown statistics table explanation")
        self.table_id = table_id
        self.setObjectName("statisticsTableExplanation")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(CONTROL_SPACING, COMPACT_SPACING, CONTROL_SPACING, COMPACT_SPACING)
        layout.setSpacing(COMPACT_SPACING)
        self.toggle = _ExplanationToggle(self)
        self.toggle.setObjectName("statisticsExplanationToggle")
        self.toggle.setText("Как читать таблицу")
        self.toggle.setIcon(QIcon(str(EXPLANATION_ICON)))
        self.toggle.setIconSize(QSize(20, 20))
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content = QLabel(explanation.html(), self)
        self.content.setObjectName("statisticsExplanationText")
        self.content.setTextFormat(Qt.TextFormat.RichText)
        self.content.setWordWrap(True)
        self.content.setMinimumWidth(0)
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.content.setSizePolicy(policy)
        self.content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self._set_expanded)
        self._theme_timer = QTimer(self)
        self._theme_timer.setSingleShot(True)
        self._theme_timer.timeout.connect(self.apply_theme)
        self._set_expanded(False)
        self.apply_theme()

    def _set_expanded(self, expanded):
        self.content.setVisible(expanded)
        self.toggle.update()
        state = "развернуто" if expanded else "свернуто"
        self.toggle.setAccessibleName("Как читать таблицу — " + state)
        self.toggle.setAccessibleDescription("Пояснение к таблице. Пробел: раскрыть или свернуть.")
        self.content.updateGeometry()
        self.updateGeometry()

    def apply_theme(self):
        surface = QColor(os.environ.get("QTMATERIAL_SECONDARYCOLOR", "#f5f5f5"))
        foreground = QColor(os.environ.get("QTMATERIAL_SECONDARYTEXTCOLOR", "#555555"))
        accent = QColor(os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#a65cf2"))
        background = surface.lighter(118) if surface.lightness() < 128 else surface.darker(104)
        style = (
            f"QFrame#statisticsTableExplanation {{background:{background.name()};border:0;border-radius:8px;}}"
            f"QToolButton#statisticsExplanationToggle {{background:transparent;color:{foreground.name()};"
            "border:1px solid transparent;border-radius:4px;padding:4px;text-align:left;}"
            f"QToolButton#statisticsExplanationToggle:focus {{border-color:{accent.name()};}}"
            f"QLabel#statisticsExplanationText {{background:transparent;color:{foreground.name()};padding:4px;}}"
        )
        if self.styleSheet() != style:
            self.setStyleSheet(style)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange) and hasattr(self, "_theme_timer"):
            self._theme_timer.start(0)
