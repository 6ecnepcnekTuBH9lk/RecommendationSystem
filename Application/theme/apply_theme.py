from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor


def build_palette(is_dark: bool) -> QPalette:

    palette = QPalette()

    if is_dark:

        palette.setColor(QPalette.ColorRole.Window, QColor(70, 70, 70))  # общий фон (да)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)  # текст на фоне (да)
        palette.setColor(QPalette.ColorRole.Base,
                         QColor(120, 120, 120))  # поля ввода (да, галочки)
        palette.setColor(QPalette.ColorRole.AlternateBase,
                         QColor(100, 100, 100))  # чередующиеся строки в тч (нет)
        palette.setColor(QPalette.ColorRole.ToolTipBase,
                         QColor(100, 100, 100))  # всплывающие подсказки (нет)
        palette.setColor(QPalette.ColorRole.ToolTipText,
                         Qt.GlobalColor.white)  # текст всплывающих подсказок (нет)
        palette.setColor(QPalette.ColorRole.Text,
                         Qt.GlobalColor.white)  # основной текст внутри полей (да)
        palette.setColor(QPalette.ColorRole.Button, QColor(100, 100, 100))  # фон кнопки (нет)
        palette.setColor(QPalette.ColorRole.ButtonText,
                         Qt.GlobalColor.white)  # текст на кнопках (нет)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)  # текст ошибки (да)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(110, 110, 110))  # выделение (да)
        palette.setColor(QPalette.ColorRole.HighlightedText,
                         Qt.GlobalColor.white)  # текст при выделении (да)
        palette.setColor(QPalette.ColorRole.PlaceholderText,
                         QColor(100, 100, 100))  # цвет текста в пустом поле (нет)

    else:

        palette.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250))  # общий фон (да)
        palette.setColor(QPalette.ColorRole.WindowText,
                         Qt.GlobalColor.black)  # текст на фоне (да)
        palette.setColor(QPalette.ColorRole.Base,
                         QColor(240, 240, 240))  # поля ввода (да, галочки)
        palette.setColor(QPalette.ColorRole.AlternateBase,
                         QColor(100, 100, 100))  # чередующиеся строки в тч (нет)
        palette.setColor(QPalette.ColorRole.ToolTipBase,
                         QColor(100, 100, 100))  # всплывающие подсказки (нет)
        palette.setColor(QPalette.ColorRole.ToolTipText,
                         Qt.GlobalColor.white)  # текст всплывающих подсказок (нет)
        palette.setColor(QPalette.ColorRole.Text,
                         Qt.GlobalColor.black)  # основной текст внутри полей (да)
        palette.setColor(QPalette.ColorRole.Button, QColor(100, 100, 100))  # фон кнопки (нет)
        palette.setColor(QPalette.ColorRole.ButtonText,
                         Qt.GlobalColor.white)  # текст на кнопках (нет)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)  # текст ошибки (да)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(210, 210, 210))  # выделение (да)
        palette.setColor(QPalette.ColorRole.HighlightedText,
                         Qt.GlobalColor.black)  # текст при выделении (да)
        palette.setColor(QPalette.ColorRole.PlaceholderText,
                         QColor(100, 100, 100))  # цвет текста в пустом поле (нет)

    return palette


def build_stylesheet(is_dark: bool) -> str:

    if is_dark:

        return """
        * {
            font-family: "Verdana"; 
            font-weight: 500;
            font-size: 14px;
            letter-spacing: 0.1px;
            color: white;
        }

        QTabWidget::pane { border: 1px solid #6E6E6E; }

        QTabBar::tab { 
            background: #5F5F5F;
            padding: 7px 10px; 
            border-radius: 10px; 
        }

        QTabBar::tab:selected { background: #6E6E6E; }

        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
            background-color: #5F5F5F;
            padding: 4px 5px; 
            border-radius: 10px;
            border: 1px solid #6E6E6E;
        }
        
        QAbstractSpinBox::up-button, 
        QAbstractSpinBox::down-button {
            subcontrol-origin: border;
            width: 19px;
            border: none;
            background: transparent;
        }
        
        QAbstractSpinBox::up-button {
            subcontrol-position: top right;
            height: 16px;        
            margin-top: 1px;       
            margin-right: 4px;
        }
        
        QAbstractSpinBox::down-button {
            subcontrol-position: bottom right;
            height: 16px;
            margin-bottom: 1px;
            margin-right: 4px;
        }
        
        QAbstractSpinBox::up-arrow {
            image: url("Картинки/ТреугольникВверх.png");
            width: 14px;
            height: 14px;
        }
        
        QAbstractSpinBox::down-arrow {
            image: url("Картинки/ТреугольникВниз.png");
            width: 14px;
            height: 14px;
        }
        
        QAbstractSpinBox::up-button:hover, 
        QAbstractSpinBox::down-button:hover {
            background: transparent;
        }
        
        QAbstractSpinBox::up-button:pressed, 
        QAbstractSpinBox::down-button:pressed {
            background: transparent;
        }
        
        QListWidget::item {
            padding: 2px 2px;
        }
            
        QListWidget::item:selected {
            background-color: #7D7D7D;
            border-radius: 10px;
        }
        
        QListWidget::item:hover {
            background-color: #6E6E6E;
            border-radius: 10px;
        }
        
        QAbstractItemView {
            outline: 0;
        }
        
        QAbstractItemView::item {
            show-decoration-selected: 0;
        }

        QComboBox::drop-down {
            width: 19px;
            border: none;
            background: transparent;
        }

        QComboBox::down-arrow {
            image: url("Картинки/Вниз.png");
            width: 17px;
            height: 17px;
        }

        QPushButton {
            padding: 5px 7px; 
            border-radius: 10px;
            border: 1px solid #9B4DFF;
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #464646,
                stop:1 #5A5A5A
            );
        }

        QPushButton:hover {
            background-color: #7D7D7D;
        }

        QPushButton:pressed {
            background-color: #8C8C8C;
        }
        
        QPushButton:disabled {
            color: #A0A0A0;
            border: 1px solid #6E6E6E;
        }
        
        QPushButton:checked {
            background-color: #4A3E73;
        }
        
        QScrollBar:vertical {
            width: 10px;
            background: transparent;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background: #6E6E6E;
            min-height: 30px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #7D7D7D;
        }
        
        QScrollBar::add-line:vertical, 
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar::add-page:vertical, 
        QScrollBar::sub-page:vertical {
            background: transparent;
        }
        
        QScrollBar:horizontal {
            height: 10px;
            background: transparent;
            margin: 0px;
        }
            
        QScrollBar::handle:horizontal {
            background: #6E6E6E;
            min-width: 30px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #7D7D7D;
        }
        
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {
            background: transparent;
        }
        
        QAbstractScrollArea::corner {
            background: none;
            border: none;
        }
        
        QTableWidget {
            background-color: #464646;
            border: 1px solid #6E6E6E;
            border-radius: 10px;
            gridline-color: #6E6E6E;
        }
        
        QHeaderView::section {
            background-color: #5F5F5F;
            border: 1px solid #6E6E6E;
            padding: 5px 0px;
        }
        
        QHeaderView::section:first {
            border-top-left-radius: 10px;
        }
        QHeaderView::section:last {
            border-top-right-radius: 10px;
        }
        
        QHeaderView {
            background: transparent;
        }
        
        QTableWidget::item:selected {
            background-color: #5F5F5F;
        }

        QProgressBar {
            border: 1px solid #6E6E6E;
            text-align: center;
            color: black;
            border-radius: 7px;
            background-color: #5F5F5F;
        }
        QProgressBar::chunk {
            background-color: #5F5F5F;
            border-radius: 7px;
        }
        QMessageBox {
            margin: 0;
            padding: 0;
        }
        QMessageBox QLabel {
            padding: 3;
            margin: 0;
        }

        QFrame[frameRole="separator"] {
            background-color: #6E6E6E;
            border: none;
            min-height: 1px;
            max-height: 1px;
        }
        
        #vSeparator {
            background-color: #6E6E6E;
        }
        
        #hSeparator {
            background-color: #6E6E6E;
        }
    """

    return """
    * {
        font-family: "Verdana"; 
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.1px;
        color: black;
    }

    QTabWidget::pane { border: 1px solid #DCDCDC; }

    QTabBar::tab { 
        background: #EBEBEB;
        padding: 7px 10px; 
        border-radius: 10px; 
    }

    QTabBar::tab:selected { background: #DCDCDC; }

    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
        background-color: #EBEBEB;
        padding: 4px 5px; 
        border-radius: 10px;
        border: 1px solid #DCDCDC;
    }
    
    QAbstractSpinBox::up-button, 
    QAbstractSpinBox::down-button {
        subcontrol-origin: border;
        width: 19px;
        border: none;
        background: transparent;
    }
    
    QAbstractSpinBox::up-button {
        subcontrol-position: top right;
        height: 16px;        
        margin-top: 1px;       
        margin-right: 4px;
    }
    
    QAbstractSpinBox::down-button {
        subcontrol-position: bottom right;
        height: 16px;
        margin-bottom: 1px;
        margin-right: 4px;
    }
    
    QAbstractSpinBox::up-arrow {
        image: url("Картинки/ТреугольникВверх.png");
        width: 14px;
        height: 14px;
    }
    
    QAbstractSpinBox::down-arrow {
        image: url("Картинки/ТреугольникВниз.png");
        width: 14px;
        height: 14px;
    }
    
    QAbstractSpinBox::up-button:hover, 
    QAbstractSpinBox::down-button:hover {
        background: transparent;
    }
    
    QAbstractSpinBox::up-button:pressed, 
    QAbstractSpinBox::down-button:pressed {
        background: transparent;
    }

    QListWidget::item {
        padding: 2px 2px;
    }

    QListWidget::item:selected {
        background-color: #CDCDCD;
        border-radius: 10px;
    }

    QListWidget::item:hover {
        background-color: #DCDCDC;
        border-radius: 10px;
    }

    QAbstractItemView {
        outline: 0;
    }

    QAbstractItemView::item {
        show-decoration-selected: 0;
    }

    QComboBox::drop-down {
        width: 19px;
        border: none;
        background: transparent;
    }

    QComboBox::down-arrow {
        image: url("Картинки/Вниз.png");
        width: 17px;
        height: 17px;
    }

    QPushButton {
        padding: 5px 7px; 
        border-radius: 10px;
        border: 1px solid #9B4DFF;
        background: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 #FFFFFF,
            stop:1 #EBEBEB
        );
    }

    QPushButton:hover {
        background-color: #EBEBEB;
    }

    QPushButton:pressed {
        background-color: #DCDCDC;
    }

    QPushButton:disabled {
        color: #A0A0A0;
        border: 1px solid #DCDCDC;
    }

    QPushButton:checked {
        background-color: #E3DBFF;
    }

    QScrollBar:vertical {
        width: 10px;
        background: transparent;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background: #DCDCDC;
        min-height: 30px;
        border-radius: 5px;
    }

    QScrollBar::handle:vertical:hover {
        background: #CDCDCD;
    }

    QScrollBar::add-line:vertical, 
    QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar::add-page:vertical, 
    QScrollBar::sub-page:vertical {
        background: transparent;
    }

    QScrollBar:horizontal {
        height: 10px;
        background: transparent;
        margin: 0px;
    }

    QScrollBar::handle:horizontal {
        background: #DCDCDC;
        min-width: 30px;
        border-radius: 5px;
    }

    QScrollBar::handle:horizontal:hover {
        background: #CDCDCD;
    }

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        width: 0px;
    }

    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {
        background: transparent;
    }

    QAbstractScrollArea::corner {
        background: none;
        border: none;
    }
    
    QTableWidget {
        background-color: #FAFAFA;
        border: 1px solid #DCDCDC;
        border-radius: 10px;
        gridline-color: #DCDCDC;
    }
    
    QHeaderView::section {
        background-color: #EBEBEB;
        border: 1px solid #DCDCDC;
        padding: 5px 0px;
    }
    
    QHeaderView::section:first {
        border-top-left-radius: 10px;
    }
    QHeaderView::section:last {
        border-top-right-radius: 10px;
    }
    
    QHeaderView {
        background: transparent;
    }
    
    QTableWidget::item:selected {
        background-color: #EBEBEB;
    }

    QProgressBar {
        border: 1px solid #D7D7D7;
        text-align: center;
        color: black;
        border-radius: 7px;
        background-color: #F0F0F0;
    }
    QProgressBar::chunk {
        background-color: #969696;
        border-radius: 7px;
    }
    QMessageBox {
        margin: 0;
        padding: 0;
    }
    QMessageBox QLabel {
        padding: 3px;
        margin: 0;
    }

    QFrame[frameRole="separator"] {
        background-color: #DCDCDC;
        border: none;
        min-height: 1px;
        max-height: 1px;
    }

    #vSeparator {
        background-color: #DCDCDC;
    }
    
    #hSeparator {
        background-color: #DCDCDC;
    }
"""
