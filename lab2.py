import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QColor, QBrush

class TridiagonalSolver:
    @staticmethod
    def solve(a, b, c, d):
        n = len(b)
        alpha = np.zeros(n)
        beta = np.zeros(n)

        if abs(b[0]) < 1e-12:
            raise ValueError("Нулевой элемент b[0]")

        alpha[0] = -c[0] / b[0]
        beta[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] + a[i] * alpha[i-1]
            if abs(denom) < 1e-12:
                raise ValueError(f"Деление на ноль на шаге i={i}")
            if i < n-1:
                alpha[i] = -c[i] / denom
            beta[i] = (d[i] - a[i] * beta[i-1]) / denom

        x = np.zeros(n)
        x[n-1] = beta[n-1]
        for i in range(n-2, -1, -1):
            x[i] = alpha[i] * x[i+1] + beta[i]
        return x

    @staticmethod
    def determinant(a, b, c):
        n = len(b)
        if n == 1:
            return b[0]
        det_prev2 = 1.0
        det_prev1 = b[0]
        for i in range(2, n+1):
            det = b[i-1] * det_prev1 - a[i-1] * c[i-2] * det_prev2
            det_prev2, det_prev1 = det_prev1, det
        return det_prev1

    @staticmethod
    def inverse(a, b, c):
        n = len(b)
        inv = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            inv[:, j] = TridiagonalSolver.solve(a, b, c, e)
        return inv


class TridiagonalInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.n = 5
        self.inputs = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Ввод трёхдиагональной СЛАУ 5x5")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QLineEdit { padding: 5px; border: 1px solid #ccc; border-radius: 3px; }
            QLineEdit:focus { border: 2px solid #4CAF50; }
            QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px 15px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:reject { background-color: #f44336; }
            QLabel { font-size: 12px; }
        """)

        layout = QVBoxLayout()
        
        title = QLabel("Введите коэффициенты трёхдиагональной системы 5×5")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)
        
        info = QLabel("Формат: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]")
        info.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info)

        grid = QGridLayout()
        grid.setSpacing(10)
        
        headers = ["i", "a[i]", "b[i]", "c[i]", "d[i]"]
        for j, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 13px;")
            grid.addWidget(lbl, 0, j)

        validator = QDoubleValidator()

        for i in range(self.n):
            # Номер уравнения
            num_label = QLabel(str(i+1))
            num_label.setAlignment(Qt.AlignCenter)
            num_label.setStyleSheet("font-weight: bold; color: #555;")
            grid.addWidget(num_label, i+1, 0)

            # a[i] (нижняя диагональ)
            a_edit = QLineEdit()
            a_edit.setValidator(validator)
            a_edit.setMaximumWidth(80)
            a_edit.setPlaceholderText("0" if i == 0 else f"a{i+1}")
            if i == 0:
                a_edit.setText("0")
                a_edit.setEnabled(False)
                a_edit.setStyleSheet("background-color: #e0e0e0;")
            grid.addWidget(a_edit, i+1, 1)

            # b[i] (главная диагональ)
            b_edit = QLineEdit()
            b_edit.setValidator(validator)
            b_edit.setMaximumWidth(80)
            b_edit.setPlaceholderText(f"b{i+1}")
            grid.addWidget(b_edit, i+1, 2)

            # c[i] (верхняя диагональ)
            c_edit = QLineEdit()
            c_edit.setValidator(validator)
            c_edit.setMaximumWidth(80)
            c_edit.setPlaceholderText("0" if i == self.n-1 else f"c{i+1}")
            if i == self.n-1:
                c_edit.setText("0")
                c_edit.setEnabled(False)
                c_edit.setStyleSheet("background-color: #e0e0e0;")
            grid.addWidget(c_edit, i+1, 3)

            # d[i] (правая часть)
            d_edit = QLineEdit()
            d_edit.setValidator(validator)
            d_edit.setMaximumWidth(80)
            d_edit.setPlaceholderText(f"d{i+1}")
            grid.addWidget(d_edit, i+1, 4)

            self.inputs[i] = {'a': a_edit, 'b': b_edit, 'c': c_edit, 'd': d_edit}

        layout.addLayout(grid)
        
        # Пример для вашей системы
        example_frame = QFrame()
        example_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 5px;
                margin-top: 10px;
                padding: 5px;
            }
            QLabel {
                color: #1565c0;
            }
        """)
        example_layout = QVBoxLayout(example_frame)
        example_label = QLabel("Пример для вашей системы:")
        example_label.setFont(QFont("Arial", 10, QFont.Bold))
        example_layout.addWidget(example_label)
        example_text = QLabel(
            "Уравнение 1: -11*x1 -9*x2 = -122\n"
            "Уравнение 2: 5*x1 -15*x2 -2*x3 = -48\n"
            "Уравнение 3: -8*x2 +11*x3 -3*x4 = -14\n"
            "Уравнение 4: 6*x3 -15*x4 +4*x5 = -50\n"
            "Уравнение 5: 3*x4 +6*x5 = 42\n\n"
            "Ввод:\n"
            "i=1: a=0, b=-11, c=-9, d=-122\n"
            "i=2: a=5, b=-15, c=-2, d=-48\n"
            "i=3: a=-8, b=11, c=-3, d=-14\n"
            "i=4: a=6, b=-15, c=4, d=-50\n"
            "i=5: a=3, b=6, c=0, d=42"
        )
        example_text.setWordWrap(True)
        example_layout.addWidget(example_text)
        layout.addWidget(example_frame)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Решить систему")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.resize(750, 550)

    def get_system(self):
        try:
            n = self.n
            a = np.zeros(n)
            b = np.zeros(n)
            c = np.zeros(n)
            d = np.zeros(n)

            for i in range(n):
                if i > 0:
                    a_val = self.inputs[i]['a'].text()
                    a[i] = float(a_val) if a_val.strip() else 0.0

                b_val = self.inputs[i]['b'].text()
                if not b_val.strip():
                    QMessageBox.warning(self, "Ошибка", f"Не заполнено b[{i+1}]")
                    return None, None, None, None
                b[i] = float(b_val)

                if i < n-1:
                    c_val = self.inputs[i]['c'].text()
                    c[i] = float(c_val) if c_val.strip() else 0.0

                d_val = self.inputs[i]['d'].text()
                if not d_val.strip():
                    QMessageBox.warning(self, "Ошибка", f"Не заполнено d[{i+1}]")
                    return None, None, None, None
                d[i] = float(d_val)

            return a, b, c, d
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Неверный формат числа:\n{str(e)}")
            return None, None, None, None


class ResultsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 15px;
                margin: 2px;
                border-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
        
        # Вкладка с решением
        self.solution_tab = QWidget()
        self.setup_solution_tab()
        self.tab_widget.addTab(self.solution_tab, "Решение системы")
        
        # Вкладка с матрицей
        self.matrix_tab = QWidget()
        self.setup_matrix_tab()
        self.tab_widget.addTab(self.matrix_tab, "Матрица системы")
        
        # Вкладка с проверкой
        self.verification_tab = QWidget()
        self.setup_verification_tab()
        self.tab_widget.addTab(self.verification_tab, "Проверка")
        
        # Вкладка с определителем
        self.det_tab = QWidget()
        self.setup_det_tab()
        self.tab_widget.addTab(self.det_tab, "Определитель")
        
        # Вкладка с обратной матрицей
        self.inv_tab = QWidget()
        self.setup_inv_tab()
        self.tab_widget.addTab(self.inv_tab, "Обратная матрица")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def setup_solution_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Решение системы методом прогонки")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin: 10px;")
        layout.addWidget(title)
        
        self.solution_table = QTableWidget()
        self.solution_table.setColumnCount(2)
        self.solution_table.setHorizontalHeaderLabels(["Переменная", "Значение"])
        self.solution_table.horizontalHeader().setStretchLastSection(True)
        self.solution_table.setAlternatingRowColors(True)
        self.solution_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.solution_table)
        
        self.solution_bars = QWidget()
        self.solution_bars_layout = QVBoxLayout()
        self.solution_bars.setLayout(self.solution_bars_layout)
        layout.addWidget(QLabel("Графическое представление:"))
        layout.addWidget(self.solution_bars)
        
        self.solution_tab.setLayout(layout)
    
    def setup_matrix_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Трёхдиагональная матрица системы")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff9800; margin: 10px;")
        layout.addWidget(title)
        
        self.matrix_table = QTableWidget()
        self.matrix_table.setAlternatingRowColors(True)
        self.matrix_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                text-align: center;
            }
            QHeaderView::section {
                background-color: #ff9800;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.matrix_table)
        
        self.matrix_tab.setLayout(layout)
    
    def setup_verification_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Проверка решения (невязка)")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2196F3; margin: 10px;")
        layout.addWidget(title)
        
        self.residual_table = QTableWidget()
        self.residual_table.setColumnCount(2)
        self.residual_table.setHorizontalHeaderLabels(["Уравнение", "Невязка"])
        self.residual_table.horizontalHeader().setStretchLastSection(True)
        self.residual_table.setAlternatingRowColors(True)
        self.residual_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.residual_table)
        
        self.residual_status = QLabel()
        self.residual_status.setAlignment(Qt.AlignCenter)
        self.residual_status.setStyleSheet("padding: 10px; font-size: 12px; border-radius: 5px;")
        layout.addWidget(self.residual_status)
        
        self.verification_tab.setLayout(layout)
    
    def setup_det_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Определитель трёхдиагональной матрицы")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #9C27B0; margin: 10px;")
        layout.addWidget(title)
        
        self.det_value = QLabel()
        self.det_value.setAlignment(Qt.AlignCenter)
        self.det_value.setFont(QFont("Arial", 24, QFont.Bold))
        self.det_value.setStyleSheet("color: #9C27B0; padding: 20px; background-color: #f3e5f5; border-radius: 10px;")
        layout.addWidget(self.det_value)
        
        self.det_warning = QLabel()
        self.det_warning.setAlignment(Qt.AlignCenter)
        self.det_warning.setWordWrap(True)
        layout.addWidget(self.det_warning)
        
        self.det_formula = QLabel()
        self.det_formula.setAlignment(Qt.AlignCenter)
        self.det_formula.setWordWrap(True)
        self.det_formula.setStyleSheet("color: #666; margin-top: 20px;")
        layout.addWidget(self.det_formula)
        
        layout.addStretch()
        self.det_tab.setLayout(layout)
    
    def setup_inv_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Обратная матрица A^{-1}")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #f44336; margin: 10px;")
        layout.addWidget(title)
        
        self.inv_table = QTableWidget()
        self.inv_table.setAlternatingRowColors(True)
        self.inv_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f44336;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.inv_table)
        
        self.verification_label = QLabel()
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setStyleSheet("padding: 10px; background-color: #ffebee; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(self.verification_label)
        
        self.inv_tab.setLayout(layout)
    
    def display_results(self, a, b, c, d, x, residual, det, inv_A):
        n = len(x)
        
        # Отображение решения
        self.solution_table.setRowCount(n)
        max_val = max(abs(x)) if len(x) > 0 else 1
        
        for i, val in enumerate(x):
            var_item = QTableWidgetItem(f"x{i+1}")
            var_item.setFont(QFont("Arial", 12, QFont.Bold))
            val_item = QTableWidgetItem(f"{val:.8f}")
            val_item.setFont(QFont("Courier New", 12))
            
            if val > 0:
                val_item.setForeground(QBrush(QColor(76, 175, 80)))
            elif val < 0:
                val_item.setForeground(QBrush(QColor(244, 67, 54)))
            
            self.solution_table.setItem(i, 0, var_item)
            self.solution_table.setItem(i, 1, val_item)
        
        self.solution_table.resizeColumnsToContents()
        
        # Графические бары
        self.clear_bars()
        for i, val in enumerate(x):
            bar_frame = QFrame()
            bar_layout = QHBoxLayout()
            label = QLabel(f"x{i+1}")
            label.setMinimumWidth(40)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(abs(val) / max_val * 100))
            bar.setFormat(f"{val:.4f}")
            
            if val > 0:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            elif val < 0:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #9e9e9e; }")
            
            bar_layout.addWidget(label)
            bar_layout.addWidget(bar)
            bar_frame.setLayout(bar_layout)
            self.solution_bars_layout.addWidget(bar_frame)
        
        # Отображение полной матрицы
        self.matrix_table.setRowCount(n)
        self.matrix_table.setColumnCount(n)
        self.matrix_table.setHorizontalHeaderLabels([f"x{j+1}" for j in range(n)])
        
        for i in range(n):
            for j in range(n):
                value = 0
                if j == i-1:
                    value = a[i]
                elif j == i:
                    value = b[i]
                elif j == i+1:
                    value = c[i]
                
                item = QTableWidgetItem(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                
                if value != 0:
                    item.setForeground(QBrush(QColor(76, 175, 80)))
                    item.setFont(QFont("Courier New", 10, QFont.Bold))
                
                self.matrix_table.setItem(i, j, item)
        
        self.matrix_table.resizeColumnsToContents()
        
        # Отображение невязки
        self.residual_table.setRowCount(len(residual))
        for i, res in enumerate(residual):
            eq_item = QTableWidgetItem(f"Уравнение {i+1}")
            res_item = QTableWidgetItem(f"{res:.2e}")
            
            if abs(res) < 1e-10:
                res_item.setForeground(QBrush(QColor(76, 175, 80)))
                res_item.setText("≈ 0")
            elif abs(res) < 1e-6:
                res_item.setForeground(QBrush(QColor(255, 152, 0)))
            else:
                res_item.setForeground(QBrush(QColor(244, 67, 54)))
            
            self.residual_table.setItem(i, 0, eq_item)
            self.residual_table.setItem(i, 1, res_item)
        
        self.residual_table.resizeColumnsToContents()
        
        max_residual = np.max(np.abs(residual))
        if max_residual < 1e-10:
            self.residual_status.setText("Отлично! Невязка в пределах машинной точности")
            self.residual_status.setStyleSheet("background-color: #c8e6c9; color: #2e7d32; padding: 10px; border-radius: 5px;")
        elif max_residual < 1e-6:
            self.residual_status.setText("Хорошо, но есть небольшая погрешность")
            self.residual_status.setStyleSheet("background-color: #fff3e0; color: #e65100; padding: 10px; border-radius: 5px;")
        else:
            self.residual_status.setText(f"Большая погрешность! Максимальная невязка: {max_residual:.2e}")
            self.residual_status.setStyleSheet("background-color: #ffebee; color: #c62828; padding: 10px; border-radius: 5px;")
        
        # Отображение определителя
        self.det_value.setText(f"det(A) = {det:.10f}")
        if abs(det) < 1e-10:
            self.det_warning.setText("Внимание! Определитель близок к нулю, матрица вырождена")
            self.det_warning.setStyleSheet("color: #f44336; font-weight: bold;")
        else:
            self.det_warning.setText("Матрица невырождена, решение существует и единственно")
            self.det_warning.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        self.det_formula.setText("Определитель вычислен по рекуррентной формуле:\ndet[1] = b[1]\ndet[i] = b[i]*det[i-1] - a[i]*c[i-1]*det[i-2]")
        
        # Отображение обратной матрицы
        n = len(inv_A)
        self.inv_table.setRowCount(n)
        self.inv_table.setColumnCount(n)
        self.inv_table.setHorizontalHeaderLabels([f"x{j+1}" for j in range(n)])
        
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{inv_A[i, j]:.8f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.inv_table.setItem(i, j, item)
        
        self.inv_table.resizeColumnsToContents()
        self.verification_label.setText("Обратная матрица вычислена методом прогонки для каждого столбца единичной матрицы")
    
    def clear_bars(self):
        while self.solution_bars_layout.count():
            child = self.solution_bars_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Решение трёхдиагональных СЛАУ методом прогонки")
        self.setGeometry(100, 100, 1100, 850)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        
        title = QLabel("Решение трёхдиагональных систем линейных уравнений")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                color: #1565c0;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_label = QLabel(
            "Метод решения: метод прогонки (алгоритм Томаса)\n"
            "Структура матрицы: трёхдиагональная\n"
            "Размерность: 5 x 5\n"
            "Преимущества: O(n) операций, высокая скорость"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_frame)
        
        self.input_button = QPushButton("Ввести трёхдиагональную систему")
        self.input_button.clicked.connect(self.input_system)
        layout.addWidget(self.input_button)
        
        self.results_widget = ResultsWidget()
        layout.addWidget(self.results_widget)
        
        clear_button = QPushButton("Очистить все")
        clear_button.setStyleSheet("background-color: #f44336;")
        clear_button.clicked.connect(self.clear_results)
        layout.addWidget(clear_button)
        
        self.statusBar().showMessage("Готов к работе")
    
    def input_system(self):
        dialog = TridiagonalInputDialog()
        if dialog.exec_():
            a, b, c, d = dialog.get_system()
            if a is not None:
                self.solve_and_display(a, b, c, d)
    
    def solve_and_display(self, a, b, c, d):
        try:
            # Решение системы
            x = TridiagonalSolver.solve(a, b, c, d)
            
            # Определитель
            det = TridiagonalSolver.determinant(a, b, c)
            
            # Обратная матрица
            inv_A = TridiagonalSolver.inverse(a, b, c)
            
            # Формирование полной матрицы для проверки
            n = len(b)
            A_full = np.zeros((n, n))
            for i in range(n):
                if i > 0:
                    A_full[i, i-1] = a[i]
                A_full[i, i] = b[i]
                if i < n-1:
                    A_full[i, i+1] = c[i]
            
            # Невязка
            residual = np.dot(A_full, x) - d
            
            # Отображение результатов
            self.results_widget.display_results(a, b, c, d, x, residual, det, inv_A)
            
            self.statusBar().showMessage("Решение успешно найдено")
            
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"{str(e)}")
            self.statusBar().showMessage("Ошибка при решении")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Непредвиденная ошибка:\n{str(e)}")
            self.statusBar().showMessage("Ошибка при решении")
    
    def clear_results(self):
        self.results_widget.solution_table.setRowCount(0)
        self.results_widget.matrix_table.setRowCount(0)
        self.results_widget.residual_table.setRowCount(0)
        self.results_widget.inv_table.setRowCount(0)
        self.results_widget.det_value.setText("—")
        self.results_widget.det_warning.setText("")
        self.results_widget.residual_status.setText("")
        self.results_widget.verification_label.setText("")
        self.results_widget.det_formula.setText("")
        self.results_widget.clear_bars()
        self.statusBar().showMessage("Результаты очищены")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()