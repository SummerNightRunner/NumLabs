import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QColor, QBrush


class IterativeSolvers:
    @staticmethod
    def is_diagonally_dominant(A):
        n = len(A)
        for i in range(n):
            diag = abs(A[i, i])
            off_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
            if diag <= off_diag:
                return False
        return True

    @staticmethod
    def has_zero_on_diagonal(A):
        n = len(A)
        for i in range(n):
            if abs(A[i, i]) < 1e-12:
                return True
        return False

    @staticmethod
    def simple_iteration(A, b, tolerance, max_iterations=10000):
        n = len(A)
        x = np.zeros(n)
        errors = []
        residual_norms = []

        for iteration in range(max_iterations):
            x_new = np.zeros(n)

            for i in range(n):
                s = 0.0
                for j in range(n):
                    if j != i:
                        s += A[i, j] * x[j]

                x_new[i] = (b[i] - s) / A[i, i]

            error = np.linalg.norm(x_new - x, np.inf)
            residual = np.dot(A, x_new) - b
            residual_norm = np.linalg.norm(residual, np.inf)

            errors.append(error)
            residual_norms.append(residual_norm)

            if error < tolerance:
                return x_new, iteration + 1, errors, residual_norms, True

            x = x_new

        return x, max_iterations, errors, residual_norms, False

    @staticmethod
    def seidel_method(A, b, tolerance, max_iterations=10000):
        n = len(A)
        x = np.zeros(n)
        errors = []
        residual_norms = []

        for iteration in range(max_iterations):
            x_new = x.copy()

            for i in range(n):
                sum1 = 0.0
                for j in range(i):
                    sum1 += A[i, j] * x_new[j]

                sum2 = 0.0
                for j in range(i + 1, n):
                    sum2 += A[i, j] * x[j]

                x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

            error = np.linalg.norm(x_new - x, np.inf)
            residual = np.dot(A, x_new) - b
            residual_norm = np.linalg.norm(residual, np.inf)

            errors.append(error)
            residual_norms.append(residual_norm)

            if error < tolerance:
                return x_new, iteration + 1, errors, residual_norms, True

            x = x_new

        return x, max_iterations, errors, residual_norms, False

    @staticmethod
    def residual(A, b, x):
        return np.dot(A, x) - b


class MatrixInputDialog(QDialog):
    def __init__(self, n=4):
        super().__init__()
        self.n = n
        self.matrix_inputs = []
        self.vector_inputs = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Ввод СЛАУ {self.n}x{self.n}")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 12px;
                background-color: white;
                color: black;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:reject {
                background-color: #f44336;
            }
            QPushButton:reject:hover {
                background-color: #da190b;
            }
            QLabel {
                font-size: 12px;
                color: black;
            }
        """)

        layout = QVBoxLayout()

        title = QLabel("Введите коэффициенты системы уравнений:")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        for j in range(self.n):
            label = QLabel(f"x{j+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 13px;")
            grid_layout.addWidget(label, 0, j + 1)

        label = QLabel("= b")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; color: #ff9800; font-size: 13px;")
        grid_layout.addWidget(label, 0, self.n + 1)

        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)

        for i in range(self.n):
            row_label = QLabel(f"Ур.{i+1}:")
            row_label.setStyleSheet("font-weight: bold; color: #555;")
            grid_layout.addWidget(row_label, i + 1, 0)

            row_inputs = []
            for j in range(self.n):
                line_edit = QLineEdit()
                line_edit.setValidator(validator)
                line_edit.setMaximumWidth(100)
                line_edit.setPlaceholderText("0")
                grid_layout.addWidget(line_edit, i + 1, j + 1)
                row_inputs.append(line_edit)
            self.matrix_inputs.append(row_inputs)

            b_edit = QLineEdit()
            b_edit.setValidator(validator)
            b_edit.setMaximumWidth(100)
            b_edit.setPlaceholderText("0")
            grid_layout.addWidget(b_edit, i + 1, self.n + 1)
            self.vector_inputs.append(b_edit)

        layout.addLayout(grid_layout)

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
        example_label = QLabel("Пример для вашей системы (4×4):")
        example_label.setFont(QFont("Arial", 10, QFont.Bold))
        example_layout.addWidget(example_label)
        example_text = QLabel(
            "Уравнение 1: 19*x1 - 4*x2 - 9*x3 - x4 = 100\n"
            "Уравнение 2: -2*x1 + 20*x2 - 2*x3 - 7*x4 = -5\n"
            "Уравнение 3: 6*x1 - 5*x2 - 25*x3 + 9*x4 = 34\n"
            "Уравнение 4: 0*x1 - 3*x2 - 9*x3 + 13*x4 = 78"
        )
        example_text.setWordWrap(True)
        example_layout.addWidget(example_text)
        layout.addWidget(example_frame)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("Далее")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(700, 500)

    def get_system(self):
        try:
            A = np.zeros((self.n, self.n))
            b = np.zeros(self.n)

            for i in range(self.n):
                for j in range(self.n):
                    text = self.matrix_inputs[i][j].text()
                    A[i, j] = float(text) if text.strip() else 0.0

                text = self.vector_inputs[i].text()
                b[i] = float(text) if text.strip() else 0.0

            return A, b
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат числа")
            return None, None


class ToleranceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Параметры решения")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 12px;
                background-color: white;
                color: black;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:reject {
                background-color: #f44336;
            }
            QPushButton:reject:hover {
                background-color: #da190b;
            }
            QLabel {
                font-size: 12px;
                color: black;
            }
            QRadioButton {
                font-size: 13px;
                margin: 8px;
                padding: 8px;
                color: black;
                background-color: #f9f9f9;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            QRadioButton:hover {
                background-color: #e3f2fd;
                border: 2px solid #4CAF50;
            }
            QRadioButton:checked {
                background-color: #c8e6c9;
                border: 2px solid #4CAF50;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                margin-right: 8px;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #4CAF50;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(15)

        title = QLabel("Выберите метод решения и точность")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px; padding: 5px;")
        layout.addWidget(title)

        method_group = QGroupBox("Метод решения")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(10)

        self.method_simple = QRadioButton("  Метод простых итераций (Якоби)")
        self.method_seidel = QRadioButton("  Метод Зейделя")
        self.method_simple.setChecked(True)

        self.method_simple.setMinimumHeight(40)
        self.method_seidel.setMinimumHeight(40)

        method_layout.addWidget(self.method_simple)
        method_layout.addWidget(self.method_seidel)
        method_layout.addStretch()

        method_group.setLayout(method_layout)
        method_group.setMinimumWidth(350)
        layout.addWidget(method_group)

        tolerance_group = QGroupBox("Точность вычислений")
        tolerance_layout = QVBoxLayout()
        tolerance_layout.setSpacing(10)

        tol_label = QLabel("Предельная погрешность:")
        tol_label.setStyleSheet("font-weight: bold;")
        tolerance_layout.addWidget(tol_label)

        self.tolerance_edit = QLineEdit()
        self.tolerance_edit.setText("0.0001")
        self.tolerance_edit.setPlaceholderText("Например: 0.0001 или 1e-6")
        tolerance_layout.addWidget(self.tolerance_edit)

        examples_label = QLabel("Примеры: 0.001 | 0.0001 | 1e-6 | 1e-8")
        examples_label.setStyleSheet("color: #666; font-size: 10px;")
        tolerance_layout.addWidget(examples_label)

        tolerance_group.setLayout(tolerance_layout)
        layout.addWidget(tolerance_group)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        ok_button = QPushButton("Решить")
        ok_button.setMinimumWidth(120)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.setMinimumWidth(120)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(500, 450)

    def get_parameters(self):
        try:
            method = "simple" if self.method_simple.isChecked() else "seidel"
            tolerance = float(self.tolerance_edit.text())

            if tolerance <= 0:
                raise ValueError("Точность должна быть положительной")
            if tolerance > 1:
                raise ValueError("Точность должна быть меньше 1")

            return method, tolerance
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Неверное значение точности:\n{str(e)}")
            return None, None


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

        self.solution_tab = QWidget()
        self.setup_solution_tab()
        self.tab_widget.addTab(self.solution_tab, "Решение")

        self.iterations_tab = QWidget()
        self.setup_iterations_tab()
        self.tab_widget.addTab(self.iterations_tab, "Сходимость")

        self.comparison_tab = QWidget()
        self.setup_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "Сравнение методов")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def setup_solution_tab(self):
        layout = QVBoxLayout()

        self.method_label = QLabel()
        self.method_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.method_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.method_label)

        self.solution_table = QTableWidget()
        self.solution_table.setColumnCount(2)
        self.solution_table.setHorizontalHeaderLabels(["Переменная", "Значение"])
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

        self.iter_info = QLabel()
        self.iter_info.setAlignment(Qt.AlignCenter)
        self.iter_info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px;")
        layout.addWidget(self.iter_info)

        self.residual_info = QLabel()
        self.residual_info.setAlignment(Qt.AlignCenter)
        self.residual_info.setStyleSheet("padding: 10px; background-color: #f3e5f5; border-radius: 5px;")
        layout.addWidget(self.residual_info)

        self.convergence_status = QLabel()
        self.convergence_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.convergence_status)

        self.solution_tab.setLayout(layout)

    def setup_iterations_tab(self):
        layout = QVBoxLayout()

        self.iter_plot_label = QLabel("История сходимости")
        self.iter_plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.iter_plot_label)

        self.iter_table = QTableWidget()
        self.iter_table.setColumnCount(3)
        self.iter_table.setHorizontalHeaderLabels(["Итерация", "Погрешность", "Норма невязки"])
        self.iter_table.setAlternatingRowColors(True)
        self.iter_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #ff9800;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.iter_table)

        self.iterations_tab.setLayout(layout)

    def setup_comparison_tab(self):
        layout = QVBoxLayout()

        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(6)
        self.comparison_table.setHorizontalHeaderLabels(["Метод", "Итераций", "x1", "x2", "x3", "x4"])
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #9C27B0;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.comparison_table)

        self.recommendation = QLabel()
        self.recommendation.setAlignment(Qt.AlignCenter)
        self.recommendation.setWordWrap(True)
        self.recommendation.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(self.recommendation)

        self.comparison_tab.setLayout(layout)

    def display_results(self, method_name, x, iterations, errors, residual_norms, residual, converged):
        self.method_label.setText(f"Результаты: {method_name}")

        n = len(x)
        self.solution_table.setRowCount(n)
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

        self.iter_info.setText(f"Количество итераций: {iterations}")

        residual_norm = np.linalg.norm(residual, np.inf)
        self.residual_info.setText(f"Максимальная невязка: {residual_norm:.2e}")

        if converged:
            self.convergence_status.setText("Сходимость достигнута")
            self.convergence_status.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
        else:
            self.convergence_status.setText("Не удалось достичь сходимости за максимальное число итераций")
            self.convergence_status.setStyleSheet("color: #f44336; font-weight: bold; padding: 5px;")

        rows = min(len(errors), 50)
        self.iter_table.setRowCount(rows)
        for i in range(rows):
            self.iter_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.iter_table.setItem(i, 1, QTableWidgetItem(f"{errors[i]:.2e}"))
            self.iter_table.setItem(i, 2, QTableWidgetItem(f"{residual_norms[i]:.2e}"))

        self.iter_table.resizeColumnsToContents()

    def display_comparison(self, simple_data, seidel_data):
        self.comparison_table.setRowCount(2)

        self.comparison_table.setItem(0, 0, QTableWidgetItem("Метод простых итераций"))
        self.comparison_table.setItem(0, 1, QTableWidgetItem(str(simple_data['iterations'])))
        for i in range(4):
            self.comparison_table.setItem(0, i + 2, QTableWidgetItem(f"{simple_data['x'][i]:.8f}"))

        self.comparison_table.setItem(1, 0, QTableWidgetItem("Метод Зейделя"))
        self.comparison_table.setItem(1, 1, QTableWidgetItem(str(seidel_data['iterations'])))
        for i in range(4):
            self.comparison_table.setItem(1, i + 2, QTableWidgetItem(f"{seidel_data['x'][i]:.8f}"))

        self.comparison_table.resizeColumnsToContents()

        if simple_data['iterations'] < seidel_data['iterations']:
            speedup = seidel_data['iterations'] / simple_data['iterations']
            recommendation = f"Метод простых итераций оказался быстрее в {speedup:.1f} раз"
        elif seidel_data['iterations'] < simple_data['iterations']:
            speedup = simple_data['iterations'] / seidel_data['iterations']
            recommendation = f"Метод Зейделя оказался быстрее в {speedup:.1f} раз"
        else:
            recommendation = "Оба метода показали одинаковую скорость сходимости"

        if simple_data['converged'] and seidel_data['converged']:
            recommendation += "\nОба метода достигли сходимости."
        elif not simple_data['converged'] and not seidel_data['converged']:
            recommendation += "\nВнимание: ни один метод не достиг сходимости. Возможно, матрица не обладает свойством диагонального преобладания."
        elif not simple_data['converged']:
            recommendation += "\nМетод простых итераций не достиг сходимости. Рекомендуется использовать метод Зейделя."
        else:
            recommendation += "\nМетод Зейделя не достиг сходимости. Рекомендуется использовать метод простых итераций."

        self.recommendation.setText(recommendation)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Решение СЛАУ итерационными методами")
        self.setGeometry(100, 100, 1100, 800)

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

        title = QLabel("Решение систем линейных уравнений")
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
            "Методы решения: метод простых итераций (Якоби) и метод Зейделя\n"
            "Размерность: 4 x 4\n"
            "Особенность: методы основаны на приведении системы к виду x = Bx + c\n"
            "Условие сходимости: матрица должна иметь диагональное преобладание"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_frame)

        self.input_button = QPushButton("Ввести коэффициенты системы")
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
        dialog = MatrixInputDialog(4)
        if dialog.exec_():
            A, b = dialog.get_system()
            if A is not None:
                self.A = A
                self.b = b
                self.choose_method()

    def choose_method(self):
        dialog = ToleranceDialog()
        if dialog.exec_():
            method, tolerance = dialog.get_parameters()
            if method is not None:
                self.solve_and_display(method, tolerance)

    def solve_and_display(self, method, tolerance):
        try:
            if IterativeSolvers.has_zero_on_diagonal(self.A):
                raise ValueError("На главной диагонали есть нулевой элемент. Итерационный метод в таком виде неприменим.")

            if not IterativeSolvers.is_diagonally_dominant(self.A):
                self.statusBar().showMessage("Внимание: матрица не имеет диагонального преобладания, сходимость не гарантирована")
                QMessageBox.warning(
                    self,
                    "Предупреждение",
                    "Матрица не обладает свойством диагонального преобладания.\n"
                    "Итерационные методы могут не сходиться или сходиться медленно."
                )

            x_simple, iter_simple, errors_simple, residuals_simple, converged_simple = IterativeSolvers.simple_iteration(
                self.A, self.b, tolerance
            )
            x_seidel, iter_seidel, errors_seidel, residuals_seidel, converged_seidel = IterativeSolvers.seidel_method(
                self.A, self.b, tolerance
            )

            residual_simple = IterativeSolvers.residual(self.A, self.b, x_simple)
            residual_seidel = IterativeSolvers.residual(self.A, self.b, x_seidel)

            simple_data = {
                'x': x_simple,
                'iterations': iter_simple,
                'converged': converged_simple,
                'residual': residual_simple
            }

            seidel_data = {
                'x': x_seidel,
                'iterations': iter_seidel,
                'converged': converged_seidel,
                'residual': residual_seidel
            }

            if method == "simple":
                x = x_simple
                iterations = iter_simple
                errors = errors_simple
                residual_norms = residuals_simple
                residual = residual_simple
                converged = converged_simple
                method_name = "Метод простых итераций (Якоби)"
            else:
                x = x_seidel
                iterations = iter_seidel
                errors = errors_seidel
                residual_norms = residuals_seidel
                residual = residual_seidel
                converged = converged_seidel
                method_name = "Метод Зейделя"

            self.results_widget.display_results(
                method_name, x, iterations, errors, residual_norms, residual, converged
            )
            self.results_widget.display_comparison(simple_data, seidel_data)

            self.statusBar().showMessage(f"Решение найдено за {iterations} итераций")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Непредвиденная ошибка:\n{str(e)}")
            self.statusBar().showMessage("Ошибка при решении")

    def clear_results(self):
        self.results_widget.solution_table.setRowCount(0)
        self.results_widget.iter_table.setRowCount(0)
        self.results_widget.comparison_table.setRowCount(0)
        self.results_widget.method_label.setText("")
        self.results_widget.iter_info.setText("")
        self.results_widget.residual_info.setText("")
        self.results_widget.convergence_status.setText("")
        self.results_widget.recommendation.setText("")
        self.statusBar().showMessage("Результаты очищены")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()