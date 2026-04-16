import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QColor, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RotationMethod:
    @staticmethod
    def is_symmetric(A, tolerance=1e-10):
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j] - A[j, i]) > tolerance:
                    return False
        return True

    @staticmethod
    def find_max_off_diag(A):
        n = len(A)
        max_val = 0.0
        p, q = 0, 1

        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        return p, q, max_val

    @staticmethod
    def calculate_rotation_angle(A, p, q):
        if abs(A[p, q]) < 1e-12:
            return 0.0

        if abs(A[p, p] - A[q, q]) < 1e-12:
            return np.pi / 4

        return 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])

    @staticmethod
    def jacobi_rotation(A, tolerance=1e-10, max_iterations=1000):
        n = len(A)
        A_k = A.copy().astype(float)
        V = np.eye(n)
        errors = []
        iterations = 0

        for iteration in range(max_iterations):
            p, q, max_off_diag = RotationMethod.find_max_off_diag(A_k)
            errors.append(max_off_diag)

            if max_off_diag < tolerance:
                break

            if abs(A_k[p, q]) < 1e-12:
                break

            if abs(A_k[p, p] - A_k[q, q]) < 1e-12:
                theta = np.pi / 4
            else:
                theta = 0.5 * np.arctan2(2 * A_k[p, q], A_k[q, q] - A_k[p, p])

            c = np.cos(theta)
            s = np.sin(theta)

            app = A_k[p, p]
            aqq = A_k[q, q]
            apq = A_k[p, q]

            for j in range(n):
                if j != p and j != q:
                    apj = A_k[p, j]
                    aqj = A_k[q, j]

                    A_k[p, j] = c * apj - s * aqj
                    A_k[j, p] = A_k[p, j]

                    A_k[q, j] = s * apj + c * aqj
                    A_k[j, q] = A_k[q, j]

            A_k[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq
            A_k[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq
            A_k[p, q] = 0.0
            A_k[q, p] = 0.0

            for i in range(n):
                vip = V[i, p]
                viq = V[i, q]
                V[i, p] = c * vip - s * viq
                V[i, q] = s * vip + c * viq

            iterations += 1

        eigenvalues = np.diag(A_k)
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = V[:, idx]

        return eigenvalues, eigenvectors, iterations, errors

    @staticmethod
    def check_eigenvectors(A, eigenvalues, eigenvectors):
        n = len(A)
        max_error = 0.0

        for i in range(n):
            Av = A @ eigenvectors[:, i]
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            error = np.linalg.norm(Av - lambda_v)
            max_error = max(max_error, error)

        return max_error


class MatrixInputDialog(QDialog):
    def __init__(self, n=3):
        super().__init__()
        self.n = n
        self.matrix_inputs = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Ввод симметрической матрицы {self.n}x{self.n}")
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
                min-width: 80px;
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

        title = QLabel("Введите симметрическую матрицу 3x3:")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)

        info = QLabel("Внимание: матрица должна быть симметричной (a[i][j] = a[j][i])")
        info.setStyleSheet("color: #ff9800; margin-bottom: 10px;")
        layout.addWidget(info)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        for j in range(self.n):
            label = QLabel(f"Столбец {j+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 13px;")
            grid_layout.addWidget(label, 0, j + 1)

        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)

        for i in range(self.n):
            row_label = QLabel(f"Строка {i+1}:")
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

        layout.addLayout(grid_layout)

        example_frame = QFrame()
        example_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QLabel {
                color: #1565c0;
            }
        """)
        example_layout = QVBoxLayout(example_frame)
        example_label = QLabel("Пример симметрической матрицы 3x3:")
        example_label.setFont(QFont("Arial", 10, QFont.Bold))
        example_layout.addWidget(example_label)
        example_text = QLabel(
            "[ 4   1   2 ]\n"
            "[ 1   3   1 ]\n"
            "[ 2   1   5 ]"
        )
        example_text.setFont(QFont("Courier New", 10))
        example_text.setAlignment(Qt.AlignCenter)
        example_layout.addWidget(example_text)
        layout.addWidget(example_frame)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("Найти собственные значения")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(600, 450)

    def get_matrix(self):
        try:
            A = np.zeros((self.n, self.n))

            for i in range(self.n):
                for j in range(self.n):
                    text = self.matrix_inputs[i][j].text()
                    A[i, j] = float(text) if text.strip() else 0.0

            if not RotationMethod.is_symmetric(A):
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    "Матрица не является симметричной.\nМетод вращений Якоби работает только для симметрических матриц."
                )
                return None

            return A
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат числа")
            return None


class ToleranceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Параметры вычислений")
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
                min-width: 250px;
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

        title = QLabel("Настройка точности вычислений")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)

        tolerance_group = QGroupBox("Параметры метода вращений")
        tolerance_layout = QVBoxLayout()
        tolerance_layout.setSpacing(10)

        tol_label = QLabel("Предельная погрешность (максимальный внедиагональный элемент):")
        tol_label.setStyleSheet("font-weight: bold; color: black;")
        tolerance_layout.addWidget(tol_label)

        self.tolerance_edit = QLineEdit()
        self.tolerance_edit.setText("0.0001")
        tolerance_layout.addWidget(self.tolerance_edit)

        examples_label = QLabel("Рекомендации:\n0.01 - низкая точность\n0.0001 - средняя точность\n1e-6 - высокая точность")
        examples_label.setStyleSheet("color: #666; font-size: 10px;")
        tolerance_layout.addWidget(examples_label)

        tolerance_group.setLayout(tolerance_layout)
        layout.addWidget(tolerance_group)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        ok_button = QPushButton("Вычислить")
        ok_button.setMinimumWidth(120)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.setMinimumWidth(120)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(500, 400)

    def get_tolerance(self):
        try:
            tolerance = float(self.tolerance_edit.text())
            if tolerance <= 0:
                raise ValueError("Точность должна быть положительной")
            if tolerance > 1:
                raise ValueError("Точность должна быть меньше 1")
            return tolerance
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Неверное значение точности:\n{str(e)}")
            return None


class ResultsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        container_layout = QVBoxLayout(container)

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
                color: black;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)

        self.eigenvalues_tab = QWidget()
        self.setup_eigenvalues_tab()
        self.tab_widget.addTab(self.eigenvalues_tab, "Собственные значения")

        self.eigenvectors_tab = QWidget()
        self.setup_eigenvectors_tab()
        self.tab_widget.addTab(self.eigenvectors_tab, "Собственные векторы")

        self.convergence_tab = QWidget()
        self.setup_convergence_tab()
        self.tab_widget.addTab(self.convergence_tab, "Сходимость")

        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "Анализ точности")

        container_layout.addWidget(self.tab_widget)
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

        self.setLayout(layout)

    def setup_eigenvalues_tab(self):
        layout = QVBoxLayout()

        title = QLabel("Собственные значения матрицы")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin: 10px;")
        layout.addWidget(title)

        self.eigenvalues_table = QTableWidget()
        self.eigenvalues_table.setColumnCount(2)
        self.eigenvalues_table.setHorizontalHeaderLabels(["N", "Собственное значение"])
        self.eigenvalues_table.setAlternatingRowColors(True)
        self.eigenvalues_table.setMinimumHeight(200)
        self.eigenvalues_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 14px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 10px;
                color: black;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.eigenvalues_table)

        self.iter_info = QLabel()
        self.iter_info.setAlignment(Qt.AlignCenter)
        self.iter_info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px; color: black; font-weight: bold;")
        layout.addWidget(self.iter_info)

        self.verification_info = QLabel()
        self.verification_info.setAlignment(Qt.AlignCenter)
        self.verification_info.setWordWrap(True)
        self.verification_info.setStyleSheet("padding: 10px; background-color: #f3e5f5; border-radius: 5px; color: black;")
        layout.addWidget(self.verification_info)

        layout.addStretch()
        self.eigenvalues_tab.setLayout(layout)

    def setup_eigenvectors_tab(self):
        layout = QVBoxLayout()

        title = QLabel("Собственные векторы матрицы")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff9800; margin: 10px;")
        layout.addWidget(title)

        self.eigenvectors_table = QTableWidget()
        self.eigenvectors_table.setAlternatingRowColors(True)
        self.eigenvectors_table.setMinimumHeight(300)
        self.eigenvectors_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                text-align: center;
                color: black;
            }
            QHeaderView::section {
                background-color: #ff9800;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.eigenvectors_table)

        layout.addStretch()
        self.eigenvectors_tab.setLayout(layout)

    def setup_convergence_tab(self):
        layout = QVBoxLayout()

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        layout.addWidget(self.canvas)

        self.convergence_table = QTableWidget()
        self.convergence_table.setColumnCount(4)
        self.convergence_table.setHorizontalHeaderLabels(["Итерация", "Погрешность", "Уменьшение", "Статус"])
        self.convergence_table.setAlternatingRowColors(True)
        self.convergence_table.setMinimumHeight(200)
        self.convergence_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 11px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 6px;
                color: black;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 6px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.convergence_table)

        explanation = QLabel(
            "Пояснение к графику:\n"
            "Синяя линия - максимальный внедиагональный элемент (погрешность)\n"
            "Красная пунктирная линия - заданная точность\n"
            "Чем быстрее линия падает вниз, тем быстрее метод сходится\n"
            "График построен в логарифмическом масштабе по вертикали"
        )
        explanation.setStyleSheet("""
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            color: #1B5E20;
            border: 1px solid #4CAF50;
        """)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.convergence_tab.setLayout(layout)

    def setup_analysis_tab(self):
        layout = QVBoxLayout()

        title = QLabel("Анализ зависимости погрешности от числа итераций")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; background-color: transparent; margin: 10px; padding: 10px;")
        layout.addWidget(title)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(500)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                font-size: 12px;
                background-color: white;
                color: black;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.analysis_text)

        self.analysis_tab.setLayout(layout)

    def display_results(self, eigenvalues, eigenvectors, iterations, errors, tolerance, verification_error):
        n = len(eigenvalues)
        self.eigenvalues_table.setRowCount(n)

        for i in range(n):
            self.eigenvalues_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            item = QTableWidgetItem(f"{eigenvalues[i]:.10f}")
            if eigenvalues[i] > 0:
                item.setForeground(QBrush(QColor(76, 175, 80)))
            elif eigenvalues[i] < 0:
                item.setForeground(QBrush(QColor(244, 67, 54)))
            self.eigenvalues_table.setItem(i, 1, item)

        self.eigenvalues_table.resizeColumnsToContents()
        self.iter_info.setText(f"Количество итераций: {iterations}")

        tolerance_reached = errors[-1] <= tolerance if errors else True

        if tolerance_reached:
            status_color = "#4CAF50"
            status_text = "Точность достигнута"
        else:
            status_color = "#f44336"
            status_text = "Точность не достигнута"

        final_error = errors[-1] if errors else 0.0

        self.verification_info.setText(
            f"Погрешность проверки A·v = λ·v: {verification_error:.2e}\n"
            f"Достигнутая точность (макс. внедиаг. элемент): {final_error:.2e}\n"
            f"Заданная точность: {tolerance:.2e}\n"
            f"{status_text}"
        )
        self.verification_info.setStyleSheet(
            f"padding: 10px; background-color: #f3e5f5; border-radius: 5px; color: black; "
            f"font-weight: bold; border-left: 4px solid {status_color};"
        )

        self.eigenvectors_table.setRowCount(n)
        self.eigenvectors_table.setColumnCount(n)
        self.eigenvectors_table.setHorizontalHeaderLabels([f"λ{i+1}" for i in range(n)])

        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{eigenvectors[i, j]:.8f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.eigenvectors_table.setItem(i, j, item)

        self.eigenvectors_table.resizeColumnsToContents()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        iterations_list = list(range(1, len(errors) + 1))

        if errors:
            ax.semilogy(iterations_list, errors, 'b-', linewidth=2, marker='o', markersize=4, label='Погрешность')
            ax.axhline(y=tolerance, color='r', linestyle='--', linewidth=2, label=f'Заданная точность: {tolerance:.0e}')
            ax.set_ylim(bottom=max(1e-16, min(errors[-1] * 0.5, tolerance * 0.5)))

        ax.set_xlabel('Номер итерации', fontsize=11)
        ax.set_ylabel('Максимальный внедиагональный элемент', fontsize=11)
        ax.set_title('Сходимость метода вращений Якоби', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        self.canvas.draw()

        display_count = min(len(errors), 30)
        self.convergence_table.setRowCount(display_count)

        for i in range(display_count):
            self.convergence_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.convergence_table.setItem(i, 1, QTableWidgetItem(f"{errors[i]:.2e}"))

            if i > 0:
                ratio = errors[i] / errors[i - 1] if errors[i - 1] > 0 else 0.0
                self.convergence_table.setItem(i, 2, QTableWidgetItem(f"{ratio:.4f}"))

                if ratio < 0.1:
                    status = "Быстрая"
                    status_color = QColor("#4CAF50")
                elif ratio < 0.5:
                    status = "Средняя"
                    status_color = QColor("#FF9800")
                else:
                    status = "Медленная"
                    status_color = QColor("#f44336")

                status_item = QTableWidgetItem(status)
                status_item.setForeground(QBrush(status_color))
                self.convergence_table.setItem(i, 3, status_item)
            else:
                self.convergence_table.setItem(i, 2, QTableWidgetItem("-"))
                self.convergence_table.setItem(i, 3, QTableWidgetItem("Начало"))

        self.convergence_table.resizeColumnsToContents()
        analysis = self.analyze_convergence_html(errors, iterations, tolerance, verification_error)
        self.analysis_text.setHtml(analysis)

    def analyze_convergence_html(self, errors, iterations, tolerance, verification_error):
        final_error = errors[-1] if errors else 0.0
        tolerance_reached = final_error <= tolerance if errors else True

        if tolerance_reached:
            main_color = "#4CAF50"
            status_text = "ДОСТИГНУТА"
        else:
            main_color = "#f44336"
            status_text = "НЕ ДОСТИГНУТА"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f2f5;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            .card {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #ddd;
            }}
            .card-title {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }}
            .metric-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 15px;
            }}
            .metric {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                min-width: 180px;
                text-align: center;
                border: 1px solid #dee2e6;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                font-size: 12px;
                color: #6c757d;
                margin-top: 5px;
            }}
            .status {{
                background-color: #f8f9fa;
                border-left: 4px solid {main_color};
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                color: #333;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .info-table th, .info-table td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
                color: #333;
            }}
            .info-table th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            .info-table tr:hover {{
                background-color: #f8f9fa;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 5px;
                font-size: 11px;
                font-weight: bold;
            }}
            .badge-success {{
                background-color: #4CAF50;
                color: white;
            }}
            .badge-warning {{
                background-color: #FF9800;
                color: white;
            }}
            .badge-danger {{
                background-color: #f44336;
                color: white;
            }}
            .recommendation-list {{
                margin: 0;
                padding-left: 20px;
                color: #333;
            }}
            .recommendation-list li {{
                margin: 8px 0;
                line-height: 1.5;
                color: #333;
            }}
        </style>
        </head>
        <body>
        <div class="container">
            <div class="card">
                <div class="card-title">Основные параметры</div>
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{iterations}</div>
                        <div class="metric-label">Всего итераций</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{tolerance:.2e}</div>
                        <div class="metric-label">Заданная точность</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{final_error:.2e}</div>
                        <div class="metric-label">Достигнутая точность</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{verification_error:.2e}</div>
                        <div class="metric-label">Проверка A·v = lambda·v</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Достижение точности</div>
                <div class="status">
                    <strong>Статус: {status_text}</strong><br>
        """

        if tolerance_reached:
            reached_iter = next((i for i, e in enumerate(errors) if e <= tolerance), -1)
            if reached_iter >= 0:
                html += f"""
                    Точность достигнута на <strong>{reached_iter + 1}</strong> итерации<br>
                    Экономия итераций: <strong>{iterations - reached_iter - 1}</strong>
                """
            else:
                html += "Точность достигнута на последней итерации"
        else:
            html += f"""
                Точность НЕ достигнута за {iterations} итераций<br>
                Рекомендуется увеличить максимальное число итераций или уменьшить требования к точности
            """

        html += """
                </div>
            </div>

            <div class="card">
                <div class="card-title">Динамика изменения погрешности</div>
                <table class="info-table">
                    <thead>
                        <tr><th>Этап</th><th>Погрешность</th><th>Уменьшение</th></tr>
                    </thead>
                    <tbody>
        """

        if errors:
            points = [
                (0, "Начальная"),
                (max(1, iterations // 4), f"25 процентов итераций ({max(1, iterations // 4)})"),
                (max(1, iterations // 2), f"50 процентов итераций ({max(1, iterations // 2)})"),
                (min(iterations - 1, int(iterations * 0.75)), f"75 процентов итераций ({min(iterations - 1, int(iterations * 0.75))})"),
                (len(errors) - 1, "Конечная")
            ]

            prev_error = None
            used = set()
            for idx, label in points:
                if idx < len(errors) and idx not in used:
                    used.add(idx)
                    error = errors[idx]
                    if prev_error is not None and error > 0:
                        reduction = prev_error / error
                        html += f"<tr><td>{label}</td><td>{error:.2e}</td><td>в {reduction:.2f} раз</td></tr>"
                    else:
                        html += f"<tr><td>{label}</td><td>{error:.2e}</td><td>-</td></tr>"
                    prev_error = error

        html += """
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-title">Скорость сходимости</div>
        """

        if len(errors) >= 5:
            ratios = [errors[i] / errors[i - 1] for i in range(1, min(len(errors), 6)) if errors[i - 1] > 0]
            if ratios:
                early_ratio = np.mean(ratios)

                if early_ratio < 0.1:
                    speed_text = "СВЕРХБЫСТРАЯ"
                    speed_badge = "badge-success"
                elif early_ratio < 0.3:
                    speed_text = "БЫСТРАЯ"
                    speed_badge = "badge-success"
                elif early_ratio < 0.6:
                    speed_text = "УМЕРЕННАЯ"
                    speed_badge = "badge-warning"
                else:
                    speed_text = "МЕДЛЕННАЯ"
                    speed_badge = "badge-danger"

                html += f"""
                    <p>Среднее уменьшение погрешности за итерацию: <strong>в {1 / early_ratio:.2f} раз</strong></p>
                    <p>Характер сходимости: <span class="badge {speed_badge}">{speed_text}</span></p>
                """

        html += """
            </div>

            <div class="card">
                <div class="card-title">Рекомендации</div>
                <ul class="recommendation-list">
        """

        if tolerance_reached:
            if iterations < 20:
                html += "<li>Отличный результат! Метод сошелся очень быстро.</li>"
                html += "<li>Для повышения точности можно уменьшить значение погрешности.</li>"
            elif iterations < 50:
                html += "<li>Хороший результат. Метод сошелся за приемлемое число итераций.</li>"
            else:
                html += "<li>Метод сошелся, но требуется много итераций.</li>"
                html += "<li>Возможно, матрица близка к вырожденной.</li>"
        else:
            html += "<li>Метод не достиг заданной точности за отведенное число итераций.</li>"
            html += "<li>Рекомендации:</li>"
            html += "<li>- Увеличьте максимальное число итераций.</li>"
            html += "<li>- Проверьте, не близка ли матрица к вырожденной.</li>"
            html += "<li>- Увеличьте допустимую погрешность.</li>"

        html += """
                </ul>
            </div>
        </div>
        </body>
        </html>
        """

        return html


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.A = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Метод вращений (Якоби) для нахождения собственных значений и векторов")
        self.setGeometry(100, 100, 1300, 900)

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

        title = QLabel("Метод вращений (Якоби) для симметрических матриц")
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
            "Метод вращений (Якоби): итерационный метод для нахождения всех собственных значений и векторов\n"
            "симметрических матриц. Размерность: 3 x 3"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_frame)

        self.input_button = QPushButton("Ввести симметрическую матрицу 3x3")
        self.input_button.clicked.connect(self.input_matrix)
        layout.addWidget(self.input_button)

        self.results_widget = ResultsWidget()
        layout.addWidget(self.results_widget)

        clear_button = QPushButton("Очистить все")
        clear_button.setStyleSheet("background-color: #f44336;")
        clear_button.clicked.connect(self.clear_results)
        layout.addWidget(clear_button)

        self.statusBar().showMessage("Готов к работе")

    def input_matrix(self):
        dialog = MatrixInputDialog(3)
        if dialog.exec_():
            A = dialog.get_matrix()
            if A is not None:
                self.A = A
                self.choose_tolerance()

    def choose_tolerance(self):
        dialog = ToleranceDialog()
        if dialog.exec_():
            tolerance = dialog.get_tolerance()
            if tolerance is not None:
                self.solve_and_display(tolerance)

    def solve_and_display(self, tolerance):
        try:
            self.statusBar().showMessage("Вычисление собственных значений и векторов...")

            eigenvalues, eigenvectors, iterations, errors = RotationMethod.jacobi_rotation(
                self.A, tolerance
            )

            verification_error = RotationMethod.check_eigenvectors(self.A, eigenvalues, eigenvectors)

            self.results_widget.display_results(
                eigenvalues, eigenvectors, iterations, errors, tolerance, verification_error
            )

            self.statusBar().showMessage(f"Вычисление завершено за {iterations} итераций")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Непредвиденная ошибка:\n{str(e)}")
            self.statusBar().showMessage("Ошибка при вычислении")

    def clear_results(self):
        self.results_widget.eigenvalues_table.setRowCount(0)
        self.results_widget.eigenvectors_table.setRowCount(0)
        self.results_widget.convergence_table.setRowCount(0)
        self.results_widget.iter_info.setText("")
        self.results_widget.verification_info.setText("")
        self.results_widget.analysis_text.clear()
        self.results_widget.figure.clear()
        self.results_widget.canvas.draw()
        self.statusBar().showMessage("Результаты очищены")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()