import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QColor, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class QRAlgorithm:
    """Класс с реализацией QR-алгоритма"""
    
    @staticmethod
    def householder_reflection(x):
        """Преобразование Хаусхолдера для вектора x"""
        n = len(x)
        v = x.copy()
        norm_x = np.linalg.norm(x)
        
        if norm_x < 1e-15:
            return np.eye(n)
        
        alpha = -np.sign(x[0]) * norm_x
        v[0] = x[0] - alpha
        v = v / np.linalg.norm(v)
        
        H = np.eye(n) - 2 * np.outer(v, v)
        return H, alpha
    
    @staticmethod
    def qr_decomposition(A):
        """
        QR-разложение методом Хаусхолдера
        Возвращает Q и R матрицы
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy().astype(float)
        
        for i in range(min(m, n)):
            x = R[i:, i]
            
            if np.linalg.norm(x) < 1e-15:
                continue
            
            H_i = np.eye(m)
            H_small, _ = QRAlgorithm.householder_reflection(x)
            H_i[i:, i:] = H_small
            
            R = H_i @ R
            Q = Q @ H_i.T
        
        return Q, R
    
    @staticmethod
    def qr_algorithm_eigenvalues(A, tolerance=1e-10, max_iterations=1000):
        """
        QR-алгоритм для нахождения собственных значений
        Возвращает собственные значения и историю итераций
        """
        n = A.shape[0]
        Ak = A.copy().astype(float)
        eigenvalues = np.zeros(n, dtype=complex)
        iterations = 0
        history = []
        
        for iteration in range(max_iterations):
            Q, R = QRAlgorithm.qr_decomposition(Ak)
            Ak_next = R @ Q
            
            off_diag_norm = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    off_diag_norm += abs(Ak[i, j])
            
            history.append({
                'iteration': iteration + 1,
                'matrix': Ak.copy(),
                'off_diag_norm': off_diag_norm
            })
            
            if off_diag_norm < tolerance:
                iterations = iteration + 1
                break
            
            Ak = Ak_next
            iterations = iteration + 1
        
        for i in range(n):
            if i < n-1 and abs(Ak[i+1, i]) > tolerance:
                a = Ak[i, i]
                b = Ak[i, i+1]
                c = Ak[i+1, i]
                d = Ak[i+1, i+1]
                
                trace = a + d
                det = a*d - b*c
                discriminant = trace**2 - 4*det
                
                if discriminant < 0:
                    real_part = trace / 2
                    imag_part = np.sqrt(-discriminant) / 2
                    eigenvalues[i] = complex(real_part, imag_part)
                    eigenvalues[i+1] = complex(real_part, -imag_part)
                else:
                    eigenvalues[i] = (trace + np.sqrt(discriminant)) / 2
                    eigenvalues[i+1] = (trace - np.sqrt(discriminant)) / 2
                
                i += 1
            else:
                eigenvalues[i] = Ak[i, i]
        
        converged = off_diag_norm < tolerance
        return eigenvalues, iterations, history, Ak, converged
    
    @staticmethod
    def verify_eigenvalues(A, eigenvalues):
        """Проверка найденных собственных значений"""
        n = len(A)
        errors = []
        
        for ev in eigenvalues:
            if np.iscomplex(ev):
                ev_real = ev.real
                ev_imag = ev.imag
                error = np.linalg.det(A - ev_real * np.eye(n))
            else:
                error = np.linalg.det(A - ev * np.eye(n))
            errors.append(abs(error))
        
        return errors


class ConvergenceGraph(FigureCanvas):
    """График сходимости QR-алгоритма"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Номер итерации')
        self.axes.set_ylabel('Норма поддиагональных элементов')
        self.axes.set_title('График сходимости QR-алгоритма')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        
    def plot_convergence(self, history, tolerance):
        """Построение графика сходимости"""
        self.axes.clear()
        
        if not history:
            self.draw()
            return
        
        iterations = [h['iteration'] for h in history]
        errors = [h['off_diag_norm'] for h in history]
        
        self.axes.semilogy(iterations, errors, 'b-', linewidth=2, label='Погрешность')
        self.axes.axhline(y=tolerance, color='r', linestyle='--', 
                         linewidth=1.5, label=f'Заданная точность ({tolerance:.1e})')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Номер итерации', fontsize=11)
        self.axes.set_ylabel('Норма поддиагональных элементов', fontsize=11)
        self.axes.set_title('График сходимости QR-алгоритма', fontsize=13, fontweight='bold')
        self.axes.legend(loc='upper right', fontsize=10)
        
        if len(iterations) > 1:
            self.axes.set_xlim(1, iterations[-1])
        
        self.figure.tight_layout()
        self.draw()


class MatrixInputDialog(QDialog):
    """Диалог для ввода матрицы 3x3"""
    
    def __init__(self):
        super().__init__()
        self.matrix_inputs = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Ввод матрицы 3x3")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
                background-color: white;
                color: black;
                min-width: 80px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QLabel {
                font-size: 12px;
                color: black;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        title = QLabel("Введите элементы матрицы 3x3")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        
        for j in range(3):
            label = QLabel(f"Столбец {j+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 12px;")
            grid_layout.addWidget(label, 0, j + 1)
        
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        
        for i in range(3):
            row_label = QLabel(f"Строка {i+1}:")
            row_label.setStyleSheet("font-weight: bold; color: #555;")
            grid_layout.addWidget(row_label, i + 1, 0)
            
            row_inputs = []
            for j in range(3):
                line_edit = QLineEdit()
                line_edit.setValidator(validator)
                line_edit.setPlaceholderText("0")
                line_edit.setAlignment(Qt.AlignCenter)
                grid_layout.addWidget(line_edit, i + 1, j + 1)
                row_inputs.append(line_edit)
            self.matrix_inputs.append(row_inputs)
        
        layout.addLayout(grid_layout)
        
        examples_frame = QFrame()
        examples_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                color: #1565c0;
            }
        """)
        examples_layout = QVBoxLayout(examples_frame)
        
        examples_label = QLabel("Примеры матриц:")
        examples_label.setFont(QFont("Arial", 11, QFont.Bold))
        examples_layout.addWidget(examples_label)
        
        example1_btn = QPushButton("Матрица 1: [[4, -1, 1], [-1, 3, -2], [1, -2, 3]]")
        example1_btn.setStyleSheet("""
            QPushButton {
                background-color: #64B5F6;
                text-align: left;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
        """)
        example1_btn.clicked.connect(lambda: self.fill_example(1))
        examples_layout.addWidget(example1_btn)
        
        example2_btn = QPushButton("Матрица 2: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]")
        example2_btn.setStyleSheet("""
            QPushButton {
                background-color: #64B5F6;
                text-align: left;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
        """)
        example2_btn.clicked.connect(lambda: self.fill_example(2))
        examples_layout.addWidget(example2_btn)
        
        example3_btn = QPushButton("Матрица 3: [[1, 2, 3], [2, 1, 2], [3, 2, 1]]")
        example3_btn.setStyleSheet("""
            QPushButton {
                background-color: #64B5F6;
                text-align: left;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
        """)
        example3_btn.clicked.connect(lambda: self.fill_example(3))
        examples_layout.addWidget(example3_btn)
        
        layout.addWidget(examples_frame)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        ok_button = QPushButton("Далее")
        ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("Отмена")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(500, 600)
    
    def fill_example(self, example_num):
        """Заполнение матрицы примером"""
        if example_num == 1:
            matrix = [[4, -1, 1], [-1, 3, -2], [1, -2, 3]]
        elif example_num == 2:
            matrix = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        else:
            matrix = [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
        
        for i in range(3):
            for j in range(3):
                self.matrix_inputs[i][j].setText(str(matrix[i][j]))
    
    def get_matrix(self):
        """Получение введенной матрицы"""
        try:
            A = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    text = self.matrix_inputs[i][j].text()
                    if text.strip():
                        A[i, j] = float(text)
                    else:
                        A[i, j] = 0
            return A
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат числа")
            return None


class ToleranceDialog(QDialog):
    """Диалог для ввода точности"""
    
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
                padding: 10px;
                border: 2px solid #ccc;
                border-radius: 5px;
                font-size: 13px;
                background-color: white;
                color: black;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
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
                color: #2196F3;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        title = QLabel("Настройка точности вычислений")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")
        layout.addWidget(title)
        
        tolerance_group = QGroupBox("Параметры точности")
        tolerance_layout = QVBoxLayout()
        tolerance_layout.setSpacing(15)
        
        tol_label = QLabel("Предельная погрешность (epsilon):")
        tol_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        tolerance_layout.addWidget(tol_label)
        
        self.tolerance_edit = QLineEdit()
        self.tolerance_edit.setText("1e-10")
        self.tolerance_edit.setPlaceholderText("Например: 1e-10, 0.0001")
        tolerance_layout.addWidget(self.tolerance_edit)
        
        examples_label = QLabel("Рекомендуемые значения:")
        examples_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        tolerance_layout.addWidget(examples_label)
        
        examples_text = QLabel(
            "1e-6  - средняя точность\n"
            "1e-10 - высокая точность\n"
            "1e-12 - максимальная точность"
        )
        examples_text.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        tolerance_layout.addWidget(examples_text)
        
        max_iter_label = QLabel("Максимальное число итераций:")
        max_iter_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        tolerance_layout.addWidget(max_iter_label)
        
        self.max_iter_edit = QLineEdit()
        self.max_iter_edit.setText("1000")
        tolerance_layout.addWidget(self.max_iter_edit)
        
        tolerance_group.setLayout(tolerance_layout)
        layout.addWidget(tolerance_group)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #fff3e0;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                color: #e65100;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_label = QLabel(
            "QR-алгоритм находит все собственные значения матрицы.\n"
            "Сходимость гарантируется для симметричных матриц.\n"
            "Для несимметричных матриц алгоритм также работает."
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_frame)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        ok_button = QPushButton("Вычислить")
        ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("Отмена")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(450, 500)
    
    def get_parameters(self):
        """Получение параметров вычислений"""
        try:
            tolerance = float(self.tolerance_edit.text())
            max_iter = int(self.max_iter_edit.text())
            
            if tolerance <= 0:
                raise ValueError("Точность должна быть положительной")
            if tolerance >= 1:
                raise ValueError("Точность должна быть меньше 1")
            if max_iter <= 0:
                raise ValueError("Число итераций должно быть положительным")
            if max_iter > 10000:
                raise ValueError("Слишком большое число итераций (макс. 10000)")
                
            return tolerance, max_iter
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"Неверное значение:\n{str(e)}")
            return None, None


class ResultsWidget(QWidget):
    """Виджет для отображения результатов"""
    
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
                padding: 10px 20px;
                margin: 2px;
                border-radius: 5px;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Вкладка "Собственные значения" с прокруткой
        self.results_tab = QWidget()
        self.setup_results_tab()
        self.tab_widget.addTab(self.create_scrollable_tab(self.results_tab), "Собственные значения")
        
        # Вкладка "QR-разложение" с прокруткой
        self.qr_tab = QWidget()
        self.setup_qr_tab()
        self.tab_widget.addTab(self.create_scrollable_tab(self.qr_tab), "QR-разложение")
        
        # Вкладка "История итераций" с прокруткой
        self.history_tab = QWidget()
        self.setup_history_tab()
        self.tab_widget.addTab(self.create_scrollable_tab(self.history_tab), "История итераций")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def create_scrollable_tab(self, content_widget):
        """Оборачивает виджет в QScrollArea для прокрутки"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll
    
    def setup_results_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.convergence_label = QLabel()
        self.convergence_label.setAlignment(Qt.AlignCenter)
        self.convergence_label.setWordWrap(True)
        self.convergence_label.setStyleSheet("""
            padding: 12px;
            background-color: #e8f5e9;
            border-radius: 5px;
            font-size: 14px;
            color: #1b5e20;
        """)
        layout.addWidget(self.convergence_label)
        
        self.eigenvalues_table = QTableWidget()
        self.eigenvalues_table.setColumnCount(4)
        self.eigenvalues_table.setHorizontalHeaderLabels([
            "Номер", "Собственное значение", "Тип", "Невязка |det(A-λI)|"
        ])
        self.eigenvalues_table.setAlternatingRowColors(True)
        self.eigenvalues_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.eigenvalues_table.setMinimumHeight(150)
        self.eigenvalues_table.horizontalHeader().setStretchLastSection(True)
        self.eigenvalues_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.eigenvalues_table)
        
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            padding: 12px;
            background-color: #f3e5f5;
            border-radius: 5px;
            font-size: 13px;
            color: #333333;
        """)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        self.results_tab.setLayout(layout)
    
    def setup_qr_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        q_label = QLabel("Ортогональная матрица Q:")
        q_label.setFont(QFont("Arial", 13, QFont.Bold))
        layout.addWidget(q_label)
        
        self.q_table = QTableWidget()
        self.q_table.setColumnCount(3)
        self.q_table.setHorizontalHeaderLabels(["Столбец 1", "Столбец 2", "Столбец 3"])
        self.q_table.setAlternatingRowColors(True)
        self.q_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.q_table.setMinimumHeight(150)
        self.q_table.horizontalHeader().setStretchLastSection(True)
        self.q_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.q_table)
        
        r_label = QLabel("Верхняя треугольная матрица R:")
        r_label.setFont(QFont("Arial", 13, QFont.Bold))
        layout.addWidget(r_label)
        
        self.r_table = QTableWidget()
        self.r_table.setColumnCount(3)
        self.r_table.setHorizontalHeaderLabels(["Столбец 1", "Столбец 2", "Столбец 3"])
        self.r_table.setAlternatingRowColors(True)
        self.r_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.r_table.setMinimumHeight(150)
        self.r_table.horizontalHeader().setStretchLastSection(True)
        self.r_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #FF9800;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.r_table)
        
        self.verification_label = QLabel()
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setWordWrap(True)
        self.verification_label.setStyleSheet("""
            padding: 12px;
            background-color: #e3f2fd;
            border-radius: 5px;
            font-size: 13px;
            color: #0d47a1;
        """)
        layout.addWidget(self.verification_label)
        
        layout.addStretch()
        self.qr_tab.setLayout(layout)
    
    def setup_history_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(2)
        self.history_table.setHorizontalHeaderLabels([
            "Итерация", "Норма поддиагональных элементов"
        ])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.history_table.setMinimumHeight(200)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setStyleSheet("""
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
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.history_table, 2)  # Растягивается больше
        
        graph_label = QLabel("График сходимости:")
        graph_label.setFont(QFont("Arial", 13, QFont.Bold))
        layout.addWidget(graph_label)
        
        self.convergence_graph = ConvergenceGraph(self)
        self.convergence_graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.convergence_graph, 3)  # Растягивается ещё больше
        
        layout.addStretch()
        self.history_tab.setLayout(layout)
    
    def display_results(self, A, eigenvalues, iterations, history, Ak_final, converged, tolerance):
        """Отображение результатов вычислений"""
        
        if converged:
            self.convergence_label.setText(
                f"Сходимость достигнута за {iterations} итераций\n"
                f"Точность: {tolerance}"
            )
            self.convergence_label.setStyleSheet("""
                padding: 12px;
                background-color: #e8f5e9;
                border-radius: 5px;
                font-size: 14px;
                color: #1b5e20;
            """)
        else:
            self.convergence_label.setText(
                f"Сходимость не достигнута за {iterations} итераций\n"
                f"Точность: {tolerance}"
            )
            self.convergence_label.setStyleSheet("""
                padding: 12px;
                background-color: #fff3e0;
                border-radius: 5px;
                font-size: 14px;
                color: #e65100;
            """)
        
        errors = QRAlgorithm.verify_eigenvalues(A, eigenvalues)
        self.eigenvalues_table.setRowCount(3)
        
        for i, (ev, error) in enumerate(zip(eigenvalues, errors)):
            self.eigenvalues_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            if np.iscomplex(ev):
                val_text = f"{ev.real:.8f} {'+' if ev.imag >= 0 else '-'} {abs(ev.imag):.8f}i"
                type_text = "Комплексное"
            else:
                val_text = f"{ev:.8f}"
                type_text = "Действительное"
            
            val_item = QTableWidgetItem(val_text)
            val_item.setFont(QFont("Courier New", 11))
            self.eigenvalues_table.setItem(i, 1, val_item)
            
            self.eigenvalues_table.setItem(i, 2, QTableWidgetItem(type_text))
            
            error_item = QTableWidgetItem(f"{error:.2e}")
            if error < tolerance:
                error_item.setForeground(QBrush(QColor(76, 175, 80)))
            else:
                error_item.setForeground(QBrush(QColor(244, 67, 54)))
            self.eigenvalues_table.setItem(i, 3, error_item)
        
        self.eigenvalues_table.resizeColumnsToContents()
        
        trace_A = np.trace(A)
        trace_eigen = sum(eigenvalues)
        det_A = np.linalg.det(A)
        det_eigen = np.prod(eigenvalues)
        
        info_text = (
            f"След матрицы A: {trace_A:.6f}\n"
            f"Сумма собственных значений: {trace_eigen:.6f}\n"
            f"Определитель A: {det_A:.6f}\n"
            f"Произведение собственных значений: {det_eigen:.6f}"
        )
        self.info_label.setText(info_text)
        
        Q, R = QRAlgorithm.qr_decomposition(A)
        
        self.q_table.setRowCount(3)
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem(f"{Q[i, j]:.8f}")
                item.setFont(QFont("Courier New", 10))
                self.q_table.setItem(i, j, item)
        
        self.r_table.setRowCount(3)
        for i in range(3):
            for j in range(3):
                if i <= j:
                    item = QTableWidgetItem(f"{R[i, j]:.8f}")
                else:
                    item = QTableWidgetItem("0")
                item.setFont(QFont("Courier New", 10))
                self.r_table.setItem(i, j, item)
        
        self.q_table.resizeColumnsToContents()
        self.r_table.resizeColumnsToContents()
        
        QR_product = Q @ R
        error_qr = np.linalg.norm(A - QR_product, np.inf)
        self.verification_label.setText(
            f"Проверка QR = A:\n"
            f"Максимальная погрешность: {error_qr:.2e}"
        )
        
        self.history_table.setRowCount(len(history))
        for i, h in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(h['iteration'])))
            
            norm_item = QTableWidgetItem(f"{h['off_diag_norm']:.2e}")
            if h['off_diag_norm'] < tolerance:
                norm_item.setForeground(QBrush(QColor(76, 175, 80)))
            self.history_table.setItem(i, 1, norm_item)
        
        self.history_table.resizeColumnsToContents()
        
        self.convergence_graph.plot_convergence(history, tolerance)


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("QR-алгоритм для нахождения собственных значений")
        self.setGeometry(100, 100, 1200, 900)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        
        # Создаем главный виджет и оборачиваем его в QScrollArea
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setSpacing(15)
        central_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("QR-алгоритм для нахождения собственных значений матриц 3x3")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333; margin: 15px;")
        central_layout.addWidget(title)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #e3f2fd;
                border-radius: 10px;
                padding: 15px;
            }
            QLabel {
                color: #1565c0;
                font-size: 13px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_label = QLabel(
            "QR-алгоритм - это итерационный метод для вычисления собственных значений матрицы.\n"
            "Алгоритм использует QR-разложение на каждой итерации: A_k = Q_k R_k, A_{k+1} = R_k Q_k\n"
            "Для матриц 3x3 алгоритм находит все собственные значения (действительные и комплексные)"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        central_layout.addWidget(info_frame)
        
        button_layout = QHBoxLayout()
        self.input_button = QPushButton("Ввести матрицу 3x3")
        self.input_button.clicked.connect(self.input_matrix)
        button_layout.addWidget(self.input_button)
        
        clear_button = QPushButton("Очистить все")
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(clear_button)
        central_layout.addLayout(button_layout)
        
        self.results_widget = ResultsWidget()
        central_layout.addWidget(self.results_widget, 1)
        
        # Создаем QScrollArea для всего содержимого
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll_area)
        
        self.statusBar().showMessage("Готов к работе")
    
    def input_matrix(self):
        """Открытие диалога ввода матрицы"""
        dialog = MatrixInputDialog()
        if dialog.exec_():
            A = dialog.get_matrix()
            if A is not None:
                self.A = A
                self.choose_tolerance()
    
    def choose_tolerance(self):
        """Выбор точности вычислений"""
        dialog = ToleranceDialog()
        if dialog.exec_():
            tolerance, max_iter = dialog.get_parameters()
            if tolerance is not None:
                self.compute_eigenvalues(tolerance, max_iter)
    
    def compute_eigenvalues(self, tolerance, max_iter):
        """Вычисление собственных значений"""
        try:
            self.statusBar().showMessage("Вычисление собственных значений...")
            
            eigenvalues, iterations, history, Ak_final, converged = QRAlgorithm.qr_algorithm_eigenvalues(
                self.A, tolerance, max_iter
            )
            
            self.results_widget.display_results(
                self.A, eigenvalues, iterations, history, Ak_final, converged, tolerance
            )
            
            if converged:
                self.statusBar().showMessage(f"Вычисления завершены успешно за {iterations} итераций")
            else:
                self.statusBar().showMessage(f"Достигнуто максимальное число итераций ({max_iter})")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при вычислении:\n{str(e)}")
            self.statusBar().showMessage("Ошибка при вычислении")
    
    def clear_results(self):
        """Очистка результатов"""
        self.results_widget.eigenvalues_table.setRowCount(0)
        self.results_widget.q_table.setRowCount(0)
        self.results_widget.r_table.setRowCount(0)
        self.results_widget.history_table.setRowCount(0)
        self.results_widget.convergence_label.setText("")
        self.results_widget.info_label.setText("")
        self.results_widget.verification_label.setText("")
        self.results_widget.convergence_graph.axes.clear()
        self.results_widget.convergence_graph.draw()
        self.statusBar().showMessage("Результаты очищены")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()