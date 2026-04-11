import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator, QColor, QBrush

class LUDecomposition:
    """Класс для LU-разложения с выбором главного элемента"""
    
    @staticmethod
    def lu_decomposition_with_pivoting(A):
        n = len(A)
        U = np.array(A, dtype=float)
        L = np.eye(n, dtype=float)
        P = np.eye(n, dtype=float)
        
        for k in range(n - 1):
            pivot_row = np.argmax(np.abs(U[k:, k])) + k
            
            if abs(U[pivot_row, k]) < 1e-10:
                raise ValueError("Матрица вырождена или близка к вырожденной")
            
            if pivot_row != k:
                U[[k, pivot_row]] = U[[pivot_row, k]]
                P[[k, pivot_row]] = P[[pivot_row, k]]
                if k > 0:
                    L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]
        
        return L, U, P
    
    @staticmethod
    def solve_system(A, b):
        n = len(A)
        L, U, P = LUDecomposition.lu_decomposition_with_pivoting(A)
        
        b_permuted = np.dot(P, b)
        
        y = np.zeros(n)
        for i in range(n):
            y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])
        
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < 1e-10:
                raise ValueError("Вырожденная матрица - решение не существует")
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        
        return x
    
    @staticmethod
    def calculate_determinant(A):
        L, U, P = LUDecomposition.lu_decomposition_with_pivoting(A)
        det_P = np.linalg.det(P)
        det_U = np.prod(np.diag(U))
        return det_P * det_U
    
    @staticmethod
    def inverse_matrix(A):
        n = len(A)
        inv = np.zeros((n, n))
        
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            inv[:, i] = LUDecomposition.solve_system(A, e)
        
        return inv


class MatrixInputDialog(QDialog):
    """Диалог для ввода матрицы и правой части"""
    
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
        
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Решить систему")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(650, 350)
    
    def get_system(self):
        try:
            A = np.zeros((self.n, self.n))
            b = np.zeros(self.n)
            
            for i in range(self.n):
                for j in range(self.n):
                    text = self.matrix_inputs[i][j].text()
                    if text.strip():
                        A[i, j] = float(text)
                    else:
                        A[i, j] = 0
                
                text = self.vector_inputs[i].text()
                if text.strip():
                    b[i] = float(text)
                else:
                    b[i] = 0
            
            return A, b
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат числа")
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
        self.tab_widget.addTab(self.solution_tab, "Решение системы")
        
        self.verification_tab = QWidget()
        self.setup_verification_tab()
        self.tab_widget.addTab(self.verification_tab, "Проверка")
        
        self.det_tab = QWidget()
        self.setup_det_tab()
        self.tab_widget.addTab(self.det_tab, "Определитель")
        
        self.inv_tab = QWidget()
        self.setup_inv_tab()
        self.tab_widget.addTab(self.inv_tab, "Обратная матрица")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def setup_solution_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Решение системы линейных уравнений")
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
    
    def setup_verification_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Проверка решения (невязка)")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff9800; margin: 10px;")
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
                background-color: #ff9800;
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
        
        title = QLabel("Определитель матрицы")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2196F3; margin: 10px;")
        layout.addWidget(title)
        
        self.det_value = QLabel()
        self.det_value.setAlignment(Qt.AlignCenter)
        self.det_value.setFont(QFont("Arial", 24, QFont.Bold))
        self.det_value.setStyleSheet("color: #2196F3; padding: 20px; background-color: #e3f2fd; border-radius: 10px;")
        layout.addWidget(self.det_value)
        
        self.det_warning = QLabel()
        self.det_warning.setAlignment(Qt.AlignCenter)
        self.det_warning.setWordWrap(True)
        layout.addWidget(self.det_warning)
        
        layout.addStretch()
        self.det_tab.setLayout(layout)
    
    def setup_inv_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("Обратная матрица A^{-1}")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #9C27B0; margin: 10px;")
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
                background-color: #9C27B0;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.inv_table)
        
        self.verification_label = QLabel()
        self.verification_label.setAlignment(Qt.AlignCenter)
        self.verification_label.setStyleSheet("padding: 10px; background-color: #f3e5f5; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(self.verification_label)
        
        self.inv_tab.setLayout(layout)
    
    def display_results(self, x, residual, det, inv_A):
        self.solution_table.setRowCount(len(x))
        for i, val in enumerate(x):
            var_item = QTableWidgetItem(f"x{i+1}")
            var_item.setFont(QFont("Arial", 12, QFont.Bold))
            val_item = QTableWidgetItem(f"{val:.6f}")
            val_item.setFont(QFont("Courier New", 12))
            
            if val > 0:
                val_item.setForeground(QBrush(QColor(76, 175, 80)))
            elif val < 0:
                val_item.setForeground(QBrush(QColor(244, 67, 54)))
            
            self.solution_table.setItem(i, 0, var_item)
            self.solution_table.setItem(i, 1, val_item)
        
        self.solution_table.resizeColumnsToContents()
        
        self.clear_bars()
        max_val = max(abs(x)) if len(x) > 0 else 1
        for i, val in enumerate(x):
            bar_frame = QFrame()
            bar_layout = QHBoxLayout()
            label = QLabel(f"x{i+1}")
            label.setMinimumWidth(40)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(abs(val) / max_val * 100))
            bar.setFormat(f"{val:.3f}")
            
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
        
        self.det_value.setText(f"det(A) = {det:.8f}")
        if abs(det) < 1e-10:
            self.det_warning.setText("Внимание! Определитель близок к нулю, матрица вырождена")
            self.det_warning.setStyleSheet("color: #f44336; font-weight: bold;")
        else:
            self.det_warning.setText("Матрица невырождена, решение существует и единственно")
            self.det_warning.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        n = len(inv_A)
        self.inv_table.setRowCount(n)
        self.inv_table.setColumnCount(n)
        self.inv_table.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n)])
        
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{inv_A[i, j]:.6f}")
                item.setTextAlignment(Qt.AlignCenter)
                self.inv_table.setItem(i, j, item)
        
        self.inv_table.resizeColumnsToContents()
        self.verification_label.setText("Проверка: A * A^{-1} должна быть единичной матрицей")
    
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
        self.setWindowTitle("Решение СЛАУ методом LU-разложения")
        self.setGeometry(100, 100, 1000, 800)
        
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
        
        title = QLabel("Решение систем линейных алгебраических уравнений")
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
            "Метод решения: LU-разложение с выбором главного элемента по столбцу\n"
            "Размерность: 4 x 4\n"
            "Высокая точность вычислений"
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
                self.solve_and_display(A, b)
    
    def solve_and_display(self, A, b):
        try:
            x = LUDecomposition.solve_system(A, b)
            det = LUDecomposition.calculate_determinant(A)
            inv_A = LUDecomposition.inverse_matrix(A)
            residual = np.dot(A, x) - b
            
            self.results_widget.display_results(x, residual, det, inv_A)
            self.statusBar().showMessage("Решение успешно найдено")
            
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", f"{str(e)}")
            self.statusBar().showMessage("Ошибка при решении")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Непредвиденная ошибка:\n{str(e)}")
            self.statusBar().showMessage("Ошибка при решении")
    
    def clear_results(self):
        self.results_widget.solution_table.setRowCount(0)
        self.results_widget.residual_table.setRowCount(0)
        self.results_widget.inv_table.setRowCount(0)
        self.results_widget.det_value.setText("—")
        self.results_widget.det_warning.setText("")
        self.results_widget.residual_status.setText("")
        self.results_widget.verification_label.setText("")
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