import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LeastSquaresSolver:
    """Класс для построения аппроксимирующих полиномов методом наименьших квадратов"""
    
    def __init__(self):
        self.x = None
        self.y = None
        self.coeffs_linear = None  # [a0, a1] для y = a0 + a1*x
        self.coeffs_quad = None    # [a0, a1, a2] для y = a0 + a1*x + a2*x^2
        self.error_linear = None
        self.error_quad = None
        
    def fit_polynomial(self, x, y, degree):
        """
        Построение полинома заданной степени методом МНК.
        Возвращает коэффициенты (от a0 до a_degree) и сумму квадратов ошибок.
        """
        n = len(x)
        # Построение матрицы Вандермонда
        A = np.zeros((n, degree + 1))
        for i in range(degree + 1):
            A[:, i] = np.array(x) ** i
        # Нормальная система: (A^T A) c = A^T y
        ATA = A.T @ A
        ATy = A.T @ np.array(y)
        coeffs = np.linalg.solve(ATA, ATy)
        # Вычисление ошибки
        y_pred = A @ coeffs
        error = np.sum((np.array(y) - y_pred) ** 2)
        return coeffs, error
    
    def build_models(self, x, y):
        """Построение линейной и квадратичной моделей"""
        self.x = np.array(x)
        self.y = np.array(y)
        self.coeffs_linear, self.error_linear = self.fit_polynomial(x, y, 1)
        self.coeffs_quad, self.error_quad = self.fit_polynomial(x, y, 2)
        
    def evaluate_linear(self, x_val):
        if self.coeffs_linear is None:
            return None
        return self.coeffs_linear[0] + self.coeffs_linear[1] * x_val
    
    def evaluate_quad(self, x_val):
        if self.coeffs_quad is None:
            return None
        return self.coeffs_quad[0] + self.coeffs_quad[1] * x_val + self.coeffs_quad[2] * x_val**2


class LeastSquaresGraph(FigureCanvas):
    """Виджет для отображения графика точек и аппроксимирующих полиномов"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(400)
        
    def plot(self, x_points, y_points, solver):
        self.axes.clear()
        if solver.coeffs_linear is None:
            self.draw()
            return
        
        # Точки
        self.axes.plot(x_points, y_points, 'ro', markersize=8, label='Исходные точки')
        
        # Линия для полиномов
        x_min, x_max = min(x_points), max(x_points)
        margin = (x_max - x_min) * 0.1
        x_plot = np.linspace(x_min - margin, x_max + margin, 500)
        
        # Линейный
        y_linear = [solver.evaluate_linear(xi) for xi in x_plot]
        self.axes.plot(x_plot, y_linear, 'b--', linewidth=2, label='Линейный (1-й степени)')
        
        # Квадратичный
        y_quad = [solver.evaluate_quad(xi) for xi in x_plot]
        self.axes.plot(x_plot, y_quad, 'g-', linewidth=2, label='Квадратичный (2-й степени)')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_title('Аппроксимация методом наименьших квадратов')
        self.axes.legend(loc='best')
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = LeastSquaresSolver()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Метод наименьших квадратов (полиномы 1-й и 2-й степени)")
        self.setGeometry(100, 100, 1300, 950)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QPushButton { background-color: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #1976D2; }
            QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 8px; margin-top: 20px; padding-top: 15px; background-color: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 20px; padding: 0 10px; color: #2196F3; background-color: white; }
            QLabel { font-size: 12px; color: #333333; }
            QLineEdit { padding: 8px; border: 2px solid #cccccc; border-radius: 5px; font-size: 12px; background-color: white; color: #000000; }
            QLineEdit:focus { border: 2px solid #2196F3; }
            QTableWidget { gridline-color: #dddddd; font-size: 12px; background-color: white; alternate-background-color: #f9f9f9; }
            QTableWidget::item { color: #000000; }
            QHeaderView::section { background-color: #2196F3; color: white; padding: 8px; font-weight: bold; }
        """)
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Аппроксимация функции полиномами 1-й и 2-й степени (МНК)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Панель ввода данных
        input_group = QGroupBox("Входные данные (6 точек)")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        
        table_label = QLabel("Введите значения Xi и Yi (ровно 6 точек):")
        input_layout.addWidget(table_label)
        
        self.input_table = QTableWidget()
        self.input_table.setRowCount(6)
        self.input_table.setColumnCount(2)
        self.input_table.setHorizontalHeaderLabels(["Xi", "Yi"])
        self.input_table.setAlternatingRowColors(True)
        self.input_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_table.horizontalHeader().setStretchLastSection(True)
        self.input_table.verticalHeader().setDefaultSectionSize(35)
        self.input_table.setMinimumHeight(260)
        
        validator = QDoubleValidator()
        for i in range(6):
            for j in range(2):
                editor = QLineEdit()
                editor.setValidator(validator)
                self.input_table.setCellWidget(i, j, editor)
        
        input_layout.addWidget(self.input_table)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Вычислить аппроксимацию")
        self.calc_btn.clicked.connect(self.calculate)
        btn_layout.addWidget(self.calc_btn)
        
        self.clear_btn = QPushButton("Очистить всё")
        self.clear_btn.setStyleSheet("background-color: #f44336;")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)
        input_layout.addLayout(btn_layout)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # График
        graph_group = QGroupBox("График")
        graph_layout = QVBoxLayout()
        self.graph = LeastSquaresGraph(self)
        graph_layout.addWidget(self.graph)
        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group, 1)
        
        # Результаты
        results_group = QGroupBox("Результаты аппроксимации")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(15)
        
        # Линейная модель
        linear_group = QGroupBox("Полином 1-й степени")
        linear_group.setStyleSheet("QGroupBox { font-weight: bold; color: #1565c0; }")
        linear_layout = QVBoxLayout()
        
        self.linear_eq_label = QLabel()
        self.linear_eq_label.setFont(QFont("Arial", 11, QFont.Bold))
        linear_layout.addWidget(self.linear_eq_label)
        
        self.linear_error_label = QLabel()
        self.linear_error_label.setFont(QFont("Arial", 10))
        linear_layout.addWidget(self.linear_error_label)
        
        linear_group.setLayout(linear_layout)
        results_layout.addWidget(linear_group)
        
        # Квадратичная модель
        quad_group = QGroupBox("Полином 2-й степени")
        quad_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2e7d32; }")
        quad_layout = QVBoxLayout()
        
        self.quad_eq_label = QLabel()
        self.quad_eq_label.setFont(QFont("Arial", 11, QFont.Bold))
        quad_layout.addWidget(self.quad_eq_label)
        
        self.quad_error_label = QLabel()
        self.quad_error_label.setFont(QFont("Arial", 10))
        quad_layout.addWidget(self.quad_error_label)
        
        quad_group.setLayout(quad_layout)
        results_layout.addWidget(quad_group)
        
        # Таблица коэффициентов
        coeffs_label = QLabel("Коэффициенты полиномов:")
        coeffs_label.setFont(QFont("Arial", 11, QFont.Bold))
        results_layout.addWidget(coeffs_label)
        
        self.coeffs_table = QTableWidget()
        self.coeffs_table.setColumnCount(4)
        self.coeffs_table.setHorizontalHeaderLabels(["Степень", "a0", "a1", "a2"])
        self.coeffs_table.setAlternatingRowColors(True)
        self.coeffs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.coeffs_table.horizontalHeader().setStretchLastSection(True)
        self.coeffs_table.setMinimumHeight(100)
        results_layout.addWidget(self.coeffs_table)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Скролл
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll_area)
        
        self.statusBar().showMessage("Готов к работе")
    
    def get_table_data(self):
        """Извлечение данных из таблицы ввода"""
        x_vals = []
        y_vals = []
        for i in range(6):
            editor_x = self.input_table.cellWidget(i, 0)
            editor_y = self.input_table.cellWidget(i, 1)
            if editor_x is None or editor_y is None:
                return None, None
            try:
                x = float(editor_x.text())
                y = float(editor_y.text())
                x_vals.append(x)
                y_vals.append(y)
            except ValueError:
                return None, None
        return x_vals, y_vals
    
    def format_polynomial(self, coeffs):
        """Форматирование полинома в строку"""
        terms = []
        for i, c in enumerate(coeffs):
            if i == 0:
                terms.append(f"{c:.6f}")
            elif i == 1:
                terms.append(f"{c:+.6f}·x")
            else:
                terms.append(f"{c:+.6f}·x^{i}")
        return " ".join(terms)
    
    def calculate(self):
        x_vals, y_vals = self.get_table_data()
        if x_vals is None:
            QMessageBox.warning(self, "Предупреждение", "Заполните все ячейки таблицы корректными числами")
            return
        
        if len(set(x_vals)) != 6:
            reply = QMessageBox.question(self, "Предупреждение", 
                                        "Значения Xi не все различны. Продолжить?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        try:
            self.solver.build_models(x_vals, y_vals)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить модели:\n{str(e)}")
            return
        
        # Отображение результатов
        # Линейная модель
        a0, a1 = self.solver.coeffs_linear
        self.linear_eq_label.setText(f"P₁(x) = {self.format_polynomial([a0, a1])}")
        self.linear_error_label.setText(f"Сумма квадратов ошибок: {self.solver.error_linear:.6e}")
        
        # Квадратичная модель
        a0, a1, a2 = self.solver.coeffs_quad
        self.quad_eq_label.setText(f"P₂(x) = {self.format_polynomial([a0, a1, a2])}")
        self.quad_error_label.setText(f"Сумма квадратов ошибок: {self.solver.error_quad:.6e}")
        
        # Таблица коэффициентов
        self.coeffs_table.setRowCount(2)
        # Линейная
        self.coeffs_table.setItem(0, 0, QTableWidgetItem("1"))
        self.coeffs_table.setItem(0, 1, QTableWidgetItem(f"{self.solver.coeffs_linear[0]:.8f}"))
        self.coeffs_table.setItem(0, 2, QTableWidgetItem(f"{self.solver.coeffs_linear[1]:.8f}"))
        self.coeffs_table.setItem(0, 3, QTableWidgetItem("—"))
        # Квадратичная
        self.coeffs_table.setItem(1, 0, QTableWidgetItem("2"))
        self.coeffs_table.setItem(1, 1, QTableWidgetItem(f"{self.solver.coeffs_quad[0]:.8f}"))
        self.coeffs_table.setItem(1, 2, QTableWidgetItem(f"{self.solver.coeffs_quad[1]:.8f}"))
        self.coeffs_table.setItem(1, 3, QTableWidgetItem(f"{self.solver.coeffs_quad[2]:.8f}"))
        self.coeffs_table.resizeColumnsToContents()
        
        # График
        self.graph.plot(x_vals, y_vals, self.solver)
        
        self.statusBar().showMessage("Аппроксимация выполнена")
    
    def clear_all(self):
        for i in range(6):
            for j in range(2):
                editor = self.input_table.cellWidget(i, j)
                if editor:
                    editor.clear()
        self.linear_eq_label.setText("")
        self.linear_error_label.setText("")
        self.quad_eq_label.setText("")
        self.quad_error_label.setText("")
        self.coeffs_table.setRowCount(0)
        self.graph.axes.clear()
        self.graph.draw()
        self.statusBar().showMessage("Очищено")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()