import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SplineSolver:
    """Класс для построения естественного кубического сплайна"""
    
    def __init__(self):
        self.x = None
        self.y = None
        self.M = None  # Моменты (вторые производные)
        self.h = None
        self.coeffs = None  # Коэффициенты полиномов на отрезках [a,b,c,d]
        
    def build_spline(self, x, y):
        """
        Построение естественного кубического сплайна для n+1 узлов (n=4 для 5 точек).
        x: массив узлов (5 значений)
        y: массив значений функции в узлах
        """
        n = len(x) - 1  # n = 4
        h = np.diff(x)
        self.h = h
        self.x = np.array(x)
        self.y = np.array(y)
        
        # Построение трёхдиагональной системы для моментов M_i (i=0..n)
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)
        
        # Естественные граничные условия: M0 = 0, Mn = 0
        A[0, 0] = 1.0
        b[0] = 0.0
        A[n, n] = 1.0
        b[n] = 0.0
        
        for i in range(1, n):
            A[i, i-1] = h[i-1] / 6.0
            A[i, i]   = (h[i-1] + h[i]) / 3.0
            A[i, i+1] = h[i] / 6.0
            b[i] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
        
        self.M = np.linalg.solve(A, b)
        
        # Вычисление коэффициентов сплайна для каждого из n отрезков
        self.coeffs = []
        for i in range(n):
            a = y[i]
            c = self.M[i] / 2.0
            d = (self.M[i+1] - self.M[i]) / (6.0 * h[i])
            b_coef = (y[i+1] - y[i]) / h[i] - h[i] * (2.0*self.M[i] + self.M[i+1]) / 6.0
            self.coeffs.append([a, b_coef, c, d])
        
        return self.coeffs
    
    def evaluate(self, x_val):
        """Вычисление значения сплайна в точке x_val"""
        if self.coeffs is None:
            return None
        x = self.x
        idx = np.searchsorted(x, x_val) - 1
        if idx < 0:
            idx = 0
        if idx >= len(x) - 1:
            idx = len(x) - 2
        h = x[idx+1] - x[idx]
        dx = x_val - x[idx]
        a, b, c, d = self.coeffs[idx]
        return a + b*dx + c*dx**2 + d*dx**3


class SplineGraph(FigureCanvas):
    """Виджет для отображения графика сплайна и узлов"""
    
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
        
    def plot_spline(self, solver, x_points, y_points, x_star=None):
        self.axes.clear()
        if solver.coeffs is None:
            self.draw()
            return
        
        x_min, x_max = min(x_points), max(x_points)
        margin = (x_max - x_min) * 0.1
        x_plot = np.linspace(x_min - margin, x_max + margin, 500)
        
        y_plot = [solver.evaluate(xi) for xi in x_plot]
        self.axes.plot(x_plot, y_plot, 'b-', linewidth=2, label='Кубический сплайн')
        self.axes.plot(x_points, y_points, 'ro', markersize=8, label='Узлы интерполяции')
        
        if x_star is not None:
            y_star = solver.evaluate(x_star)
            self.axes.plot(x_star, y_star, 'ms', markersize=10, label=f'X* = {x_star:.4f}')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_title('Естественный кубический сплайн (5 узлов)')
        self.axes.legend(loc='best')
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = SplineSolver()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Построение кубического сплайна (5 узлов)")
        self.setGeometry(100, 100, 1250, 950)
        
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
        
        title = QLabel("Построение естественного кубического сплайна по 5 узлам")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Панель ввода данных
        input_group = QGroupBox("Входные данные")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        
        # X*
        xstar_layout = QHBoxLayout()
        xstar_layout.addWidget(QLabel("Точка X*:"))
        self.xstar_edit = QLineEdit()
        self.xstar_edit.setPlaceholderText("Например: 1.5")
        xstar_layout.addWidget(self.xstar_edit)
        xstar_layout.addStretch()
        input_layout.addLayout(xstar_layout)
        
        # Таблица узлов (5 строк)
        table_label = QLabel("Введите значения Xi и Yi (ровно 5 точек):")
        input_layout.addWidget(table_label)
        
        self.input_table = QTableWidget()
        self.input_table.setRowCount(5)
        self.input_table.setColumnCount(2)
        self.input_table.setHorizontalHeaderLabels(["Xi", "Yi"])
        self.input_table.setAlternatingRowColors(True)
        self.input_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_table.horizontalHeader().setStretchLastSection(True)
        self.input_table.verticalHeader().setDefaultSectionSize(35)
        self.input_table.setMinimumHeight(220)
        
        validator = QDoubleValidator()
        for i in range(5):
            for j in range(2):
                editor = QLineEdit()
                editor.setValidator(validator)
                self.input_table.setCellWidget(i, j, editor)
        
        input_layout.addWidget(self.input_table)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Построить сплайн и вычислить S(X*)")
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
        graph_group = QGroupBox("График сплайна")
        graph_layout = QVBoxLayout()
        self.graph = SplineGraph(self)
        graph_layout.addWidget(self.graph)
        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group, 1)
        
        # Результаты
        results_group = QGroupBox("Результаты вычислений")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(15)
        
        # Блок с основной информацией
        info_frame = QFrame()
        info_frame.setStyleSheet("QFrame { background-color: #e8f5e9; border-radius: 8px; padding: 10px; }")
        info_layout = QVBoxLayout(info_frame)
        
        self.x_star_label = QLabel()
        self.x_star_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.x_star_label.setStyleSheet("color: #1b5e20;")
        info_layout.addWidget(self.x_star_label)
        
        self.s_value_label = QLabel()
        self.s_value_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.s_value_label.setStyleSheet("color: #1b5e20;")
        info_layout.addWidget(self.s_value_label)
        
        results_layout.addWidget(info_frame)
        
        # Таблица узлов (после сортировки)
        nodes_label = QLabel("Узлы интерполяции (отсортированы):")
        nodes_label.setFont(QFont("Arial", 11, QFont.Bold))
        results_layout.addWidget(nodes_label)
        
        self.nodes_table = QTableWidget()
        self.nodes_table.setColumnCount(2)
        self.nodes_table.setHorizontalHeaderLabels(["Xi", "Yi"])
        self.nodes_table.setAlternatingRowColors(True)
        self.nodes_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.nodes_table.horizontalHeader().setStretchLastSection(True)
        self.nodes_table.setMinimumHeight(150)
        results_layout.addWidget(self.nodes_table)
        
        # Таблица коэффициентов сплайна
        coeffs_label = QLabel("Коэффициенты кубических полиномов на отрезках:")
        coeffs_label.setFont(QFont("Arial", 11, QFont.Bold))
        results_layout.addWidget(coeffs_label)
        
        self.coeffs_table = QTableWidget()
        self.coeffs_table.setColumnCount(5)
        self.coeffs_table.setHorizontalHeaderLabels(["Отрезок", "a", "b", "c", "d"])
        self.coeffs_table.setAlternatingRowColors(True)
        self.coeffs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.coeffs_table.horizontalHeader().setStretchLastSection(True)
        self.coeffs_table.setMinimumHeight(150)
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
        for i in range(5):
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
    
    def calculate(self):
        # Получение X*
        try:
            x_star = float(self.xstar_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Предупреждение", "Введите корректное числовое значение X*")
            return
        
        # Получение данных из таблицы
        x_vals, y_vals = self.get_table_data()
        if x_vals is None:
            QMessageBox.warning(self, "Предупреждение", "Заполните все ячейки таблицы корректными числами")
            return
        
        # Сортировка по возрастанию x
        points = sorted(zip(x_vals, y_vals))
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Проверка уникальности x
        if len(set(x_vals)) != 5:
            QMessageBox.warning(self, "Предупреждение", "Значения Xi должны быть различными")
            return
        
        # Построение сплайна
        try:
            self.solver.build_spline(x_vals, y_vals)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить сплайн:\n{str(e)}")
            return
        
        # Вычисление значения в X*
        S_star = self.solver.evaluate(x_star)
        
        # Отображение основной информации
        self.x_star_label.setText(f"Точка X* = {x_star:.6f}")
        self.s_value_label.setText(f"Значение сплайна S(X*) = {S_star:.10f}")
        
        # Заполнение таблицы узлов
        self.nodes_table.setRowCount(5)
        for i in range(5):
            self.nodes_table.setItem(i, 0, QTableWidgetItem(f"{x_vals[i]:.6f}"))
            self.nodes_table.setItem(i, 1, QTableWidgetItem(f"{y_vals[i]:.6f}"))
        self.nodes_table.resizeColumnsToContents()
        
        # Заполнение таблицы коэффициентов
        self.coeffs_table.setRowCount(4)
        for i in range(4):
            a, b, c, d = self.solver.coeffs[i]
            segment = f"[{x_vals[i]:.4f}, {x_vals[i+1]:.4f}]"
            self.coeffs_table.setItem(i, 0, QTableWidgetItem(segment))
            self.coeffs_table.setItem(i, 1, QTableWidgetItem(f"{a:.8f}"))
            self.coeffs_table.setItem(i, 2, QTableWidgetItem(f"{b:.8f}"))
            self.coeffs_table.setItem(i, 3, QTableWidgetItem(f"{c:.8f}"))
            self.coeffs_table.setItem(i, 4, QTableWidgetItem(f"{d:.8f}"))
        self.coeffs_table.resizeColumnsToContents()
        
        # График
        self.graph.plot_spline(self.solver, x_vals, y_vals, x_star)
        self.statusBar().showMessage("Сплайн построен, значение вычислено")
    
    def clear_all(self):
        self.xstar_edit.clear()
        for i in range(5):
            for j in range(2):
                editor = self.input_table.cellWidget(i, j)
                if editor:
                    editor.clear()
        self.nodes_table.setRowCount(0)
        self.coeffs_table.setRowCount(0)
        self.x_star_label.setText("")
        self.s_value_label.setText("")
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