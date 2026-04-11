import sys
import numpy as np
import sympy as sp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class InterpolationSolver:
    """Класс для вычисления интерполяционных многочленов Лагранжа и Ньютона"""
    
    def __init__(self):
        self.x_sym = sp.Symbol('x')
        self.f_expr = None
        self.f_lambdified = None
        
    def set_function(self, f_str):
        """Установка функции y = f(x)"""
        try:
            local_dict = {
                'pi': sp.pi,
                'e': sp.E,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'exp': sp.exp,
                'log': sp.log,
                'sqrt': sp.sqrt,
            }
            self.f_expr = sp.sympify(f_str, locals=local_dict)
            self.f_lambdified = sp.lambdify(self.x_sym, self.f_expr, modules=['numpy', 'math'])
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def evaluate_function(self, x_vals):
        """Вычисление значений функции в точках x_vals"""
        try:
            y_vals = [float(self.f_lambdified(x)) for x in x_vals]
            return y_vals
        except:
            return None
    
    def lagrange_polynomial(self, x_points, y_points):
        """Построение многочлена Лагранжа в символьном виде"""
        n = len(x_points)
        x = self.x_sym
        L = 0
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if j != i:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            L += term
        return sp.simplify(L)
    
    def newton_polynomial(self, x_points, y_points):
        """Построение многочлена Ньютона с разделёнными разностями"""
        n = len(x_points)
        x = self.x_sym
        div_diff = [[0]*n for _ in range(n)]
        for i in range(n):
            div_diff[i][0] = y_points[i]
        for j in range(1, n):
            for i in range(n - j):
                div_diff[i][j] = (div_diff[i+1][j-1] - div_diff[i][j-1]) / (x_points[i+j] - x_points[i])
        coeffs = [div_diff[0][j] for j in range(n)]
        N = coeffs[0]
        prod = 1
        for i in range(1, n):
            prod *= (x - x_points[i-1])
            N += coeffs[i] * prod
        return sp.simplify(N)
    
    def interpolation_error(self, x_star, x_points, y_points, poly_expr):
        """Вычисление погрешности интерполяции: |f(x*) - P(x*)|"""
        try:
            f_star = float(self.f_lambdified(x_star))
            poly_lambdified = sp.lambdify(self.x_sym, poly_expr, modules=['numpy', 'math'])
            P_star = float(poly_lambdified(x_star))
            return abs(f_star - P_star)
        except:
            return None


class InterpolationGraph(FigureCanvas):
    """График исходной функции и интерполяционных многочленов"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(9, 6), dpi=100)
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
        
    def plot(self, f_lambdified, x_points, y_points, L_expr, N_expr, x_range=None, x_star=None):
        self.axes.clear()
        if f_lambdified is None:
            self.draw()
            return
        
        if x_range is None:
            x_min, x_max = min(x_points) - 1, max(x_points) + 1
        else:
            x_min, x_max = x_range
        x_plot = np.linspace(x_min, x_max, 500)
        
        try:
            y_plot = f_lambdified(x_plot)
            self.axes.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x) исходная')
        except:
            pass
        
        self.axes.plot(x_points, y_points, 'ro', markersize=8, label='Узлы интерполяции')
        
        if L_expr is not None:
            L_lambdified = sp.lambdify(sp.Symbol('x'), L_expr, modules=['numpy', 'math'])
            try:
                yL = L_lambdified(x_plot)
                self.axes.plot(x_plot, yL, 'b--', linewidth=2, label='Лагранж L(x)')
            except:
                pass
        
        if N_expr is not None:
            N_lambdified = sp.lambdify(sp.Symbol('x'), N_expr, modules=['numpy', 'math'])
            try:
                yN = N_lambdified(x_plot)
                self.axes.plot(x_plot, yN, 'g-.', linewidth=2, label='Ньютон N(x)')
            except:
                pass
        
        if x_star is not None:
            try:
                y_star = f_lambdified(x_star)
                self.axes.plot(x_star, y_star, 'ms', markersize=10, label=f'X* = {x_star:.4f}')
            except:
                pass
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_title('Интерполяционные многочлены')
        self.axes.legend(loc='best')
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = InterpolationSolver()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Интерполяция многочленами Лагранжа и Ньютона")
        self.setGeometry(100, 100, 1400, 950)
        
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
            QTabWidget::pane { border: 1px solid #cccccc; background: white; border-radius: 5px; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin: 2px; border-radius: 5px; color: #000000; }
            QTabBar::tab:selected { background-color: #2196F3; color: white; }
        """)
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Интерполяция многочленами Лагранжа и Ньютона по 4 точкам")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        func_group = QGroupBox("Функция y = f(x)")
        func_layout = QHBoxLayout()
        func_layout.addWidget(QLabel("f(x) ="))
        self.f_edit = QLineEdit()
        self.f_edit.setPlaceholderText("Например: sin(pi*x), exp(x), x**3 - 2*x + 1")
        func_layout.addWidget(self.f_edit)
        self.set_func_btn = QPushButton("Установить функцию")
        self.set_func_btn.clicked.connect(self.set_function)
        func_layout.addWidget(self.set_func_btn)
        func_group.setLayout(func_layout)
        main_layout.addWidget(func_group)
        
        self.task_tabs = QTabWidget()
        
        self.task_a_tab = QWidget()
        self.setup_task_a()
        self.task_tabs.addTab(self.task_a_tab, "Задача (а)")
        
        self.task_b_tab = QWidget()
        self.setup_task_b()
        self.task_tabs.addTab(self.task_b_tab, "Задача (б)")
        
        main_layout.addWidget(self.task_tabs)
        
        graph_group = QGroupBox("График")
        graph_layout = QVBoxLayout()
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("X min:"))
        self.xmin_edit = QLineEdit("-2")
        self.xmin_edit.setValidator(QDoubleValidator())
        range_layout.addWidget(self.xmin_edit)
        range_layout.addWidget(QLabel("X max:"))
        self.xmax_edit = QLineEdit("5")
        self.xmax_edit.setValidator(QDoubleValidator())
        range_layout.addWidget(self.xmax_edit)
        self.update_graph_btn = QPushButton("Обновить график")
        self.update_graph_btn.clicked.connect(self.update_graph)
        range_layout.addWidget(self.update_graph_btn)
        range_layout.addStretch()
        graph_layout.addLayout(range_layout)
        
        self.graph = InterpolationGraph(self)
        graph_layout.addWidget(self.graph)
        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group, 1)
        
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("background-color: white; color: #000000;")
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        btn_layout = QHBoxLayout()
        self.calc_a_btn = QPushButton("Вычислить для задачи (а)")
        self.calc_a_btn.clicked.connect(lambda: self.calculate('a'))
        btn_layout.addWidget(self.calc_a_btn)
        self.calc_b_btn = QPushButton("Вычислить для задачи (б)")
        self.calc_b_btn.clicked.connect(lambda: self.calculate('b'))
        btn_layout.addWidget(self.calc_b_btn)
        self.clear_btn = QPushButton("Очистить всё")
        self.clear_btn.setStyleSheet("background-color: #f44336;")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)
        main_layout.addLayout(btn_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll_area)
        
        self.statusBar().showMessage("Готов к работе")
    
    def setup_task_a(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        label = QLabel("Введите четыре значения Xi (через пробел или запятую).\nМожно использовать выражения: pi, 0.1*pi, 1/3, sqrt(2) и т.д.")
        layout.addWidget(label)
        self.x_a_edit = QLineEdit()
        self.x_a_edit.setPlaceholderText("0 0.1*pi 0.2*pi 0.3*pi")
        layout.addWidget(self.x_a_edit)
        self.table_a = QTableWidget()
        self.table_a.setColumnCount(2)
        self.table_a.setHorizontalHeaderLabels(["Xi", "Yi = f(Xi)"])
        self.table_a.setAlternatingRowColors(True)
        layout.addWidget(self.table_a)
        layout.addStretch()
        self.task_a_tab.setLayout(layout)
    
    def setup_task_b(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        label_x = QLabel("Введите четыре значения Xi (через пробел или запятую).\nМожно использовать выражения: pi, 0.1*pi, 1/3, sqrt(2) и т.д.")
        layout.addWidget(label_x)
        self.x_b_edit = QLineEdit()
        self.x_b_edit.setPlaceholderText("0 0.1*pi 0.2*pi 0.3*pi")
        layout.addWidget(self.x_b_edit)
        label_xstar = QLabel("Введите значение X* (можно выражение: 0.25*pi, 1/2, sqrt(3) и т.д.):")
        layout.addWidget(label_xstar)
        self.xstar_edit = QLineEdit()
        self.xstar_edit.setPlaceholderText("0.25*pi")
        # Убираем валидатор чисел, разрешаем любые символы
        layout.addWidget(self.xstar_edit)
        self.table_b = QTableWidget()
        self.table_b.setColumnCount(2)
        self.table_b.setHorizontalHeaderLabels(["Xi", "Yi = f(Xi)"])
        self.table_b.setAlternatingRowColors(True)
        layout.addWidget(self.table_b)
        layout.addStretch()
        self.task_b_tab.setLayout(layout)
    
    def set_function(self):
        f_str = self.f_edit.text().strip()
        if not f_str:
            QMessageBox.warning(self, "Предупреждение", "Введите функцию f(x)")
            return
        success, msg = self.solver.set_function(f_str)
        if not success:
            QMessageBox.critical(self, "Ошибка", f"Некорректное выражение:\n{msg}")
        else:
            self.statusBar().showMessage("Функция установлена")
    
    def parse_expression(self, text):
        """Преобразует строку с математическим выражением в число с помощью sympy."""
        try:
            local_dict = {'pi': sp.pi, 'e': sp.E}
            val = float(sp.sympify(text, locals=local_dict))
            return val
        except:
            return None
    
    def parse_x_values(self, text):
        """Парсинг строки с числами или выражениями (поддерживает pi, дроби и т.д.)"""
        text = text.replace(',', ' ')
        parts = text.split()
        values = []
        for p in parts:
            val = self.parse_expression(p)
            if val is None:
                return None
            values.append(val)
        return values
    
    def calculate(self, task):
        if self.solver.f_lambdified is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала установите функцию f(x)")
            return
        
        if task == 'a':
            x_str = self.x_a_edit.text().strip()
            x_vals = self.parse_x_values(x_str)
            if x_vals is None or len(x_vals) != 4:
                QMessageBox.warning(self, "Предупреждение", "Введите ровно 4 значения Xi (можно использовать выражения)")
                return
            x_star = None
            table = self.table_a
        else:
            x_str = self.x_b_edit.text().strip()
            x_vals = self.parse_x_values(x_str)
            if x_vals is None or len(x_vals) != 4:
                QMessageBox.warning(self, "Предупреждение", "Введите ровно 4 значения Xi (можно использовать выражения)")
                return
            x_star_str = self.xstar_edit.text().strip()
            if not x_star_str:
                QMessageBox.warning(self, "Предупреждение", "Введите значение X*")
                return
            x_star = self.parse_expression(x_star_str)
            if x_star is None:
                QMessageBox.warning(self, "Предупреждение", "Некорректное выражение для X*")
                return
            table = self.table_b
        
        y_vals = self.solver.evaluate_function(x_vals)
        if y_vals is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось вычислить значения функции")
            return
        
        table.setRowCount(4)
        for i in range(4):
            table.setItem(i, 0, QTableWidgetItem(f"{x_vals[i]:.6f}"))
            table.setItem(i, 1, QTableWidgetItem(f"{y_vals[i]:.6f}"))
        
        try:
            L_expr = self.solver.lagrange_polynomial(x_vals, y_vals)
            N_expr = self.solver.newton_polynomial(x_vals, y_vals)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении многочленов:\n{str(e)}")
            return
        
        results = f"Задача ({task})\n"
        results += f"Узлы интерполяции: Xi = {x_vals}\nYi = {[f'{y:.6f}' for y in y_vals]}\n\n"
        results += f"Многочлен Лагранжа L(x) = {L_expr}\n"
        results += f"Многочлен Ньютона N(x) = {N_expr}\n"
        results += f"Проверка совпадения: L(x) - N(x) = {sp.simplify(L_expr - N_expr)}\n\n"
        
        if x_star is not None:
            err_L = self.solver.interpolation_error(x_star, x_vals, y_vals, L_expr)
            err_N = self.solver.interpolation_error(x_star, x_vals, y_vals, N_expr)
            results += f"Точка X* = {x_star:.6f}\n"
            if err_L is not None:
                results += f"Погрешность Лагранжа |f(X*) - L(X*)| = {err_L:.6e}\n"
            if err_N is not None:
                results += f"Погрешность Ньютона |f(X*) - N(X*)| = {err_N:.6e}\n"
        else:
            results += "X* не задано, погрешность не вычислялась.\n"
        
        self.results_text.setText(results)
        
        self.current_L = L_expr
        self.current_N = N_expr
        self.current_x_points = x_vals
        self.current_y_points = y_vals
        self.current_x_star = x_star
        self.update_graph()
        
        self.statusBar().showMessage(f"Вычисление для задачи ({task}) завершено")
    
    def update_graph(self):
        if self.solver.f_lambdified is None:
            return
        try:
            x_min = float(self.xmin_edit.text())
            x_max = float(self.xmax_edit.text())
        except:
            x_min, x_max = -2, 5
        
        L_expr = getattr(self, 'current_L', None)
        N_expr = getattr(self, 'current_N', None)
        x_points = getattr(self, 'current_x_points', [])
        y_points = getattr(self, 'current_y_points', [])
        x_star = getattr(self, 'current_x_star', None)
        
        self.graph.plot(self.solver.f_lambdified, x_points, y_points, 
                        L_expr, N_expr, (x_min, x_max), x_star)
    
    def clear_all(self):
        self.f_edit.clear()
        self.x_a_edit.clear()
        self.x_b_edit.clear()
        self.xstar_edit.clear()
        self.table_a.setRowCount(0)
        self.table_b.setRowCount(0)
        self.results_text.clear()
        self.current_L = None
        self.current_N = None
        self.current_x_points = []
        self.current_y_points = []
        self.current_x_star = None
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