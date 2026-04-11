import sys
import numpy as np
import sympy as sp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class EquationSolver:
    """Класс для работы с уравнением и методами решения"""
    
    def __init__(self):
        self.x_symbol = sp.Symbol('x')
        self.f_expr = None
        self.phi_expr = None
        self.f_lambdified = None
        self.f_prime_lambdified = None
        self.phi_lambdified = None
        self.phi_options = []  # Список вариантов phi(x)
        
    def set_equation(self, f_str):
        """Установка уравнения f(x)=0"""
        try:
            self.f_expr = sp.sympify(f_str, evaluate=False)
            self.f_lambdified = sp.lambdify(self.x_symbol, self.f_expr, modules=['numpy', 'math'])
            # Вычисляем производную
            f_prime_expr = sp.diff(self.f_expr, self.x_symbol)
            self.f_prime_lambdified = sp.lambdify(self.x_symbol, f_prime_expr, modules=['numpy', 'math'])
            # Генерируем варианты phi(x)
            self.generate_phi_options()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def generate_phi_options(self):
        """Генерация возможных выражений phi(x) из f(x)=0 (улучшенная версия)"""
        self.phi_options = []
        if self.f_expr is None:
            return
        
        x = self.x_symbol
        f = self.f_expr
        
        # 1. Стандартный релаксационный: x = x - f(x)
        phi1 = x - f
        self.phi_options.append(("Релаксация: x - f(x)", phi1))
        
        # 2. x = x - f(x)/f'(x) (по сути Ньютон, но как итерационная функция)
        f_prime = sp.diff(f, x)
        if f_prime != 0:
            phi2 = x - f / f_prime
            self.phi_options.append(("Ньютоновский: x - f(x)/f'(x)", phi2))
        
        # 3. Попытка решить уравнение аналитически (может не сработать)
        try:
            solutions = sp.solve(f, x)
            for sol in solutions:
                if not sol.has(sp.I):  # без комплексных
                    self.phi_options.append((f"Аналитическое: x = {sol}", sol))
        except:
            pass
        
        # 4. Эвристики: разложение на слагаемые и попытка выразить x
        try:
            expanded = f.expand()
        except:
            expanded = f
        terms = list(sp.Add.make_args(expanded))
        
        for i, term in enumerate(terms):
            rest = -sum(t for j, t in enumerate(terms) if j != i)
            
            if term.is_Pow and term.args[0] == x:
                n = term.args[1]
                if n != 0:
                    phi_candidate = sp.Pow(rest, 1/n)
                    if not phi_candidate.has(sp.I):
                        self.phi_options.append((f"Из x**{n} = {rest}", phi_candidate))
            elif term.is_Mul:
                coeff, x_part = term.as_coeff_Mul()
                if x_part == x:
                    phi_candidate = rest / coeff
                    self.phi_options.append((f"Из {term} = {rest}", phi_candidate))
            elif (term.is_Pow and term.args[0] != x) or (hasattr(term, 'func') and term.func == sp.exp):
                base = term.args[0] if term.is_Pow else sp.E
                exponent = term.args[1] if term.is_Pow else term.args[0]
                if exponent.has(x):
                    if base == sp.E:
                        log_base = 1
                    else:
                        log_base = sp.log(base)
                    new_eq = sp.Eq(exponent, sp.log(rest) / log_base)
                    try:
                        sol = sp.solve(new_eq, x)
                        for s in sol:
                            if not s.has(sp.I):
                                self.phi_options.append((f"Из {term} = {rest}", s))
                    except:
                        pass
            elif hasattr(term, 'func') and term.func == sp.log:
                inner = term.args[0]
                if inner.has(x):
                    new_eq = sp.Eq(inner, sp.exp(rest))
                    try:
                        sol = sp.solve(new_eq, x)
                        for s in sol:
                            if not s.has(sp.I):
                                self.phi_options.append((f"Из log({inner}) = {rest}", s))
                    except:
                        pass
            elif hasattr(term, 'func') and term.func in (sp.sin, sp.cos, sp.tan):
                inv_map = {sp.sin: sp.asin, sp.cos: sp.acos, sp.tan: sp.atan}
                inv_func = inv_map.get(term.func)
                if inv_func:
                    inner = term.args[0]
                    if inner.has(x):
                        new_eq = sp.Eq(inner, inv_func(rest))
                        try:
                            sol = sp.solve(new_eq, x)
                            for s in sol:
                                if not s.has(sp.I):
                                    self.phi_options.append((f"Из {term.func}({inner}) = {rest}", s))
                        except:
                            pass
        
        if len(self.phi_options) <= 1:
            for c in [0.5, 0.1, 1.5]:
                phi_c = x - c * f
                self.phi_options.append((f"Релаксация: x - {c}*f(x)", phi_c))
        
        seen = set()
        unique_options = []
        for name, expr in self.phi_options:
            key = str(expr)
            if key not in seen:
                seen.add(key)
                unique_options.append((name, expr))
        self.phi_options = unique_options
    
    def set_phi(self, phi_expr):
        """Установка выбранного выражения phi(x)"""
        try:
            self.phi_expr = phi_expr
            self.phi_lambdified = sp.lambdify(self.x_symbol, self.phi_expr, modules=['numpy', 'math'])
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def f(self, x):
        try:
            return float(self.f_lambdified(x))
        except:
            return None
    
    def f_prime(self, x):
        try:
            return float(self.f_prime_lambdified(x))
        except:
            return None
    
    def phi(self, x):
        try:
            return float(self.phi_lambdified(x))
        except:
            return None
    
    def check_convergence_condition(self, x0, delta=0.1):
        if self.phi_expr is None:
            return None
        try:
            phi_prime_expr = sp.diff(self.phi_expr, self.x_symbol)
            phi_prime_lambdified = sp.lambdify(self.x_symbol, phi_prime_expr, modules=['numpy', 'math'])
            val = abs(phi_prime_lambdified(x0))
            return val
        except:
            return None
    
    def newton_method(self, x0, tolerance, max_iter):
        history = []
        fx0 = self.f(x0)
        history.append({
            'iteration': 0,
            'x': x0,
            'fx': fx0,
            'error': None
        })
        x = x0
        for i in range(max_iter):
            fx = self.f(x)
            fpx = self.f_prime(x)
            if fx is None or fpx is None or abs(fpx) < 1e-15:
                break
            x_new = x - fx / fpx
            error = abs(x_new - x)
            fx_new = self.f(x_new)
            history.append({
                'iteration': i + 1,
                'x': x_new,
                'fx': fx_new,
                'error': error
            })
            if error < tolerance:
                return x_new, i+1, history, True
            x = x_new
        return x, max_iter, history, False
    
    def simple_iteration_method(self, x0, tolerance, max_iter):
        if self.phi_lambdified is None:
            return None, 0, [], False
        history = []
        fx0 = self.f(x0) if self.f_lambdified else None
        history.append({
            'iteration': 0,
            'x': x0,
            'fx': fx0,
            'error': None
        })
        x = x0
        for i in range(max_iter):
            try:
                x_new = self.phi(x)
            except:
                break
            error = abs(x_new - x)
            fx_new = self.f(x_new) if self.f_lambdified else None
            history.append({
                'iteration': i + 1,
                'x': x_new,
                'fx': fx_new,
                'error': error
            })
            if error < tolerance:
                return x_new, i+1, history, True
            x = x_new
        return x, max_iter, history, False


class FunctionGraph(FigureCanvas):
    """Виджет для отображения графика функции с итерациями"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('f(x)')
        self.axes.set_title('График функции f(x)')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        
        self.mpl_connect('button_press_event', self.on_click)
        self.click_callback = None
        
    def plot_function(self, f_lambdified, x_min, x_max, history=None, num_points=1000):
        self.axes.clear()
        if f_lambdified is None:
            self.draw()
            return
        
        x = np.linspace(x_min, x_max, num_points)
        try:
            y = f_lambdified(x)
            y = np.array([float(val) if np.isreal(val) else np.nan for val in y])
        except:
            y = np.full_like(x, np.nan)
        
        self.axes.plot(x, y, 'b-', linewidth=2, label='f(x)')
        self.axes.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        if history and len(history) > 0:
            xs = [h['x'] for h in history]
            ys = [h['fx'] for h in history]
            # Фильтруем None
            points = [(x_val, y_val) for x_val, y_val in zip(xs, ys) if y_val is not None]
            if points:
                xs_f, ys_f = zip(*points)
                self.axes.plot(xs_f, ys_f, 'g-o', linewidth=2, markersize=5, label='Итерации')
                if len(xs_f) > 0:
                    self.axes.plot(xs_f[0], ys_f[0], 'go', markersize=10, label='Начало')
                if len(xs_f) > 1:
                    self.axes.plot(xs_f[-1], ys_f[-1], 'r*', markersize=12, label='Решение')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('f(x)')
        self.axes.set_title('График функции f(x) с итерациями')
        self.axes.legend()
        self.figure.tight_layout()
        self.draw()
    
    def on_click(self, event):
        if event.xdata is not None and self.click_callback is not None:
            self.click_callback(event.xdata)


class ConvergenceGraph(FigureCanvas):
    """График сходимости (погрешность от итерации)"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Номер итерации')
        self.axes.set_ylabel('Погрешность |x_{k+1} - x_k|')
        self.axes.set_title('График сходимости')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(250)
        
    def plot_convergence(self, history, tolerance):
        self.axes.clear()
        if not history:
            self.draw()
            return
        
        iter_history = [h for h in history if h['error'] is not None]
        if not iter_history:
            self.draw()
            return
        
        iterations = [h['iteration'] for h in iter_history]
        errors = [h['error'] for h in iter_history]
        
        self.axes.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=4, label='Погрешность')
        self.axes.axhline(y=tolerance, color='r', linestyle='--', 
                         linewidth=1.5, label=f'Заданная точность ({tolerance:.1e})')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Номер итерации', fontsize=10)
        self.axes.set_ylabel('Погрешность', fontsize=10)
        self.axes.set_title('Зависимость погрешности от номера итерации', fontsize=12, fontweight='bold')
        self.axes.legend(loc='upper right', fontsize=9)
        
        if len(iterations) > 1:
            self.axes.set_xlim(1, iterations[-1])
        
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = EquationSolver()
        self.current_history = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Решение нелинейных уравнений")
        self.setGeometry(100, 100, 1300, 900)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QPushButton { background-color: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #1976D2; }
            QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 8px; margin-top: 20px; padding-top: 15px; background-color: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 20px; padding: 0 10px 0 10px; color: #2196F3; background-color: white; }
            QLabel { font-size: 12px; color: #333333; }
            QLineEdit { padding: 8px; border: 2px solid #cccccc; border-radius: 5px; font-size: 12px; background-color: white; color: #000000; selection-background-color: #2196F3; }
            QLineEdit:focus { border: 2px solid #2196F3; }
            QTableWidget { gridline-color: #dddddd; font-size: 12px; background-color: white; alternate-background-color: #f9f9f9; }
            QTableWidget::item { color: #000000; }
            QHeaderView::section { background-color: #2196F3; color: white; padding: 8px; font-weight: bold; }
            QComboBox { padding: 6px; border: 2px solid #cccccc; border-radius: 5px; background-color: white; color: #000000; }
            QComboBox:focus { border: 2px solid #2196F3; }
            QTabWidget::pane { border: 1px solid #cccccc; background: white; border-radius: 5px; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin: 2px; border-radius: 5px; color: #000000; }
            QTabBar::tab:selected { background-color: #2196F3; color: white; }
        """)
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Решение нелинейных уравнений методами простой итерации и Ньютона")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333333; margin: 15px;")
        main_layout.addWidget(title)
        
        input_group = QGroupBox("Входные данные")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(15, 20, 15, 15)
        
        f_layout = QHBoxLayout()
        f_label = QLabel("Уравнение f(x) = 0:")
        f_label.setMinimumWidth(150)
        f_layout.addWidget(f_label)
        self.f_edit = QLineEdit()
        self.f_edit.setPlaceholderText("Например: 2**x - x**2 - 0.5")
        f_layout.addWidget(self.f_edit)
        input_layout.addLayout(f_layout)
        
        method_layout = QHBoxLayout()
        method_label = QLabel("Метод:")
        method_label.setMinimumWidth(150)
        method_layout.addWidget(method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Метод Ньютона", "Метод простой итерации"])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        input_layout.addLayout(method_layout)
        
        phi_layout = QHBoxLayout()
        phi_label = QLabel("phi(x) для итераций:")
        phi_label.setMinimumWidth(150)
        phi_layout.addWidget(phi_label)
        self.phi_combo = QComboBox()
        self.phi_combo.setEditable(True)
        self.phi_combo.setInsertPolicy(QComboBox.NoInsert)
        self.phi_combo.lineEdit().setPlaceholderText("Выберите или введите phi(x)")
        phi_layout.addWidget(self.phi_combo)
        self.refresh_phi_btn = QPushButton("Обновить варианты")
        self.refresh_phi_btn.clicked.connect(self.refresh_phi_options)
        phi_layout.addWidget(self.refresh_phi_btn)
        input_layout.addLayout(phi_layout)
        
        self.phi_info_label = QLabel("")
        self.phi_info_label.setWordWrap(True)
        self.phi_info_label.setStyleSheet("color: #555555; font-size: 11px;")
        input_layout.addWidget(self.phi_info_label)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Начальное приближение x0:"))
        self.x0_edit = QLineEdit("1.0")
        self.x0_edit.setValidator(QDoubleValidator())
        params_layout.addWidget(self.x0_edit)
        params_layout.addWidget(QLabel("Точность ε:"))
        self.tol_edit = QLineEdit("1e-6")
        params_layout.addWidget(self.tol_edit)
        params_layout.addWidget(QLabel("Макс. итераций:"))
        self.max_iter_edit = QLineEdit("100")
        params_layout.addWidget(self.max_iter_edit)
        input_layout.addLayout(params_layout)
        
        btn_layout = QHBoxLayout()
        self.plot_btn = QPushButton("Построить график f(x)")
        self.plot_btn.clicked.connect(self.plot_function)
        btn_layout.addWidget(self.plot_btn)
        self.solve_btn = QPushButton("Решить уравнение")
        self.solve_btn.clicked.connect(self.solve_equation)
        self.solve_btn.setStyleSheet("background-color: #4CAF50;")
        btn_layout.addWidget(self.solve_btn)
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setStyleSheet("background-color: #f44336;")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)
        input_layout.addLayout(btn_layout)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        self.tab_widget = QTabWidget()
        
        self.graph_tab = QWidget()
        graph_layout = QVBoxLayout(self.graph_tab)
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("X min:"))
        self.xmin_edit = QLineEdit("-5")
        self.xmin_edit.setValidator(QDoubleValidator())
        interval_layout.addWidget(self.xmin_edit)
        interval_layout.addWidget(QLabel("X max:"))
        self.xmax_edit = QLineEdit("5")
        self.xmax_edit.setValidator(QDoubleValidator())
        interval_layout.addWidget(self.xmax_edit)
        self.update_graph_btn = QPushButton("Обновить график")
        self.update_graph_btn.clicked.connect(self.plot_function)
        interval_layout.addWidget(self.update_graph_btn)
        interval_layout.addStretch()
        graph_layout.addLayout(interval_layout)
        
        self.function_graph = FunctionGraph(self)
        self.function_graph.click_callback = self.on_graph_click
        graph_layout.addWidget(self.function_graph)
        
        info_label = QLabel("Кликните по графику, чтобы установить начальное приближение x0")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #555555; font-style: italic; padding: 5px;")
        graph_layout.addWidget(info_label)
        self.tab_widget.addTab(self.graph_tab, "График f(x)")
        
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        self.result_label = QLabel("Результаты вычислений")
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_label)
        self.solution_info = QLabel()
        self.solution_info.setWordWrap(True)
        self.solution_info.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px; color: #1b5e20;")
        results_layout.addWidget(self.solution_info)
        self.iter_table = QTableWidget()
        self.iter_table.setColumnCount(4)
        self.iter_table.setHorizontalHeaderLabels(["Итерация", "x", "f(x)", "Погрешность"])
        self.iter_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.iter_table)
        self.tab_widget.addTab(self.results_tab, "Результаты")
        
        self.convergence_tab = QWidget()
        conv_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_graph = ConvergenceGraph(self)
        conv_layout.addWidget(self.convergence_graph)
        self.tab_widget.addTab(self.convergence_tab, "График сходимости")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll_area)
        
        self.statusBar().showMessage("Готов к работе")
        self.on_method_changed(self.method_combo.currentText())
    
    def on_method_changed(self, method):
        is_simple_iter = (method == "Метод простой итерации")
        self.phi_combo.setEnabled(is_simple_iter)
        self.refresh_phi_btn.setEnabled(is_simple_iter)
        if not is_simple_iter:
            self.phi_info_label.setText("")
    
    def refresh_phi_options(self):
        self.phi_combo.clear()
        if not self.solver.phi_options:
            if self.solver.f_expr is not None:
                self.solver.generate_phi_options()
        if self.solver.phi_options:
            for name, expr in self.solver.phi_options:
                self.phi_combo.addItem(f"{name}: {expr}", str(expr))
            self.phi_combo.setCurrentIndex(0)
            self.update_phi_info()
        else:
            self.phi_combo.setEditText("")
            self.phi_info_label.setText("Не удалось автоматически сгенерировать варианты. Введите phi(x) вручную.")
    
    def update_phi_info(self):
        try:
            x0 = float(self.x0_edit.text())
        except:
            x0 = 1.0
        idx = self.phi_combo.currentIndex()
        if idx >= 0 and self.phi_combo.currentData():
            expr_str = self.phi_combo.currentData()
            if expr_str:
                expr = sp.sympify(expr_str)
                self.solver.set_phi(expr)
                cond = self.solver.check_convergence_condition(x0)
                if cond is not None:
                    if cond < 1:
                        self.phi_info_label.setText(f"|φ'({x0:.3f})| = {cond:.4f} < 1 → вероятна сходимость")
                        self.phi_info_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
                    else:
                        self.phi_info_label.setText(f"|φ'({x0:.3f})| = {cond:.4f} >= 1 → возможна расходимость")
                        self.phi_info_label.setStyleSheet("color: #c62828; font-size: 11px;")
                else:
                    self.phi_info_label.setText("Не удалось вычислить производную φ'(x)")
                    self.phi_info_label.setStyleSheet("color: #555555; font-size: 11px;")
        else:
            text = self.phi_combo.currentText().strip()
            if text:
                try:
                    expr = sp.sympify(text)
                    self.solver.set_phi(expr)
                    cond = self.solver.check_convergence_condition(x0)
                    if cond is not None:
                        if cond < 1:
                            self.phi_info_label.setText(f"|φ'({x0:.3f})| = {cond:.4f} < 1 → вероятна сходимость")
                            self.phi_info_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
                        else:
                            self.phi_info_label.setText(f"|φ'({x0:.3f})| = {cond:.4f} >= 1 → возможна расходимость")
                            self.phi_info_label.setStyleSheet("color: #c62828; font-size: 11px;")
                except:
                    self.phi_info_label.setText("Некорректное выражение phi(x)")
                    self.phi_info_label.setStyleSheet("color: #c62828; font-size: 11px;")
    
    def plot_function(self):
        f_str = self.f_edit.text().strip()
        if not f_str:
            QMessageBox.warning(self, "Предупреждение", "Введите уравнение f(x)")
            return
        success, msg = self.solver.set_equation(f_str)
        if not success:
            QMessageBox.critical(self, "Ошибка", f"Некорректное выражение f(x):\n{msg}")
            return
        try:
            x_min = float(self.xmin_edit.text())
            x_max = float(self.xmax_edit.text())
        except:
            x_min, x_max = -5, 5
        
        self.function_graph.plot_function(self.solver.f_lambdified, x_min, x_max, history=self.current_history)
        self.tab_widget.setCurrentIndex(0)
        self.statusBar().showMessage("График построен. Кликните по графику для выбора x0")
        self.refresh_phi_options()
    
    def on_graph_click(self, x):
        self.x0_edit.setText(f"{x:.6f}")
        self.statusBar().showMessage(f"Установлено x0 = {x:.6f}")
        self.update_phi_info()
    
    def solve_equation(self):
        f_str = self.f_edit.text().strip()
        if not f_str:
            QMessageBox.warning(self, "Предупреждение", "Введите уравнение f(x)")
            return
        success, msg = self.solver.set_equation(f_str)
        if not success:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в выражении f(x):\n{msg}")
            return
        
        method = self.method_combo.currentText()
        if method == "Метод простой итерации":
            phi_text = self.phi_combo.currentText().strip()
            if self.phi_combo.currentIndex() >= 0 and self.phi_combo.currentData():
                phi_expr = sp.sympify(self.phi_combo.currentData())
            else:
                if not phi_text:
                    QMessageBox.warning(self, "Предупреждение", "Для метода простой итерации необходимо задать phi(x)")
                    return
                try:
                    phi_expr = sp.sympify(phi_text)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка в выражении phi(x):\n{str(e)}")
                    return
            success, msg = self.solver.set_phi(phi_expr)
            if not success:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при установке phi(x):\n{msg}")
                return
        
        try:
            x0 = float(self.x0_edit.text())
            tol = float(self.tol_edit.text())
            max_iter = int(self.max_iter_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Предупреждение", "Некорректные числовые параметры")
            return
        
        self.statusBar().showMessage("Вычисление...")
        try:
            if method == "Метод Ньютона":
                root, iters, history, converged = self.solver.newton_method(x0, tol, max_iter)
            else:
                root, iters, history, converged = self.solver.simple_iteration_method(x0, tol, max_iter)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при вычислении:\n{str(e)}")
            self.statusBar().showMessage("Ошибка вычисления")
            return
        
        self.current_history = history
        
        if converged:
            self.solution_info.setText(
                f"Метод: {method}\n"
                f"Найденный корень: x = {root:.10f}\n"
                f"Значение f(x) = {self.solver.f(root):.2e}\n"
                f"Число итераций: {iters}\n"
                f"Достигнута точность: {tol}"
            )
            self.solution_info.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px; color: #1b5e20;")
        else:
            self.solution_info.setText(
                f"Метод: {method}\n"
                f"Решение не сошлось за {max_iter} итераций.\n"
                f"Последнее приближение: x = {root:.10f}\n"
                f"Значение f(x) = {self.solver.f(root):.2e}"
            )
            self.solution_info.setStyleSheet("padding: 10px; background-color: #fff3e0; border-radius: 5px; color: #e65100;")
        
        self.iter_table.setRowCount(len(history))
        for i, h in enumerate(history):
            self.iter_table.setItem(i, 0, QTableWidgetItem(str(h['iteration'])))
            self.iter_table.setItem(i, 1, QTableWidgetItem(f"{h['x']:.8f}"))
            fx_val = h['fx'] if h['fx'] is not None else self.solver.f(h['x'])
            self.iter_table.setItem(i, 2, QTableWidgetItem(f"{fx_val:.2e}" if fx_val is not None else "—"))
            if h['error'] is not None:
                self.iter_table.setItem(i, 3, QTableWidgetItem(f"{h['error']:.2e}"))
            else:
                self.iter_table.setItem(i, 3, QTableWidgetItem("—"))
        self.iter_table.resizeColumnsToContents()
        
        self.convergence_graph.plot_convergence(history, tol)
        
        # Обновить график с итерациями
        try:
            x_min = float(self.xmin_edit.text())
            x_max = float(self.xmax_edit.text())
        except:
            x_min, x_max = -5, 5
        self.function_graph.plot_function(self.solver.f_lambdified, x_min, x_max, history=history)
        
        self.tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage(f"Решение завершено. Итераций: {iters}")
    
    def clear_all(self):
        self.f_edit.clear()
        self.phi_combo.clear()
        self.phi_combo.setEditText("")
        self.phi_info_label.setText("")
        self.x0_edit.setText("1.0")
        self.tol_edit.setText("1e-6")
        self.max_iter_edit.setText("100")
        self.solution_info.setText("")
        self.solution_info.setStyleSheet("")
        self.iter_table.setRowCount(0)
        self.function_graph.axes.clear()
        self.function_graph.draw()
        self.convergence_graph.axes.clear()
        self.convergence_graph.draw()
        self.current_history = []
        self.statusBar().showMessage("Очищено")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()