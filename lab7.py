import sys
import numpy as np
import sympy as sp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SystemSolver:
    """Класс для работы с системой нелинейных уравнений"""
    
    def __init__(self):
        self.x_sym, self.y_sym, self.a_sym = sp.symbols('x y a')
        self.f1_expr = None
        self.f2_expr = None
        self.phi1_expr = None
        self.phi2_expr = None
        self.f1_lambdified = None
        self.f2_lambdified = None
        self.phi1_lambdified = None
        self.phi2_lambdified = None
        self.jacobian_lambdified = None
        self.phi_options = []  # Список вариантов (φ1, φ2)
        
    def set_equations(self, f1_str, f2_str):
        """Установка уравнений системы F(x,y) = (0,0)"""
        try:
            self.f1_expr = sp.sympify(f1_str, evaluate=False)
            self.f2_expr = sp.sympify(f2_str, evaluate=False)
            self.f1_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), self.f1_expr, modules=['numpy', 'math'])
            self.f2_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), self.f2_expr, modules=['numpy', 'math'])
            J = sp.Matrix([[sp.diff(self.f1_expr, self.x_sym), sp.diff(self.f1_expr, self.y_sym)],
                           [sp.diff(self.f2_expr, self.x_sym), sp.diff(self.f2_expr, self.y_sym)]])
            self.jacobian_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), J, modules=['numpy', 'math'])
            self.generate_phi_options()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def generate_phi_options(self):
        """Автоматическая генерация вариантов φ1, φ2"""
        self.phi_options = []
        if self.f1_expr is None or self.f2_expr is None:
            return
        
        x, y, a = self.x_sym, self.y_sym, self.a_sym
        f1, f2 = self.f1_expr, self.f2_expr
        
        # Вариант 1: релаксация x = x - c1*f1, y = y - c2*f2
        for c1, c2 in [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0)]:
            phi1 = x - c1 * f1
            phi2 = y - c2 * f2
            self.phi_options.append((f"Релаксация: x - {c1}*f1, y - {c2}*f2", phi1, phi2))
        
        # Попытка решить аналитически одно уравнение относительно x, другое относительно y
        try:
            sol_x = sp.solve(f1, x)
            for sx in sol_x:
                if not sx.has(sp.I):
                    f2_sub = f2.subs(x, sx)
                    sol_y = sp.solve(f2_sub, y)
                    for sy in sol_y:
                        if not sy.has(sp.I):
                            self.phi_options.append((f"x = {sx}, y = {sy}", sx, sy))
        except:
            pass
        
        try:
            sol_y = sp.solve(f1, y)
            for sy in sol_y:
                if not sy.has(sp.I):
                    f2_sub = f2.subs(y, sy)
                    sol_x = sp.solve(f2_sub, x)
                    for sx in sol_x:
                        if not sx.has(sp.I):
                            self.phi_options.append((f"y = {sy}, x = {sx}", sx, sy))
        except:
            pass
        
        try:
            sols = sp.solve([f1, f2], [x, y], dict=True)
            for sol in sols:
                sx = sol.get(x, x)
                sy = sol.get(y, y)
                if not sx.has(sp.I) and not sy.has(sp.I):
                    self.phi_options.append((f"Точное решение: x={sx}, y={sy}", sx, sy))
        except:
            pass
        
        for c in [0.05, 0.2, 0.8]:
            phi1 = x - c * f1
            phi2 = y - c * f2
            self.phi_options.append((f"Релаксация: x - {c}*f1, y - {c}*f2", phi1, phi2))
        
        seen = set()
        unique = []
        for name, phi1, phi2 in self.phi_options:
            key = (str(phi1), str(phi2))
            if key not in seen:
                seen.add(key)
                unique.append((name, phi1, phi2))
        self.phi_options = unique
    
    def set_phi(self, phi1_expr, phi2_expr):
        """Установка выбранных выражений φ1, φ2"""
        try:
            self.phi1_expr = phi1_expr
            self.phi2_expr = phi2_expr
            self.phi1_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), self.phi1_expr, modules=['numpy', 'math'])
            self.phi2_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), self.phi2_expr, modules=['numpy', 'math'])
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def F(self, x, y, a):
        try:
            return np.array([float(self.f1_lambdified(x, y, a)), float(self.f2_lambdified(x, y, a))])
        except:
            return None
    
    def J(self, x, y, a):
        try:
            return np.array(self.jacobian_lambdified(x, y, a), dtype=float)
        except:
            return None
    
    def phi(self, x, y, a):
        try:
            return np.array([float(self.phi1_lambdified(x, y, a)), float(self.phi2_lambdified(x, y, a))])
        except:
            return None
    
    def phi_jacobian(self, x, y, a):
        if self.phi1_expr is None or self.phi2_expr is None:
            return None
        try:
            J_phi = sp.Matrix([[sp.diff(self.phi1_expr, self.x_sym), sp.diff(self.phi1_expr, self.y_sym)],
                               [sp.diff(self.phi2_expr, self.x_sym), sp.diff(self.phi2_expr, self.y_sym)]])
            J_phi_lambdified = sp.lambdify((self.x_sym, self.y_sym, self.a_sym), J_phi, modules=['numpy', 'math'])
            return np.array(J_phi_lambdified(x, y, a), dtype=float)
        except:
            return None
    
    def check_convergence(self, x0, y0, a):
        J_phi = self.phi_jacobian(x0, y0, a)
        if J_phi is None:
            return None
        try:
            eigenvalues = np.linalg.eigvals(J_phi)
            spectral_radius = max(abs(ev) for ev in eigenvalues)
            return spectral_radius
        except:
            return None
    
    def newton_method(self, x0, y0, a, tolerance, max_iter):
        history = []
        F0 = self.F(x0, y0, a)
        history.append({
            'iteration': 0,
            'x': x0, 'y': y0,
            'F_norm': np.linalg.norm(F0) if F0 is not None else None,
            'error': None
        })
        x, y = x0, y0
        for i in range(max_iter):
            F_val = self.F(x, y, a)
            J_val = self.J(x, y, a)
            if F_val is None or J_val is None:
                break
            try:
                delta = np.linalg.solve(J_val, -F_val)
            except:
                break
            x_new, y_new = x + delta[0], y + delta[1]
            error = np.linalg.norm(delta)
            F_new = self.F(x_new, y_new, a)
            history.append({
                'iteration': i+1,
                'x': x_new, 'y': y_new,
                'F_norm': np.linalg.norm(F_new) if F_new is not None else None,
                'error': error
            })
            if error < tolerance:
                return (x_new, y_new), i+1, history, True
            x, y = x_new, y_new
        return (x, y), max_iter, history, False
    
    def simple_iteration_method(self, x0, y0, a, tolerance, max_iter):
        if self.phi1_lambdified is None or self.phi2_lambdified is None:
            return None, 0, [], False
        history = []
        F0 = self.F(x0, y0, a)
        history.append({
            'iteration': 0,
            'x': x0, 'y': y0,
            'F_norm': np.linalg.norm(F0) if F0 is not None else None,
            'error': None
        })
        x, y = x0, y0
        for i in range(max_iter):
            phi_val = self.phi(x, y, a)
            if phi_val is None:
                break
            x_new, y_new = phi_val
            error = np.linalg.norm([x_new - x, y_new - y])
            F_new = self.F(x_new, y_new, a)
            history.append({
                'iteration': i+1,
                'x': x_new, 'y': y_new,
                'F_norm': np.linalg.norm(F_new) if F_new is not None else None,
                'error': error
            })
            if error < tolerance:
                return (x_new, y_new), i+1, history, True
            x, y = x_new, y_new
        return (x, y), max_iter, history, False


class SystemGraph(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 6), dpi=100)
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
        self.mpl_connect('button_press_event', self.on_click)
        self.click_callback = None
        
    def plot_system(self, f1, f2, a, x_range, y_range, history=None, num_points=100):
        self.axes.clear()
        if f1 is None or f2 is None:
            self.draw()
            return
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        try:
            Z1 = f1(X, Y, a)
            Z2 = f2(X, Y, a)
            Z1 = np.array(Z1, dtype=float)
            Z2 = np.array(Z2, dtype=float)
        except:
            self.draw()
            return
        cs1 = self.axes.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        cs2 = self.axes.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, linestyles='--')
        if cs1.collections:
            cs1.collections[0].set_label('f₁(x,y)=0')
        if cs2.collections:
            cs2.collections[0].set_label('f₂(x,y)=0')
        
        if history and len(history) > 0:
            xs = [h['x'] for h in history]
            ys = [h['y'] for h in history]
            self.axes.plot(xs, ys, 'g-o', linewidth=2, markersize=4, label='Итерации')
            if len(xs) > 0:
                self.axes.plot(xs[0], ys[0], 'go', markersize=10, label='Начало')
            if len(xs) > 1:
                self.axes.plot(xs[-1], ys[-1], 'r*', markersize=12, label='Решение')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_title(f'Система при a = {a}')
        self.axes.legend()
        self.figure.tight_layout()
        self.draw()
    
    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None and self.click_callback is not None:
            self.click_callback(event.xdata, event.ydata)


class ConvergenceGraph(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Номер итерации')
        self.axes.set_ylabel('Погрешность')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(250)
        
    def plot_convergence(self, history, tolerance):
        self.axes.clear()
        if not history:
            self.draw()
            return
        # Фильтруем только итерации с ошибкой (начиная с 1)
        iter_history = [h for h in history if h['error'] is not None]
        if not iter_history:
            self.draw()
            return
        iterations = [h['iteration'] for h in iter_history]
        errors = [h['error'] for h in iter_history]
        self.axes.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=4, label='Погрешность')
        self.axes.axhline(y=tolerance, color='r', linestyle='--', linewidth=1.5, label=f'Точность ({tolerance:.1e})')
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
        self.solver = SystemSolver()
        self.current_history = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Решение систем нелинейных уравнений")
        self.setGeometry(100, 100, 1450, 950)
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
            QComboBox { padding: 6px; border: 2px solid #cccccc; border-radius: 5px; background-color: white; color: #000000; }
            QTabWidget::pane { border: 1px solid #cccccc; background: white; border-radius: 5px; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin: 2px; border-radius: 5px; color: #000000; }
            QTabBar::tab:selected { background-color: #2196F3; color: white; }
        """)
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Решение систем нелинейных уравнений (Ньютон и простая итерация)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Панель ввода
        input_group = QGroupBox("Входные данные")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(15, 20, 15, 15)
        
        eq_layout = QHBoxLayout()
        eq_layout.addWidget(QLabel("f₁(x,y) = 0:"))
        self.f1_edit = QLineEdit()
        self.f1_edit.setPlaceholderText("x**2 + y**2 - 4")
        eq_layout.addWidget(self.f1_edit)
        eq_layout.addWidget(QLabel("f₂(x,y) = 0:"))
        self.f2_edit = QLineEdit()
        self.f2_edit.setPlaceholderText("x*y - 1")
        eq_layout.addWidget(self.f2_edit)
        input_layout.addLayout(eq_layout)
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Параметр a:"))
        self.a_edit = QLineEdit("1.0")
        self.a_edit.setValidator(QDoubleValidator())
        param_layout.addWidget(self.a_edit)
        param_layout.addStretch()
        input_layout.addLayout(param_layout)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Метод:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Метод Ньютона", "Метод простой итерации"])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        input_layout.addLayout(method_layout)
        
        self.phi_group = QGroupBox("Функции для метода простой итерации")
        phi_layout = QVBoxLayout()
        
        phi_select_layout = QHBoxLayout()
        phi_select_layout.addWidget(QLabel("Выберите вариант:"))
        self.phi_combo = QComboBox()
        self.phi_combo.setEditable(True)
        self.phi_combo.currentIndexChanged.connect(self.update_phi_info)
        phi_select_layout.addWidget(self.phi_combo)
        self.refresh_phi_btn = QPushButton("Обновить варианты")
        self.refresh_phi_btn.clicked.connect(self.refresh_phi_options)
        phi_select_layout.addWidget(self.refresh_phi_btn)
        phi_layout.addLayout(phi_select_layout)
        
        self.phi_info_label = QLabel("")
        self.phi_info_label.setWordWrap(True)
        self.phi_info_label.setStyleSheet("color: #555555; font-size: 11px;")
        phi_layout.addWidget(self.phi_info_label)
        
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Или введите вручную:"))
        manual_layout.addStretch()
        phi_layout.addLayout(manual_layout)
        
        phi1_layout = QHBoxLayout()
        phi1_layout.addWidget(QLabel("φ₁(x,y):"))
        self.phi1_edit = QLineEdit()
        phi1_layout.addWidget(self.phi1_edit)
        phi_layout.addLayout(phi1_layout)
        
        phi2_layout = QHBoxLayout()
        phi2_layout.addWidget(QLabel("φ₂(x,y):"))
        self.phi2_edit = QLineEdit()
        phi2_layout.addWidget(self.phi2_edit)
        phi_layout.addLayout(phi2_layout)
        
        self.phi_group.setLayout(phi_layout)
        input_layout.addWidget(self.phi_group)
        
        init_layout = QHBoxLayout()
        init_layout.addWidget(QLabel("Начальное x₀:"))
        self.x0_edit = QLineEdit("1.0")
        self.x0_edit.setValidator(QDoubleValidator())
        init_layout.addWidget(self.x0_edit)
        init_layout.addWidget(QLabel("Начальное y₀:"))
        self.y0_edit = QLineEdit("1.0")
        self.y0_edit.setValidator(QDoubleValidator())
        init_layout.addWidget(self.y0_edit)
        input_layout.addLayout(init_layout)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Точность ε:"))
        self.tol_edit = QLineEdit("1e-6")
        params_layout.addWidget(self.tol_edit)
        params_layout.addWidget(QLabel("Макс. итераций:"))
        self.max_iter_edit = QLineEdit("100")
        params_layout.addWidget(self.max_iter_edit)
        input_layout.addLayout(params_layout)
        
        btn_layout = QHBoxLayout()
        self.plot_btn = QPushButton("Построить графики")
        self.plot_btn.clicked.connect(self.plot_system)
        btn_layout.addWidget(self.plot_btn)
        self.solve_btn = QPushButton("Решить систему")
        self.solve_btn.clicked.connect(self.solve_system)
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
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("X min:"))
        self.xmin_edit = QLineEdit("-5")
        self.xmin_edit.setValidator(QDoubleValidator())
        range_layout.addWidget(self.xmin_edit)
        range_layout.addWidget(QLabel("X max:"))
        self.xmax_edit = QLineEdit("5")
        range_layout.addWidget(self.xmax_edit)
        range_layout.addWidget(QLabel("Y min:"))
        self.ymin_edit = QLineEdit("-5")
        range_layout.addWidget(self.ymin_edit)
        range_layout.addWidget(QLabel("Y max:"))
        self.ymax_edit = QLineEdit("5")
        range_layout.addWidget(self.ymax_edit)
        self.update_graph_btn = QPushButton("Обновить")
        self.update_graph_btn.clicked.connect(self.plot_system)
        range_layout.addWidget(self.update_graph_btn)
        range_layout.addStretch()
        graph_layout.addLayout(range_layout)
        
        self.system_graph = SystemGraph(self)
        self.system_graph.click_callback = self.on_graph_click
        graph_layout.addWidget(self.system_graph)
        info_label = QLabel("Кликните по графику для выбора начального приближения")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #555555; font-style: italic;")
        graph_layout.addWidget(info_label)
        self.tab_widget.addTab(self.graph_tab, "Графики")
        
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        self.solution_info = QLabel("Результаты будут отображены здесь")
        self.solution_info.setWordWrap(True)
        self.solution_info.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px; color: #1b5e20;")
        results_layout.addWidget(self.solution_info)
        self.iter_table = QTableWidget()
        self.iter_table.setColumnCount(5)
        self.iter_table.setHorizontalHeaderLabels(["Итерация", "x", "y", "||F||", "Погрешность"])
        self.iter_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.iter_table)
        self.tab_widget.addTab(self.results_tab, "Результаты")
        
        self.convergence_tab = QWidget()
        conv_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_graph = ConvergenceGraph(self)
        conv_layout.addWidget(self.convergence_graph)
        self.tab_widget.addTab(self.convergence_tab, "Сходимость")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll_area)
        
        self.statusBar().showMessage("Готов к работе")
        self.on_method_changed(self.method_combo.currentText())
    
    def on_method_changed(self, method):
        is_simple = (method == "Метод простой итерации")
        self.phi_group.setEnabled(is_simple)
        if not is_simple:
            self.phi_info_label.setText("")
    
    def refresh_phi_options(self):
        self.phi_combo.clear()
        if self.solver.phi_options:
            for name, phi1, phi2 in self.solver.phi_options:
                self.phi_combo.addItem(f"{name}: x = {phi1}, y = {phi2}", (str(phi1), str(phi2)))
            self.phi_combo.setCurrentIndex(0)
            self.update_phi_info()
        else:
            self.phi_combo.setEditText("")
            self.phi_info_label.setText("Варианты не найдены. Введите вручную.")
    
    def update_phi_info(self):
        idx = self.phi_combo.currentIndex()
        if idx >= 0 and self.phi_combo.currentData():
            phi1_str, phi2_str = self.phi_combo.currentData()
            self.phi1_edit.setText(phi1_str)
            self.phi2_edit.setText(phi2_str)
            try:
                phi1_expr = sp.sympify(phi1_str)
                phi2_expr = sp.sympify(phi2_str)
                self.solver.set_phi(phi1_expr, phi2_expr)
                try:
                    x0 = float(self.x0_edit.text())
                    y0 = float(self.y0_edit.text())
                    a = float(self.a_edit.text())
                except:
                    x0 = y0 = 1.0
                    a = 1.0
                rho = self.solver.check_convergence(x0, y0, a)
                if rho is not None:
                    if rho < 1:
                        self.phi_info_label.setText(f"Спектральный радиус ρ = {rho:.4f} < 1 → вероятна сходимость")
                        self.phi_info_label.setStyleSheet("color: #2e7d32; font-size: 11px;")
                    else:
                        self.phi_info_label.setText(f"Спектральный радиус ρ = {rho:.4f} >= 1 → возможна расходимость")
                        self.phi_info_label.setStyleSheet("color: #c62828; font-size: 11px;")
                else:
                    self.phi_info_label.setText("Не удалось оценить сходимость")
            except:
                self.phi_info_label.setText("Ошибка в выражениях")
        else:
            pass
    
    def plot_system(self):
        f1_str = self.f1_edit.text().strip()
        f2_str = self.f2_edit.text().strip()
        if not f1_str or not f2_str:
            QMessageBox.warning(self, "Предупреждение", "Введите оба уравнения")
            return
        success, msg = self.solver.set_equations(f1_str, f2_str)
        if not success:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в уравнениях:\n{msg}")
            return
        try:
            a = float(self.a_edit.text())
            x_min, x_max = float(self.xmin_edit.text()), float(self.xmax_edit.text())
            y_min, y_max = float(self.ymin_edit.text()), float(self.ymax_edit.text())
        except:
            QMessageBox.warning(self, "Предупреждение", "Некорректные границы")
            return
        self.system_graph.plot_system(self.solver.f1_lambdified, self.solver.f2_lambdified, a,
                                      (x_min, x_max), (y_min, y_max), history=self.current_history)
        self.tab_widget.setCurrentIndex(0)
        self.statusBar().showMessage("Графики построены")
        self.refresh_phi_options()
    
    def on_graph_click(self, x, y):
        self.x0_edit.setText(f"{x:.6f}")
        self.y0_edit.setText(f"{y:.6f}")
        self.statusBar().showMessage(f"x₀ = {x:.6f}, y₀ = {y:.6f}")
        self.update_phi_info()
    
    def solve_system(self):
        f1_str = self.f1_edit.text().strip()
        f2_str = self.f2_edit.text().strip()
        if not f1_str or not f2_str:
            QMessageBox.warning(self, "Предупреждение", "Введите оба уравнения")
            return
        success, msg = self.solver.set_equations(f1_str, f2_str)
        if not success:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в уравнениях:\n{msg}")
            return
        method = self.method_combo.currentText()
        if method == "Метод простой итерации":
            if self.phi_combo.currentIndex() >= 0 and self.phi_combo.currentData():
                phi1_str, phi2_str = self.phi_combo.currentData()
            else:
                phi1_str = self.phi1_edit.text().strip()
                phi2_str = self.phi2_edit.text().strip()
                if not phi1_str or not phi2_str:
                    QMessageBox.warning(self, "Предупреждение", "Введите φ₁ и φ₂")
                    return
            try:
                phi1_expr = sp.sympify(phi1_str)
                phi2_expr = sp.sympify(phi2_str)
                self.solver.set_phi(phi1_expr, phi2_expr)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка в φ: {e}")
                return
        try:
            x0 = float(self.x0_edit.text())
            y0 = float(self.y0_edit.text())
            a = float(self.a_edit.text())
            tol = float(self.tol_edit.text())
            max_iter = int(self.max_iter_edit.text())
        except:
            QMessageBox.warning(self, "Предупреждение", "Некорректные параметры")
            return
        self.statusBar().showMessage("Вычисление...")
        try:
            if method == "Метод Ньютона":
                (rx, ry), iters, history, conv = self.solver.newton_method(x0, y0, a, tol, max_iter)
            else:
                (rx, ry), iters, history, conv = self.solver.simple_iteration_method(x0, y0, a, tol, max_iter)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")
            return
        
        self.current_history = history
        
        if conv:
            self.solution_info.setText(
                f"Метод: {method}\nРешение: x = {rx:.10f}, y = {ry:.10f}\n"
                f"||F|| = {np.linalg.norm(self.solver.F(rx, ry, a)):.2e}\n"
                f"Итераций: {iters}, точность: {tol}"
            )
            self.solution_info.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px; color: #1b5e20;")
        else:
            self.solution_info.setText(
                f"Метод: {method}\nНе сошлось за {max_iter} итераций.\n"
                f"Последнее: x = {rx:.10f}, y = {ry:.10f}"
            )
            self.solution_info.setStyleSheet("padding: 10px; background-color: #fff3e0; border-radius: 5px; color: #e65100;")
        
        # Заполняем таблицу, включая нулевую итерацию
        self.iter_table.setRowCount(len(history))
        for i, h in enumerate(history):
            self.iter_table.setItem(i, 0, QTableWidgetItem(str(h['iteration'])))
            self.iter_table.setItem(i, 1, QTableWidgetItem(f"{h['x']:.8f}"))
            self.iter_table.setItem(i, 2, QTableWidgetItem(f"{h['y']:.8f}"))
            fn = f"{h['F_norm']:.2e}" if h['F_norm'] is not None else "—"
            self.iter_table.setItem(i, 3, QTableWidgetItem(fn))
            if h['error'] is not None:
                self.iter_table.setItem(i, 4, QTableWidgetItem(f"{h['error']:.2e}"))
            else:
                self.iter_table.setItem(i, 4, QTableWidgetItem("—"))
        self.iter_table.resizeColumnsToContents()
        
        self.convergence_graph.plot_convergence(history, tol)
        
        try:
            a = float(self.a_edit.text())
            x_min, x_max = float(self.xmin_edit.text()), float(self.xmax_edit.text())
            y_min, y_max = float(self.ymin_edit.text()), float(self.ymax_edit.text())
            self.system_graph.plot_system(self.solver.f1_lambdified, self.solver.f2_lambdified, a,
                                          (x_min, x_max), (y_min, y_max), history=history)
        except:
            pass
        
        self.tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage(f"Завершено. Итераций: {iters}")
    
    def clear_all(self):
        self.f1_edit.clear()
        self.f2_edit.clear()
        self.a_edit.setText("1.0")
        self.phi_combo.clear()
        self.phi1_edit.clear()
        self.phi2_edit.clear()
        self.phi_info_label.setText("")
        self.x0_edit.setText("1.0")
        self.y0_edit.setText("1.0")
        self.tol_edit.setText("1e-6")
        self.max_iter_edit.setText("100")
        self.solution_info.setText("Результаты будут отображены здесь")
        self.iter_table.setRowCount(0)
        self.system_graph.axes.clear()
        self.system_graph.draw()
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