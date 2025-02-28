import sys
import sqlite3
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # 创建用户表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                UID INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def add_user(self, role, password):
        # 添加用户
        self.cursor.execute("INSERT INTO users (role, password) VALUES (?, ?)", (role, password))
        self.conn.commit()

    def get_user(self, uid):
        # 根据 UID 获取用户信息
        self.cursor.execute("SELECT * FROM users WHERE UID = ?", (uid,))
        return self.cursor.fetchone()

    def get_user_by_role_and_password(self, role, password):
        # 根据身份和密码获取用户信息
        self.cursor.execute("SELECT * FROM users WHERE role = ? AND password = ?", (role, password))
        return self.cursor.fetchone()

    def close(self):
        # 关闭数据库连接
        self.conn.close()

# 模拟一个训练好的模型（实际使用时替换为你的模型）
class DummyModel:
    def predict(self, image_path):
        # 这里模拟一个预测结果
        return "This is a dummy prediction result for image: " + image_path

# 登录页面
class LoginPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("登录页面")

        # 布局
        layout = QVBoxLayout()

        # 公告区域
        self.notice_label = QLabel("公告：欢迎使用本系统！")
        layout.addWidget(self.notice_label)

        # 身份选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["管理员", "用户", "游客"])
        layout.addWidget(QLabel("选择身份："))
        layout.addWidget(self.role_combo)

        # 用户名和密码输入
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        layout.addWidget(QLabel("用户名："))
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.password_input)

        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.on_login)
        layout.addWidget(self.login_button)

        # 注册按钮
        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def on_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        role = self.role_combo.currentText()

        # 简单的登录验证（实际使用时需要连接数据库或调用API）
        if username and password:
            QMessageBox.information(self, "登录成功", f"欢迎 {role} {username}！")
            self.open_prediction_page()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码不能为空！")

    def on_register(self):
        QMessageBox.information(self, "注册", "注册功能尚未实现！")

    def open_prediction_page(self):
        self.prediction_page = PredictionPage()
        self.prediction_page.show()
        self.close()

# 预测页面
class PredictionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = DummyModel()  # 模拟的模型

    def initUI(self):
        self.setWindowTitle("预测页面")

        # 布局
        layout = QVBoxLayout()

        # 上传图片按钮
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.on_upload)
        layout.addWidget(self.upload_button)

        # 显示图片
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # 预测结果
        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def on_upload(self):
        # 打开文件对话框选择图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # 显示图片
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

            # 调用模型进行预测
            prediction = self.model.predict(file_path)
            self.result_label.setText(f"预测结果：{prediction}")

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统登录")
        self.setGeometry(100, 100, 400, 300)

        # 设置登录页面为主页面
        self.login_page = LoginPage()
        self.setCentralWidget(self.login_page)

# 运行程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())