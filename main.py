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

    def add_user(self, name, identity, password):
        # 添加用户
        self.cursor.execute("INSERT INTO users (UNAME, UIDENTITY, UPD) VALUES (?, ?, ?)", (name, identity, password))
        self.conn.commit()

    def get_user(self, uid):
        # 根据 UID 获取用户信息
        self.cursor.execute("SELECT * FROM users WHERE UID = ?", (uid,))
        return self.cursor.fetchone()

    def get_user_by_role_and_password(self, uid, name, identity, password):
        # 根据身份和密码获取用户信息
        self.cursor.execute("SELECT * FROM users WHERE UID = ? AND UNAME = ? AND UIDENTITY = ? AND UPD = ?", (uid, name, identity, password))
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
    def __init__(self, db):  # 接受 db 参数
        super().__init__()
        self.db = db  # 将 db 保存为实例变量
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
        self.role_combo.addItems(["管理员", "用户"])
        layout.addWidget(QLabel("选择身份："))
        layout.addWidget(self.role_combo)

        # UID 输入
        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("请输入用户id")
        layout.addWidget(QLabel("用户id："))
        layout.addWidget(self.uid_input)

        # 名称输入
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入用户名")
        layout.addWidget(QLabel("用户名称："))
        layout.addWidget(self.name_input)

        # 密码输入
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
        uid = self.uid_input.text()
        name = self.name_imput.text()
        password = self.password_input.text()
        identity = self.role_combo.currentText()

        if not uid or not password:
            QMessageBox.warning(self, "登录失败", "UID 或密码不能为空！")
            return

        # 检查用户是否存在
        user = self.db.get_user_by_role_and_password(uid, name, identity, password)
        if user:  # 检查用户是否存在
            QMessageBox.information(self, "登录成功", f"欢迎 {identity} {uid}！")
            self.open_prediction_page()
        else:
            QMessageBox.warning(self, "登录失败", "UID 或密码错误！")

    def on_register(self):
        # 打开注册页面
        self.register_page = RegisterPage(self.db)
        self.register_page.show()

    def open_prediction_page(self):
        # 打开预测页面
        self.prediction_page = PredictionPage()
        self.prediction_page.show()
        self.close()

# 注册页面
class RegisterPage(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.initUI()

    def initUI(self):
        self.setWindowTitle("注册页面")

        # 布局
        layout = QVBoxLayout()

        # 身份选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["管理员", "用户"])
        layout.addWidget(QLabel("选择身份："))
        layout.addWidget(self.role_combo)

        # 密码输入
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.password_input)

        # 注册按钮
        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def on_register(self):
        identity = self.role_combo.currentText()
        password = self.password_input.text()

        if not password:
            QMessageBox.warning(self, "注册失败", "密码不能为空！")
            return

        # 添加用户到数据库
        self.db.add_user(identity, password)
        QMessageBox.information(self, "注册成功", "用户注册成功！")
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
    def __init__(self, db):  # 接受 db 参数
        super().__init__()
        self.setWindowTitle("系统登录")
        self.setGeometry(100, 100, 400, 300)
        self.db = db

        # 设置登录页面为主页面
        self.login_page = LoginPage(self.db)  # 将 db 传递给 LoginPage
        self.setCentralWidget(self.login_page)

# 运行程序
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 初始化数据库（使用现有的数据库文件）
    db = Database("database.db")

    # 创建主窗口
    window = MainWindow(db)
    window.show()

    # 运行应用
    sys.exit(app.exec())

    # 关闭数据库连接
    db.close()