import sys
import sqlite3

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
from ultralytics import YOLO

# 数据库操作类
class Database:
    def __init__(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        except sqlite3.Error as e:
            print("数据库连接失败：", e)

    def add_user(self, name, identity, password):
        try:
            self.cursor.execute("INSERT INTO user (UNAME, UIDENTITY, UPD) VALUES (?, ?, ?)", (name, identity, password))
            self.conn.commit()
            print("用户添加成功")
        except sqlite3.Error as e:
            print("用户添加失败：", e)

    def get_user(self, uid):
        try:
            self.cursor.execute("SELECT * FROM user WHERE UID = ?", (uid,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("查询用户失败：", e)
            return None

    def get_user_by_role_and_password(self, uid, name, identity, password):
        try:
            self.cursor.execute("SELECT * FROM user WHERE UID = ? AND UNAME = ? AND UIDENTITY = ? AND UPD = ?", (uid, name, identity, password))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("查询用户失败：", e)
            return None

    def get_all_users(self):
        try:
            self.cursor.execute("SELECT * FROM user")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("查询所有用户失败：", e)
            return []

    def update_user(self, uid, name, identity, password):
        try:
            self.cursor.execute("UPDATE user SET UNAME = ?, UIDENTITY = ?, UPD = ? WHERE UID = ?", (name, identity, password, uid))
            self.conn.commit()
            print("用户信息更新成功")
        except sqlite3.Error as e:
            print("用户信息更新失败：", e)

    def close(self):
        self.conn.close()
        print("数据库连接关闭")

# YOLOv8 模型类
class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        # 预测并保存结果
        results = self.model.predict(source=image_path, conf=0.1, save=True, show=False)
        # 返回预测结果（类别名称、置信度）和保存的图片路径
        if len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                class_id = int(boxes.cls[0])
                confidence = float(boxes.conf[0])
                class_name = results[0].names[class_id]
                return class_name, confidence, results[0].save_dir
        return None, None, None

# 登录页面
class LoginPage(QWidget):
    def __init__(self, db, main_window):
        super().__init__()
        self.db = db
        self.main_window = main_window  # 保存主窗口引用
        self.initUI()

    def initUI(self):
        self.setWindowTitle("登录页面")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        # 布局
        layout = QVBoxLayout()

        # 公告区域
        self.notice_label = QLabel("公告：欢迎使用本系统！")
        self.notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.notice_label)

        # 身份选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Administrator", "User"])
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
        name = self.name_input.text()
        password = self.password_input.text()
        identity = self.role_combo.currentText()

        if not uid or not password:
            QMessageBox.warning(self, "登录失败", "UID 或密码不能为空！")
            return

        try:
            user = self.db.get_user_by_role_and_password(uid, name, identity, password)
            if user:
                QMessageBox.information(self, "登录成功", f"欢迎 {identity} {uid}！")
                self.main_window.current_user = user  # 保存当前用户信息
                self.main_window.open_home_page()
                self.close()  # 关闭登录页面
            else:
                QMessageBox.warning(self, "登录失败", "UID 或密码错误！")
        except Exception as e:
            print("登录出错：", e)

    def on_register(self):
        self.register_page = RegisterPage(self.db)
        self.register_page.show()

# 注册页面
class RegisterPage(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.initUI()

    def initUI(self):
        self.setWindowTitle("注册页面")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        # 布局
        layout = QVBoxLayout()

        # 身份选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["管理员", "用户"])
        layout.addWidget(QLabel("选择身份："))
        layout.addWidget(self.role_combo)

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

        # 注册按钮
        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def on_register(self):
        identity = self.role_combo.currentText()
        name = self.name_input.text()
        password = self.password_input.text()

        if not name or not password:
            QMessageBox.warning(self, "注册失败", "用户名和密码不能为空！")
            return

        try:
            self.db.add_user(name, identity, password)
            QMessageBox.information(self, "注册成功", "用户注册成功！")
            self.close()
        except Exception as e:
            print("注册出错：", e)

# 主页
class HomePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("主页")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QTextEdit {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        # 布局
        layout = QVBoxLayout()

        # 公告区域
        self.notice_label = QLabel("公告：欢迎使用本系统！")
        self.notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.notice_label)

        # 编辑公告按钮（仅管理员可见）
        if self.main_window.current_user[2] == "Administrator":
            self.edit_notice_button = QPushButton("编辑公告")
            self.edit_notice_button.clicked.connect(self.on_edit_notice)
            layout.addWidget(self.edit_notice_button)

        # 个人信息按钮
        self.profile_button = QPushButton("个人信息")
        self.profile_button.clicked.connect(self.on_profile)
        layout.addWidget(self.profile_button)

        # 预测按钮
        self.predict_button = QPushButton("进行预测")
        self.predict_button.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def on_edit_notice(self):
        # 打开编辑公告页面
        self.edit_notice_page = EditNoticePage(self.main_window)
        self.edit_notice_page.show()

    def on_profile(self):
        # 打开个人信息页面
        self.profile_page = ProfilePage(self.main_window)
        self.profile_page.show()

    def on_predict(self):
        # 打开预测页面
        self.main_window.setCentralWidget(PredictionPage(self.main_window))
        self.close()

# 编辑公告页面
class EditNoticePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("编辑公告")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QTextEdit {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

        # 布局
        layout = QVBoxLayout()

        # 公告编辑框
        self.notice_edit = QTextEdit()
        self.notice_edit.setPlainText("公告：欢迎使用本系统！")
        layout.addWidget(self.notice_edit)

        # 保存按钮
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.on_save)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def on_save(self):
        # 保存公告
        notice = self.notice_edit.toPlainText()
        self.main_window.home_page.notice_label.setText(notice)
        self.close()

# 个人信息页面
class ProfilePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("个人信息")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

        # 布局
        layout = QVBoxLayout()

        # 显示当前用户信息
        self.uid_label = QLabel(f"用户ID：{self.main_window.current_user[0]}")
        layout.addWidget(self.uid_label)

        self.name_input = QLineEdit(self.main_window.current_user[1])
        layout.addWidget(QLabel("用户名："))
        layout.addWidget(self.name_input)

        self.identity_input = QLineEdit(self.main_window.current_user[2])
        layout.addWidget(QLabel("身份："))
        layout.addWidget(self.identity_input)

        self.password_input = QLineEdit(self.main_window.current_user[3])
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.password_input)

        # 保存按钮
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.on_save)
        layout.addWidget(self.save_button)

        # 如果是管理员，显示所有用户信息
        if self.main_window.current_user[2] == "Administrator":
            self.user_table = QTableWidget()
            self.user_table.setColumnCount(4)
            self.user_table.setHorizontalHeaderLabels(["用户ID", "用户名", "身份", "密码"])
            self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.load_user_table()
            layout.addWidget(self.user_table)

        self.setLayout(layout)

    def load_user_table(self):
        # 加载所有用户信息
        users = self.main_window.db.get_all_users()
        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            for j, item in enumerate(user):
                self.user_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_save(self):
        # 保存当前用户信息
        uid = self.main_window.current_user[0]
        name = self.name_input.text()
        identity = self.identity_input.text()
        password = self.password_input.text()

        self.main_window.db.update_user(uid, name, identity, password)
        QMessageBox.information(self, "保存成功", "用户信息已更新！")
        self.close()

# 预测页面
class PredictionPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()
        self.model = YOLOModel(model_path='./best.pt')  # 加载 YOLOv8 模型

    def initUI(self):
        self.setWindowTitle("预测页面")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

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

        # 返回按钮
        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def on_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # 显示原始图片
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

            # 调用 YOLOv8 模型进行预测
            class_name, confidence, save_dir = self.model.predict(file_path)
            if class_name and confidence:
                # 显示预测结果图片
                result_image_path = f"{save_dir}/{file_path.split('/')[-1]}"  # 假设保存的图片名称不变
                result_pixmap = QPixmap(result_image_path)
                self.image_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")
            else:
                self.result_label.setText("未检测到目标")

    def on_back(self):
        # 返回主页
        self.main_window.setCentralWidget(HomePage(self.main_window))
        self.close()

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self, db):
        super().__init__()
        self.setWindowTitle("系统登录")
        self.setGeometry(100, 100, 400, 300)
        self.db = db
        self.current_user = None  # 当前登录用户
        self.home_page = None  # 主页

        # 设置登录页面为主页面
        self.login_page = LoginPage(self.db, self)
        self.setCentralWidget(self.login_page)

    def open_home_page(self):
        # 打开主页
        self.home_page = HomePage(self)
        self.setCentralWidget(self.home_page)

# 运行程序
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 初始化数据库
    db = Database("database.db")

    # 创建主窗口
    window = MainWindow(db)
    window.show()

    # 运行应用
    sys.exit(app.exec())

    # 关闭数据库连接
    db.close()