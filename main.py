import sys
import sqlite3
import cv2
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QSlider
)
from PyQt6.QtCore import Qt, QTimer
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

    def get_latest_notice(self):
        try:
            self.cursor.execute("SELECT NOTICE FROM notice ORDER BY TIME DESC LIMIT 1")
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("获取最新公告失败：", e)
            return None

    def add_feedback(self, serve_id, user_id, feedback):
        try:
            # 插入新反馈记录
            self.cursor.execute(
                "INSERT INTO serve (SERVEID, USERID, FEEDBACK, SERVETIME, FINISH) VALUES (?, ?, ?, datetime('now'), ?)",
                (serve_id, user_id, feedback, False))
            self.conn.commit()
            print("反馈提交成功")
        except sqlite3.Error as e:
            print("反馈提交失败：", e)

    def update_feedback_status(self, serve_id):
        try:
            # 将 FINISH 列更新为 True
            self.cursor.execute("UPDATE serve SET FINISH = ? WHERE SERVEID = ?", (True, serve_id))
            self.conn.commit()
            print("反馈状态更新成功")
        except sqlite3.Error as e:
            print("反馈状态更新失败：", e)
    def get_all_feedback(self):
        try:
            self.cursor.execute("SELECT * FROM serve")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("获取反馈失败：", e)
            return []


    def update_notice(self, notice, operator_id):
        try:
            # 插入新的公告记录
            self.cursor.execute("INSERT INTO notice (NOTICE, OPRATORID, TIME) VALUES (?, ?, datetime('now'))",
                                (notice, operator_id))
            self.conn.commit()
            print("公告更新成功")
        except sqlite3.Error as e:
            print("公告更新失败：", e)

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

# YOLOv11 模型类
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
                font-size: 16px;
                color: #333;
            }
            QLineEdit {
                padding: 10px;
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

class FeedbackPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("反馈页面")
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

        # 反馈输入框
        self.feedback_input = QTextEdit()
        self.feedback_input.setPlaceholderText("请输入您的反馈内容")
        layout.addWidget(self.feedback_input)

        # 提交按钮
        self.submit_button = QPushButton("提交")
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        if self.main_window.current_user[2] == "Administrator":
            self.handle_feedback_button = QPushButton("处理反馈")
            self.handle_feedback_button.clicked.connect(self.on_handle_feedback)
            layout.addWidget(self.handle_feedback_button)

        self.setLayout(layout)

    def on_submit(self):
        # 获取反馈内容
        feedback = self.feedback_input.toPlainText()
        if not feedback:
            QMessageBox.warning(self, "提交失败", "反馈内容不能为空！")
            return

        # 获取用户 ID 和服务 ID（假设服务 ID 是自动生成的）
        user_id = self.main_window.current_user[0]
        serve_id = self.generate_serve_id()  # 生成唯一的服务 ID

        # 保存反馈到数据库
        self.main_window.db.add_feedback(serve_id, user_id, feedback)
        QMessageBox.information(self, "提交成功", "反馈已提交！")
        self.close()

    def generate_serve_id(self):
        # 生成唯一的服务 ID（可以根据需求自定义）
        import uuid
        return str(uuid.uuid4())

    def on_handle_feedback(self):
        # 打开处理反馈页面
        self.handle_feedback_page = HandleFeedbackPage(self.main_window)
        self.handle_feedback_page.show()

class HandleFeedbackPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("处理反馈")
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QTableWidget {
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

        # 反馈表格
        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(5)
        self.feedback_table.setHorizontalHeaderLabels(["服务ID", "用户ID", "反馈内容", "提交时间", "处理状态"])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.load_feedback_table()
        layout.addWidget(self.feedback_table)

        # 处理按钮
        self.handle_button = QPushButton("标记为已处理")
        self.handle_button.clicked.connect(self.on_handle)
        layout.addWidget(self.handle_button)

        self.setLayout(layout)

    def load_feedback_table(self):
        # 加载所有反馈
        feedbacks = self.main_window.db.get_all_feedback()
        self.feedback_table.setRowCount(len(feedbacks))
        for i, feedback in enumerate(feedbacks):
            for j, item in enumerate(feedback):
                self.feedback_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_handle(self):
        # 获取选中的反馈
        selected_row = self.feedback_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "操作失败", "请选择一条反馈！")
            return

        # 获取服务 ID
        serve_id = self.feedback_table.item(selected_row, 0).text()

        # 更新反馈状态
        self.main_window.db.update_feedback_status(serve_id)
        QMessageBox.information(self, "操作成功", "反馈已标记为已处理！")
        self.load_feedback_table()  # 刷新表格
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
        self.role_combo.addItems(["Administrator", "User"])
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
        latest_notice = self.main_window.db.get_latest_notice()
        if latest_notice:
            self.notice_label.setText(latest_notice[0])  # 显示公告内容
        else:
            self.notice_label.setText("公告：暂无公告")


        # 编辑公告按钮（仅管理员可见）
        if self.main_window.current_user[2] == "Administrator":
            self.edit_notice_button = QPushButton("编辑公告")
            self.edit_notice_button.clicked.connect(self.on_edit_notice)
            layout.addWidget(self.edit_notice_button)

        # 个人信息按钮
        self.profile_button = QPushButton("个人信息")
        self.profile_button.clicked.connect(self.on_profile)
        layout.addWidget(self.profile_button)

        # 处理反馈按钮
        if self.main_window.current_user[2] == "Administrator":
            self.handle_feedback_button = QPushButton("处理反馈")
            self.handle_feedback_button.clicked.connect(self.on_handle_feedback)
            layout.addWidget(self.handle_feedback_button)
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

    def on_handle_feedback(self):
        # 打开处理反馈页面
        self.handle_feedback_page = HandleFeedbackPage(self.main_window)
        self.handle_feedback_page.show()

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
        # 获取公告内容和操作者 ID
        notice = self.notice_edit.toPlainText()
        operator_id = self.main_window.current_user[0]  # 当前用户的 UID

        # 保存公告到数据库
        self.main_window.db.update_notice(notice, operator_id)

        # 更新主页的公告显示
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
        self.model = YOLOModel(model_path='./best.pt')  # 加载 YOLOv11 模型
        self.video_path = None  # 视频文件路径
        self.video_capture = None  # 视频捕获对象
        self.current_frame = None  # 当前帧
        self.timer = QTimer()  # 定时器，用于播放视频
        self.timer.timeout.connect(self.update_frame)  # 定时器触发时更新帧

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

        # 上传视频按钮
        self.upload_video_button = QPushButton("上传视频")
        self.upload_video_button.clicked.connect(self.on_upload_video)
        layout.addWidget(self.upload_video_button)

        # 视频播放控制
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.video_slider)

        # 播放/暂停按钮
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.on_play)
        layout.addWidget(self.play_button)

        # 截图按钮
        self.snapshot_button = QPushButton("截图并识别")
        self.snapshot_button.clicked.connect(self.on_snapshot)
        layout.addWidget(self.snapshot_button)

        # 显示视频帧
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # 上传图片按钮
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.on_upload)
        layout.addWidget(self.upload_button)

        # 反馈按钮
        self.feedback_button = QPushButton("提交反馈")
        self.feedback_button.clicked.connect(self.on_feedback)  # 绑定点击事件
        layout.addWidget(self.feedback_button)

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

    def on_upload_video(self):
        # 打开文件对话框，选择视频文件
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Videos (*.mp4 *.avi *.mov)")
        if self.video_path:
            # 初始化视频捕获对象
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                QMessageBox.warning(self, "错误", "无法打开视频文件！")
                return

            # 获取视频总帧数和帧率
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            # 设置滑块的最大值
            self.video_slider.setMaximum(self.total_frames)

            # 显示第一帧
            self.update_frame()

    def update_frame(self):
        # 读取当前帧
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            # 将 OpenCV 帧转换为 QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            # 显示帧
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def on_slider_changed(self, value):
        # 当滑块值改变时，跳转到指定帧
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.update_frame()

    def on_play(self):
        # 播放/暂停视频
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            self.timer.start(1000 // self.fps)  # 根据帧率设置定时器间隔
            self.play_button.setText("暂停")

    def on_snapshot(self):
        # 截取当前帧并进行识别
        if self.current_frame is not None:
            # 保存当前帧为临时文件
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, self.current_frame)

            # 调用 YOLO 模型进行识别
            class_name, confidence, save_dir = self.model.predict(temp_image_path)
            if class_name and confidence:
                # 显示识别结果
                result_image_path = f"{save_dir}/{temp_image_path.split('/')[-1]}"  # 假设保存的图片名称不变
                result_pixmap = QPixmap(result_image_path)
                self.video_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")
            else:
                self.result_label.setText("未检测到目标")

    def on_back(self):
        # 返回主页
        if self.video_capture:
            self.video_capture.release()  # 释放视频捕获对象
        self.main_window.setCentralWidget(HomePage(self.main_window))
        self.close()

    def on_feedback(self):
        # 打开反馈页面
        self.feedback_page = FeedbackPage(self.main_window)
        self.feedback_page.show()
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
        self.setGeometry(100, 100, 800, 600)  # 调整窗口大小
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