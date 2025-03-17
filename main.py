import sys
import sqlite3
import uuid

import cv2
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QSlider, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 数据库操作类
class Database:
    def __init__(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        except sqlite3.Error as e:
            print("数据库连接失败：", e)

    def get_all_predictions(self):
        """获取所有预测记录"""
        try:
            self.cursor.execute("SELECT * FROM prediction")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("获取预测记录失败：", e)
            return []

    def add_prediction(self, user_id, image_path, result):
        """插入预测记录"""
        try:
            predict_id = str(uuid.uuid4())  # 生成唯一的预测 ID
            self.cursor.execute(
                "INSERT INTO prediction (PREDICTID, USERID, IMAGEPATH, RESULT, PREDICTTIME) VALUES (?, ?, ?, ?, datetime('now'))",
                (predict_id, user_id, image_path, result)
            )
            self.conn.commit()
            print("预测记录保存成功")
        except sqlite3.Error as e:
            print("预测记录保存失败：", e)

    def get_user_count(self):
        """获取用户数量"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM user")
            return self.cursor.fetchone()[0]
        except sqlite3.Error as e:
            print("获取用户数量失败：", e)
            return 0

    def get_feedback_count(self):
        """获取反馈数量"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM serve")
            return self.cursor.fetchone()[0]
        except sqlite3.Error as e:
            print("获取反馈数量失败：", e)
            return 0

    def get_prediction_count(self):
        """获取预测次数"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM prediction")
            return self.cursor.fetchone()[0]
        except sqlite3.Error as e:
            print("获取预测次数失败：", e)
            return 0

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
        print("模型类别名称：", self.model.names)  # 打印类别名称

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
                print(f"检测到目标：{class_name}, 置信度：{confidence}")
                return class_name, confidence, results[0].save_dir
        print("未检测到目标")
        return None, None, None


# 登录页面
class LoginPage(QWidget):
    def __init__(self, db, main_window):
        super().__init__()
        self.db = db
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("登录页面")
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                font-size: 16px;
                color: #333;
            }
            QLineEdit {
                padding: 10px;
                font-size: 14px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #007bff;
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
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 卡片式布局
        card_frame = QFrame()
        card_layout = QVBoxLayout()
        card_frame.setLayout(card_layout)

        # 标题
        title_label = QLabel("用户登录")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #007bff;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title_label)

        # 身份选择
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Administrator", "User"])
        card_layout.addWidget(QLabel("选择身份："))
        card_layout.addWidget(self.role_combo)

        # UID 输入
        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("请输入用户 ID")
        card_layout.addWidget(QLabel("用户 ID："))
        card_layout.addWidget(self.uid_input)

        # 名称输入
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入用户名")
        card_layout.addWidget(QLabel("用户名称："))
        card_layout.addWidget(self.name_input)

        # 密码输入
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        card_layout.addWidget(QLabel("密码："))
        card_layout.addWidget(self.password_input)

        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.on_login)
        card_layout.addWidget(self.login_button)

        # 注册按钮
        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        card_layout.addWidget(self.register_button)

        # 将卡片添加到主布局
        main_layout.addWidget(card_frame)
        self.setLayout(main_layout)


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
                   background-color: #f5f5f5;
                   font-family: 'Segoe UI', sans-serif;
               }
               QLabel {
                   font-size: 16px;
                   color: #333;
               }
               QPushButton {
                   padding: 15px;
                   font-size: 14px;
                   color: white;
                   background-color: #007bff;
                   border: none;
                   border-radius: 5px;
               }
               QPushButton:hover {
                   background-color: #0056b3;
               }
               QFrame {
                   background-color: white;
                   border-radius: 10px;
                   padding: 20px;
                   margin: 10px;
                   border: 1px solid #ddd;
               }
           """)

        # 主布局
        main_layout = QVBoxLayout()

        # 公告区域
        notice_frame = QFrame()
        notice_layout = QVBoxLayout()
        notice_frame.setLayout(notice_layout)

        # 获取最新公告
        latest_notice = self.main_window.db.get_latest_notice()
        notice_text = latest_notice[0] if latest_notice else "暂无公告"

        self.notice_label = QLabel(f"公告：{notice_text}")
        self.notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice_layout.addWidget(self.notice_label)

        main_layout.addWidget(notice_frame)

        # 数据统计面板
        stats_frame = QFrame()
        stats_layout = QHBoxLayout()
        stats_frame.setLayout(stats_layout)

        # 用户数量卡片
        user_card = QFrame()
        user_card.setStyleSheet("background-color: #e3f2fd; border-radius: 10px; padding: 15px;")
        user_layout = QVBoxLayout()
        user_card.setLayout(user_layout)
        user_label = QLabel("用户数量")
        user_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        user_count = QLabel(str(self.main_window.db.get_user_count()))
        user_count.setStyleSheet("font-size: 24px; color: #007bff;")
        user_layout.addWidget(user_label)
        user_layout.addWidget(user_count)
        stats_layout.addWidget(user_card)

        # 反馈数量卡片
        feedback_card = QFrame()
        feedback_card.setStyleSheet("background-color: #fff3e0; border-radius: 10px; padding: 15px;")
        feedback_layout = QVBoxLayout()
        feedback_card.setLayout(feedback_layout)
        feedback_label = QLabel("反馈数量")
        feedback_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        feedback_count = QLabel(str(self.main_window.db.get_feedback_count()))
        feedback_count.setStyleSheet("font-size: 24px; color: #ff9800;")
        feedback_layout.addWidget(feedback_label)
        feedback_layout.addWidget(feedback_count)
        stats_layout.addWidget(feedback_card)

        # 预测次数卡片
        predict_card = QFrame()
        predict_card.setStyleSheet("background-color: #f0f4c3; border-radius: 10px; padding: 15px;")
        predict_layout = QVBoxLayout()
        predict_card.setLayout(predict_layout)
        predict_label = QLabel("预测次数")
        predict_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        predict_count = QLabel(str(self.main_window.db.get_prediction_count()))
        predict_count.setStyleSheet("font-size: 24px; color: #8bc34a;")
        predict_layout.addWidget(predict_label)
        predict_layout.addWidget(predict_count)
        stats_layout.addWidget(predict_card)

        main_layout.addWidget(stats_frame)

        # 功能按钮区域
        button_frame = QFrame()
        button_layout = QGridLayout()
        button_frame.setLayout(button_layout)

        self.setLayout(main_layout)

        # 个人信息按钮
        self.profile_button = QPushButton("个人信息")
        self.profile_button.clicked.connect(self.on_profile)
        button_layout.addWidget(self.profile_button, 0, 0)

        # 处理反馈按钮（仅管理员可见）
        if self.main_window.current_user[2] == "Administrator":
            self.handle_feedback_button = QPushButton("处理反馈")
            self.handle_feedback_button.clicked.connect(self.on_handle_feedback)
            button_layout.addWidget(self.handle_feedback_button, 0, 1)

        # 预测按钮
        self.predict_button = QPushButton("进行预测")
        self.predict_button.clicked.connect(self.on_predict)
        button_layout.addWidget(self.predict_button, 1, 0)

        # 预测记录按钮

        # 编辑公告按钮（仅管理员可见）
        if self.main_window.current_user[2] == "Administrator":
            self.edit_notice_button = QPushButton("编辑公告")
            self.edit_notice_button.clicked.connect(self.on_edit_notice)
            button_layout.addWidget(self.edit_notice_button, 1, 1)

        # 退出登录按钮
        self.logout_button = QPushButton("退出登录")
        self.logout_button.clicked.connect(self.on_logout)
        button_layout.addWidget(self.logout_button, 2, 1)

        if self.main_window.current_user[2] == "Administrator":
            self.prediction_record_button = QPushButton("预测记录")
            self.prediction_record_button.clicked.connect(self.on_prediction_record)
            button_layout.addWidget(self.prediction_record_button, 2, 0)


        main_layout.addWidget(button_frame)
        self.setLayout(main_layout)
    def on_prediction_record(self):
        """打开预测记录页面"""
        self.prediction_record_page = PredictionRecordPage(self.main_window)
        self.main_window.setCentralWidget(self.prediction_record_page)

    def on_logout(self):
        """退出登录"""
        self.main_window.current_user = None  # 清空当前用户信息
        self.main_window.setCentralWidget(LoginPage(self.main_window.db, self.main_window))
        self.close()
    def load_prediction_table(self):
        """加载预测记录"""
        predictions = self.main_window.db.get_all_predictions()
        self.prediction_table.setRowCount(len(predictions))
        # 确保列数已正确初始化为5，此处无需再次设置

        for i, prediction in enumerate(predictions):
            for j, item in enumerate(prediction):  # 遍历每一列数据
                self.prediction_table.setItem(i, j, QTableWidgetItem(str(item)))

        # 设置表格自适应列宽
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

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


class PredictionRecordPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("预测记录")
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                font-size: 16px;
                color: #333;
            }
            QTableWidget {
                font-size: 14px;
                border: 1px solid #ddd;
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

        # 主布局
        layout = QVBoxLayout()

        # 预测记录表格
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(5)
        self.prediction_table.setHorizontalHeaderLabels(["预测ID", "用户ID", "图片路径", "预测结果", "预测时间"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.load_prediction_table()
        layout.addWidget(self.prediction_table)

        # 返回按钮
        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def load_prediction_table(self):
        """加载预测记录"""
        predictions = self.main_window.db.get_all_predictions()
        self.prediction_table.setRowCount(len(predictions))
        for i, prediction in enumerate(predictions):
            for j, item in enumerate(prediction):
                self.prediction_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_back(self):
        """返回主页"""
        self.main_window.setCentralWidget(HomePage(self.main_window))
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

            # 添加选择用户的功能
            self.select_user_button = QPushButton("选择用户")
            self.select_user_button.clicked.connect(self.on_select_user)
            layout.addWidget(self.select_user_button)

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

    def on_select_user(self):
        # 获取选中的用户
        selected_row = self.user_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "操作失败", "请选择一个用户！")
            return

        # 获取选中的用户信息
        selected_uid = self.user_table.item(selected_row, 0).text()
        selected_name = self.user_table.item(selected_row, 1).text()
        selected_identity = self.user_table.item(selected_row, 2).text()
        selected_password = self.user_table.item(selected_row, 3).text()

        # 更新输入框中的信息
        self.uid_label.setText(f"用户ID：{selected_uid}")
        self.name_input.setText(selected_name)
        self.identity_input.setText(selected_identity)
        self.password_input.setText(selected_password)

        # 修改保存按钮的功能，使其更新选中的用户信息
        self.save_button.disconnect()
        self.save_button.clicked.connect(lambda: self.on_save_selected_user(selected_uid))

    def on_save_selected_user(self, uid):
        # 保存选中的用户信息
        name = self.name_input.text()
        identity = self.identity_input.text()
        password = self.password_input.text()

        self.main_window.db.update_user(uid, name, identity, password)
        QMessageBox.information(self, "保存成功", "用户信息已更新！")
        self.load_user_table()  # 刷新用户表格

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
        self.auto_tracking = False  # 自动跟踪标志

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
            QSlider {
                padding: 10px;
            }
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
            }
        """)

        # 布局
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_frame = QFrame()
        left_frame.setLayout(left_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        left_layout.addWidget(self.video_label)


        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border-radius: 10px;")
        left_layout.addWidget(self.image_label)

        # 右侧布局（按钮和控件）
        right_layout = QVBoxLayout()
        right_frame = QFrame()
        right_frame.setLayout(right_layout)

        # 自动跟踪按钮
        self.auto_track_button = QPushButton("自动跟踪")
        self.auto_track_button.clicked.connect(self.on_auto_track)
        right_layout.addWidget(self.auto_track_button)

        # 上传视频按钮
        self.upload_video_button = QPushButton("上传视频")
        self.upload_video_button.clicked.connect(self.on_upload_video)
        right_layout.addWidget(self.upload_video_button)

        # 视频播放控制
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.valueChanged.connect(self.on_slider_changed)
        right_layout.addWidget(self.video_slider)

        # 播放/暂停按钮
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.on_play)
        right_layout.addWidget(self.play_button)

        # 截图按钮
        self.snapshot_button = QPushButton("截图并识别")
        self.snapshot_button.clicked.connect(self.on_snapshot)
        right_layout.addWidget(self.snapshot_button)

        # 上传图片按钮
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.on_upload)
        right_layout.addWidget(self.upload_button)

        # 反馈按钮
        self.feedback_button = QPushButton("提交反馈")
        self.feedback_button.clicked.connect(self.on_feedback)
        right_layout.addWidget(self.feedback_button)

        # 预测结果
        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.result_label)

        # 返回按钮
        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        right_layout.addWidget(self.back_button)

        # 将左右的布局添加到主布局
        main_layout.addWidget(left_frame, 70)  # 左侧占 70% 宽度
        main_layout.addWidget(right_frame, 30)  # 右侧占 30% 宽度

        self.setLayout(main_layout)

    def on_auto_track(self):
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先上传视频！")
            return

        # 如果定时器未启动，则启动定时器
        if not self.timer.isActive():
            self.timer.start(1000 // self.fps)  # 根据帧率设置定时器间隔
            self.play_button.setText("暂停")

        # 设置自动跟踪标志
        self.auto_tracking = True
        QMessageBox.information(self, "提示", "自动跟踪已启动！")
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
        if self.video_capture is None:
            return

        # 读取当前帧
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()  # 停止定时器
            QMessageBox.information(self, "提示", "视频播放完毕！")
            return

        self.current_frame = frame

        # 如果启用了自动跟踪
        if self.auto_tracking:
            # 使用 YOLO 模型进行目标检测
            results = self.model.model.track(frame, persist=True)
            print("模型输出：", results)  # 打印模型输出
            if len(results) > 0:
                boxes = results[0].boxes
                print("检测框：", boxes)  # 打印检测框信息
                if len(boxes) > 0:
                    for box in boxes:
                        # 提取类别 ID 和类别名称
                        class_id = int(box.cls[0])  # 类别 ID
                        class_name = self.model.model.names[class_id]  # 类别名称
                        confidence = float(box.conf[0])  # 置信度
                        print(f"类别 ID: {class_id}, 类别名称: {class_name}, 置信度: {confidence}")  # 打印类别信息

                        # 绘制检测框
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色矩形框
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 显示类别名称和置信度

        # 将 OpenCV 帧转换为 QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        # 显示帧（保持原始分辨率）
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

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
        """截取当前帧并进行识别"""
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
                self.image_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")

                # 保存预测记录到数据库
                user_id = self.main_window.current_user[0]  # 当前用户 ID
                self.main_window.db.add_prediction(user_id, temp_image_path, class_name)
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
        """上传图片并进行预测"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # 显示原始图片
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

            # 调用 YOLO 模型进行预测
            class_name, confidence, save_dir = self.model.predict(file_path)
            if class_name and confidence:
                # 显示预测结果图片
                result_image_path = f"{save_dir}/{file_path.split('/')[-1]}"  # 假设保存的图片名称不变
                result_pixmap = QPixmap(result_image_path)
                self.image_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")

                # 保存预测记录到数据库
                user_id = self.main_window.current_user[0]  # 当前用户 ID
                self.main_window.db.add_prediction(user_id, file_path, class_name)
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