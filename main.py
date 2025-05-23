import sys
import sqlite3
import uuid
from dotenv import load_dotenv, set_key
import cv2
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QSlider, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 加载 .env 文件
load_dotenv()

# 获取模型路径
MODEL_PATH = os.getenv("MODEL_PATH", "./best.pt")

# 全局样式表
GLOBAL_STYLESHEET = """
    QWidget {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #000000;
    }
    QLabel#contact_label {
        font-size: 25px;
        color: #444444;
        padding: 5px;
        background-color: #f8f8f8;
        border-radius: 4px;
    }
    QLabel#support_label {
        font-size: 25px;
        color: #888888;
        margin-top: 10px;
    }
    QLabel {
        color: #000000;
        font-size: 25px;
    }
    QLineEdit, QTextEdit, QComboBox {
        padding: 8px;
        font-size: 25px;
        border: 1px solid #cccccc;
        border-radius: 6px;
        background-color: #f9f9f9;
        color: #000000;
    }
    QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
        border: 1px solid #0078d4;
        background-color: #ffffff;
    }
    QPushButton {
        padding: 10px;
        font-size: 25px;
        color: #000000;
        background-color: #e0e0e0;
        border: none;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    QPushButton:hover {
        background-color: #d0d0d0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    QPushButton:pressed {
        background-color: #c0c0c0;
    }
    QTableWidget {
        font-size: 25px;
        border: 1px solid #cccccc;
        border-radius: 6px;
        background-color: #ffffff;
        color: #000000;
    }
    QSlider {
        padding: 8px;
    }
    QFrame {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
    }
    QTableWidget::item {
        padding: 8px;
    }
    QHeaderView::section {
        background-color: #f0f0f0;
        padding: 4px;
        border: 1px solid #e0e0e0;
        font-size: 14px;
        min-height: 40px;
        text-align: center;
    }
"""


# 数据库操作类
class Database:
    def __init__(self, db_name):
        """初始化数据库连接并创建所有必要的表"""
        try:
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()

            # 创建用户表（仅限普通用户）
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user (
                    UID TEXT PRIMARY KEY,
                    UNAME TEXT,
                    UIDENTITY TEXT,
                    UPD TEXT,
                    EMAIL TEXT
                )
            """)

            # 创建管理员表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin (
                    AID TEXT PRIMARY KEY,
                    ANAME TEXT,
                    APD TEXT,
                    EMAIL TEXT
                )
            """)

            # 创建预测记录表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction (
                    PREDICTID TEXT PRIMARY KEY,
                    USERID TEXT,
                    IMAGEPATH TEXT,
                    RESULT TEXT,
                    PREDICTTIME TEXT
                )
            """)

            # 创建反馈表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS serve (
                    SERVEID TEXT PRIMARY KEY,
                    USERID TEXT,
                    FEEDBACK TEXT,
                    SERVETIME TEXT,
                    FINISH BOOLEAN
                )
            """)

            # 创建公告表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS notice (
                    NOTICE TEXT,
                    OPRATORID TEXT,
                    TIME TEXT
                )
            """)

            self.conn.commit()
            print("数据库连接成功，所有表已创建")
        except sqlite3.Error as e:
            print("数据库连接或表创建失败：", e)

    def get_all_predictions(self):
        """获取所有预测记录"""
        try:
            self.cursor.execute("SELECT PREDICTID, USERID, IMAGEPATH, RESULT, PREDICTTIME FROM prediction")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("获取预测记录失败：", e)
            return []

    def get_next_uid(self):
        """获取下一个可用的用户ID，从1001开始自增"""
        try:
            self.cursor.execute("SELECT MAX(CAST(UID AS INTEGER)) FROM user")
            max_uid = self.cursor.fetchone()[0]
            if max_uid is None:
                return "1001"  # 如果表为空，从1001开始
            next_uid = int(max_uid) + 1
            if next_uid < 1001:
                return "1001"  # 确保最小ID为1001
            return str(next_uid)
        except sqlite3.Error as e:
            print("获取下一个UID失败：", e)
            return "1001"  # 默认返回1001以确保系统继续运行

    def get_next_aid(self):
        """获取下一个可用的管理员ID，从1001开始自增"""
        try:
            self.cursor.execute("SELECT MAX(CAST(AID AS INTEGER)) FROM admin")
            max_aid = self.cursor.fetchone()[0]
            if max_aid is None:
                return "1001"  # 如果表为空，从1001开始
            next_aid = int(max_aid) + 1
            if next_aid < 1001:
                return "1001"  # 确保最小ID为1001
            return str(next_aid)
        except sqlite3.Error as e:
            print("获取下一个AID失败：", e)
            return "1001"  # 默认返回1001以确保系统继续运行

    def add_prediction(self, user_id, image_path, result):
        """添加预测记录"""
        try:
            predict_id = str(uuid.uuid4())
            self.cursor.execute(
                "INSERT INTO prediction (PREDICTID, USERID, IMAGEPATH, RESULT, PREDICTTIME) VALUES (?, ?, ?, ?, datetime('now'))",
                (predict_id, user_id, image_path, result)
            )
            self.conn.commit()
            print("预测记录保存成功")
        except sqlite3.Error as e:
            print("预测记录保存失败：", e)

    def get_user_count(self):
        """获取用户和管理员总数"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM user")
            user_count = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT COUNT(*) FROM admin")
            admin_count = self.cursor.fetchone()[0]
            return user_count + admin_count
        except sqlite3.Error as e:
            print("获取用户数量失败：", e)
            return 0

    def get_feedback_count(self):
        """获取反馈总数"""
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
        """获取最新公告"""
        try:
            self.cursor.execute("SELECT NOTICE FROM notice ORDER BY TIME DESC LIMIT 1")
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("获取最新公告失败：", e)
            return None

    def add_feedback(self, serve_id, user_id, feedback):
        """添加用户反馈"""
        try:
            self.cursor.execute(
                "INSERT INTO serve (SERVEID, USERID, FEEDBACK, SERVETIME, FINISH) VALUES (?, ?, ?, datetime('now'), ?)",
                (serve_id, user_id, feedback, False))
            self.conn.commit()
            print("反馈提交成功")
        except sqlite3.Error as e:
            print("反馈提交失败：", e)

    def update_feedback_status(self, serve_id):
        """更新反馈状态为已处理"""
        try:
            self.cursor.execute("UPDATE serve SET FINISH = ? WHERE SERVEID = ?", (True, serve_id))
            self.conn.commit()
            print("反馈状态更新成功")
        except sqlite3.Error as e:
            print("反馈状态更新失败：", e)

    def get_all_feedback(self):
        """获取所有反馈记录"""
        try:
            self.cursor.execute("SELECT * FROM serve")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("获取反馈失败：", e)
            return []

    def update_notice(self, notice, operator_id):
        """更新公告内容"""
        try:
            self.cursor.execute("INSERT INTO notice (NOTICE, OPRATORID, TIME) VALUES (?, ?, datetime('now'))",
                                (notice, operator_id))
            self.conn.commit()
            print("公告更新成功")
        except sqlite3.Error as e:
            print("公告更新失败：", e)

    def add_user(self, name, identity, password, email):
        """添加新用户（仅普通用户）"""
        try:
            uid = self.get_next_uid()
            self.cursor.execute("INSERT INTO user (UID, UNAME, UIDENTITY, UPD, EMAIL) VALUES (?, ?, ?, ?, ?)",
                                (uid, name, identity, password, email))
            self.conn.commit()
            print("用户添加成功")
            return uid
        except sqlite3.Error as e:
            print("用户添加失败：", e)
            return None

    def add_admin(self, name, password, email):
        """添加新管理员"""
        try:
            aid = self.get_next_aid()
            self.cursor.execute("INSERT INTO admin (AID, ANAME, APD, EMAIL) VALUES (?, ?, ?, ?)",
                                (aid, name, password, email))
            self.conn.commit()
            print("管理员添加成功")
            return aid
        except sqlite3.Error as e:
            print("管理员添加失败：", e)
            return None

    def get_user(self, uid):
        """根据用户ID获取用户信息"""
        try:
            self.cursor.execute("SELECT UID, UNAME, UIDENTITY, UPD, EMAIL FROM user WHERE UID = ?", (uid,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("查询用户失败：", e)
            return None

    def get_admin(self, aid):
        """根据管理员ID获取管理员信息"""
        try:
            self.cursor.execute("SELECT AID, ANAME, 'Administrator', APD, EMAIL FROM admin WHERE AID = ?", (aid,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("查询管理员失败：", e)
            return None

    def get_user_by_role_and_password(self, uid, name, identity, password):
        """根据用户信息验证用户或管理员"""
        try:
            if identity == "Administrator":
                self.cursor.execute(
                    "SELECT AID, ANAME, 'Administrator', APD, EMAIL FROM admin WHERE AID = ? AND ANAME = ? AND APD = ?",
                    (uid, name, password))
            else:
                self.cursor.execute(
                    "SELECT UID, UNAME, UIDENTITY, UPD, EMAIL FROM user WHERE UID = ? AND UNAME = ? AND UIDENTITY = ? AND UPD = ?",
                    (uid, name, identity, password))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print("查询用户失败：", e)
            return None

    def get_all_users(self):
        """获取所有普通用户信息"""
        try:
            self.cursor.execute("SELECT UID, UNAME, UIDENTITY, UPD, EMAIL FROM user")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("查询所有用户失败：", e)
            return []

    def get_all_admins(self):
        """获取所有管理员信息"""
        try:
            self.cursor.execute("SELECT AID, ANAME, 'Administrator', APD, EMAIL FROM admin")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("查询所有管理员失败：", e)
            return []

    def get_users_by_role(self, role):
        """根据角色获取用户信息"""
        try:
            if role == "Administrator":
                self.cursor.execute("SELECT AID, ANAME, 'Administrator', APD, EMAIL FROM admin")
            else:
                self.cursor.execute("SELECT UID, UNAME, UIDENTITY, UPD, EMAIL FROM user WHERE UIDENTITY = ?", (role,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"查询{role}用户失败：", e)
            return []

    def update_user(self, uid, name, identity, password, email):
        """更新用户信息"""
        try:
            self.cursor.execute("UPDATE user SET UNAME = ?, UIDENTITY = ?, UPD = ?, EMAIL = ? WHERE UID = ?",
                                (name, identity, password, email, uid))
            self.conn.commit()
            print("用户信息更新成功")
        except sqlite3.Error as e:
            print("用户信息更新失败：", e)

    def update_admin(self, aid, name, password, email):
        """更新管理员信息"""
        try:
            self.cursor.execute("UPDATE admin SET ANAME = ?, APD = ?, EMAIL = ? WHERE AID = ?",
                                (name, password, email, aid))
            self.conn.commit()
            print("管理员信息更新成功")
        except sqlite3.Error as e:
            print("管理员信息更新失败：", e)

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
        print("数据库连接关闭")


# YOLOv11 模型类（未修改）
class YOLOModel:
    def __init__(self, model_path):
        """初始化YOLO模型"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.class_name_map = {
            "Chihuahua": "吉娃娃",
            "Japanese_spaniel": "日本猎犬",
            "Maltese_dog": "马尔济斯犬",
            "Pekinese": "北京犬",
            "Shih-Tzu": "西施犬",
            "Blenheim_spaniel": "布伦海姆西班牙猎犬",
            "papillon": "蝴蝶犬",
            "toy_terrier": "玩具梗犬",
            "Rhodesian_ridgeback": "罗得西亚背脊犬",
            "Afghan_hound": "阿富汗猎犬",
            "basset": "巴吉度猎犬",
            "beagle": "比格犬",
            "bloodhound": "寻血猎犬",
            "bluetick": "蓝蒂克猎犬",
            "black-and-tan_coonhound": "黑棕浣熊猎犬",
            "Walker_hound": "沃克猎犬",
            "English_foxhound": "英国猎狐犬",
            "redbone": "红骨猎犬",
            "borzoi": "波索尔猎犬",
            "Irish_wolfhound": "爱尔兰狼犬",
            "Italian_greyhound": "意大利灰狗",
            "whippet": "惠比特犬",
            "Ibizan_hound": "伊比莎猎犬",
            "Norwegian_elkhound": "挪威猎犬",
            "otterhound": "水獭猎犬",
            "Saluki": "萨路基犬",
            "Scottish_deerhound": "苏格兰鹿犬",
            "Weimaraner": "魏玛犬",
            "Staffordshire_bullterrier": "斯塔福郡斗牛梗",
            "American_Staffordshire_terrier": "美国斯塔福郡梗",
            "Bedlington_terrier": "贝灵顿梗",
            "Border_terrier": "边境梗",
            "Kerry_blue_terrier": "凯利蓝梗",
            "Irish_terrier": "爱尔兰梗",
            "Norfolk_terrier": "诺福克梗",
            "Norwich_terrier": "诺维奇梗",
            "Yorkshire_terrier": "约克夏梗",
            "wire-haired_fox_terrier": "刚毛狐梗",
            "Lakeland_terrier": "莱克兰梗",
            "Sealyham_terrier": "西利哈姆梗",
            "Airedale": "艾尔代尔梗",
            "cairn": "凯恩梗",
            "Australian_terrier": "澳大利亚梗",
            "Dandie_Dinmont": "丹迪丁蒙梗",
            "Boston_bull": "波士顿斗牛犬",
            "miniature_schnauzer": "迷你雪纳瑞",
            "giant_schnauzer": "巨型雪纳瑞",
            "standard_schnauzer": "标准雪纳瑞",
            "Scotch_terrier": "苏格兰梗",
            "Tibetan_terrier": "西藏梗",
            "silky_terrier": "丝毛梗",
            "soft-coated_wheaten_terrier": "软毛麦色梗",
            "West_Highland_white_terrier": "西高地白梗",
            "Lhasa": "拉萨犬",
            "flat-coated_retriever": "平毛寻回犬",
            "curly-coated_retriever": "卷毛寻回犬",
            "golden_retriever": "金毛寻回犬",
            "Labrador_retriever": "拉布拉多寻回犬",
            "Chesapeake_Bay_retriever": "切萨皮克湾寻回犬",
            "German_short-haired_pointer": "德国短毛指示犬",
            "vizsla": "维兹拉犬",
            "English_setter": "英国塞特犬",
            "Irish_setter": "爱尔兰塞特犬",
            "Gordon_setter": "戈登塞特犬",
            "Brittany_spaniel": "布列塔尼猎犬",
            "clumber": "克拉姆猎犬",
            "English_springer": "英国史宾格犬",
            "Welsh_springer_spaniel": "威尔士史宾格犬",
            "cocker_spaniel": "可卡犬",
            "Sussex_spaniel": "苏塞克斯猎犬",
            "Irish_water_spaniel": "爱尔兰水猎犬",
            "kuvasz": "库瓦斯犬",
            "schipperke": "斯希珀基犬",
            "groenendael": "格罗宁达尔犬",
            "malinois": "马里努阿犬",
            "briard": "布里亚德犬",
            "kelpie": "凯尔皮犬",
            "komondor": "科蒙多犬",
            "Old_English_sheepdog": "古老英国牧羊犬",
            "Shetland_sheepdog": "设得兰牧羊犬",
            "collie": "柯利犬",
            "Border_collie": "边境柯利犬",
            "Bouvier_des_Flandres": "弗兰德斯牧牛犬",
            "Rottweiler": "罗威纳犬",
            "German_shepherd": "德国牧羊犬",
            "Doberman": "杜宾犬",
            "miniature_pinscher": "迷你杜宾犬",
            "Greater_Swiss_Mountain_dog": "大瑞士山地犬",
            "Bernese_mountain_dog": "伯恩山犬",
            "Appenzeller": "阿彭策尔山犬",
            "EntleBucher": "恩特布赫山犬",
            "boxer": "拳师犬",
            "bull_mastiff": "斗牛獒",
            "Tibetan_mastiff": "藏獒",
            "French_bulldog": "法国斗牛犬",
            "Great_Dane": "大丹犬",
            "Saint_Bernard": "圣伯纳犬",
            "Eskimo_dog": "爱斯基摩犬",
            "malamute": "马拉缪犬",
            "Siberian_husky": "西伯利亚哈士奇",
            "affenpinscher": "阿芬犬",
            "basenji": "巴辛吉犬",
            "pug": "巴哥犬",
            "Leonberg": "莱昂伯格犬",
            "Newfoundland": "纽芬兰犬",
            "Great_Pyrenees": "大白熊犬",
            "Samoyed": "萨摩耶犬",
            "Pomeranian": "博美犬",
            "chow": "松狮犬",
            "keeshond": "荷兰毛狮犬",
            "Brabancon_griffon": "布拉邦松格里芬犬",
            "Pembroke": "彭布罗克威尔士柯基",
            "Cardigan": "卡迪根威尔士柯基",
            "toy_poodle": "玩具贵宾犬",
            "miniature_poodle": "迷你贵宾犬",
            "standard_poodle": "标准贵宾犬",
            "Mexican_hairless": "墨西哥无毛犬",
            "dingo": "澳洲野犬",
            "dhole": "亚洲野犬",
            "African_hunting_dog": "非洲猎犬"
        }
        print("模型类别名称：", self.model.names)
        self._is_tracking = False

    def predict(self, image_path):
        """对图片进行预测，确保禁用跟踪"""
        self.reset_model()
        results = self.model.predict(source=image_path, conf=0.1, save=True, show=False, stream=False)
        if len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                class_id = int(boxes.cls[0])
                confidence = float(boxes.conf[0])
                class_name = results[0].names[class_id]
                chinese_class_name = self.class_name_map.get(class_name, class_name)
                print(f"检测到目标：{chinese_class_name}, 置信度：{confidence}")
                return chinese_class_name, confidence, results[0].save_dir
        print("未检测到目标")
        return None, None, None

    def track(self, frame):
        """对视频帧进行跟踪"""
        results = self.model.track(source=frame, conf=0.1, persist=True, stream=False)
        return results

    def reset_model(self):
        """重置模型以清除所有跟踪状态"""
        self.model = YOLO(self.model_path)
        if hasattr(self.model, 'trackers'):
            self.model.trackers = None
        print("模型状态已重置")


# 登录页面
class LoginPage(QWidget):
    def __init__(self, db, main_window):
        """初始化登录页面"""
        super().__init__()
        self.db = db
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("登录页面")
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(20)

        card_frame = QFrame()
        card_layout = QVBoxLayout()
        card_frame.setLayout(card_layout)

        title_label = QLabel("用户登录")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #000000;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title_label)

        self.role_combo = QComboBox()
        self.role_combo.addItems(["Administrator", "User"])
        card_layout.addWidget(QLabel("选择身份："))
        card_layout.addWidget(self.role_combo)

        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("请输入用户 ID")
        card_layout.addWidget(QLabel("用户 ID："))
        card_layout.addWidget(self.uid_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入用户名")
        card_layout.addWidget(QLabel("用户名称："))
        card_layout.addWidget(self.name_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        card_layout.addWidget(QLabel("密码："))
        card_layout.addWidget(self.password_input)

        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.on_login)
        card_layout.addWidget(self.login_button)

        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        card_layout.addWidget(self.register_button)

        self.guest_button = QPushButton("游客登录")
        self.guest_button.clicked.connect(self.on_guest_login)
        card_layout.addWidget(self.guest_button)

        main_layout.addWidget(card_frame)
        self.setLayout(main_layout)

    def on_guest_login(self):
        """游客登录逻辑"""
        self.main_window.current_user = ("guest", "Guest", "Guest", "", "guest@example.com")
        self.main_window.open_home_page()
        self.close()

    def on_login(self):
        """用户登录逻辑"""
        uid = self.uid_input.text()
        name = self.name_input.text()
        password = self.password_input.text()
        identity = self.role_combo.currentText()

        if not uid or not name or not password:
            QMessageBox.warning(self, "登录失败", "用户ID、用户名或密码不能为空！")
            return

        try:
            user = self.db.get_user_by_role_and_password(uid, name, identity, password)
            if user:
                QMessageBox.information(self, "登录成功", f"欢迎 {identity} {uid}！")
                self.main_window.current_user = user
                self.main_window.open_home_page()
                self.close()
            else:
                QMessageBox.warning(self, "登录失败", "用户ID、用户名、身份或密码错误！")
        except Exception as e:
            print("登录出错：", e)
            QMessageBox.critical(self, "登录错误", f"登录失败：{str(e)}")

    def on_register(self):
        """打开注册页面"""
        self.register_page = RegisterPage(self.db)
        self.register_page.show()


# 注册页面
class RegisterPage(QWidget):
    def __init__(self, db):
        """初始化注册页面"""
        super().__init__()
        self.db = db
        self.initUI()

    def initUI(self):
        self.setWindowTitle("注册页面")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.role_combo = QComboBox()
        self.role_combo.addItems(["User", "Administrator"])
        self.role_combo.currentTextChanged.connect(self.toggle_invite_code_field)
        layout.addWidget(QLabel("选择身份："))
        layout.addWidget(self.role_combo)

        self.uid_label = QLabel(f"您的用户ID将是: {self.db.get_next_uid()}")
        self.uid_label.setStyleSheet("font-size: 14px; color: #000000; font-weight: bold;")
        self.uid_label.setVisible(self.role_combo.currentText() == "User")
        layout.addWidget(self.uid_label)

        self.aid_label = QLabel(f"您的管理员ID将是: {self.db.get_next_aid()}")
        self.aid_label.setStyleSheet("font-size: 14px; color: #000000; font-weight: bold;")
        self.aid_label.setVisible(self.role_combo.currentText() == "Administrator")
        layout.addWidget(self.aid_label)

        self.invite_code_label = QLabel("邀请码：")
        self.invite_code_input = QLineEdit()
        self.invite_code_input.setPlaceholderText("请输入管理员邀请码")
        self.invite_code_label.setVisible(False)
        self.invite_code_input.setVisible(False)

        invite_code_layout = QHBoxLayout()
        invite_code_layout.addWidget(self.invite_code_label)
        invite_code_layout.addWidget(self.invite_code_input)
        layout.addLayout(invite_code_layout)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入用户名")
        layout.addWidget(QLabel("用户名称："))
        layout.addWidget(self.name_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("请输入邮箱")
        layout.addWidget(QLabel("邮箱："))
        layout.addWidget(self.email_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.password_input)

        self.register_button = QPushButton("注册")
        self.register_button.clicked.connect(self.on_register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def toggle_invite_code_field(self, role):
        """根据选择的角色显示/隐藏邀请码输入框和ID标签"""
        self.invite_code_label.setVisible(role == "Administrator")
        self.invite_code_input.setVisible(role == "Administrator")
        self.uid_label.setVisible(role == "User")
        self.aid_label.setVisible(role == "Administrator")
        self.uid_label.setText(f"您的用户ID将是: {self.db.get_next_uid()}")
        self.aid_label.setText(f"您的管理员ID将是: {self.db.get_next_aid()}")

    def on_register(self):
        """用户注册逻辑"""
        identity = self.role_combo.currentText()
        name = self.name_input.text()
        email = self.email_input.text()
        password = self.password_input.text()
        invite_code = self.invite_code_input.text() if identity == "Administrator" else ""

        if not name or not password or not email:
            QMessageBox.warning(self, "注册失败", "用户名、密码和邮箱不能为空！")
            return

        if "@" not in email or "." not in email:
            QMessageBox.warning(self, "注册失败", "请输入有效的邮箱地址！")
            return

        if identity == "Administrator" and invite_code != "walkx030724":
            QMessageBox.warning(self, "注册失败", "管理员邀请码不正确！")
            return

        try:
            if identity == "Administrator":
                assigned_id = self.db.add_admin(name, password, email)
                id_label = "管理员ID"
            else:
                assigned_id = self.db.add_user(name, identity, password, email)
                id_label = "用户ID"

            if assigned_id is None:
                raise Exception("ID分配失败")

            confirm_msg = f"您将注册为:\n\n{id_label}: {assigned_id}\n用户名: {name}\n邮箱: {email}\n身份: {identity}"
            reply = QMessageBox.question(
                self, "确认注册", confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("注册成功")
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setText(
                    f"注册成功！\n\n{id_label}: {assigned_id}\n用户名: {name}\n邮箱: {email}\n身份: {identity}")
                copy_button = msg_box.addButton("复制ID", QMessageBox.ButtonRole.ActionRole)
                copy_button.setStyleSheet("padding: 5px;")
                msg_box.addButton(QMessageBox.StandardButton.Ok)
                msg_box.exec()

                if msg_box.clickedButton() == copy_button:
                    clipboard = QApplication.clipboard()
                    clipboard.setText(assigned_id)
                    QMessageBox.information(self, "已复制", "ID已复制到剪贴板！")
                self.close()

        except Exception as e:
            print("注册出错：", e)
            QMessageBox.critical(self, "注册错误", f"注册过程中发生错误: {str(e)}")


# 反馈页面（未修改）
class FeedbackPage(QWidget):
    def __init__(self, main_window):
        """初始化反馈页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("反馈页面")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        contact_label = QLabel("如需紧急帮助，请联系管理员: a@likaix.in")
        contact_label.setStyleSheet("font-size: 14px; color: #FF0000;")
        layout.addWidget(contact_label)

        self.feedback_input = QTextEdit()
        self.feedback_input.setPlaceholderText("请输入您的反馈内容")
        layout.addWidget(self.feedback_input)

        self.submit_button = QPushButton("提交")
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        if self.main_window.current_user[2] == "Administrator":
            self.handle_feedback_button = QPushButton("处理反馈")
            self.handle_feedback_button.clicked.connect(self.on_handle_feedback)
            layout.addWidget(self.handle_feedback_button)

        self.setLayout(layout)

    def on_submit(self):
        """提交反馈逻辑"""
        feedback = self.feedback_input.toPlainText()
        if not feedback:
            QMessageBox.warning(self, "提交失败", "反馈内容不能为空！")
            return

        user_id = self.main_window.current_user[0]
        serve_id = str(uuid.uuid4())
        self.main_window.db.add_feedback(serve_id, user_id, feedback)
        QMessageBox.information(self, "提交成功", "反馈已提交！")
        self.close()

    def on_handle_feedback(self):
        """打开处理反馈页面"""
        self.handle_feedback_page = HandleFeedbackPage(self.main_window)
        self.handle_feedback_page.show()


# 处理反馈页面（未修改）
class HandleFeedbackPage(QWidget):
    def __init__(self, main_window):
        """初始化处理反馈页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("处理反馈")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(5)
        self.feedback_table.setHorizontalHeaderLabels(["服务ID", "用户ID", "反馈内容", "提交时间", "处理状态"])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feedback_table.horizontalHeader().setMinimumHeight(40)
        self.feedback_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_table.horizontalHeader().setDefaultSectionSize(150)
        self.load_feedback_table()
        layout.addWidget(self.feedback_table)

        self.handle_button = QPushButton("标记为已处理")
        self.handle_button.clicked.connect(self.on_handle)
        layout.addWidget(self.handle_button)

        self.setLayout(layout)

    def load_feedback_table(self):
        """加载反馈表格数据"""
        feedbacks = self.main_window.db.get_all_feedback()
        self.feedback_table.setRowCount(len(feedbacks))
        for i, feedback in enumerate(feedbacks):
            for j, item in enumerate(feedback):
                self.feedback_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_handle(self):
        """标记反馈为已处理"""
        selected_row = self.feedback_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "操作失败", "请选择一条反馈！")
            return

        serve_id = self.feedback_table.item(selected_row, 0).text()
        self.main_window.db.update_feedback_status(serve_id)
        QMessageBox.information(self, "操作成功", "反馈已标记为已处理！")
        self.load_feedback_table()


# 主页（未修改）
class HomePage(QWidget):
    def __init__(self, main_window):
        """初始化主页"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("主页")
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)

        notice_frame = QFrame()
        notice_layout = QVBoxLayout()
        notice_frame.setLayout(notice_layout)

        latest_notice = self.main_window.db.get_latest_notice()
        notice_text = latest_notice[0] if latest_notice else "暂无公告"
        self.notice_label = QLabel(f"{notice_text}")
        self.notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice_layout.addWidget(self.notice_label)
        main_layout.addWidget(notice_frame)

        stats_frame = QFrame()
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        stats_frame.setLayout(stats_layout)

        user_card = QFrame()
        user_layout = QVBoxLayout()
        user_card.setLayout(user_layout)
        user_label = QLabel("用户数量")
        user_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        user_count = QLabel(str(self.main_window.db.get_user_count()))
        user_count.setStyleSheet("font-size: 28px; color: #000000;")
        user_layout.addWidget(user_label)
        user_layout.addWidget(user_count)
        stats_layout.addWidget(user_card)

        feedback_card = QFrame()
        feedback_layout = QVBoxLayout()
        feedback_card.setLayout(feedback_layout)
        feedback_label = QLabel("反馈数量")
        feedback_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        feedback_count = QLabel(str(self.main_window.db.get_feedback_count()))
        feedback_count.setStyleSheet("font-size: 28px; color: #000000;")
        feedback_layout.addWidget(feedback_label)
        feedback_layout.addWidget(feedback_count)
        stats_layout.addWidget(feedback_card)

        predict_card = QFrame()
        predict_layout = QVBoxLayout()
        predict_card.setLayout(predict_layout)
        predict_label = QLabel("预测次数")
        predict_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        predict_count = QLabel(str(self.main_window.db.get_prediction_count()))
        predict_count.setStyleSheet("font-size: 28px; color: #000000;")
        predict_layout.addWidget(predict_label)
        predict_layout.addWidget(predict_count)
        stats_layout.addWidget(predict_card)

        main_layout.addWidget(stats_frame)

        button_frame = QFrame()
        button_layout = QGridLayout()
        button_layout.setSpacing(10)
        button_frame.setLayout(button_layout)

        if self.main_window.current_user[2] != "Guest":
            self.profile_button = QPushButton("个人信息")
            self.profile_button.clicked.connect(self.on_profile)
            button_layout.addWidget(self.profile_button, 0, 0)

        if self.main_window.current_user[2] == "Administrator":
            self.handle_feedback_button = QPushButton("处理反馈")
            self.handle_feedback_button.clicked.connect(self.on_handle_feedback)
            button_layout.addWidget(self.handle_feedback_button, 0, 1)

        if self.main_window.current_user[2] != "Guest":
            self.predict_button = QPushButton("进行预测")
            self.predict_button.clicked.connect(self.on_predict)
            button_layout.addWidget(self.predict_button, 1, 0)

        if self.main_window.current_user[2] == "User":
            self.view_feedback_button = QPushButton("查看反馈状态")
            self.view_feedback_button.clicked.connect(self.on_view_feedback)
            button_layout.addWidget(self.view_feedback_button, 1, 1)

        if self.main_window.current_user[2] == "Administrator":
            self.edit_notice_button = QPushButton("编辑公告")
            self.edit_notice_button.clicked.connect(self.on_edit_notice)
            button_layout.addWidget(self.edit_notice_button, 2, 0)

        if self.main_window.current_user[2] == "Administrator":
            self.upload_model_button = QPushButton("上传模型")
            self.upload_model_button.clicked.connect(self.on_upload_model)
            button_layout.addWidget(self.upload_model_button, 2, 1)

        self.logout_button = QPushButton("退出登录")
        self.logout_button.clicked.connect(self.on_logout)
        button_layout.addWidget(self.logout_button, 3, 1)

        if self.main_window.current_user[2] == "Administrator":
            self.prediction_record_button = QPushButton("预测记录")
            self.prediction_record_button.clicked.connect(self.on_prediction_record)
            button_layout.addWidget(self.prediction_record_button, 3, 0)

        main_layout.addWidget(button_frame)

        contact_frame = QFrame()
        contact_layout = QVBoxLayout()
        contact_frame.setLayout(contact_layout)

        contact_label = QLabel("管理员联系方式: a@likaix.in")
        contact_label.setStyleSheet("font-size: 25px; color: #666666;")
        contact_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        contact_layout.addWidget(contact_label)

        main_layout.addWidget(contact_frame)
        self.setLayout(main_layout)

    def on_upload_model(self):
        """上传模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt)")
        if file_path:
            set_key(".env", "MODEL_PATH", file_path)
            QMessageBox.information(self, "上传成功", "模型文件已上传并更新！")

    def on_view_feedback(self):
        """查看反馈状态"""
        self.view_feedback_page = ViewFeedbackPage(self.main_window)
        self.main_window.setCentralWidget(self.view_feedback_page)

    def on_prediction_record(self):
        """查看预测记录"""
        self.prediction_record_page = PredictionRecordPage(self.main_window)
        self.main_window.setCentralWidget(self.prediction_record_page)

    def on_logout(self):
        """退出登录"""
        self.main_window.current_user = None
        self.main_window.setCentralWidget(LoginPage(self.main_window.db, self.main_window))
        self.close()

    def on_edit_notice(self):
        """编辑公告"""
        self.edit_notice_page = EditNoticePage(self.main_window)
        self.edit_notice_page.show()

    def on_profile(self):
        """查看个人信息"""
        self.profile_page = ProfilePage(self.main_window)
        self.profile_page.show()

    def on_handle_feedback(self):
        """处理反馈"""
        self.handle_feedback_page = HandleFeedbackPage(self.main_window)
        self.handle_feedback_page.show()

    def on_predict(self):
        """进行预测"""
        self.main_window.setCentralWidget(PredictionPage(self.main_window))
        self.close()


# 查看反馈页面（未修改）
class ViewFeedbackPage(QWidget):
    def __init__(self, main_window):
        """初始化查看反馈页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("查看反馈状态")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(5)
        self.feedback_table.setHorizontalHeaderLabels(["服务ID", "用户ID", "反馈内容", "提交时间", "处理状态"])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feedback_table.horizontalHeader().setMinimumHeight(40)
        self.feedback_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_table.horizontalHeader().setDefaultSectionSize(150)
        self.load_feedback_table()
        layout.addWidget(self.feedback_table)

        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def load_feedback_table(self):
        """加载用户反馈表格"""
        user_id = self.main_window.current_user[0]
        try:
            self.main_window.db.cursor.execute("SELECT * FROM serve WHERE USERID = ?", (user_id,))
            feedbacks = self.main_window.db.cursor.fetchall()
            self.feedback_table.setRowCount(len(feedbacks))
            for i, feedback in enumerate(feedbacks):
                for j, item in enumerate(feedback):
                    self.feedback_table.setItem(i, j, QTableWidgetItem(str(item)))
        except sqlite3.Error as e:
            print("加载反馈记录失败：", e)

    def on_back(self):
        """返回主页"""
        self.main_window.setCentralWidget(HomePage(self.main_window))
        self.close()


# 预测记录页面（未修改）
class PredictionRecordPage(QWidget):
    def __init__(self, main_window):
        """初始化预测记录页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("预测记录")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(5)
        self.prediction_table.setHorizontalHeaderLabels(["预测ID", "用户ID", "图片路径", "预测结果", "预测时间"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prediction_table.horizontalHeader().setMinimumHeight(40)
        self.prediction_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_table.horizontalHeader().setDefaultSectionSize(150)
        self.load_prediction_table()
        layout.addWidget(self.prediction_table)

        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def load_prediction_table(self):
        """加载预测记录表格"""
        predictions = self.main_window.db.get_all_predictions()
        self.prediction_table.setRowCount(len(predictions))
        for i, prediction in enumerate(predictions):
            for j, item in enumerate(prediction):
                self.prediction_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_back(self):
        """返回主页"""
        self.main_window.setCentralWidget(HomePage(self.main_window))
        self.close()


# 编辑公告页面（未修改）
class EditNoticePage(QWidget):
    def __init__(self, main_window):
        """初始化编辑公告页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("编辑公告")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.notice_edit = QTextEdit()
        self.notice_edit.setPlainText("公告：欢迎使用本系统！")
        layout.addWidget(self.notice_edit)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.on_save)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def on_save(self):
        """保存公告"""
        notice = self.notice_edit.toPlainText()
        operator_id = self.main_window.current_user[0]
        self.main_window.db.update_notice(notice, operator_id)

        current_home_page = self.main_window.centralWidget()
        if isinstance(current_home_page, HomePage):
            current_home_page.notice_label.setText(notice)

        self.close()


# 个人信息页面
class ProfilePage(QWidget):
    def __init__(self, main_window):
        """初始化个人信息页面"""
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("个人信息")
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 仅管理员可见：ID 输入框和加载按钮（用于搜索用户ID）
        if self.main_window.current_user[2] == "Administrator":
            self.id_input = QLineEdit()
            self.id_input.setPlaceholderText("请输入用户ID")
            layout.addWidget(QLabel("目标用户ID："))
            layout.addWidget(self.id_input)

            self.load_button = QPushButton("加载用户信息")
            self.load_button.clicked.connect(self.on_load_user)
            layout.addWidget(self.load_button)

        # 用户信息输入框
        self.uid_label = QLabel(f"{'管理员ID' if self.main_window.current_user[2] == 'Administrator' else '用户ID'}：{self.main_window.current_user[0]}")
        layout.addWidget(self.uid_label)

        self.name_input = QLineEdit(self.main_window.current_user[1])
        layout.addWidget(QLabel("用户名："))
        layout.addWidget(self.name_input)

        self.email_input = QLineEdit(self.main_window.current_user[4])
        layout.addWidget(QLabel("邮箱："))
        layout.addWidget(self.email_input)

        self.identity_input = QLineEdit(self.main_window.current_user[2])
        layout.addWidget(QLabel("身份："))
        layout.addWidget(self.identity_input)

        self.password_input = QLineEdit(self.main_window.current_user[3])
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("密码："))
        layout.addWidget(self.password_input)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.on_save)
        layout.addWidget(self.save_button)

        # 管理员修改个人信息按钮
        if self.main_window.current_user[2] == "Administrator":
            self.edit_self_button = QPushButton("修改我的信息")
            self.edit_self_button.clicked.connect(self.on_edit_self)
            layout.addWidget(self.edit_self_button)

        # 用户表格（仅管理员可见）
        if self.main_window.current_user[2] == "Administrator":
            self.user_table_label = QLabel("用户列表")
            self.user_table_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            layout.addWidget(self.user_table_label)

            self.user_table = QTableWidget()
            self.user_table.setColumnCount(5)
            self.user_table.setHorizontalHeaderLabels(["用户ID", "用户名", "身份", "密码", "邮箱"])
            self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.user_table.horizontalHeader().setMinimumHeight(40)
            self.user_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            self.user_table.horizontalHeader().setDefaultSectionSize(150)
            self.load_user_table()
            layout.addWidget(self.user_table)

        self.setLayout(layout)

    def load_user_table(self):
        """加载用户表格"""
        users = self.main_window.db.get_all_users()
        self.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            for j, item in enumerate(user):
                self.user_table.setItem(i, j, QTableWidgetItem(str(item)))

    def on_load_user(self):
        """根据输入的用户ID加载用户信息（仅查询user表，仅管理员可用）"""
        target_id = self.id_input.text().strip()
        if not target_id:
            QMessageBox.warning(self, "加载失败", "请输入用户ID！")
            return

        # 仅查询用户表
        user = self.main_window.db.get_user(target_id)
        if user:
            self.uid_label.setText(f"用户ID：{user[0]}")
            self.name_input.setText(user[1])
            self.identity_input.setText(user[2])
            self.password_input.setText(user[3])
            self.email_input.setText(user[4])
            print(f"加载普通用户信息: ID={user[0]}, 名字={user[1]}, 身份={user[2]}")  # 调试输出
        else:
            QMessageBox.warning(self, "加载失败", "用户ID不存在！")

    def on_edit_self(self):
        """加载当前管理员自己的信息进行编辑（仅管理员可用）"""
        admin = self.main_window.db.get_admin(self.main_window.current_user[0])
        if admin:
            self.uid_label.setText(f"管理员ID：{admin[0]}")
            self.name_input.setText(admin[1])
            self.identity_input.setText("Administrator")
            self.password_input.setText(admin[3])
            self.email_input.setText(admin[4])
            self.id_input.clear()  # 清空目标ID输入框
            print(f"加载当前管理员信息: ID={admin[0]}, 名字={admin[1]}, 身份=Administrator")  # 调试输出
        else:
            QMessageBox.critical(self, "错误", "无法加载管理员信息！")

    def on_save(self):
        """保存用户信息"""
        target_id = self.id_input.text().strip() if hasattr(self, 'id_input') else ""
        name = self.name_input.text()
        email = self.email_input.text()
        identity = self.identity_input.text()
        password = self.password_input.text()

        if not email or "@" not in email or "." not in email:
            QMessageBox.warning(self, "保存失败", "请输入有效的邮箱地址！")
            return

        if not name or not password:
            QMessageBox.warning(self, "保存失败", "用户名和密码不能为空！")
            return

        try:
            if target_id and self.main_window.current_user[2] == "Administrator":
                # 管理员保存普通用户信息（仅限user表）
                user = self.main_window.db.get_user(target_id)
                if user:
                    self.main_window.db.update_user(target_id, name, identity, password, email)
                    QMessageBox.information(self, "保存成功", f"用户 {target_id} 信息已更新！")
                    print(f"更新普通用户信息: ID={target_id}, 名字={name}, 身份={identity}")  # 调试输出
                else:
                    QMessageBox.warning(self, "保存失败", "用户ID不存在！")
                    return
            else:
                # 保存当前用户信息（管理员或普通用户）
                current_id = self.main_window.current_user[0]
                if self.main_window.current_user[2] == "Administrator":
                    # 管理员保存自己的信息（admin表）
                    self.main_window.db.update_admin(current_id, name, password, email)
                    self.main_window.current_user = (current_id, name, "Administrator", password, email)
                    QMessageBox.information(self, "保存成功", f"管理员 {current_id} 信息已更新！")
                    print(f"更新当前管理员信息: ID={current_id}, 名字={name}, 身份=Administrator")  # 调试输出
                else:
                    # 普通用户保存自己的信息（user表）
                    self.main_window.db.update_user(current_id, name, identity, password, email)
                    self.main_window.current_user = (current_id, name, identity, password, email)
                    QMessageBox.information(self, "保存成功", f"用户 {current_id} 信息已更新！")
                    print(f"更新普通用户信息: ID={current_id}, 名字={name}, 身份={identity}")  # 调试输出

            # 刷新用户表格（仅管理员）
            if self.main_window.current_user[2] == "Administrator":
                self.load_user_table()

            # 恢复输入框为当前用户信息
            if hasattr(self, 'id_input'):
                self.id_input.clear()
            self.uid_label.setText(f"{'管理员ID' if self.main_window.current_user[2] == 'Administrator' else '用户ID'}：{self.main_window.current_user[0]}")
            self.name_input.setText(self.main_window.current_user[1])
            self.identity_input.setText(self.main_window.current_user[2])
            self.password_input.setText(self.main_window.current_user[3])
            self.email_input.setText(self.main_window.current_user[4])

            # 确保界面刷新
            self.update()

        except Exception as e:
            print("保存出错：", e)
            QMessageBox.critical(self, "保存错误", f"保存失败：{str(e)}")

# 预测页面（未修改）
class PredictionPage(QWidget):
    def __init__(self, main_window):
        """初始化预测页面"""
        super().__init__()
        self.main_window = main_window
        self.model_path = os.getenv("MODEL_PATH", "./best.pt")
        self.initUI()
        self.model = YOLOModel(model_path=self.model_path)
        self.video_path = None
        self.video_capture = None
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.auto_tracking = False

    def update_model(self, new_model_path):
        """更新模型路径并重新加载模型"""
        self.model_path = new_model_path
        self.model = YOLOModel(model_path=self.model_path)
        QMessageBox.information(self, "模型更新", "模型已成功更新！")

    def initUI(self):
        self.setWindowTitle("预测页面")
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        left_layout = QVBoxLayout()
        left_frame = QFrame()
        left_frame.setLayout(left_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        self.video_label.setMinimumSize(400, 400)
        left_layout.addWidget(self.video_label)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_frame = QFrame()
        right_frame.setLayout(right_layout)

        self.auto_track_button = QPushButton("开始自动跟踪")
        self.auto_track_button.setCheckable(True)
        self.auto_track_button.clicked.connect(self.on_auto_track)
        right_layout.addWidget(self.auto_track_button)

        self.upload_video_button = QPushButton("上传视频")
        self.upload_video_button.clicked.connect(self.on_upload_video)
        right_layout.addWidget(self.upload_video_button)

        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.valueChanged.connect(self.on_slider_changed)
        right_layout.addWidget(self.video_slider)

        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.on_play)
        right_layout.addWidget(self.play_button)

        self.snapshot_button = QPushButton("截图并识别")
        self.snapshot_button.clicked.connect(self.on_snapshot)
        right_layout.addWidget(self.snapshot_button)

        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.on_upload)
        right_layout.addWidget(self.upload_button)

        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self.on_clear)
        right_layout.addWidget(self.clear_button)

        self.feedback_button = QPushButton("提交反馈")
        self.feedback_button.clicked.connect(self.on_feedback)
        right_layout.addWidget(self.feedback_button)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 25px; color: #000000;")
        right_layout.addWidget(self.result_label)

        self.back_button = QPushButton("返回主页")
        self.back_button.clicked.connect(self.on_back)
        right_layout.addWidget(self.back_button)

        main_layout.addWidget(left_frame, 70)
        main_layout.addWidget(right_frame, 30)
        self.setLayout(main_layout)

    def on_auto_track(self):
        """自动跟踪功能开关"""
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先上传视频！")
            self.auto_track_button.setChecked(False)
            self.auto_track_button.setText("开始自动跟踪")
            return

        if self.auto_track_button.isChecked():
            self.auto_tracking = True
            if not self.timer.isActive():
                self.timer.start(1000 // self.fps)
                self.play_button.setText("暂停")
            self.auto_track_button.setText("停止自动跟踪")
            self.result_label.setText("自动跟踪中...")
        else:
            self.auto_tracking = False
            self.auto_track_button.setText("开始自动跟踪")
            self.result_label.setText("预测结果将显示在这里")
            self.model.reset_model()

    def on_upload_video(self):
        """上传视频文件"""
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Videos (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                QMessageBox.warning(self, "错误", "无法打开视频文件！")
                return

            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_slider.setMaximum(self.total_frames)
            self.auto_tracking = False
            self.auto_track_button.setChecked(False)
            self.auto_track_button.setText("开始自动跟踪")
            self.result_label.setText("预测结果将显示在这里")
            self.update_frame()

    def update_frame(self):
        """更新视频帧并进行自动跟踪"""
        if self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            QMessageBox.information(self, "提示", "视频播放完毕！")
            return

        self.current_frame = frame

        if self.auto_tracking:
            results = self.model.track(frame)
            detected_objects = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.model.names[class_id]
                    chinese_class_name = self.model.class_name_map.get(class_name, class_name)
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    try:
                        font = ImageFont.truetype("SimHei.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    label = f"{class_name} {confidence:.2f}"
                    draw.text((x1, y1 - 30), label, font=font, fill=(0, 255, 0, 255))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    detected_objects.append(f"{chinese_class_name} ({confidence:.2f})")
                if detected_objects:
                    self.result_label.setText("检测到: " + ", ".join(detected_objects))
                else:
                    self.result_label.setText("未检测到目标")

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def on_slider_changed(self, value):
        """滑动条改变视频帧位置"""
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.update_frame()

    def on_play(self):
        """播放或暂停视频"""
        if self.video_capture is None:
            QMessageBox.warning(self, "错误", "请先上传视频！")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            self.timer.start(1000 // self.fps)
            self.play_button.setText("暂停")

    def on_snapshot(self):
        """截图并进行识别"""
        if self.current_frame is None:
            QMessageBox.warning(self, "错误", "没有可截图的视频帧！")
            return

        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, self.current_frame)

        class_name, confidence, save_dir = self.model.predict(temp_image_path)
        if class_name and confidence:
            result_image_path = f"{save_dir}/{temp_image_path.split('/')[-1]}"
            result_pixmap = QPixmap(result_image_path)
            self.video_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")

            user_id = self.main_window.current_user[0]
            self.main_window.db.add_prediction(user_id, temp_image_path, class_name)
        else:
            self.result_label.setText("未检测到目标")

    def on_upload(self):
        """上传图片并进行预测"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.video_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

            class_name, confidence, save_dir = self.model.predict(file_path)
            if class_name and confidence:
                result_image_path = f"{save_dir}/{file_path.split('/')[-1]}"
                result_pixmap = QPixmap(result_image_path)
                self.video_label.setPixmap(result_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.result_label.setText(f"预测结果：{class_name}，置信度：{confidence:.2f}")

                user_id = self.main_window.current_user[0]
                self.main_window.db.add_prediction(user_id, file_path, class_name)
            else:
                self.result_label.setText("未检测到目标")

    def on_clear(self):
        """清空当前内容"""
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.video_path = None
        self.current_frame = None
        self.auto_tracking = False
        self.auto_track_button.setChecked(False)
        self.auto_track_button.setText("开始自动跟踪")
        self.play_button.setText("播放")
        self.video_slider.setValue(0)
        self.video_label.clear()
        self.result_label.setText("预测结果将显示在这里")
        self.model.reset_model()
        QMessageBox.information(self, "提示", "已清空当前内容！")

    def on_feedback(self):
        """打开反馈页面"""
        self.feedback_page = FeedbackPage(self.main_window)
        self.feedback_page.show()

    def on_back(self):
        if self.video_capture:
            self.video_capture.release()
        if hasattr(self.model, 'trackers'):
            self.model.reset_model()
        self.main_window.open_home_page()
        self.close()


# 主窗口（未修改）
class MainWindow(QMainWindow):
    def __init__(self, db):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("系统登录")
        self.setGeometry(100, 100, 800, 600)
        self.db = db
        self.current_user = None
        self.home_page = None
        self.login_page = LoginPage(self.db, self)
        self.setCentralWidget(self.login_page)
        self.setStyleSheet(GLOBAL_STYLESHEET)

    def open_home_page(self):
        """打开主页"""
        self.home_page = HomePage(self)
        self.setCentralWidget(self.home_page)


# 运行程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    db = Database("database.db")
    window = MainWindow(db)
    window.show()
    sys.exit(app.exec())
    db.close()