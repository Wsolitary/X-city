import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import os
import random
import json
import uuid
import pandas as pd
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QMenu, QSystemTrayIcon, QPushButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QFont, QAction, QIcon, QBrush, QPen, QCursor

# ================= 配置与常量 =================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17] 
MAR_THRESHOLD = 0.6
CONSECUTIVE_FRAMES = 10

# 数据存储路径 - F盘专属
DATA_DIR = r"F:\Vigil\data"
LOG_FILE = os.path.join(DATA_DIR, "focus_history.csv")
REPORT_FILE = os.path.join(DATA_DIR, "reports.json")

# API 配置
API_KEY = "sk-y8LGmh4LtgB3A2Dy5kRL9NZbXfdhWdLNpz8zT2v92Z2OTDv2"
API_URL = "https://api.moonshot.cn/v1/chat/completions"

SYSTEM_PROMPT = {
    "role": "system", 
    "content": (
        "你是 CodeVigilante 系统的健康守护 AI，代号 'Vigil'。"
        "1. 当用户疲劳时：用关怀但坚定的语气劝导休息。**严禁重复相同的话术**。"
        "2. 当用户专注时：给予极简的肯定，或者保持沉默。"
        "3. 你的目标是让用户保持可持续的高效。"
        "回复限制在 30 字以内。"
    )
}

# ================= 优化后的 EAR 算法类 =================
class AdaptiveEARTracker:
    def __init__(self, calibration_seconds=30):
        self.calibration_seconds = calibration_seconds
        self.baseline_ear_sum = 0.0
        self.baseline_count = 0
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        
        # 个人化阈值（初始值，会在校准后更新）
        self.personal_threshold = 0.22
        self.personal_baseline = 0.3
        self.focused_threshold = 0.3  # 新增：专注阈值
        
    def update(self, current_ear):
        """更新 EAR 状态，返回 (is_drowsy, confidence)"""
        current_time = time.time()
        
        # 校准阶段：收集个人基准数据
        if self.is_calibrating and (current_time - self.calibration_start_time) < self.calibration_seconds:
            self.baseline_ear_sum += current_ear
            self.baseline_count += 1
            
            # 实时更新阈值（基于当前平均值）
            if self.baseline_count > 10:  # 至少有10个样本
                self.personal_baseline = self.baseline_ear_sum / self.baseline_count
                self.personal_threshold = self.personal_baseline * 0.75  # 调整敏感度：基准的75%即视为疲劳 (0.7太难触发, 0.8太易触发)
                self.focused_threshold = self.personal_baseline * 1.0   # 降低难度：达到基准值即视为专注
            
            return False, 0.5  # 校准中，不确定状态
        
        # 校准完成
        if self.is_calibrating:
            self.is_calibrating = False
            if self.baseline_count > 0:
                self.personal_baseline = self.baseline_ear_sum / self.baseline_count
                self.personal_threshold = self.personal_baseline * 0.75
                self.focused_threshold = self.personal_baseline * 1.0
                print(f"✅ 个人化校准完成！")
                print(f"   基准EAR: {self.personal_baseline:.3f}")
                print(f"   专注阈值: > {self.focused_threshold:.3f}")
                print(f"   疲劳阈值: < {self.personal_threshold:.3f}")
        
        # 使用个人化阈值判断疲劳
        # 优化：加入防抖动，只有连续多帧低 EAR 才算疲劳，避免目光偏移造成的误判
        # 且降低灵敏度：从 0.75 降到 0.65 (只有明显闭眼或极度眯眼才触发)
        if self.is_calibrating:
             pass # 校准时不修改阈值
        elif self.baseline_count > 0:
             # 运行时动态微调：如果用户觉得太敏感，我们手动降低系数
             self.personal_threshold = self.personal_baseline * 0.65
        
        is_drowsy = current_ear < self.personal_threshold
        
        # 计算置信度（离阈值越远，置信度越高）
        if is_drowsy:
            # 疲劳区间：0 到 threshold
            confidence = min(0.95, (self.personal_threshold - current_ear) / self.personal_threshold)
        else:
            confidence = 0.0
        
        return is_drowsy, confidence

# 滑动平均滤波器
class MovingAverageFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values) if self.values else value

# ================= 核心算法 =================
def calculate_ear(landmarks, indices):
    try:
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        return (A + B) / (2.0 * C)
    except: return 0.0

def calculate_mar(landmarks, indices):
    try:
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
        A = np.linalg.norm(points[2] - points[3])
        B = np.linalg.norm(points[4] - points[5])
        C = np.linalg.norm(points[0] - points[1])
        return (A + B) / (2.0 * C)
    except: return 0.0

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        try: os.makedirs(DATA_DIR)
        except: pass

def log_data(status_label, ear, mar):
    ensure_data_dir()
    now = datetime.now()
    data = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "status": status_label,
        "ear": round(ear, 3),
        "mar": round(mar, 3)
    }
    df = pd.DataFrame([data])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False, mode='w')
    else:
        df.to_csv(LOG_FILE, index=False, mode='a', header=False)

# ================= 线程类 =================
class AIThread(QThread):
    response_received = pyqtSignal(str)

    def __init__(self, user_msg):
        super().__init__()
        self.user_msg = user_msg

    def run(self):
        try:
            messages = [SYSTEM_PROMPT, {"role": "user", "content": self.user_msg}]
            data = {"model": "moonshot-v1-8k", "messages": messages, "temperature": 0.7}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
            response = requests.post(API_URL, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content']
                self.response_received.emit(reply)
        except:
            pass

class ReportThread(QThread):
    finished = pyqtSignal(str) # 返回生成状态消息

    def run(self):
        # 1. 读取数据
        if not os.path.exists(LOG_FILE): return
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty: return
            
            df['datetime'] = pd.to_datetime(df['timestamp'])
            last_time = df['datetime'].max()
            start_time = last_time - timedelta(minutes=30)
            df_recent = df[df['datetime'] >= start_time]
            
            if df_recent.empty: return

            # 统计指标
            rec_total = len(df_recent)
            rec_focused = len(df_recent[df_recent['status'] == 'Focused'])
            rec_drowsy = len(df_recent[df_recent['status'].isin(['Drowsy', 'Yawning'])])
            rec_ear = df_recent['ear'].mean()

            # 构建 Prompt
            prompt = (
                f"请分析用户最近30分钟的精力状态 (自动定期报告)：\n"
                f"- 记录时长: {rec_total * 5 // 60} 分钟\n"
                f"- 专注时长: {rec_focused * 5 // 60} 分钟\n"
                f"- 疲劳/打哈欠次数: {rec_drowsy} 次\n"
                f"- 平均专注度(EAR): {rec_ear:.3f}\n\n"
                f"请给出简短的当前状态评估和接下来的行动建议。"
            )

            # 调用 API
            messages = [
                {"role": "system", "content": "你是 Vigil 系统的智能效能分析师。"},
                {"role": "user", "content": prompt}
            ]
            data = {"model": "moonshot-v1-8k", "messages": messages, "temperature": 0.7}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
            
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # 读取旧报告
                reports = []
                if os.path.exists(REPORT_FILE):
                    try:
                        with open(REPORT_FILE, 'r', encoding='utf-8') as f:
                            reports = json.load(f)
                    except: pass
                
                # 添加新报告
                new_report = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "30min",
                    "content": content
                }
                reports.insert(0, new_report)
                
                # 写入文件
                with open(REPORT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(reports, f, ensure_ascii=False, indent=2)
                    
                self.finished.emit("📝 已自动生成 30分钟效能报告")
        except Exception as e:
            print(f"Auto report failed: {e}")

class VideoThread(QThread):
    status_update = pyqtSignal(str, float, float) # status, ear, mar
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # 初始化优化算法
        self.ear_tracker = AdaptiveEARTracker(calibration_seconds=30)
        self.ear_filter = MovingAverageFilter(window_size=10)

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_counter = 0
        last_log_time = time.time()
        data_buffer = {"ear": [], "mar": [], "status": []}

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                self.status_update.emit("No Face", 0.0, 0.0)
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            status = "No Face"
            ear = 0.0
            mar = 0.0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # 计算原始 EAR
                    raw_ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2.0
                    
                    # 应用滑动平均滤波
                    filtered_ear = self.ear_filter.update(raw_ear)
                    
                    # 使用自适应算法判断疲劳
                    is_drowsy, confidence = self.ear_tracker.update(filtered_ear)
                    
                    mar = calculate_mar(landmarks, MOUTH)

                    # 状态判定逻辑
                    # 疲劳判定：AdaptiveEARTracker 说是疲劳，且必须连续多帧确认 (防抖)
                    # CONSECUTIVE_FRAMES 原为10 (约0.5秒)，增加到 20 (约1秒)，过滤短暂低头或眨眼
                    if is_drowsy: 
                        frame_counter += 1
                    else:
                        frame_counter = 0
                    
                    if frame_counter >= 20: # 提高判定门槛
                        status = "Drowsy"
                    elif mar > MAR_THRESHOLD:
                        status = "Yawning"
                    elif filtered_ear > self.ear_tracker.focused_threshold: # 使用动态专注阈值
                        status = "Focused"
                    else:
                        status = "Normal"
                    
                    ear = filtered_ear
                    
                    # 缓冲数据
                    data_buffer["ear"].append(ear)
                    data_buffer["mar"].append(mar)
                    data_buffer["status"].append(status)

            # 发送信号更新 UI
            self.status_update.emit(status, ear, mar)

            # 数据记录 (5秒聚合)
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                if data_buffer["ear"]:
                    avg_ear = sum(data_buffer["ear"]) / len(data_buffer["ear"])
                    avg_mar = sum(data_buffer["mar"]) / len(data_buffer["mar"])
                    from collections import Counter
                    most_common = Counter(data_buffer["status"]).most_common(1)[0][0]
                    log_data(most_common, avg_ear, avg_mar)
                
                data_buffer = {"ear": [], "mar": [], "status": []}
                last_log_time = current_time

            time.sleep(0.05) # 降低 CPU 占用

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# ================= UI 类 =================
class BubbleLabel(QLabel):
    def __init__(self, target_widget=None):
        super().__init__(None) # 设置为顶级窗口，避免被父窗口裁剪
        self.target = target_widget
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 240);
                color: #333333;
                border-radius: 10px;
                padding: 12px;
                border: 2px solid #3498db;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.setWordWrap(True) # 允许换行
        self.setMaximumWidth(250) # 限制最大宽度
        self.hide()
        
        # 自动消失定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hide)

    def show_message(self, text, duration=5000):
        self.setText(text)
        self.adjustSize()
        
        # 智能定位：优先显示在左侧，如果左侧不够显示在右侧
        if self.target:
            target_geo = self.target.geometry()
            screen_geo = QApplication.primaryScreen().geometry()
            
            # 尝试放在左侧
            x = target_geo.x() - self.width() - 15
            y = target_geo.y()
            
            # 如果左侧超出屏幕 (即 x < 0)，则放在右侧
            if x < 0:
                x = target_geo.x() + target_geo.width() + 15
                
            # 防止底部溢出
            if y + self.height() > screen_geo.height():
                y = screen_geo.height() - self.height() - 10
                
            self.move(x, y)
            
        self.show()
        self.timer.start(duration)

class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initLogic()
        
    def initUI(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(100, 100) # 初始大小
        
        # 状态指示颜色
        self.current_color = QColor(200, 200, 200) # 默认灰色
        self.status_text = "..."

        # 气泡 (传入 self 作为定位目标)
        self.bubble = BubbleLabel(self)
        
        # 移动相关
        self.dragging = False
        self.offset = QPoint()

        # 右键菜单
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        
        # 放到屏幕右下角
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 150, screen.height() - 200)

    def initLogic(self):
        # 显示启动提示
        self.bubble.show_message("👋 嗨！请正常睁眼，眨眼30秒进行校准...", 5000)
        
        # 视频线程
        self.video_thread = VideoThread()
        self.video_thread.status_update.connect(self.update_status)
        self.video_thread.start()
        
        # 自动巡航计时器
        self.check_timer = QTimer(self)
        self.check_timer.timeout.connect(self.auto_check)
        self.check_timer.start(5000) # 改为每5秒检查一次，以便更及时地捕捉状态
        
        self.last_ear = 0.0
        self.last_mar = 0.0
        self.current_status = "Normal"
        
        # 历史状态记录 (用于趋势分析)
        self.status_history = []  # 存储 (timestamp, status)
        self.last_ai_trigger_time = 0 # 上次触发 AI 的时间
        self.last_report_time = time.time() # 上次生成报告的时间
        
        # 免打扰逻辑
        self.focused_start_time = None
        self.in_dnd_mode = False
        self.dnd_threshold = 60 # 连续专注60秒进入免打扰 
        
        # 校准状态追踪
        self.is_calibrating = True
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.check_calibration_status)
        self.calibration_timer.start(1000) # 每秒检查一次校准状态

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制圆形背景
        painter.setBrush(QBrush(self.current_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(10, 10, 60, 60)
        
        # 绘制眼睛 (简单的拟人化)
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawEllipse(25, 30, 10, 10) # 左眼
        painter.drawEllipse(45, 30, 10, 10) # 右眼
        
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        # 根据状态改变眼球位置/大小
        if self.current_status == "Drowsy":
            # 闭眼
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawLine(25, 35, 35, 35)
            painter.drawLine(45, 35, 55, 35)
        else:
            painter.drawEllipse(27, 32, 5, 5)
            painter.drawEllipse(47, 32, 5, 5)
        
        # 添加状态文字
        # 优化：校准模式(黄色)下用黑色文字，否则白色
        if self.current_status == "Calibrating":
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
        else:
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        
        # 根据状态显示不同文字
        if self.current_status == "Calibrating":
            text = "校准中..."
        elif self.current_status == "Drowsy":
            text = "疲劳警告"
        elif self.current_status == "Yawning":
            text = "正在打哈欠"
        elif self.current_status == "Focused":
            text = "专注中"
        elif self.current_status == "Normal":
            text = "状态正常"
        else:
            text = "状态正常" 
        
        # 计算文字位置（居中）
        text_rect = painter.fontMetrics().boundingRect(text)
        text_x = (80 - text_rect.width()) // 2
        text_y = 85
        
        painter.drawText(text_x, text_y, text)

    def update_status(self, status, ear, mar):
        # 优先处理校准状态
        is_calibrating = False
        if hasattr(self, 'video_thread') and hasattr(self.video_thread, 'ear_tracker'):
            is_calibrating = self.video_thread.ear_tracker.is_calibrating

        if is_calibrating:
            self.last_ear = ear
            self.last_mar = mar
            
          
            self.current_status = "Calibrating"
            self.current_color = QColor(255, 255, 0)
            
            # 控制台输出
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [校准中] 状态: {status} | EAR: {ear:.3f} | MAR: {mar:.3f}")
            self.update()
            return

        self.current_status = status
        self.last_ear = ear
        self.last_mar = mar
        
        # 控制台输出
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 状态: {status} | EAR: {ear:.3f} | MAR: {mar:.3f}")
        
        # 记录历史状态 (用于15分钟趋势分析)
        self.status_history.append((time.time(), status))
        
        # 追踪专注时间
        if status == "Focused":
            if self.focused_start_time is None:
                self.focused_start_time = time.time()
            elif time.time() - self.focused_start_time > self.dnd_threshold and not self.in_dnd_mode:
                self.in_dnd_mode = True
                # 进入免打扰时提示一下
                self.bubble.show_message("🌙 进入深度免打扰模式", 3000)
        else:
            # 状态中断，重置计时
            # 只有当状态变成“疲劳”或“打哈欠”时才打断，如果是偶尔的“Normal”可以容忍（这里简化为一律打断）
            if status != "Normal": # 允许短暂的 Normal 状态不打断心流
                self.focused_start_time = None
                if self.in_dnd_mode:
                    self.in_dnd_mode = False
                    self.bubble.show_message("☀️ 退出免打扰模式", 3000)
        
        if status == "Drowsy":
            self.current_color = QColor(255, 80, 80) # 红
            self.setToolTip("警告：疲劳！")
        elif status == "Yawning":
            self.current_color = QColor(255, 165, 0) # 橙
            self.setToolTip("正在打哈欠")
        elif status == "Focused":
            self.current_color = QColor(100, 255, 100) # 绿
            if self.in_dnd_mode:
                 self.setToolTip("高度专注 (免打扰中)")
                 # 免打扰模式下，颜色稍微变深一点，表示沉浸
                 self.current_color = QColor(0, 200, 0)
            else:
                 self.setToolTip("高度专注")
        elif status == "Normal":
            self.current_color = QColor(100, 200, 255) # 蓝
            self.setToolTip("状态正常")
        else:
            self.current_color = QColor(200, 200, 200) # 灰
            self.setToolTip("未检测到人脸")
            
        self.update() # 重绘

    def check_calibration_status(self):
        """检查校准状态并更新显示"""
        if hasattr(self.video_thread, 'ear_tracker'):
            tracker = self.video_thread.ear_tracker
            
            # 如果还在校准状态
            if tracker.is_calibrating:
                elapsed = time.time() - tracker.calibration_start_time
                remaining = max(0, 30 - int(elapsed))
                
                # 安全网：如果超过32秒还没结束（说明可能没人脸导致 update 没被调用），强制结束
                if elapsed > 32:
                    print("⚠️ 校准超时，强制结束校准...")
                    tracker.is_calibrating = False
                    # 再次调用以进入 else 分支
                    self.check_calibration_status()
                    return

                # 更新悬浮窗显示
                self.current_status = "Calibrating"
                self.current_color = QColor(255, 255, 0)  # 黄色表示校准中
                self.setToolTip(f"校准中... 剩余 {remaining} 秒")
                self.update()  # 强制重绘
                
                # 每10秒更新一次提示
                if remaining % 10 == 0 and remaining > 0:
                    self.bubble.show_message(f"🔄 校准中... 请保持正常眨眼，剩余 {remaining} 秒", 3000)
                return

        # 校准完成 (或者 ear_tracker 不存在)
        if hasattr(self, 'calibration_timer'):
            self.calibration_timer.stop()
        
        self.bubble.show_message("✅ 校准完成！Vigil 已就绪", 3000)
        self.current_status = "Normal"  # 重置为正常状态
        self.update()  # 强制重绘

    def auto_check(self):
        """智能分析逻辑：基于最近15分钟的数据进行 AI 干预"""
        current_time = time.time()
        
        # 1. 清理过期数据 (保留最近15分钟 / 900秒)
        cutoff_time = current_time - 900
        # 简单优化：如果列表太长，先从头清理
        while self.status_history and self.status_history[0][0] < cutoff_time:
            self.status_history.pop(0)
            
        # 2. 计算各状态持续时间
        drowsy_duration = 0.0
        focused_duration = 0.0
        
        # 只有当有足够的历史数据时才计算
        if len(self.status_history) > 1:
            # 遍历历史记录计算时长 (简单的积分：时间差 * 状态)
            # 注意：status_history 是 (timestamp, status
            for i in range(len(self.status_history) - 1):
                t1, s1 = self.status_history[i]
                t2, _ = self.status_history[i+1]
                dt = t2 - t1
                
                # 过滤异常的大间隔 (比如程序卡顿或休眠)，限制最大间隔为 1秒
                if dt > 1.0: dt = 0.05
                
                if s1 in ["Drowsy", "Yawning"]:
                    drowsy_duration += dt
                elif s1 == "Focused":
                    focused_duration += dt

        # 3. 触发逻辑
        # 只有距离上次触发超过 5分钟 (300秒) 才允许再次触发，避免唠叨
        # 例外：如果正在打哈欠，且距离上次触发超过 1分钟，可以触发
        
        msg = ""
        trigger = False
        
        time_since_last = current_time - self.last_ai_trigger_time
        
        # 优先级 1: 严重疲劳趋势 (15分钟内累计 > 10分钟 / 600秒)
        # 为了演示效果，这里先把阈值设低一点，比如 1分钟 (60秒) 方便测试，
        # 实际使用请改为 600 (10分钟)
        FATIGUE_THRESHOLD = 600 # 10分钟
        FOCUS_THRESHOLD = 600   # 10分钟
        
        if drowsy_duration > FATIGUE_THRESHOLD and time_since_last > 300:
            minutes = int(drowsy_duration / 60)
            msg = f"用户在过去15分钟内有 {minutes} 分钟处于疲劳状态。请给出简短的休息建议，语气要关怀。"
            trigger = True
            
        # 优先级 2: 高度专注趋势
        elif focused_duration > FOCUS_THRESHOLD and time_since_last > 300:
            minutes = int(focused_duration / 60)
            # 只有在非免打扰模式下，或者专注刚结束时才夸奖，避免打断心流
            # 这里简单处理：如果专注时间很长，给个轻轻的夸奖
            if not self.in_dnd_mode:
                msg = f"用户在过去15分钟内保持了 {minutes} 分钟的高效专注。请给予简短的表扬和鼓励。"
                trigger = True

        # 优先级 3: 瞬时打哈欠 (降低频率，至少间隔60秒)
        elif self.current_status == "Yawning" and time_since_last > 60:
            msg = "用户刚才打了个哈欠。请用幽默的方式提醒用户注意精力管理。"
            trigger = True
            
        if trigger:
            print(f"🤖 触发 AI: {msg}")
            self.last_ai_trigger_time = current_time
            # 启动 AI 线程
            self.ai_thread = AIThread(msg)
            self.ai_thread.response_received.connect(self.show_ai_message)
            self.ai_thread.start()
            
        # 4. 自动定期报告 (每30分钟 / 1800秒)
        if current_time - self.last_report_time > 1800:
            print("📊 触发自动定期报告...")
            self.last_report_time = current_time
            self.report_thread = ReportThread()
            self.report_thread.finished.connect(lambda msg: self.bubble.show_message(msg, 5000))
            self.report_thread.start()

    def show_ai_message(self, text):
        # 核心逻辑：直接显示，不需要点击
        # 调用 BubbleLabel 的 show_message 方法，它会自动弹窗
        self.bubble.show_message(text, duration=10000) # 显示10秒，让用户有足够时间看

    # ================= 鼠标事件 (拖拽) =================
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPosition().toPoint() - self.offset)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
    def show_menu(self, pos):
        menu = QMenu(self)
        
        stats_action = QAction("📊 打开数据统计 (Web)", self)
        stats_action.triggered.connect(self.open_web_stats)
        menu.addAction(stats_action)
        
        quit_action = QAction("❌ 退出 Vigil", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        
        test_ai_action = QAction("🤖 测试 AI 对话", self)
        test_ai_action.triggered.connect(self.test_ai_dialog)
        menu.addAction(test_ai_action)
        
        menu.exec(self.mapToGlobal(pos))
        
    def test_ai_dialog(self):
        self.bubble.show_message("🤖 测试成功！我是 Vigil，你的精力守护者。我会在这里给你发送休息建议和专注鼓励。", 5000)

    def open_web_stats(self):
        # 启动 Streamlit 看板 (只读模式)
        import subprocess
        # 这里只是简单打开，实际场景可能需要更复杂的联动
        subprocess.Popen(["streamlit", "run", "app.py"])

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec())