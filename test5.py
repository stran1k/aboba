"""realtime_gaze.py — Трекинг взгляда + PyQt6 оверлей. C=точка, R=сброс, Q=выход."""

import cv2, numpy as np, torch, mediapipe as mp, sys, keyboard
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.interpolate import LinearNDInterpolator
from test_model import GazeNet

MODEL_PATH = "gaze_model_test.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCREEN_PTS = np.array([[0,0],[.5,0],[1,0],[0,.5],[.5,.5],[1,.5],[0,1],[.5,1],[1,1]], dtype=float)
L_IDX = [362,385,387,263,373,380]
R_IDX = [33,160,158,133,153,144]
MSGS  = ["ВЕРХНИЙ ЛЕВЫЙ","ВЕРХ ЦЕНТР","ВЕРХНИЙ ПРАВЫЙ",
         "ЦЕНТР СЛЕВА","ЦЕНТР","ЦЕНТР СПРАВА",
         "НИЖНИЙ ЛЕВЫЙ","НИЗ ЦЕНТР","НИЖНИЙ ПРАВЫЙ"]


class GazeWorker(QThread):
    """Фоновый поток: захват, предсказание взгляда, калибровка."""
    gaze_point = pyqtSignal(int, int)
    calib_msg  = pyqtSignal(str, int)
    stability  = pyqtSignal(bool)
    reset_sig  = pyqtSignal()

    def __init__(self):
        """Загружает MediaPipe и модель GazeNet."""
        super().__init__()
        self.fm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
                num_faces=1))
        self.model = GazeNet().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        self._reset()

    def _reset(self):
        """Сбрасывает калибровку и буферы."""
        self.pts, self.calibrated = [], False
        self.prev = np.array([.5, .5])
        self.ix = self.iy = None
        self.buf = []

    def _roi(self, frame, lm, idx):
        """Вырезает ROI глаза с отступами."""
        h, w = frame.shape[:2]
        p = np.array([(lm[i].x*w, lm[i].y*h) for i in idx], dtype=np.int32)
        x0,y0 = np.min(p,0); x1,y1 = np.max(p,0)
        dx = int((x1-x0)*.4); dy_t = int((y1-y0)*.3); dy_b = int((y1-y0)*.8)
        return frame[max(0,y0-dy_t):min(h,y1+dy_b), max(0,x0-dx):min(w,x1+dx)]

    def _pre(self, img):
        """BGR -> grayscale -> equalize -> resize(60,36) -> тензор (1,1,36,60)."""
        if img is None or img.size == 0: return None
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return torch.from_numpy(cv2.resize(g,(60,36)).astype(np.float32)/255.).unsqueeze(0).unsqueeze(0)

    def _tta(self, t):
        """TTA: 10 проходов с шумом, усреднение. Возвращает (2,2)."""
        with torch.no_grad():
            return np.mean([self.model((t+torch.randn_like(t)*.02).clamp(0,1).to(DEVICE)).cpu().numpy()
                            for _ in range(10)], axis=0)

    def _stable(self):
        """True если std взгляда за 10 кадров < 0.01."""
        if len(self.buf) < 10: return False
        return np.std(self.buf[-10:], axis=0).max() < 0.01

    def _smooth(self, cur):
        """Адаптивное EMA: быстрое при движении, медленное при покое."""
        a = .6 if np.linalg.norm(cur - self.prev) > .05 else .15
        return self.prev * (1-a) + cur * a

    def _map(self, yw, pt):
        """Интерполяция (yaw,pitch) -> нормализованные координаты экрана [0,1]."""
        p = np.array([[yw, pt]])
        nx, ny = self.ix(p)[0], self.iy(p)[0]
        if np.isnan(nx) or np.isnan(ny):
            i = np.argmin(np.linalg.norm(np.array(self.pts)-[yw,pt], axis=1))
            nx, ny = SCREEN_PTS[i]
        return np.clip(float(nx),0,1), np.clip(float(ny),0,1)

    def run(self):
        """Основной цикл захвата и трекинга. C=калибровка, R=сброс, Q=выход."""
        cap = cv2.VideoCapture(0)
        geo = QApplication.primaryScreen().geometry()
        W, H = geo.width(), geo.height()
        self.calib_msg.emit(MSGS[0]+" — жми C", 0)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            if keyboard.is_pressed('r'):
                self._reset(); self.reset_sig.emit()
                self.calib_msg.emit(MSGS[0]+" — жми C", 0)
                while keyboard.is_pressed('r'): pass

            res = self.fm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                lt = self._pre(self._roi(frame, lm, L_IDX))
                rt = self._pre(self._roi(frame, lm, R_IDX))

                if lt is not None and rt is not None:
                    out = self._tta(torch.cat([lt, rt]))
                    yw  = (out[0][1]+out[1][1])/2.
                    pt  = (out[0][0]+out[1][0])/2.

                    self.buf.append((yw, pt))
                    if len(self.buf) > 20: self.buf.pop(0)
                    self.stability.emit(self._stable())

                    if keyboard.is_pressed('c') and not self.calibrated:
                        self.pts.append((yw, pt))
                        n = len(self.pts)
                        if n < 9:
                            self.calib_msg.emit(MSGS[n]+" — жми C", n)
                        else:
                            p = np.array(self.pts)
                            self.ix = LinearNDInterpolator(p, SCREEN_PTS[:,0])
                            self.iy = LinearNDInterpolator(p, SCREEN_PTS[:,1])
                            self.calibrated = True
                            self.calib_msg.emit("ГОТОВО  |  R = сброс", -1)
                        while keyboard.is_pressed('c'): pass

                    if self.calibrated:
                        nx, ny = self._map(yw, pt)
                        self.prev = self._smooth(np.array([nx, ny]))
                        self.gaze_point.emit(int(self.prev[0]*W), int(self.prev[1]*H))

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release(); cv2.destroyAllWindows()


class OverlayWindow(QWidget):
    """Прозрачный полноэкранный оверлей с курсором взгляда."""

    def __init__(self):
        """Создаёт прозрачный виджет поверх всех окон."""
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        geo = QApplication.primaryScreen().geometry()
        self.setGeometry(geo)
        self.pt   = QPoint(geo.width()//2, geo.height()//2)
        self.msg  = ""
        self.idx  = 0
        self.stab = False

    def paintEvent(self, _):
        """Рисует курсор, калибровочную точку и подсказку."""
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setBrush(QColor(255,0,0,200)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.pt, 12, 12)
        if self.msg:
            p.setPen(QColor(255,255,255)); p.setFont(QFont("Segoe UI", 20))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.msg)
            if 0 <= self.idx < 9:
                cx,cy = [(50,50),(w//2,50),(w-50,50),(50,h//2),(w//2,h//2),
                          (w-50,h//2),(50,h-50),(w//2,h-50),(w-50,h-50)][self.idx]
                p.setBrush(QColor(0,255,0) if self.stab else QColor(255,200,0))
                p.drawEllipse(QPoint(cx,cy), 25, 25)
                p.setPen(QColor(0,0,0)); p.setFont(QFont("Segoe UI",14,QFont.Weight.Bold))
                p.drawText(QPoint(cx+30,cy+5), f"{self.idx+1}/9")
                p.setPen(QColor(255,255,100)); p.setFont(QFont("Segoe UI",14))
                p.drawText(QPoint(w//2-180,h-80),
                           "✓ Стабильно — жми C" if self.stab else "Зафиксируй взгляд...")

    def update_data(self, x, y):
        """Обновляет позицию курсора (x, y в пикселях)."""
        self.pt = QPoint(x, y); self.update()

    def update_msg(self, text, idx):
        """Обновляет инструкцию и индекс активной точки."""
        self.msg = text; self.idx = idx; self.update()

    def update_stability(self, s):
        """Обновляет цвет точки: True=зелёная, False=жёлтая."""
        self.stab = s; self.update()

    def on_reset(self):
        """Очищает интерфейс при сбросе."""
        self.msg = ""; self.idx = 0; self.update()


if __name__ == "__main__":
    from PyQt6.QtGui import QGuiApplication
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    ov  = OverlayWindow(); ov.show()
    w   = GazeWorker()
    w.gaze_point.connect(ov.update_data)
    w.calib_msg.connect(ov.update_msg)
    w.stability.connect(ov.update_stability)
    w.reset_sig.connect(ov.on_reset)
    w.start()
    sys.exit(app.exec())