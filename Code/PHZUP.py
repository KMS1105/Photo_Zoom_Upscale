import cv2
from cv2 import dnn_superres
import time
import os

class ImageViewer:
    def __init__(self, image_path, mirror=False):
        self.data = cv2.imread(image_path)
        if self.data is None:
            raise ValueError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        
        self.WIDTH, self.HEIGHT = self.data.shape[1], self.data.shape[0]
        self.center_x, self.center_y = self.WIDTH // 2, self.HEIGHT // 2
        self.scale = 1.0
        self.mirror = mirror
        self.is_upscaling = False
        self.upscaled_image = self.data.copy()
        self.cap_folder = r"C:\Users\user\Desktop\P\captured"
        os.makedirs(self.cap_folder, exist_ok=True)

        if self.mirror:
            self.data = cv2.flip(self.data, 1)

        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.load_model()

    def load_model(self):
        model_path = 'ESPCN_x2.pb'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"업스케일 모델 파일을 찾을 수 없습니다: {model_path}")
        self.sr.readModel(model_path)
        self.sr.setModel('espcn', 2)

        """ # CUDA 활성화 (OpenCV가 CUDA를 지원하는 경우)
        self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # CUDA 백엔드 사용
        self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # GPU 타겟 사용"""

    def show(self):
        while True:
            cv2.imshow('Image Viewer', self.upscaled_image)
            cv2.setMouseCallback('Image Viewer', self.mouse_callback)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK and not self.is_upscaling:
            self.scale = min(3.0, self.scale * 1.2)
            self.center_x, self.center_y = x, y
            self.process_image()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.scale = max(1.0, self.scale / 1.2)
            self.center_x, self.center_y = x, y

    def process_image(self):
        self.is_upscaling = True
        zoomed_image = self.__zoom(self.data, (self.center_x, self.center_y))
        self.upscaled_image = self.sr.upsample(zoomed_image)
        self.is_upscaling = False
        self.capture(self.upscaled_image)

    def __zoom(self, img, center):
        height, width = img.shape[:2]
        scale_factor = 1 / self.scale
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        center_x, center_y = center
        min_x, max_x = max(0, center_x - new_width // 2), min(width, center_x + new_width // 2)
        min_y, max_y = max(0, center_y - new_height // 2), min(height, center_y + new_height // 2)
        cropped = img[min_y:max_y, min_x:max_x]
        return cv2.resize(cropped, (width, height))

    def capture(self, frame):
        filename = os.path.join(self.cap_folder, f'captured_image_{int(time.time())}.png')
        cv2.imwrite(filename, frame)
        print(f'이미지 저장 완료: {filename}')

if __name__ == '__main__':
    viewer = ImageViewer('sc01.png', mirror=True)
    viewer.show()
