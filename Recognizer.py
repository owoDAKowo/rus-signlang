import av
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
from collections import Counter, deque
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from streamlit_webrtc import VideoProcessorBase
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extract_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(8, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        )

        in_features = 1568
        self.clf_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2048, out_features=num_classes),
        )

        image_size = 64
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    
    def predict(self, img: np.ndarray):
        with torch.no_grad():
            img = self.transform(img.copy()).unsqueeze(0)
            out = self.forward(img)
            return torch.argmax(out.softmax(1), 1).item(), out.softmax(1)

    def forward(self, x):
        x = self.feature_extract_layers(x)
        x = self.clf_layers(x)
        return x

class SignLanguageRecognizer(VideoProcessorBase):
    def __init__(self):
        model_path = 'model_weights.pth'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.classes = [
            'Ё', 'А', 'Б', 'В', 'Г', 'Е', 'Ж', 'И', 'Л', 'М', 'Н', 'О', 'П', 'Р',
            'С', 'Т', 'У', 'Ф', 'Х', 'Ч', 'Ш', 'Ы', 'Э', 'Ю', 'Я'
        ]

        self.letters = deque(maxlen=20)
        
        device = torch.device(device)
        self.model = CNN(len(self.classes)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.85,
            max_num_hands=1)

    @staticmethod
    def draw_region(image, x, y, color=(0, 255, 0), pad=20):
        image_height, image_width = image.shape[:2]

        min_max_x = (np.array([np.min(x), np.max(x)]) * image_width).astype(np.int32)
        min_max_y = (np.array([np.min(y), np.max(y)]) * image_height).astype(np.int32)

        x0, y0 = max(0, min_max_x[0] - pad), max(0, min_max_y[0] - pad)
        x1, y1 = min(image_width, min_max_x[1] + pad), min(image_height, min_max_y[1] + pad)
        new_image = cv.rectangle(image.copy(), (x0, y0), (x1, y1), color, 2)
        return image[y0:y1, x0:x1], new_image, (x0, y0, x1, y1)
    
    @staticmethod
    def to_rgb(img):
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    @staticmethod
    def normalize_brightness(img, target_mean=65):
        current_mean = np.mean(img)
        adjustment_factor = target_mean / current_mean
        adjusted_img = np.clip(img * adjustment_factor, 0, 255).astype(np.uint8)
        return adjusted_img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = cv.flip(frame.to_ndarray(format="rgb24"), 1)
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = [landmark.x for landmark in hand_landmarks.landmark]
                y = [landmark.y for landmark in hand_landmarks.landmark]

                cropped_image, annotated_image, coords = self.draw_region(image, x, y, (0, 255, 0), pad=25)
                pred, _ = self.model.predict(self.normalize_brightness(cropped_image))
                image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(image)
                self.letters.append(self.classes[pred])
                most_common_letter = f"Буква: {Counter(self.letters).most_common(1)[0][0]}"
                
                x0, y0, x1, y1 = coords
                y0 = y0 - 35 if y0 > 0 else y0
                draw.text((x0, y0), most_common_letter, fill=(255, 255, 255),
                          font=ImageFont.truetype("Roboto-Regular.ttf", 35, encoding='UTF-8'))

        return av.VideoFrame.from_ndarray(np.array(image), format="rgb24")