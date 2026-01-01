from datetime import datetime

import cv2
import numpy as np
import torch
import albumentations as albu

class SurfaceDetector:
    def __init__(self):
        # Определение классов и размеров изображения
        self._CLASSES = ["garbage"]
        self._INFER_WIDTH = 256
        self._INFER_HEIGHT = 256

        # Определение устройства для вычислений
        self._DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка JIT модели
        self._best_model = torch.jit.load('../../../res/models/unet.pth', map_location=self._DEVICE)

    def get_validation_augmentation(self):
        """Получить аугментации для валидации."""
        test_transform = [
            albu.LongestMaxSize(max_size=self._INFER_HEIGHT),
            albu.PadIfNeeded(min_height=self._INFER_HEIGHT, min_width=self._INFER_WIDTH),
            albu.Normalize(),
        ]
        return albu.Compose(test_transform)

    def run_model_on_image(self, image_path, cam_id):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width, _ = image.shape

        # Применение аугментаций
        augmentation = self.get_validation_augmentation()
        augmented = augmentation(image=image)
        image_transformed = augmented['image']

        # Преобразование изображения в PyTorch тензор и перемещение на устройство
        x_tensor = torch.from_numpy(image_transformed).to(self._DEVICE).unsqueeze(0).permute(0, 3, 1, 2).float()

        # Прогон изображения через модель
        self._best_model.eval()
        with torch.no_grad():
            pr_mask = self._best_model(x_tensor)

        # Преобразование вывода в массив numpy и удаление размерности пакета
        pr_mask = pr_mask.squeeze().cpu().detach().numpy()

        # Получение класса с наивысшей вероятностью для каждого пикселя
        label_mask = np.argmax(pr_mask, axis=0)

        # Определение количества пикселей, которые будут появляться по бокам от паддингов, и их обрезка
        if original_height > original_width:
            delta_pixels = int(((original_height-original_width)/2)/original_height * self._INFER_HEIGHT)
            mask_cropped = label_mask[:, delta_pixels + 1 : self._INFER_WIDTH - delta_pixels - 1]
        elif original_height < original_width:
            delta_pixels = int(((original_width-original_height)/2)/original_width * self._INFER_WIDTH)
            mask_cropped = label_mask[delta_pixels + 1: self._INFER_HEIGHT - delta_pixels - 1, :]
        else:
            mask_cropped = label_mask

        # Изменение размера маски обратно к исходному размеру изображения
        label_mask_real_size = cv2.resize(
            mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        )

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        cv2.imwrite(f'../../../res/snapshots/{cam_id}/unet-{timestamp}.png', label_mask_real_size)