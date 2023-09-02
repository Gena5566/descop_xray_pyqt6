import cv2
import skimage.io as skio
import skimage.exposure as ske
import skimage.filters as skf
import numpy as np
import skimage.transform as skt
import shutil
from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6 import QtWidgets
import os
import ultralytics
ultralytics.checks()
import ultralytics
ultralytics.checks()
import matplotlib.patches as patches
from ultralytics import SAM
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import matplotlib.pyplot as plt
from model_paths import model_path_yolov8n, model_sam_segment





class Ui_Dialog(object):
    def __init__(self):
        self.processed_image_paths = []  # Объявление списка для путей к обработанным изображениям
        self.predicted_image_labels = []  # Список для хранения QLabel'ов с предсказаниями
        self.image = QImage()  # Создание объекта QImage для работы с изображением


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1400, 620)

        # В методе setupUi(Dialog)
        self.processedLabels = []
        for j in range(4):
            label = QtWidgets.QLabel(Dialog)
            label.setGeometry(QtCore.QRect(240 + j * 200, 0, 210, 550))
            label.setObjectName(f"processedLabel{j}")
            label.setScaledContents(True)
            self.processedLabels.append(label)

        for j in range(4):
            label = QtWidgets.QLabel(Dialog)
            label.setGeometry(QtCore.QRect(j * 200, 300, 210, 550))  # координаты по вертикали и горизонтали
            label.setObjectName(f"predictedLabel{j}")
            label.setScaledContents(True)
            self.predicted_image_labels.append(label)  # Добавление QLabel в список

        self.imageLabel = QtWidgets.QLabel(Dialog)
        self.imageLabel.setGeometry(QtCore.QRect(0, 0, 210, 550))
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setScaledContents(True)

        # Создаем QLabel для пятого изображения
        self.fifthLabel = QtWidgets.QLabel(Dialog)
        self.fifthLabel.setGeometry(QtCore.QRect(240 + 4 * 200, 0, 210, 550))
        self.fifthLabel.setObjectName("fifthLabel")
        self.fifthLabel.setScaledContents(True)
        self.processedLabels.append(self.fifthLabel)  # Добавляем его в список processedLabels

        button_width = 180

        self.browseButton = QtWidgets.QPushButton(Dialog)
        self.browseButton.setGeometry(QtCore.QRect(40, 560, button_width, 40))
        self.browseButton.setObjectName("browseButton")
        self.browseButton.setText("Выбрать изображение")
        self.browseButton.clicked.connect(self.openFileDialog)

        self.processButton = QtWidgets.QPushButton(Dialog)
        self.processButton.setGeometry(QtCore.QRect(40 + button_width + 40, 560, button_width, 40))
        self.processButton.setObjectName("processButton")
        self.processButton.setText("Применить фильтры")
        self.processButton.clicked.connect(self.process_single_image)

        self.predictButton = QtWidgets.QPushButton(Dialog)
        self.predictButton.setGeometry(QtCore.QRect(40 + 2 * (button_width + 40), 560, button_width, 40))
        self.predictButton.setObjectName("predictButton")
        self.predictButton.setText("Детекция изображения")
        self.predictButton.clicked.connect(self.predict_processed_image)


        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(40, 570, 1120, 40))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.segmentationButton = QtWidgets.QPushButton(Dialog)
        self.segmentationButton.setGeometry(QtCore.QRect(40 + 3 * (button_width + 40), 560, button_width, 40))
        self.segmentationButton.setObjectName("segmentationButton")
        self.segmentationButton.setText("Сегментация изображения")
        self.segmentationButton.clicked.connect(self.segmentation_predict)

        # Подключаем обработчики событий к кнопкам
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

    def openFileDialog(self):
        destination_folder = "original_image"
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Выбрать изображение", "",
                                                             "Images (*.png *.jpg *.tif)")
        if file_path:
            image_filename = os.path.basename(file_path)
            self.destination_path = os.path.join(destination_folder, image_filename)
            shutil.copyfile(file_path, self.destination_path)

            # Отображение выбранного оригинального изображения на экране
            pixmap = QtGui.QPixmap(self.destination_path)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)

    def process_single_image(self):
        # Создание папки для сохранения обработанных изображений в формате "processed_images"
        processed_images_dir = "processed_images"
        if not os.path.exists(processed_images_dir):
            os.makedirs(processed_images_dir)

        # Загрузка оригинального изображения
        original_image = skio.imread(self.destination_path)

        # Изменение размера изображения до 950x384 пикселей
        image_resized = skt.resize(original_image, (950, 384), anti_aliasing=True)

        # Определение параметров коррекции
        dehaze_values = [0.035, 0.037, 0.04, 0.045]  # Значения для дефоггинга (от 0.5 до 1.0 с шагом 0.25)
        brightness_values = [0.11, 0.12, 0.14, 0.16]  # Значения для яркости (от 0.75 до 1.0 с шагом 0.05)
        contrast_values = [5.2, 5.5, 5.6, 6.0]  # Значения для контраста (от 0.75 до 1.0 с шагом 0.05)
        fill_light_values = [2.7, 2.8, 2.9, 3.0]  # Значения для заполняющего света (от 0.75 до 1.0 с шагом 0.05)
        exposure_range = np.arange(2.1, 3.5, 4.0)  # Диапазон значений для экспозиции (от 1.0 до 2.0 с шагом 0.25)

        # Обработка и сохранение четырех изображений с заданными параметрами
        for j in range(4):
            # Дефоггинг
            dehazed_image = ske.adjust_gamma(image_resized, gamma=dehaze_values[j])

            # Изменение яркости
            brightness_corrected = ske.adjust_gamma(dehazed_image, gamma=brightness_values[j])

            # Изменение контраста
            contrast_corrected = ske.adjust_gamma(brightness_corrected, gamma=contrast_values[j])

            # Изменение заполняющего света
            fill_light_corrected = ske.adjust_gamma(contrast_corrected, gamma=fill_light_values[j])

            # Применение фильтра увеличения резкости
            sharpened_images = []
            for exposure_value in exposure_range:
                exposure_corrected = ske.adjust_gamma(fill_light_corrected, gamma=exposure_value)
                sharpened_image = skf.unsharp_mask(exposure_corrected, radius=5, amount=5)
                sharpened_images.append(sharpened_image)

            # Генерация имени файла для сохранения обработанного изображения
            filename = f"{str(j + 1).zfill(2)}_{'items'}.tif"
            save_path = os.path.join(processed_images_dir, filename)

            # Сохранение откорректированного изображения в формате TIF с помощью OpenCV
            cv2.imwrite(save_path, (sharpened_images[-1] * 255).astype(
                np.uint8))  # Умножаем на 255, так как OpenCV ожидает значения в диапазоне [0, 255]

        # Применение фильтра увеличения резкости к исходному изображению
        W = 190
        L = 0.5
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # ядро выделения границ

        img_filtered = cv2.filter2D(original_image, -1, kernel)  # фильтрация с ядром kernel
        img_filtered[(img_filtered < L - W / 2)] = L - W / 2
        img_filtered[(img_filtered > L + W / 2)] = L + W / 2

        # Приведение значений пикселей к допустимому диапазону [0, 255]
        img_filtered = ske.rescale_intensity(img_filtered, in_range=(img_filtered.min(), img_filtered.max()),
                                             out_range=(0, 255))

        # Изменение размера изображения до (950, 384) пикселей
        resized_filtered_image = skt.resize(img_filtered, (950, 384), anti_aliasing=True)

        # Генерация имени файла для сохранения обработанного изображения
        filename = f"05_items.tif"
        save_path = os.path.join(processed_images_dir, filename)

        # Сохранение откорректированного и измененного изображения в формате TIF с помощью OpenCV
        cv2.imwrite(save_path, resized_filtered_image.astype(np.uint8))

        # Обновление отображения обработанных изображений
        for j, label in enumerate(self.processedLabels):
            pixmap = QtGui.QPixmap(os.path.join(processed_images_dir, f"{str(j + 1).zfill(2)}_items.tif"))
            label.setPixmap(pixmap)

            # Получение списока путей к обработанным изображениям
            self.processed_image_paths = []  # Сначала очистите список
            for j in range(4):
                filename = f"{str(j + 1).zfill(2)}_items.tif"
                self.processed_image_paths.append(os.path.join(processed_images_dir, filename))

        for img_path in self.processed_image_paths:
            # Вывод пути к текущему изображению
            print(f"Processing image: {img_path}")


            # Обновление отображения обработанных изображений
            for j, label in enumerate(self.processedLabels):
                pixmap = QtGui.QPixmap(os.path.join(processed_images_dir, f"{str(j + 1).zfill(2)}_items.tif"))
                label.setPixmap(pixmap)


    def show_processed_images(self):
        for label in self.processedLabels:
            label.show()

    def show_fifth_image(self):
        processed_images_dir = "processed_images"
        filename = "05_items.tif"
        path = os.path.join(processed_images_dir, filename)

        pixmap = QtGui.QPixmap(path)
        self.fifthLabel.setPixmap(pixmap)


    def predict_processed_image(self, img_pred):
        # Загрузка модели YOLO
        model = YOLO(model_path_yolov8n)

        # Выбираем изображение из списка
        first_image_path = self.processed_image_paths[2]

        # Выполняем предсказания на выбранном изображении
        results = model([first_image_path])

        # Результат предсказания модели YOLO
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs

            # Вывод результатов для отладки
            #print(results)
            #print(boxes)

            # Определение класса
            classes = {0: 'Опасные', 1: 'Внимание'}  # Пример классов, замените на свои

            # Создайте список для хранения всех осей
            all_axes = []

            # Создание полотна для отображения
            fig = plt.figure(figsize=(25, 20))

            for idx, result in enumerate(results):
                boxes = result.boxes.data.tolist()
                class_labels = result.boxes.data[:, 5].tolist()

                # Добавьте новую ось на полотно
                ax = fig.add_subplot(1, len(results), idx + 1)
                all_axes.append(ax)

                ax.imshow(result.orig_img)  # Использование атрибута orig_img для вывода исходного изображения
                ax.axis('off')  # Без оси

                for box, class_label in zip(boxes, class_labels):
                    x1, y1, x2, y2 = box[:4]  # Координаты границ bbox
                    class_name = classes[int(class_label)]  # Имя класса по индексу

                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, class_name, color='red', fontsize=5, backgroundcolor="white")  # Отобразите класс

            ax.set_title(f"Image {idx}")

        plt.tight_layout()
        plt.show()  # Показать на полотне

    def segmentation_predict(self):
        image_path = self.processed_image_paths[2]
        # Путь к изображению

        # Загрузка модели YOLO
        model = YOLO(model_path_yolov8n)
        results = model.predict(source=image_path, conf=0.25)

        # Загрузка модели SAM
        model_sam = SAM(model_sam_segment)

        # Вывод информации о модели для отладки
        #model_sam.info()

        # Список для хранения масок сегментации
        segmentation_masks = []

        # Результат предсказания модели YOLO
        for result in results:
            boxes = result.boxes
            bbox_coordinates = boxes.xyxy.tolist()

            for bbox in bbox_coordinates:
                x_min, y_min, x_max, y_max = bbox
                bboxes = [x_min, y_min, x_max, y_max]

                # Предсказание сегментации с помощью модели SAM
                results_list = model_sam.predict(image_path, bboxes=[bboxes])
                results = results_list[0]  # Извлечь результат для первого изображения

                # Извлечение масок сегментации
                masks = results.masks
                segmentation_masks.append(masks)

        # Загрузка изображения
        image = cv2.imread(image_path)

        # Создание копии изображения для наложения сегментации
        overlay = image.copy()
        alpha = 0.5  # Прозрачность сегментации

        # Проход по маскам сегментации и наложение их на изображение
        for masks in segmentation_masks:
            masks_np = masks.data.cpu().numpy()  # Преобразование в массив numpy
            for mask in masks_np:
                mask_np = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)  # Зеленый цвет для контуров

        # Наложение изображения с сегментацией на оригинальное с прозрачностью
        output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        plt.figure(figsize=(10, 8))
        # Отображение результата с помощью matplotlib
        plt.imshow(cv2.cvtColor(output,
                                cv2.COLOR_BGR2RGB))  # OpenCV использует BGR, поэтому нужно преобразовать в RGBplt.axis('off')
        plt.axis('off')  # Отключение осей
        plt.show()



if __name__ == "__main__":
    import sys
    from PyQt6 import QtWidgets, QtGui

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())