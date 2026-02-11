"""
Класс для обработки изображений различными методами
"""
import cv2
import numpy as np


class ImageProcessor:
    """Класс для применения различных методов обработки изображений"""
    
    def __init__(self):
        # Загрузка Haar Cascade для детекции лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    # ============= ОСНОВНЫЕ МЕТОДЫ (7) =============
    
    def canny(self, image):
        """
        Детектор границ Canny
        Двухпороговый алгоритм выделения границ
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def laplace(self, image):
        """
        Оператор Лапласа
        Выделяет области быстрого изменения интенсивности (вторая производная)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    def sobel(self, image):
        """
        Оператор Собеля
        Вычисляет градиент изображения (первая производная)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    
    def roberts(self, image):
        """
        Оператор Робертса
        Простой оператор для выделения границ (диагональные градиенты)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ядра Робертса
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        roberts_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        roberts = np.sqrt(roberts_x**2 + roberts_y**2)
        roberts = np.uint8(roberts)
        return cv2.cvtColor(roberts, cv2.COLOR_GRAY2BGR)
    
    def prewitt(self, image):
        """
        Оператор Превитта
        Аналог Собеля с равными весами
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ядра Превитта
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt = np.uint8(prewitt)
        return cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
    
    def scharr(self, image):
        """
        Оператор Шарра
        Улучшенная версия Собеля с большей точностью
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr = np.sqrt(scharr_x**2 + scharr_y**2)
        scharr = np.uint8(scharr)
        return cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
    
    def otsu(self, image):
        """
        Пороговая обработка Оцу
        Автоматический выбор порога бинаризации
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    
    # ============= ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ (4) =============
    
    def log_filter(self, image):
        """
        LoG (Laplacian of Gaussian)
        Сначала размытие, затем Лапласиан - хорошо выделяет края
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        log = np.uint8(np.absolute(log))
        return cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)
    
    def gaussian_blur(self, image):
        """
        Гауссово размытие
        Сглаживание изображения с сохранением важных деталей
        """
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred
    
    def bilateral_filter(self, image):
        """
        Билатеральный фильтр
        Сглаживание с сохранением границ объектов
        """
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        return bilateral
    
    def median_filter(self, image):
        """
        Медианный фильтр
        Эффективен для удаления импульсного шума (соль и перец)
        """
        median = cv2.medianBlur(image, 5)
        return median
    
    # ============= FACE DETECTION =============
    
    def haar_face_detection(self, image):
        """
        Face Detection с помощью Haar Cascades
        Детектирует лица и рисует прямоугольники вокруг них
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем эквализацию гистограммы для улучшения контраста
        gray = cv2.equalizeHist(gray)
        
        # Детекция лиц с улучшенными параметрами
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Меньший шаг для лучшей детекции
            minNeighbors=3,     # Меньше соседей для большей чувствительности
            minSize=(20, 20),   # Меньший минимальный размер
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        result = image.copy()
        
        # Если лица не найдены, добавим сообщение
        if len(faces) == 0:
            cv2.putText(result, 'No faces detected', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Рисуем прямоугольники вокруг найденных лиц
            for (x, y, w, h) in faces:
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(result, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result
    
    def get_all_methods(self):
        """Возвращает словарь всех доступных методов"""
        return {
            # Основные методы
            'Canny (Кенни)': self.canny,
            'Laplace (Лаплас)': self.laplace,
            'Sobel (Собель)': self.sobel,
            'Roberts (Робертс)': self.roberts,
            'Prewitt (Преситта)': self.prewitt,
            'Scharr (Шарра)': self.scharr,
            'Otsu (Оцу)': self.otsu,
            
            # Дополнительные методы
            'LoG (Laplacian of Gaussian)': self.log_filter,
            'Gaussian Blur (Гауссово размытие)': self.gaussian_blur,
            'Bilateral Filter (Билатеральный)': self.bilateral_filter,
            'Median Filter (Медианный)': self.median_filter,
            
            # Face Detection
            'Haar Face Detection (Детекция лиц)': self.haar_face_detection
        }
