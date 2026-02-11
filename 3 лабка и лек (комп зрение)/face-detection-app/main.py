"""
Приложение для обработки изображений с различными методами и Face Detection
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from image_processor import ImageProcessor


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений и Face Detection")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        self.processor = ImageProcessor()
        self.original_image = None
        self.current_image = None
        self.video_capture = None
        self.is_video_playing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Создание интерфейса"""
        # Верхняя панель с кнопками
        top_frame = tk.Frame(self.root, bg='#2b2b2b', padx=10, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Кнопки загрузки
        btn_style = {'bg': '#4CAF50', 'fg': 'white', 'font': ('Arial', 12, 'bold'),
                     'padx': 15, 'pady': 8, 'relief': 'raised', 'bd': 2}
        
        tk.Button(top_frame, text="📁 Загрузить изображение", 
                 command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="🎥 Загрузить видео", 
                 command=self.load_video, 
                 bg='#2196F3', fg='white', font=('Arial', 12, 'bold'),
                 padx=15, pady=8, relief='raised', bd=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="⏹ Остановить видео", 
                 command=self.stop_video,
                 bg='#f44336', fg='white', font=('Arial', 12, 'bold'),
                 padx=15, pady=8, relief='raised', bd=2).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top_frame, text="🔄 Сбросить", 
                 command=self.reset_image,
                 bg='#FF9800', fg='white', font=('Arial', 12, 'bold'),
                 padx=15, pady=8, relief='raised', bd=2).pack(side=tk.LEFT, padx=5)
        
        # Основной контейнер
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - изображения
        images_frame = tk.Frame(main_container, bg='#2b2b2b')
        images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Оригинал
        original_container = tk.Frame(images_frame, bg='#3c3c3c', relief='solid', bd=2)
        original_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(original_container, text="ОРИГИНАЛ", 
                bg='#3c3c3c', fg='white', font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.original_label = tk.Label(original_container, bg='#1e1e1e')
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Результат
        result_container = tk.Frame(images_frame, bg='#3c3c3c', relief='solid', bd=2)
        result_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(result_container, text="РЕЗУЛЬТАТ", 
                bg='#3c3c3c', fg='white', font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.result_label = tk.Label(result_container, bg='#1e1e1e')
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Правая панель - методы
        methods_frame = tk.Frame(main_container, bg='#3c3c3c', width=300, relief='solid', bd=2)
        methods_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        methods_frame.pack_propagate(False)
        
        tk.Label(methods_frame, text="МЕТОДЫ ОБРАБОТКИ", 
                bg='#3c3c3c', fg='white', font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Скроллбар для методов
        canvas = tk.Canvas(methods_frame, bg='#3c3c3c', highlightthickness=0)
        scrollbar = ttk.Scrollbar(methods_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#3c3c3c')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Кнопки методов
        methods = self.processor.get_all_methods()
        
        # Группы методов
        groups = {
            'ОСНОВНЫЕ МЕТОДЫ': list(methods.keys())[:7],
            'ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ': list(methods.keys())[7:11],
            'FACE DETECTION': list(methods.keys())[11:]
        }
        
        for group_name, method_names in groups.items():
            # Заголовок группы
            tk.Label(scrollable_frame, text=group_name, 
                    bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                    pady=5).pack(fill=tk.X, padx=10, pady=(10, 5))
            
            # Кнопки методов
            for method_name in method_names:
                btn = tk.Button(
                    scrollable_frame,
                    text=method_name,
                    command=lambda m=method_name: self.apply_method(m),
                    bg='#555555',
                    fg='white',
                    font=('Arial', 10),
                    pady=8,
                    relief='raised',
                    bd=1,
                    cursor='hand2'
                )
                btn.pack(fill=tk.X, padx=10, pady=2)
                
                # Эффект наведения
                btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#777777'))
                btn.bind('<Leave>', lambda e, b=btn: b.config(bg='#555555'))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def load_image(self):
        """Загрузка изображения"""
        self.stop_video()
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # Исправление для кириллических путей
            try:
                # Читаем файл через numpy для поддержки кириллицы
                with open(file_path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                self.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if self.original_image is None:
                    messagebox.showerror("Ошибка", "Не удалось загрузить изображение!")
                    return
                    
                self.current_image = self.original_image.copy()
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")
            
    def load_video(self):
        """Загрузка видео"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.stop_video()
            self.video_capture = cv2.VideoCapture(file_path)
            self.is_video_playing = True
            self.play_video()
            
    def play_video(self):
        """Воспроизведение видео"""
        if self.is_video_playing and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                self.original_image = frame
                self.current_image = frame.copy()
                self.display_images()
                self.root.after(30, self.play_video)
            else:
                # Видео закончилось, начать сначала
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.play_video()
                
    def stop_video(self):
        """Остановка видео"""
        self.is_video_playing = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
    def reset_image(self):
        """Сброс к оригиналу"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_images()
            
    def apply_method(self, method_name):
        """Применение выбранного метода"""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return
            
        methods = self.processor.get_all_methods()
        if method_name in methods:
            try:
                self.current_image = methods[method_name](self.original_image)
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка применения метода: {str(e)}")
                
    def display_images(self):
        """Отображение изображений"""
        if self.original_image is not None:
            # Оригинал
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb)
            original_pil = self.resize_image(original_pil, 600, 600)
            original_photo = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_photo)
            self.original_label.image = original_photo
            
        if self.current_image is not None:
            # Результат
            result_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            result_pil = self.resize_image(result_pil, 600, 600)
            result_photo = ImageTk.PhotoImage(result_pil)
            self.result_label.config(image=result_photo)
            self.result_label.image = result_photo
            
    def resize_image(self, image, max_width, max_height):
        """Изменение размера изображения с сохранением пропорций"""
        width, height = image.size
        ratio = min(max_width/width, max_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
