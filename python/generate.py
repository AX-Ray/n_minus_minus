#!/usr/bin/env python3
"""
Генератор датасета геометрических фигур (32x32 пикселя).
Классы: квадрат, треугольник, круг.
Исправленная версия с защитой от ошибок.
"""

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_shape_image(size=32, shape_type='square', max_shape_size=None):
    """
    Генерирует изображение 32x32 с одной фигурой.
    
    Args:
        size: размер изображения (32 для 32x32)
        shape_type: 'square', 'triangle', 'circle'
        max_shape_size: максимальный размер фигуры (если None, то 60% от размера)
    
    Returns:
        PIL.Image: изображение с фигурой
    """
    width = height = size
    
    # Создаём чёрный фон (0 = черный)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Определяем максимальный размер фигуры (уменьшаем до 60% для безопасности)
    if max_shape_size is None:
        max_shape_size = int(size * 0.6)  # 60% от 32 = ~19 пикселей
    
    # Случайный размер (от 20% до 100% от max_shape_size)
    # Минимальный размер 5 пикселей, чтобы фигура была видна
    min_shape_size = max(5, int(max_shape_size * 0.25))
    shape_size = random.randint(min_shape_size, max_shape_size)
    
    # Отступ от края (не меньше 2 пикселей)
    margin = max(2, shape_size // 4 + 1)
    
    if shape_type == 'square':
        # Квадрат: задаём размер стороны
        side = shape_size
        # Проверяем, что есть место для фигуры
        max_x = width - side - margin
        min_x = margin
        
        if max_x <= min_x:
            # Если места нет, уменьшаем размер
            side = min(side, width - 2 * margin)
            max_x = width - side - margin
            min_x = margin
        
        x0 = random.randint(min_x, max_x)
        y0 = random.randint(min_x, max_x)  # одинаковые ограничения для квадрата
        x1 = x0 + side
        y1 = y0 + side
        draw.rectangle([x0, y0, x1, y1], fill=255)
    
    elif shape_type == 'triangle':
        # Равносторонний треугольник
        side = shape_size
        max_center_offset = min(
            width//2 - side//2 - margin,
            height//2 - side//2 - margin
        )
        
        if max_center_offset < margin:
            # Если места нет, уменьшаем размер
            side = min(side, min(width, height) - 2 * margin)
            max_center_offset = max(0, min(
                width//2 - side//2 - margin,
                height//2 - side//2 - margin
            ))
        
        # Смещение от центра
        offset_x = random.randint(-max_center_offset, max_center_offset)
        offset_y = random.randint(-max_center_offset, max_center_offset)
        
        center_x = width // 2 + offset_x
        center_y = height // 2 + offset_y
        
        # Высота равностороннего треугольника
        height_triangle = int(side * 0.866)
        
        points = [
            (center_x, center_y - height_triangle//2),
            (center_x - side//2, center_y + height_triangle//2),
            (center_x + side//2, center_y + height_triangle//2)
        ]
        draw.polygon(points, fill=255)
    
    elif shape_type == 'circle':
        radius = shape_size // 2
        # Радиус должен быть меньше половины размера
        max_radius = min(width, height) // 2 - margin
        if radius > max_radius:
            radius = max_radius
        
        cx = random.randint(margin + radius, width - radius - margin)
        cy = random.randint(margin + radius, height - radius - margin)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    # === АУГМЕНТАЦИЯ ===
    
    # 1. Случайное размытие
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    # 2. Случайный шум
    if random.random() < 0.2:
        arr = np.array(img)
        noise_mask = np.random.random(arr.shape) < 0.02
        noise = np.random.randint(0, 40, arr.shape, dtype=np.uint8)
        arr[noise_mask] = np.clip(arr[noise_mask] + noise[noise_mask], 0, 255)
        img = Image.fromarray(arr)
    
    # 3. Инверсия цвета
    if random.random() < 0.1:
        arr = np.array(img)
        arr = 255 - arr
        img = Image.fromarray(arr)
    
    return img

def generate_dataset(output_dir, num_per_class=1000, image_size=32, max_shape_size=None):
    """
    Генерирует датасет и сохраняет в подпапки.
    """
    shapes = ['square', 'triangle', 'circle']
    
    # Создаём папки для каждого класса
    for shape in shapes:
        class_dir = os.path.join(output_dir, shape)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"Генерация датасета {image_size}x{image_size}...")
    print(f"На класс: {num_per_class} изображений")
    print(f"Всего: {num_per_class * len(shapes)} изображений")
    print("-" * 40)
    
    # Генерируем изображения
    for shape in shapes:
        class_dir = os.path.join(output_dir, shape)
        print(f"\nКласс '{shape}':")
        
        for i in range(num_per_class):
            # Генерируем изображение с защитой от ошибок
            try:
                img = generate_shape_image(image_size, shape, max_shape_size)
            except Exception as e:
                print(f"  Ошибка при генерации {i}: {e}")
                # Используем запасной вариант - маленькую фигуру в центре
                img = generate_safe_image(image_size, shape)
            
            # Сохраняем в BMP
            filename = os.path.join(class_dir, f"{i:05d}.bmp")
            img.save(filename, format='BMP')
            
            # Прогресс
            if (i + 1) % 200 == 0 or i == num_per_class - 1:
                print(f"  {i+1}/{num_per_class} сохранено")
    
    print("\n" + "=" * 40)
    print(f"✅ Датасет сохранён в: {output_dir}")
    print(f"   - Всего файлов: {num_per_class * len(shapes)}")
    print(f"   - Размер изображений: {image_size}x{image_size}")
    print("=" * 40)

def generate_safe_image(size, shape_type):
    """
    Запасной вариант: создаёт изображение с фигурой в центре.
    Используется, если основной генератор выдал ошибку.
    """
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    # Маленькая фигура в центре (40% от размера)
    shape_size = int(size * 0.4)
    center = size // 2
    
    if shape_type == 'square':
        side = shape_size
        x0 = center - side//2
        y0 = center - side//2
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=255)
    
    elif shape_type == 'triangle':
        side = shape_size
        height = int(side * 0.866)
        points = [
            (center, center - height//2),
            (center - side//2, center + height//2),
            (center + side//2, center + height//2)
        ]
        draw.polygon(points, fill=255)
    
    elif shape_type == 'circle':
        radius = shape_size // 2
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], fill=255)
    
    return img

# === ВИЗУАЛИЗАЦИЯ ДЛЯ ПРОВЕРКИ ===

def visualize_sample(output_dir, num_samples=5):
    """
    Показывает примеры сгенерированных изображений.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️ Для визуализации установите matplotlib: pip install matplotlib")
        return
    
    shapes = ['square', 'triangle', 'circle']
    fig, axes = plt.subplots(len(shapes), num_samples, figsize=(num_samples*2, len(shapes)*2))
    
    for i, shape in enumerate(shapes):
        class_dir = os.path.join(output_dir, shape)
        if not os.path.exists(class_dir):
            continue
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.bmp')])
        
        for j in range(min(num_samples, len(files))):
            img_path = os.path.join(class_dir, files[j])
            img = Image.open(img_path)
            if len(shapes) > 1:
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(shape, fontsize=12)
            else:
                axes[j].imshow(img, cmap='gray')
                axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор датасета геометрических фигур 32x32")
    parser.add_argument('--output', type=str, default='dataset_32x32',
                        help='Папка для сохранения датасета (по умолчанию: dataset_32x32)')
    parser.add_argument('--num', type=int, default=1000,
                        help='Количество изображений на класс (по умолчанию: 1000)')
    parser.add_argument('--size', type=int, default=32,
                        help='Размер изображения (по умолчанию: 32)')
    parser.add_argument('--max_shape', type=int, default=None,
                        help='Максимальный размер фигуры (по умолчанию: 60% от размера)')
    parser.add_argument('--visualize', action='store_true',
                        help='Показать примеры изображений после генерации')
    
    args = parser.parse_args()
    
    # Генерируем датасет
    generate_dataset(
        output_dir=args.output,
        num_per_class=args.num,
        image_size=args.size,
        max_shape_size=args.max_shape
    )
    
    # Визуализация
    if args.visualize:
        print("\nВизуализация примеров...")
        visualize_sample(args.output, num_samples=5)
