import os
import matplotlib.pyplot as plt
import numpy as np

log_path = "../logs/log.txt"

if not os.path.exists(log_path):
    print(f"Ошибка: Файл {log_path} не найден!")
    exit(1)

epochs = []
losses = []
l_rs = []

step = 100 

with open(log_path, "r") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            losses.append(float(parts[0]))
            l_rs.append(float(parts[1]))
            epochs.append(idx * step)

epochs = np.array(epochs)
losses = np.array(losses)
l_rs = np.array(l_rs)

log_losses = np.log10(losses)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

points = ax.scatter(epochs, l_rs, log_losses, c=epochs, cmap='jet', s=15, label='Шаги обучения')
ax.plot(epochs, l_rs, log_losses, color='gray', alpha=0.5, linewidth=1.5) # Соединительная линия

ax.set_title("3D Траектория обучения нейросети", fontsize=14, fontweight='bold')
ax.set_xlabel("Текущая эпоха (Time)", fontsize=11, labelpad=10)
ax.set_ylabel("Скорость обучения (Learning Rate)", fontsize=11, labelpad=10)
ax.set_zlabel("Логарифм ошибки log10(Loss)", fontsize=11, labelpad=10)

cbar = fig.colorbar(points, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Шкала эпох (Прогресс обучения)')

ax.grid(True)

ax.view_init(elev=20, azim=-45)

plt.tight_layout()
plt.show()
