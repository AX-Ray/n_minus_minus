import matplotlib.pyplot as plt
import pandas as pd

# Читаем лог
df = pd.read_csv('../training_log.txt')

# Создаём графики
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# График потерь
axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss')
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# График точности
axes[0, 1].plot(df['epoch'], df['val_accuracy'] * 100)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Validation Accuracy')
axes[0, 1].grid(True)

# График learning rate
axes[1, 0].plot(df['epoch'], df['lr'])
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

# График потерь (логарифмический)
axes[1, 1].plot(df['epoch'], df['train_loss'], label='Train Loss')
axes[1, 1].plot(df['epoch'], df['val_loss'], label='Val Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss (log scale)')
axes[1, 1].set_title('Loss (Log Scale)')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()