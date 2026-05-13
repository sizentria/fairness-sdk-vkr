import json
import os
import glob
import matplotlib.pyplot as plt

def plot_latest_history():
    list_of_files = glob.glob('reports/history_*.json')
    if not list_of_files:
        print("Ошибка: Файлы истории не найдены.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['task_loss']) + 1)

    # Настройка шрифтов для соответствия научному стилю
    plt.rcParams['font.family'] = 'sans-serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- График А: Сходимость обучения (Task Loss) ---
    color_task = '#1f77b4' # Глубокий синий
    ax1.plot(epochs, history['task_loss'], color=color_task, linewidth=2, 
             marker='o', markevery=5, markersize=4, label='Потери обучения')
    ax1.fill_between(epochs, history['task_loss'], color=color_task, alpha=0.1)
    
    ax1.set_title('а) Динамика обучения модели', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Эпоха', fontsize=10)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- График Б: Этический паритет (MI Estimate) ---
    color_mi = '#ff7f0e' # Насыщенный оранжевый
    ax2.plot(epochs, history['mi_loss'], color=color_mi, linewidth=2, 
             marker='s', markevery=5, markersize=4, label='Взаимная информация (Z;Y)')
    ax2.fill_between(epochs, history['mi_loss'], color=color_mi, alpha=0.1)
    
    ax2.set_title('б) Контроль информационной утечки', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Эпоха', fontsize=10)
    ax2.set_ylabel('MI Estimate (nats)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # Финальная корректировка по ГОСТ
    plt.tight_layout(pad=4.0)
    
    # Центрированная подпись рисунка под графиками
    fig.text(0.5, 0.02, 'Рисунок 5.2 – Динамика показателей состязательного обучения нейронной сети', 
             ha='center', fontsize=11, style='italic')

    output_name = 'training_dynamics_plot_v2.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Улучшенный график сохранен как: {output_name}")
    plt.show()

if __name__ == "__main__":
    plot_latest_history()