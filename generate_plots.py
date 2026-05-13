import matplotlib.pyplot as plt
import numpy as np

def generate_fairness_plots():
    # Данные для визуализации
    labels = ['Исходные\nданные', 'Стандартная\nмодель', 'Модель\nMINE']
    spd_values = [-0.1945, -0.2227, -0.0606]
    di_values = [0.3597, 0.1114, 0.5772]
    
    colors = ['#95a5a6', '#e74c3c', '#27ae60']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- ГРАФИК 1: Статистический паритет (SPD) ---
    bars1 = ax1.bar(labels, spd_values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('Статистический паритет (SPD) ↓', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Значение (Идеал: 0.0)', fontsize=12)
    ax1.set_ylim(-0.3, 0.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    ax1.axhline(y=-0.1, color='blue', linestyle='--', linewidth=1.5, label='Порог справедливости (-0.1)')
    ax1.legend(loc='lower left')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    # --- ГРАФИК 2: Диспропорция воздействия (DI) ---
    bars2 = ax2.bar(labels, di_values, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('Диспропорция воздействия (DI) ↑', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Значение (Идеал: 1.0)', fontsize=12)
    ax2.set_ylim(0, 1.2)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    ax2.axhline(y=0.8, color='blue', linestyle='--', linewidth=1.5, label='Порог справедливости (0.8)')
    ax2.axhline(y=1.0, color='gold', linestyle='-', linewidth=1, alpha=0.5, label='Идеал (1.0)')
    ax2.legend(loc='upper left')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    # Компоновка и заголовок
    plt.suptitle('Сравнительный анализ метрик справедливости моделей', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig('final_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_fairness_plots()