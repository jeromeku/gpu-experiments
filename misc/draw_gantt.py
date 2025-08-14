import matplotlib.pyplot as plt
import numpy as np

TITLE = "ATTN FWD TIMELINE"
QK0 = [
    (0, 830),
    (1826, 4861),
    (5864, 9382),
    (10495, 13797),
]
SM0 = [
    (830, 4767),
    (4861, 9199),
    (9382, 13688),
    (13797, 17857),
]
AK0 = [
    (4767, 7615),
    (9199, 12052),
    (13688, 16566),
    (17857, 18679),
]
QK1 = [
    (697, 1537),
    (2668, 6151),
    (7017, 10889),
    (11618, 15463),
]
SM1 = [
    (1537, 5923),
    (6151, 10659),
    (10889, 15368),
    (15463, 19278),
]
AK1 = [
    (5923, 8833),
    (10659, 13328),
    (15368, 17802),
    (19278, 20357),
]


def draw_gantt(title: str, data: dict):    
    # Auto-generate colors
    color_list = plt.cm.Set3(np.linspace(0, 1, len(data)))
    colors = {label: color_list[i] for i, label in enumerate(data.keys())}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each row
    for i, (label, intervals) in enumerate(data.items()):
        y_pos = len(data) - i - 1
        for start, end in intervals:
            duration = end - start
            ax.barh(
                y_pos, duration, left=start, 
                color=colors[label],
                alpha=0.7, 
                edgecolor='black', 
                linewidth=0.5
            )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Tasks')
    ax.set_title(title)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(reversed(data.keys()))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = "gantt_chart.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gantt chart saved to {filename}")


if __name__ == "__main__":
    # this is horrible but convenient!
    data = {}
    for name, value in list(globals().items()):
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            data[name] = value
    draw_gantt(title=TITLE, data=data)
