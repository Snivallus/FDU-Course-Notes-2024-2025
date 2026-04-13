import numpy as np
import matplotlib.pyplot as plt

def bezier_point(points, t):
    current = points.copy()
    n = len(current)
    for k in range(1, n):
        for i in range(n - k):
            current[i] = (1 - t) * current[i] + t * current[i + 1]
    return current[0]

# 用户输入控制点数量
n = int(input("Number of control points: "))
np.random.seed(51)
points = np.random.rand(n, 2) * 10  # 在0-10范围内生成随机点

fig, ax = plt.subplots()
ax.set_title(f"Bézier Curve with {n} control points")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 添加控制点连线（虚线灰色）
control_line, = ax.plot(points[:, 0], points[:, 1], 'k--', alpha=0.3, label='Control Polygon')
scatter = ax.scatter(points[:, 0], points[:, 1], c='red', picker=5, label='Control point')
bezier_line, = ax.plot([], [], 'b-', label='Bézier Curve')
ax.legend()

selected_point = None

def on_pick(event):
    global selected_point
    if event.artist != scatter:
        return
    selected_point = event.ind[0]
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

def on_motion(event):
    if selected_point is None or event.inaxes != ax:
        return
    points[selected_point] = [event.xdata, event.ydata]
    scatter.set_offsets(points)
    update_curve()
    fig.canvas.draw_idle()

def on_release(event):
    global selected_point
    selected_point = None

def update_curve():
    # 更新控制点连线
    control_line.set_data(points[:, 0], points[:, 1])
    
    # 更新贝塞尔曲线
    if len(points) >= 2:
        t_values = np.linspace(0, 1, 100)
        curve = np.array([bezier_point(points, t) for t in t_values])
        bezier_line.set_data(curve[:, 0], curve[:, 1])
    else:
        bezier_line.set_data([], [])

fig.canvas.mpl_connect('pick_event', on_pick)
update_curve()
plt.show()