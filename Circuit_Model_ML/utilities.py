import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def rotate_points(xy_pairs, origin, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated = []
    ox, oy = origin
    for x, y in xy_pairs:
        vec = np.array([x - ox, y - oy])
        rx, ry = rot_matrix @ vec + np.array([ox, oy])
        rotated.append((rx, ry))
    return rotated

def draw_diode_symbol(ax, x=0, y=0, color="black", up_or_down="down", is_LED=False, rotation=0):
    origin = (x, y)
    dir = 1 if up_or_down == "down" else -1

    # Diode circle for LED
    if is_LED:
        circle = patches.Circle(origin, 0.17, edgecolor=color, facecolor='white', linewidth=1.5, fill=False)
        ax.add_patch(circle)

    # Diode triangle arrowhead
    arrow_start = (x, y + 0.075 * dir)
    arrow_end = (x, y + 0.074 * dir - 0.001 * dir)  # Very short shaft
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.15, head_length=0.15, fc=color, ec=color)

    # Diode bar
    bar_pts = rotate_points([(x - 0.075, y - 0.08 * dir), (x + 0.075, y - 0.08 * dir)], origin, rotation)
    ax.add_line(plt.Line2D([bar_pts[0][0], bar_pts[1][0]], [bar_pts[0][1], bar_pts[1][1]], color=color, linewidth=2))

    # LED rays
    if is_LED:
        ray1 = rotate_points([(x - 0.05, y - 0.05 * dir), (x - 0.2, y - 0.2 * dir)], origin, rotation)
        ray2 = rotate_points([(x - 0.075, y + 0.025 * dir), (x - 0.225, y - 0.125 * dir)], origin, rotation)
        ax.arrow(*ray1[0], ray1[1][0] - ray1[0][0], ray1[1][1] - ray1[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')
        ax.arrow(*ray2[0], ray2[1][0] - ray2[0][0], ray2[1][1] - ray2[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')

    # Terminals
    term_top = rotate_points([(x, y + 0.08), (x, y + 0.5)], origin, rotation)
    term_bot = rotate_points([(x, y - 0.08), (x, y - 0.5)], origin, rotation)
    ax.add_line(plt.Line2D([term_top[0][0], term_top[1][0]], [term_top[0][1], term_top[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([term_bot[0][0], term_bot[1][0]], [term_bot[0][1], term_bot[1][1]], color=color, linewidth=1.5))

def draw_CC_symbol(ax, x=0, y=0, color="black", rotation=0):
    origin = (x, y)

    # Draw rotated circle
    circle = patches.Circle((x, y), 0.17, edgecolor=color, facecolor="white", linewidth=2)
    ax.add_patch(circle)

    # Arrow inside circle (from lower to upper)
    arrow_start = (x, y - 0.12)
    arrow_end = (x, y + 0.02)
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.1, head_length=0.1, width=0.01, fc=color, ec=color)

    # Vertical terminals (above and below the circle)
    line1 = rotate_points([(x, y + 0.18), (x, y + 0.5)], origin, rotation)
    line2 = rotate_points([(x, y - 0.18), (x, y - 0.5)], origin, rotation)
    ax.add_line(plt.Line2D([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color=color, linewidth=1.5))

def draw_resistor_symbol(ax, x=0, y=0, color="black", rotation=0):
    dx = 0.075
    dy = 0.02
    ystart = y + 0.15
    origin = (x, y)

    # Top and bottom vertical leads
    segments = [
        [(x, y+0.15), (x, y+0.5)],
        [(x, y-0.09), (x, y-0.5)],
    ]

    # Zigzag segments
    for _ in range(3):
        segments += [
            [(x, ystart), (x+dx, ystart-dy)],
            [(x+dx, ystart-dy), (x-dx, ystart-3*dy)],
            [(x-dx, ystart-3*dy), (x, ystart-4*dy)]
        ]
        ystart -= 4 * dy

    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=1.5))

def draw_earth_symbol(ax, x=0, y=0, color="black", rotation=0):
    origin = (x, y)
    segments = []
    # Vertical line
    segments.append([(x, y - 0.05), (x, y + 0.1)])
    # Horizontal lines
    for i in range(3):
        x1 = x - 0.03 * (i + 1)
        x2 = x + 0.03 * (i + 1)
        y_level = y - 3*0.05 + 0.05 * i
        segments.append([(x1, y_level), (x2, y_level)])
    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=2))

def draw_pos_terminal_symbol(ax, x=0, y=0, color="black"):
    circle = patches.Circle((x, y), 0.04, edgecolor=color,facecolor="white",linewidth=2, fill=True)
    ax.add_patch(circle)


