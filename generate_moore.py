import numpy as np
import matplotlib.pyplot as plt
import math

def generate_lsystem(axiom, rules, iterations):
    result = axiom
    for _ in range(iterations):
        next_result = ''
        for char in result:
            next_result += rules.get(char, char)
        result = next_result
    return result

def generate_moore_curve(order, area, step_size=1.0, angle_deg=90):
    """
    Generate the Moore space-filling curve of a given order.
    
    Parameters:
    - order (int): The L-system order.
    - step_size (float): The length of each forward move.
    - angle_deg (float): The turning angle in degrees.
    
    Returns:
    - np.ndarray: An array of (x, y) points.
    """
    axiom = 'LFL+F+LFL'
    rules = {
        'L': '-RF+LFL+FR-',
        'R': '+LF-RFR-FL+'
    }
    instructions = generate_lsystem(axiom, rules, order)

    angle_rad = math.radians(angle_deg)
    heading = 0.0  # Facing right
    x, y = 0.0, 0.0
    points = [(x, y)]

    for cmd in instructions:
        if cmd == 'F':
            x += step_size * math.cos(heading)
            y += step_size * math.sin(heading)
            points.append((x, y))
        elif cmd == '+':
            heading += angle_rad
        elif cmd == '-':
            heading -= angle_rad
        # L and R are ignored in drawing
    points = np.array(points)

    # scale to area given
    points_normalized = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * math.sqrt(area)

    print('Length of Moore curve: ', np.sum(np.linalg.norm(np.diff(points_normalized, axis=0), axis=1)))
    return points_normalized

# # Parameters
# order = 5
# step_size = 5

# # Generate and plot the curve
# curve = generate_moore_curve(order, step_size)

# plt.figure(figsize=(8, 8))
# plt.plot(curve[:, 0], curve[:, 1], linewidth=0.8)
# plt.title(f"Moore Space-Filling Curve (Order {order})")
# plt.axis('equal')
# plt.axis('off')
# plt.show()

# # Optional: Save to CSV
# np.savetxt(f'Moore_curve_{order}.csv', curve, delimiter=',', header='x,y', comments='')
