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

def generate_sierpinski_curve(order, area, step_size=1.0, angle_deg=45):
    """
    Generate the Sierpinski space-filling curve of a given order.
    
    Parameters:
    order (int): The L-system order.
    step_size (float): The length of each forward move.
    angle_deg (float): The turning angle in degrees.
    
    Returns:
    np.ndarray: An array of (x, y) points.
    """
    axiom = 'F--XF--F--XF'
    rules = {'X': 'XF+F+XF--F--XF+F+X'}
    instructions = generate_lsystem(axiom, rules, order)
    
    angle_rad = math.radians(angle_deg)
    heading = -math.radians(45)
    x, y = 0.0, 0.0
    points = [(x, y)]
    
    for cmd in instructions:
        if cmd == 'F':
            x += step_size * math.cos(heading)
            y += step_size * math.sin(heading)
            points.append((x, y))
        elif cmd == '+':
            heading -= angle_rad
        elif cmd == '-':
            heading += angle_rad
        # Ignore 'X'
    
    #return np.array(points)
    points = np.array(points)

    # Normalize curve length to 1
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = np.sum(segment_lengths)
    points_normalized = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * math.sqrt(area)
    # points_normalized = points # Comment this out if you want to normalize the curve length

    print('Length of Sierpinski curve: ', np.sum(np.linalg.norm(np.diff(points_normalized, axis=0), axis=1)))

    return points_normalized

# # Parameters
# order = 6  # You can adjust this, from 8 onwards it will look just like a filled squared, curve not visible
# step_size = 5  # Length of each step in the curve

# # Generate curve
# curve = generate_sierpinski_curve(order, step_size, area=14400)

# # Plotting
# plt.figure(figsize=(8, 8))
# plt.plot(curve[:, 0], curve[:, 1], linewidth=0.8)
# plt.title(f"Sierpinski Space-Filling Curve (Order {order})")
# plt.axis('equal')
# plt.axis('off')
# plt.show()

# # Save the curve to a CSV file
# np.savetxt(f'Sierpinski_curve_{order}.csv', curve, delimiter=',', header='x,y', comments='')
