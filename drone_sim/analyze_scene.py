"""analyze_scene.py - Analyze Blocks scene obstacle layout (no AirSim needed)"""
import numpy as np

# From query_scene_objects.py output, blocks at flight path
# Each block: scale=10x10x5 => size 10x10x5m, center at position
# Half-extents: 5x5x2.5m
blocks = [
    # Row 1: X=28.1 center
    ('R1', 28.1, -16.5, -1.5), ('R1', 28.1, -16.5, -6.5),
    ('R1', 28.1, -6.5, -1.5),  ('R1', 28.1, -6.5, -6.5),
    ('R1', 28.1, 3.5, -1.5),   ('R1', 28.1, 3.5, -6.5),
    ('R1', 28.1, 13.5, -1.5),  ('R1', 28.1, 13.5, -6.5),
    # Row 2: X=38.1 center
    ('R2', 38.1, -16.5, -1.5), ('R2', 38.1, -16.5, -6.5),
    ('R2', 38.1, 13.5, -1.5),  ('R2', 38.1, 13.5, -6.5),
    # Also at X=38.1 but no blocks at Y=-6.5, 3.5 => GAP in middle
    # Row 3: X=48.1 center
    ('R3', 48.1, -16.5, -1.5), ('R3', 48.1, -16.5, -6.5),
    ('R3', 48.1, 13.5, -1.5),  ('R3', 48.1, 13.5, -6.5),
    ('R3', 48.1, 3.5, -6.5),   # only at Z=-6.5, not at Z=-1.5
    # Row 4: X=58.1 center
    ('R4', 58.1, -16.5, -1.5), ('R4', 58.1, -16.5, -6.5),
    ('R4', 58.1, -6.5, -1.5),  ('R4', 58.1, -6.5, -6.5),
    ('R4', 58.1, 3.5, -1.5),   ('R4', 58.1, 3.5, -6.5),
    ('R4', 58.1, 13.5, -1.5),  ('R4', 58.1, 13.5, -6.5),
]

flight_z = -3.0
print("=== Blocks scene layout at flight height Z=-3.0 ===")
print("(each block: 10x10x5m, half-extent 5x5x2.5m)")
print()

for row_name in ['R1', 'R2', 'R3', 'R4']:
    row_blocks = [b for b in blocks if b[0] == row_name]
    active = []
    for name, cx, cy, cz in row_blocks:
        z_min, z_max = cz - 2.5, cz + 2.5
        if z_min <= flight_z <= z_max:
            y_min, y_max = cy - 5.0, cy + 5.0
            x_min, x_max = cx - 5.0, cx + 5.0
            active.append((x_min, x_max, y_min, y_max, cz))

    if active:
        x_range = (min(a[0] for a in active), max(a[1] for a in active))
        y_ranges = sorted(set((a[2], a[3]) for a in active))
        print(f"{row_name}: X=[{x_range[0]:.1f}, {x_range[1]:.1f}]")
        merged = []
        for yr in y_ranges:
            if merged and yr[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], yr[1]))
            else:
                merged.append(list(yr))
        for m in merged:
            print(f"  Wall Y=[{m[0]:.1f}, {m[1]:.1f}]")
        if len(merged) > 1:
            for i in range(len(merged) - 1):
                gap_start = merged[i][1]
                gap_end = merged[i + 1][0]
                print(f"  >>> GAP Y=[{gap_start:.1f}, {gap_end:.1f}] width={gap_end - gap_start:.1f}m")
        else:
            print(f"  No gaps (solid wall)")
        print()
    else:
        print(f"{row_name}: No blocks at Z={flight_z}")
        print()

# Goal analysis
print("=== Goal point analysis ===")
for gx in [35.0, 45.0, 55.0]:
    goal = np.array([gx, 0.0, flight_z])
    in_block = False
    for name, cx, cy, cz in blocks:
        x_min, x_max = cx - 5, cx + 5
        y_min, y_max = cy - 5, cy + 5
        z_min, z_max = cz - 2.5, cz + 2.5
        if x_min <= goal[0] <= x_max and y_min <= goal[1] <= y_max and z_min <= goal[2] <= z_max:
            in_block = True
            print(f"Goal ({gx}, 0, {flight_z}): INSIDE block {name} at ({cx},{cy},{cz})")
            print(f"  Block X=[{x_min},{x_max}] Y=[{y_min},{y_max}] Z=[{z_min},{z_max}]")
    if not in_block:
        min_dist = float('inf')
        for name, cx, cy, cz in blocks:
            z_min, z_max = cz - 2.5, cz + 2.5
            if z_min <= flight_z <= z_max:
                dist = np.sqrt((goal[0] - cx) ** 2 + (goal[1] - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (name, cx, cy, cz)
        print(f"Goal ({gx}, 0, {flight_z}): SAFE, nearest={nearest[0]} at ({nearest[1]},{nearest[2]}), dist={min_dist:.1f}m")

# Recommended goal
print()
print("=== Recommendation ===")
print("R1 is solid wall Y=[-21.5, 18.5] - must go around edge")
print("R2 has gap in middle (no blocks at Y=[-6.5, 3.5] at Z=-1.5)")
print("R2 gap: Y=[-11.5, 8.5] (20m wide)")
print()
print("Strategy: go around R1 edge (Y>18.5 or Y<-21.5), then through R2 gap")
print("Goal should be PAST all rows, in clear space")
