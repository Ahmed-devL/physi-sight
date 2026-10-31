"""Render deterministic demos locally: .venv/bin/python scripts/generate_demo_videos.py."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp


OUTPUT = Path(__file__).resolve().parents[1] / "backend" / "demo_clips"
FPS, FRAMES, SIZE = 30, 120, 512
TIMES = np.arange(FRAMES) / FPS


def integrate(rhs, initial):
    solution = solve_ivp(rhs, (0, TIMES[-1]), initial, t_eval=TIMES,
                         rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def bouncing_heights(g=9.81, e=0.82, height=2.4):
    """Integrate downward floor events, reset velocity, then restart."""
    def floor(_t, state):
        return state[0]
    floor.terminal = True
    floor.direction = -1
    start, state = 0.0, [height, 0.0]
    heights = np.empty(FRAMES)
    while start < TIMES[-1]:
        solution = solve_ivp(lambda _t, z: [z[1], -g], (start, TIMES[-1]),
                             state, events=floor, dense_output=True,
                             rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError(solution.message)
        end = solution.t[-1]
        selected = (TIMES >= start) & (TIMES <= end)
        heights[selected] = solution.sol(TIMES[selected])[0]
        if not len(solution.t_events[0]):
            break
        start = end
        state = [0.0, -e * solution.y_events[0][0, 1]]
    return heights


def studio():
    yy, xx = np.mgrid[:SIZE, :SIZE]
    gray = np.clip(240 - 12 * ((xx-230)**2 + (yy-180)**2) / SIZE**2,
                   0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def shaded_sphere(image, center, radius=17, color=(48, 54, 190)):
    """Shade a sphere in BGR with a soft upper-left highlight."""
    cx, cy = center
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dx, dy = (xx-cx)/radius, (yy-cy)/radius
    inside = dx*dx + dy*dy <= 1
    nz = np.sqrt(np.maximum(0, 1-dx*dx-dy*dy))
    light = np.clip(-0.35*dx - 0.45*dy + 0.82*nz, 0, 1)
    rgb = np.asarray(color)[None, None, :] * (0.32+0.68*light[:, :, None])
    rgb += 55 * light[:, :, None]**24
    image[inside] = np.clip(rgb[inside], 0, 255).astype(np.uint8)
    cv2.circle(image, (round(cx), round(cy)), radius, (44, 53, 85), 1, cv2.LINE_AA)


def render(path, centers, *, pendulum=False, floor=False):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create MP4: {path}")
    try:
        for cx, cy in centers:
            frame = studio()
            if floor:
                cv2.line(frame, (0, 470), (511, 470), (155, 160, 165), 2)
                cv2.ellipse(frame, (round(cx), 472), (24, 4), 0, 0, 360,
                            (205, 205, 205), -1, cv2.LINE_AA)
            if pendulum:
                cv2.line(frame, (216, 40), (296, 40), (130, 135, 140), 5, cv2.LINE_AA)
                cv2.circle(frame, (256, 48), 5, (75, 85, 95), -1, cv2.LINE_AA)
                cv2.line(frame, (256, 48), (round(cx), round(cy)), (100, 112, 125), 2, cv2.LINE_AA)
                cv2.circle(frame, (round(cx), round(cy)), 20, (65, 145, 184), -1, cv2.LINE_AA)
            shaded_sphere(frame, (cx, cy))
            writer.write(frame)
    finally:
        writer.release()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    g, length, gamma, theta0 = 9.81, 0.50, 0.15, 0.9
    theta = integrate(lambda _t, z: [z[1], -gamma*z[1] - g/length*np.sin(z[0])],
                      [theta0, 0.0])[0]
    pendulum = np.column_stack((256+280*np.sin(theta), 48+280*np.cos(theta)))
    # Downward distance, acceleration = g - k*v*abs(v); k has units 1/m.
    k = 0.65
    distance = integrate(lambda _t, z: [z[1], g-k*z[1]*abs(z[1])], [0, 0])[0]
    drag_scale = 26.0
    drag = np.column_stack((np.full(FRAMES, 256), 45+drag_scale*distance))
    heights = bouncing_heights(g=g)
    bounce = np.column_stack((np.full(FRAMES, 256), 453-160*heights))
    render(OUTPUT / "damped_pendulum.mp4", pendulum, pendulum=True)
    render(OUTPUT / "falling_drag_sphere.mp4", drag)
    render(OUTPUT / "bouncing_ball.mp4", bounce, floor=True)
    manifest = {
        "fps": FPS, "frames": FRAMES, "duration_seconds": FRAMES/FPS,
        "width": SIZE, "height": SIZE,
        "clips": {
            "damped_pendulum.mp4": {
                "L_meters": length, "gamma": gamma, "g": g,
                "omega0_squared": g/length, "theta0": theta0, "angular_velocity0": 0,
                "pivot_screen_pixels": [256, 48], "L_pixels": 280,
                "equation": "theta_ddot = -gamma*theta_dot - (g/L)*sin(theta)",
            },
            "falling_drag_sphere.mp4": {
                "g": g, "quadratic_drag_per_meter": k, "distance0": 0, "velocity0": 0,
                "pixels_per_meter": drag_scale, "origin_screen_y": 45,
                "equation": "distance_ddot = g - k*velocity*abs(velocity)",
            },
            "bouncing_ball.mp4": {
                "g": g, "e": 0.82, "initial_height_meters": 2.4, "velocity0": 0,
                "pixels_per_meter": 160, "floor_center_screen_y": 453,
                "equation": "height_ddot = -g; at height=0, v_after=-e*v_before",
            },
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered three {FRAMES}-frame clips and manifest in {OUTPUT}")


if __name__ == "__main__":
    main()
