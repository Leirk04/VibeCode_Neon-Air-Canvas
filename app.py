"""
Air Canvas — Gesture-based drawing app using MediaPipe Hand Landmarker + OpenCV.

Gestures:
  - INDEX finger only     → DRAW mode (neon brush strokes with bloom)
  - INDEX + MIDDLE finger → ERASE mode (circular eraser)
  - OPEN PALM (all 5)     → CLEAR entire canvas
  - FIST then reopen      → CYCLE brush color
  - THUMB + INDEX pinch    → ADJUST brush size

Controls:
  Q  — Quit
  S  — Save canvas as PNG
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import time
import random
import math
import os

# ──────────────────────────────────────────────
#  Particle class for sparkle trail effect
# ──────────────────────────────────────────────
class Particle:
    """Small fading dot that trails behind the brush cursor."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = random.randint(2, 6)
        self.life = 255                       # fades from 255 → 0
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 15

    def draw(self, img):
        if self.life > 0:
            center = (int(self.x), int(self.y))
            r = max(1, int(self.radius * (self.life / 255.0)))
            alpha = self.life / 255.0
            col = tuple(int(c * alpha) for c in self.color)
            cv2.circle(img, center, r, col, -1)


# ──────────────────────────────────────────────
#  Finger-state helpers
# ──────────────────────────────────────────────
def get_finger_states(landmarks, w, h):
    """
    Return a list of 5 booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended.
    landmarks: list of NormalizedLandmark from MediaPipe HandLandmarker.
    """
    lm = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    fingers = []

    # Thumb — compare tip-to-pinky-base distance vs IP-to-pinky-base distance
    dist_tip  = math.hypot(lm[4][0] - lm[17][0], lm[4][1] - lm[17][1])
    dist_ip   = math.hypot(lm[3][0] - lm[17][0], lm[3][1] - lm[17][1])
    fingers.append(dist_tip > dist_ip)

    # Index, Middle, Ring, Pinky — tip above PIP joint (y decreases upward)
    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip_id][1] < lm[pip_id][1])

    return fingers, lm


# ──────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────
def main():
    # ── Camera setup ──────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── MediaPipe Hand Landmarker (Tasks API) ─
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # ── Drawing state ─────────────────────────
    canvas = None                              # allocated on first frame

    # Neon Color Palette (BGR)
    colors = [
        (200, 0, 255),   # Hot Pink
        (255, 255, 0),   # Cyan
        (0, 255, 0),     # Lime
        (0, 165, 255),   # Orange
        (255, 0, 127),   # Violet
    ]
    color_idx   = 0
    draw_color  = colors[color_idx]

    prev_x, prev_y = 0, 0
    brush_size      = 8
    eraser_size     = 40
    current_mode    = "STANDBY"
    fist_detected   = False

    particles = []
    pTime     = 0
    frame_ts  = 0                              # monotonic timestamp for Tasks API

    print("+======================================+")
    print("|   Air Canvas - Gesture Drawing App   |")
    print("|   Q = Quit  |  S = Save Canvas       |")
    print("+======================================+")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)             # mirror for natural interaction

        if canvas is None:
            canvas = np.zeros_like(frame)

        h, w, _ = frame.shape

        # ── Hand detection ────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33                         # ~30 fps increment (ms)
        result = landmarker.detect_for_video(mp_image, frame_ts)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            fingers, lm = get_finger_states(landmarks, w, h)

            x1, y1 = lm[8]                    # index fingertip
            up_count = sum(fingers[1:])        # non-thumb raised fingers

            # ── 1. FIST → color cycle trigger ─
            if up_count == 0 and not fingers[0]:
                if not fist_detected:
                    fist_detected = True
                current_mode = "CYCLE COLOR"
                prev_x, prev_y = 0, 0

            else:
                if fist_detected:
                    color_idx  = (color_idx + 1) % len(colors)
                    draw_color = colors[color_idx]
                    fist_detected = False

                # ── 2. CLEAR ALL (open palm — 4-5 fingers) ─
                if up_count >= 4:
                    canvas = np.zeros_like(frame)
                    current_mode = "CLEAR ALL"
                    prev_x, prev_y = 0, 0

                # ── 3. ERASE (index + middle only) ─
                elif fingers[1] and fingers[2] and up_count == 2:
                    current_mode = "ERASE"
                    # Show eraser cursor
                    cv2.circle(frame, (x1, y1), eraser_size, (255, 255, 255), 2)
                    # Wipe strokes
                    cv2.circle(canvas, (x1, y1), eraser_size, (0, 0, 0), -1)
                    prev_x, prev_y = 0, 0

                # ── 4. DRAW (index only) ─
                elif fingers[1] and up_count == 1:
                    current_mode = "DRAW"
                    # Cursor dot on live feed
                    cv2.circle(frame, (x1, y1), brush_size, draw_color, cv2.FILLED)

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1

                    # Smooth interpolated line on canvas
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, brush_size)

                    # Particle sparkle trail
                    for _ in range(3):
                        particles.append(Particle(x1, y1, draw_color))

                    prev_x, prev_y = x1, y1

                # ── 5. ADJUST BRUSH (thumb + index pinch) ─
                elif fingers[0] and fingers[1] and up_count == 1:
                    current_mode = "ADJUST BRUSH"
                    tx, ty = lm[4]             # thumb tip
                    length = math.hypot(x1 - tx, y1 - ty)
                    brush_size = max(2, min(50, int(length / 5)))

                    cx, cy = (x1 + tx) // 2, (y1 + ty) // 2
                    cv2.circle(frame, (cx, cy), brush_size, draw_color, cv2.FILLED)
                    cv2.line(frame, (x1, y1), (tx, ty), (255, 255, 255), 2)
                    prev_x, prev_y = 0, 0

                else:
                    prev_x, prev_y = 0, 0
                    if current_mode not in ("CLEAR ALL", "CYCLE COLOR"):
                        current_mode = "STANDBY"

            # Translucent fingertip follower
            if current_mode not in ("ERASE", "ADJUST BRUSH"):
                cv2.circle(frame, (x1, y1), 8, (255, 255, 255),
                           max(1, brush_size // 3))
        else:
            prev_x, prev_y = 0, 0

        # ── Particle update & render ──────────
        for p in particles:
            p.update()
            p.draw(frame)
        particles = [p for p in particles if p.life > 0]

        # ── Neon bloom on canvas ──────────────
        blur_canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
        neon_canvas = cv2.add(canvas, blur_canvas)

        # Composite onto webcam frame (additive blend)
        frame = cv2.add(frame, neon_canvas)

        # ── HUD (dark glassmorphism panel) ────
        cTime = time.time()
        fps   = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        hx, hy, hw, hh = 20, 20, 280, 140
        roi = frame[hy:hy+hh, hx:hx+hw]
        if roi.size > 0:
            # Frosted glass effect
            glass = cv2.GaussianBlur(roi, (45, 45), 0)
            dark  = np.zeros_like(glass)
            glass = cv2.addWeighted(glass, 0.25, dark, 0.75, 0)
            frame[hy:hy+hh, hx:hx+hw] = glass

            # Border
            cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (255, 255, 255), 1)

            # Mode badge
            mode_icon = {
                "DRAW": "DRAW",
                "ERASE": "ERASE",
                "CLEAR ALL": "CLEAR ALL",
                "CYCLE COLOR": "COLOR SWITCH",
                "ADJUST BRUSH": "BRUSH SIZE",
                "STANDBY": "STANDBY",
            }
            mode_colors = {
                "DRAW":         (0, 255, 100),
                "ERASE":        (80, 80, 255),
                "CLEAR ALL":    (0, 200, 255),
                "CYCLE COLOR":  (255, 100, 255),
                "ADJUST BRUSH": (255, 200, 0),
                "STANDBY":      (180, 180, 180),
            }
            label = mode_icon.get(current_mode, current_mode)
            col   = mode_colors.get(current_mode, (200, 200, 200))
            cv2.putText(frame, "MODE:", (hx+15, hy+30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, label, (hx+90, hy+30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 2)

            # FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (hx+15, hy+65),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

            # Brush size bar
            cv2.putText(frame, f"SIZE: {brush_size}px", (hx+15, hy+95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            bar_x = hx + 150
            bar_w = int((brush_size / 50) * 100)
            cv2.rectangle(frame, (bar_x, hy+82), (bar_x+100, hy+98),
                          (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, hy+82), (bar_x+bar_w, hy+98),
                          draw_color, -1)

            # Color swatch
            cv2.putText(frame, "COLOR:", (hx+15, hy+125),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            swatch_x = hx + 110
            for i, c in enumerate(colors):
                cx = swatch_x + i * 30
                cy = hy + 118
                r  = 10 if i == color_idx else 7
                cv2.circle(frame, (cx, cy), r, c, -1)
                if i == color_idx:
                    cv2.circle(frame, (cx, cy), r + 3, (255, 255, 255), 2)

        # ── Display ──────────────────────────
        cv2.imshow("Air Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"canvas_export_{int(time.time())}.png"
            cv2.imwrite(filename, neon_canvas)
            print(f"Canvas saved as {filename}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
