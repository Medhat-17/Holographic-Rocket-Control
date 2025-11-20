import cv2
import numpy as np
import math
import time

def ensure_alpha(img):
    """Ensure image has alpha channel."""
    if img is None:
        raise FileNotFoundError("rocket.png not found or failed to load.")
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge([b,g,r,alpha])
    return img

def rotate_image_expand(img, angle):
    """
    Rotate image while expanding canvas to contain whole rotated image.
    Preserves alpha channel.
    """
    (h, w) = img.shape[:2]
    # rotation about center
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # compute new bounds
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def overlay_transparent(bg, fg, x, y):
    """Overlay fg (with alpha) onto bg at position x,y. Crops if out of bounds."""
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    # compute overlap region
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + fw, bw)
    y2 = min(y + fh, bh)

    if x1 >= x2 or y1 >= y2:
        return bg  # no overlap

    # corresponding fg region
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    roi_bg = bg[y1:y2, x1:x2].astype(float)
    roi_fg = fg[fy1:fy2, fx1:fx2].astype(float)

    alpha = roi_fg[:, :, 3] / 255.0
    alpha = alpha[:, :, np.newaxis]

    # blend
    roi = roi_fg[:, :, :3] * alpha + roi_bg * (1 - alpha)
    bg[y1:y2, x1:x2] = roi.astype(np.uint8)
    return bg

# ---------------------
# Load rocket
# ---------------------
rocket = cv2.imread("rocket.png", cv2.IMREAD_UNCHANGED)
rocket = ensure_alpha(rocket)

# webcam
cap = cv2.VideoCapture(0)
time.sleep(0.2)

# smoothing / state
smoothed_scale = 0.8
smoothed_angle = 0.0
angle_vel = 2.0  # degrees per frame (base rotation speed)
lerp_alpha = 0.15  # smoothing factor (0..1)

# last-good fallback (if detection fails briefly)
last_good_scale = smoothed_scale
last_seen = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ---------------------
    # Color detection (red)
    # ---------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # tune these if your red looks different
    lower1 = np.array([0, 100, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 100, 70])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # denoise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # choose the two largest contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    centers = []
    for c in contours[:4]:  # consider top 4, but we'll take two largest valid ones
        area = cv2.contourArea(c)
        if area < 250:  # ignore tiny
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + cw//2
        cy = y + ch//2
        centers.append((cx, cy, area))
        cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

    # Need 2 reliable centers. If more than 2, pick top two areas.
    if len(centers) >= 2:
        # pick two largest by area
        centers = sorted(centers, key=lambda t: t[2], reverse=True)
        (x1, y1, _), (x2, y2, _) = centers[0], centers[1]

        # measured distance
        dist = math.hypot(x2 - x1, y2 - y1)

        # raw scale mapping (tweak denominator to change sensitivity)
        raw_scale = max(0.25, min(dist / 220.0, 3.0))

        # smoothing (simple lerp)
        smoothed_scale = smoothed_scale * (1 - lerp_alpha) + raw_scale * lerp_alpha

        # increase rotation speed a little when fingers move faster apart (optional)
        # compute angle of the line between fingers for a small rotation effect
        finger_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # interpolate angle towards finger_angle (gives a subtle orientation)
        smoothed_angle = smoothed_angle * (1 - lerp_alpha) + (finger_angle) * lerp_alpha

        last_good_scale = smoothed_scale
        last_seen = time.time()
    else:
        # if detection briefly fails, gently keep last_good_scale and keep rotating
        # slowly decay to a base scale
        if time.time() - last_seen < 1.0:
            smoothed_scale = smoothed_scale * 0.98 + last_good_scale * 0.02
        else:
            # if no hands for a while, shrink slightly
            smoothed_scale = smoothed_scale * 0.96 + 0.6 * 0.04

    # always rotate the rocket (add rotation velocity)
    smoothed_angle = (smoothed_angle + angle_vel * 0.5) % 360

    # ---------------------
    # Prepare hologram canvas (black)
    # ---------------------
    holo = np.zeros_like(frame)

    # Optional: subtle circular glow behind rocket
    center_x = w // 2
    center_y = h // 2
    glow = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(glow, (center_x, center_y + 50), int(120 * smoothed_scale), (30, 50, 90), -1)
    glow = cv2.GaussianBlur(glow, (0,0), sigmaX=40 * smoothed_scale, sigmaY=40 * smoothed_scale)
    holo = cv2.add(holo, glow)

    # ---------------------
    # Scale & rotate rocket (expanded)
    # ---------------------
    rocket_scaled = cv2.resize(rocket, None, fx=smoothed_scale, fy=smoothed_scale, interpolation=cv2.INTER_AREA)
    rocket_rot = rotate_image_expand(rocket_scaled, smoothed_angle)

    # Add a 'holographic' tint: clone rocket_rot and tint its color in the RGB channels, but keep alpha.
    tint = rocket_rot.copy()
    # multiply RGB by factors for cyan-ish hologram
    tint[:, :, 0] = np.clip(tint[:, :, 0].astype(int) * 1.2, 0, 255)  # B
    tint[:, :, 1] = np.clip(tint[:, :, 1].astype(int) * 1.5, 0, 255)  # G
    tint[:, :, 2] = np.clip(tint[:, :, 2].astype(int) * 0.9, 0, 255)  # R
    # soften alpha a bit for hologram feel
    if tint.shape[2] == 4:
        alpha_channel = tint[:, :, 3].astype(np.float32)
        alpha_channel = np.clip(alpha_channel * 0.85, 0, 255)
        tint[:, :, 3] = alpha_channel.astype(np.uint8)

    # place rocket center on screen center
    rx = center_x - (tint.shape[1] // 2)
    ry = center_y - (tint.shape[0] // 2) - 20  # slightly upwards

    # overlay onto holo canvas
    holo = overlay_transparent(holo, tint, rx, ry)

    # Optional: draw faint grid lines on holo for effect
    step = 40
    for gx in range(0, w, step):
        cv2.line(holo, (gx, 0), (gx, h), (10, 10, 10), 1)
    for gy in range(0, h, step):
        cv2.line(holo, (0, gy), (w, gy), (10, 10, 10), 1)
    output = holo

    cv2.imshow("Holographic Rocket - stable", output)
    cv2.imshow("mask (debug)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
