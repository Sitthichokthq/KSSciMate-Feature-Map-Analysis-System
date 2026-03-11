"""
Feature Map Analysis System — Web Version
โดย วิทชภณ พวงแก้ว
Flask + OpenCV Backend
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import base64
import math
import io
import os
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# ─── In-memory session state (single-user dev mode) ───────────────────────────
SESSION = {
    "original_cv": None,
    "convoluted_cv": None,
    "detected_circle": None,
    "contours_data": None,
    "img_shape": None,
}


# ─────────────────────────── HELPERS ──────────────────────────────────────────
def fit_circle_algebraic(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 3:
        raise ValueError("Need ≥ 3 points")
    mx, my = xs.mean(), ys.mean()
    u, v = xs - mx, ys - my
    Suu = (u**2).sum(); Svv = (v**2).sum(); Suv = (u*v).sum()
    Suuu = (u**3).sum(); Svvv = (v**3).sum()
    Suuv = (u**2*v).sum(); Suvv = (u*v**2).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5*(Suuu+Suvv), 0.5*(Svvv+Suuv)])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = 0.0, 0.0
    cx = uc + mx; cy = vc + my
    r = float(np.sqrt(uc**2 + vc**2 + (Suu+Svv)/n))
    return cx, cy, r


def cv_to_b64(img_array, is_gray=True):
    """Convert numpy array to base64 PNG string."""
    if is_gray:
        pil = Image.fromarray(img_array.astype(np.uint8), mode='L')
    else:
        pil = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def apply_filter_cv(gray, mode):
    if mode == "Cross-Correlation":
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        return cv2.filter2D(gray, -1, kernel)
    elif mode == "Sobel Edge":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sx, sy).astype(np.uint8)
    elif mode == "Laplacian":
        return cv2.Laplacian(gray, cv2.CV_8U)
    else:  # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(gray, -1, kernel)


# ─────────────────────────── ROUTES ───────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files['file']
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Cannot decode image"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    SESSION["original_cv"] = rgb
    SESSION["detected_circle"] = None
    SESSION["contours_data"] = None

    mode = request.form.get("filter", "Cross-Correlation")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = apply_filter_cv(gray, mode)
    SESSION["convoluted_cv"] = filtered
    SESSION["img_shape"] = gray.shape

    h, w = gray.shape
    return jsonify({
        "original_b64":  cv_to_b64(rgb, is_gray=False),
        "filtered_b64":  cv_to_b64(filtered, is_gray=True),
        "width": w, "height": h,
        "filter": mode
    })


@app.route('/api/filter', methods=['POST'])
def change_filter():
    if SESSION["original_cv"] is None:
        return jsonify({"error": "No image loaded"}), 400
    mode = request.json.get("filter", "Cross-Correlation")
    gray = cv2.cvtColor(SESSION["original_cv"], cv2.COLOR_RGB2GRAY)
    filtered = apply_filter_cv(gray, mode)
    SESSION["convoluted_cv"] = filtered
    h, w = filtered.shape
    return jsonify({
        "filtered_b64": cv_to_b64(filtered, is_gray=True),
        "width": w, "height": h, "filter": mode
    })


@app.route('/api/detect', methods=['POST'])
def detect_circles():
    if SESSION["original_cv"] is None:
        return jsonify({"error": "No image loaded"}), 400

    body = request.json or {}
    lo = int(body.get("canny_lo", 20))
    hi = int(body.get("canny_hi", 80))

    gray = cv2.cvtColor(SESSION["original_cv"], cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    h_img, w_img = gray.shape
    min_r = max(10, min(h_img, w_img) // 20)
    max_r = min(h_img, w_img) // 2

    method_used = ""
    contour_pts = []
    cx2 = cy2 = r2 = 0

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, min(h_img, w_img)//4),
        param1=max(hi, 30), param2=max(15, hi//2),
        minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        cx2, cy2, r2 = map(int, max(circles, key=lambda c: c[2]))
        method_used = "HoughCircles"
        n_pts = 360
        for i in range(n_pts):
            angle = 2*math.pi*i/n_pts
            px = int(cx2 + r2*math.cos(angle))
            py = int(cy2 + r2*math.sin(angle))
            contour_pts.append([max(0,min(px,w_img-1)), max(0,min(py,h_img-1))])
    else:
        edges = cv2.Canny(blurred, lo, hi)
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_m, iterations=4)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return jsonify({"error": "No circle found. Try adjusting thresholds."}), 404
        biggest = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        (fcx, fcy), fr = cv2.minEnclosingCircle(hull)
        cx2, cy2, r2 = int(fcx), int(fcy), int(fr)
        method_used = "ConvexHull"
        contour_pts = [[int(p[0][0]), int(p[0][1])] for p in hull]

    SESSION["detected_circle"] = (cx2, cy2, r2)
    SESSION["contours_data"] = contour_pts

    area = math.pi * r2 * r2
    peri = 2 * math.pi * r2

    return jsonify({
        "method": method_used,
        "cx": cx2, "cy": cy2, "r": r2,
        "area": round(area, 1),
        "perimeter": round(peri, 1),
        "contour_pts": contour_pts,
        "n_points": len(contour_pts)
    })


@app.route('/api/roi_stats', methods=['POST'])
def roi_stats():
    if SESSION["convoluted_cv"] is None:
        return jsonify({"error": "No image"}), 400
    body = request.json
    x1, y1, x2, y2 = body['x1'], body['y1'], body['x2'], body['y2']
    h_img, w_img = SESSION["convoluted_cv"].shape
    x1 = max(0, min(x1, w_img-1)); x2 = max(0, min(x2, w_img-1))
    y1 = max(0, min(y1, h_img-1)); y2 = max(0, min(y2, h_img-1))
    crop = SESSION["convoluted_cv"][y1:y2, x1:x2]
    if crop.size == 0:
        return jsonify({"error": "Empty ROI"}), 400
    return jsonify({
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": x2-x1, "height": y2-y1,
        "area": (x2-x1)*(y2-y1),
        "mean": round(float(np.mean(crop)), 2),
        "std": round(float(np.std(crop)), 2),
        "min": int(np.min(crop)),
        "max": int(np.max(crop))
    })


@app.route('/api/pixel_info', methods=['POST'])
def pixel_info():
    body = request.json
    x, y = body['x'], body['y']
    if SESSION["convoluted_cv"] is None:
        return jsonify({"error": "No image"}), 400
    h, w = SESSION["convoluted_cv"].shape
    if not (0 <= x < w and 0 <= y < h):
        return jsonify({"error": "Out of bounds"}), 400
    val = int(SESSION["convoluted_cv"][y, x])
    orig_rgb = None
    if SESSION["original_cv"] is not None:
        orig_rgb = SESSION["original_cv"][y, x].tolist()
    return jsonify({"x": x, "y": y, "intensity": val,
                    "normalized": round(val/255, 4), "rgb": orig_rgb})


@app.route('/api/brightness_profile', methods=['GET'])
def brightness_profile():
    if not SESSION["contours_data"]:
        return jsonify({"error": "No contour"}), 400

    gray = cv2.cvtColor(SESSION["original_cv"], cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    pts = SESSION["contours_data"]
    step = max(1, len(pts)//720)
    pts_s = pts[::step]

    xs, ys, brightness = [], [], []
    for pt in pts_s:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w_img and 0 <= y < h_img:
            xs.append(x); ys.append(y)
            brightness.append(int(gray[y, x]))

    arc_len = [0.0]
    for i in range(1, len(xs)):
        arc_len.append(arc_len[-1] + math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]))

    mb = float(np.mean(brightness)); sb = float(np.std(brightness))

    # histogram
    counts, edges = np.histogram(brightness, bins=32, range=(0, 256))

    return jsonify({
        "arc_len": arc_len,
        "xs": xs, "ys": ys,
        "brightness": brightness,
        "mean": round(mb, 2), "std": round(sb, 2),
        "min": min(brightness), "max": max(brightness),
        "total_arc": round(arc_len[-1], 1),
        "hist_counts": counts.tolist(),
        "hist_edges": edges.tolist(),
        "circle": SESSION["detected_circle"]
    })


@app.route('/api/radial_profile', methods=['GET'])
def radial_profile():
    if SESSION["detected_circle"] is None:
        return jsonify({"error": "No circle detected"}), 400

    gray = cv2.cvtColor(SESSION["original_cv"], cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    cx, cy, r = SESSION["detected_circle"]
    cx, cy, r = float(cx), float(cy), float(r)

    n_angles = 360
    sample_w = int(r * 0.60)
    radii_arr = list(range(-sample_w, sample_w+1))

    all_profiles = []
    for i in range(n_angles):
        angle = 2*math.pi*i/n_angles
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        row = []
        for dr in radii_arr:
            px = cx + (r+dr)*cos_a
            py = cy + (r+dr)*sin_a
            xi, yi = int(round(px)), int(round(py))
            if 0 <= xi < w_img and 0 <= yi < h_img:
                row.append(float(gray[yi, xi]))
            else:
                row.append(None)
        all_profiles.append(row)

    profiles_np = np.array([[v if v is not None else np.nan for v in row]
                             for row in all_profiles], dtype=float)
    mean_profile = np.nanmean(profiles_np, axis=0).tolist()
    std_profile  = np.nanstd(profiles_np,  axis=0).tolist()
    mp_arr = np.array(mean_profile)
    grad_profile = np.gradient(mp_arr).tolist()
    edge_dr_idx  = int(np.argmin(np.gradient(mp_arr)))
    edge_dr      = float(radii_arr[edge_dr_idx])
    r_true        = r + edge_dr

    return jsonify({
        "radii": radii_arr,
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "grad_profile": grad_profile,
        "edge_dr": round(edge_dr, 2),
        "r_fitted": round(r, 3),
        "r_true": round(r_true, 3),
        "cx": round(cx, 2), "cy": round(cy, 2),
        "profile_map": [[v for v in row] for row in all_profiles]  # subset for heatmap
    })


@app.route('/api/sobel_fit', methods=['POST'])
def sobel_fit():
    if SESSION["original_cv"] is None:
        return jsonify({"error": "No image"}), 400

    gray = cv2.cvtColor(SESSION["original_cv"], cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 1.5)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)

    thresh_val = np.percentile(mag, 97)
    edge_mask  = (mag >= thresh_val).astype(np.uint8)

    h_img, w_img = gray.shape
    if SESSION["detected_circle"]:
        cx0, cy0, r0 = SESSION["detected_circle"]
        margin = max(30, int(r0*0.15))
        Y, X = np.ogrid[:h_img, :w_img]
        dist = np.sqrt((X-cx0)**2 + (Y-cy0)**2)
        ring = ((dist >= r0-margin) & (dist <= r0+margin)).astype(np.uint8)
        edge_mask &= ring

    ys_e, xs_e = np.where(edge_mask > 0)
    if len(xs_e) < 6:
        return jsonify({"error": "Not enough Sobel edge points. Run Auto-Detect first."}), 400

    cx, cy, r = fit_circle_algebraic(xs_e, ys_e)
    cx, cy, r = float(cx), float(cy), float(r)
    SESSION["detected_circle"] = (int(round(cx)), int(round(cy)), int(round(r)))

    n_pts = 720
    contour_pts = []
    for i in range(n_pts):
        angle = 2*math.pi*i/n_pts
        px = int(cx + r*math.cos(angle))
        py = int(cy + r*math.sin(angle))
        contour_pts.append([max(0,min(px,w_img-1)), max(0,min(py,h_img-1))])
    SESSION["contours_data"] = contour_pts

    return jsonify({
        "cx": round(cx,2), "cy": round(cy,2), "r": round(r,3),
        "diameter": round(r*2,3),
        "area": round(math.pi*r*r, 1),
        "n_edge_pts": int(len(xs_e)),
        "contour_pts": contour_pts
    })


@app.route('/api/apply_correction', methods=['POST'])
def apply_correction():
    body = request.json
    r_true = float(body['r_true'])
    cx = float(body['cx']); cy = float(body['cy'])
    if SESSION["original_cv"] is None:
        return jsonify({"error": "No image"}), 400
    h_img, w_img = SESSION["original_cv"].shape[:2]
    SESSION["detected_circle"] = (int(round(cx)), int(round(cy)), int(round(r_true)))
    n_pts = 720
    contour_pts = []
    for i in range(n_pts):
        angle = 2*math.pi*i/n_pts
        px = int(cx + r_true*math.cos(angle))
        py = int(cy + r_true*math.sin(angle))
        contour_pts.append([max(0,min(px,w_img-1)), max(0,min(py,h_img-1))])
    SESSION["contours_data"] = contour_pts
    return jsonify({"ok": True, "r_true": round(r_true,3),
                    "cx": round(cx,2), "cy": round(cy,2),
                    "contour_pts": contour_pts})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)