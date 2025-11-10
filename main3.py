import time, math
from pathlib import Path
import numpy as np
import cv2
import torch
import joblib
from PIL import Image
import face_alignment  # pip install face-alignment
import json

# ======== CONFIG ========
PI_IP = "192.168.43.231"                                # Rpi IP address
URL = f"http://{PI_IP}:5000/video_feed"                 # URL for video feed

CKPT_PATH = "cnn_model/best.pt"                         # CNN checkpoint
STATS_MODEL_PATH = "stats_model/rf_stats_model.joblib"  # joblib bundle
FRAME_SAMPLE_EVERY = 2                                  # infer every Nth frame
CLASSES = ["drowsy", "non_drowsy"]                      # keep consistent across both models

# ---------- MQTT CONFIG ----------
ENABLE_MQTT = True
MQTT_BROKER = "192.168.43.93"                           # IP address of this computer
MQTT_PORT   = 1883
MQTT_TOPIC  = "drowsy/predictions/pi-01"                # Topic name

# ---------- MQTT SETUP ----------
if ENABLE_MQTT:
    import paho.mqtt.client as mqtt
    mqtt_client = mqtt.Client(client_id="inference-pc")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
else:
    mqtt_client = None

# ---------- CNN ----------
def build_cnn_from_ckpt(ckpt_path, device):
    from torchvision import models
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "resnet18")
    img_size = ckpt.get("img_size", 224)
    class_names = ckpt.get("class_names", CLASSES)

    model_ctor = {"resnet18": models.resnet18,
                  "resnet34": models.resnet34,
                  "resnet50": models.resnet50}.get(arch)
    if model_ctor is None:
        raise ValueError(f"Unsupported arch in ckpt: {arch}")

    m = model_ctor(weights=None)
    in_f = m.fc.in_features
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_f, len(class_names)))
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(device)
    return m, img_size, class_names

# ---------- Resizing ----------
def center_crop_resize(pil_img, out_size):
    w, h = pil_img.size
    s = min(w, h)
    L, T = (w - s) // 2, (h - s) // 2
    return pil_img.crop((L, T, L + s, T + s)).resize((out_size, out_size), Image.Resampling.LANCZOS)


# ---------- Predict CNN via BGR(OpenCV preferred way) ----------
def predict_cnn_from_bgr(bgr, model, img_size, class_names, device):
    from torchvision import transforms as T
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = center_crop_resize(Image.fromarray(rgb), img_size)
    x = T.Compose([T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])])(pil).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        p = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    probs = {class_names[i]: float(p[i]) for i in range(len(class_names))}
    lab = max(probs, key=probs.get)
    return lab, probs[lab], probs


# ---------- Stats (FAN / face-alignment) ----------
# Handle enum name across versions
LT = getattr(face_alignment.LandmarksType, "TWO_D", face_alignment.LandmarksType.TWO_D)
_fa = face_alignment.FaceAlignment(
    LT,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    flip_input=False
)

# iBUG 68 indices
LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
NOSE_TIP  = 30
L_OUT, R_OUT = 36, 45

# ---------- Eucidian Distance ----------
def _eu(a, b): return float(np.linalg.norm(a - b))

# ---------- Eye aspect ratio ----------
def _ear(pts, idx):
    p1,p2,p3,p4,p5,p6 = [pts[i] for i in idx]
    return (_eu(p2,p6) + _eu(p3,p5)) / (2.0 * _eu(p1,p4) + 1e-6)

# ---------- Mouth aspect ratio ----------
def _mar(pts):
    # mouth width 48-54; vertical avg of several pairs
    horiz = _eu(pts[48], pts[54]) + 1e-6
    vertical_pairs = [(50,58), (52,56), (49,59), (51,57)]
    vert = np.mean([_eu(pts[a], pts[b]) for a,b in vertical_pairs])
    return vert / horiz

def compute_features_from_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lms_list = _fa.get_landmarks(rgb)
    if not lms_list:
        return None
    pts = np.asarray(lms_list[0], dtype=np.float32)  # (68,2)

    ear_l = _ear(pts, LEFT_EYE)
    ear_r = _ear(pts, RIGHT_EYE)
    ear   = (ear_l + ear_r) / 2.0
    mar   = _mar(pts)

    io = _eu(pts[L_OUT], pts[R_OUT]) + 1e-6
    mid = (pts[L_OUT] + pts[R_OUT]) / 2.0
    nose = pts[NOSE_TIP]
    nose_dx = float((nose[0] - mid[0]) / io)
    nose_dy = float((nose[1] - mid[1]) / io)

    dy, dx = (pts[R_OUT][1] - pts[L_OUT][1]), (pts[R_OUT][0] - pts[L_OUT][0])
    roll = math.degrees(math.atan2(float(dy), float(dx)))

    return {
        "EAR_L": float(ear_l), "EAR_R": float(ear_r), "EAR": float(ear),
        "MAR": float(mar), "NOSE_DX": nose_dx, "NOSE_DY": nose_dy, "ROLL_DEG": float(roll)
    }

# ---------- Predict Statis via BGR(OpenCV preferred way) ----------
def predict_stats_from_bgr(bgr, clf, class_names):
    feats = compute_features_from_bgr(bgr)
    if feats is None:
        # neutral if no face found
        neut = 1.0 / len(class_names)
        return "unknown", 0.0, {c: neut for c in class_names}, None
    keys = [k for k in feats.keys() if k.startswith(("EAR","MAR","NOSE_D","ROLL"))]
    x = np.array([[feats.get(k, 0.0) for k in keys]], dtype=np.float32)
    p = clf.predict_proba(x)[0]
    probs = {class_names[i]: float(p[i]) for i in range(len(class_names))}
    lab = max(probs, key=probs.get)
    return lab, probs[lab], probs, feats


# ---------- Fusion ----------
def fuse_probs(cnn_probs: dict, stats_probs: dict, classes: list, alpha=None, min_floor=0.05):
    def norm(d): 
        s = sum(d.values()) + 1e-8
        return {k: max(0.0, v) / s for k, v in d.items()}
    c = norm(cnn_probs); s = norm(stats_probs)
    if alpha is None:
        wc, ws = max(c.values()), max(s.values())
        tot = wc + ws + 1e-8
        w_c, w_s = wc / tot, ws / tot
    else:
        w_c, w_s = float(alpha), 1.0 - float(alpha)
    fused = {cls: max(min_floor, w_c * c.get(cls, 0.0) + w_s * s.get(cls, 0.0)) for cls in classes}
    Z = sum(fused.values()) + 1e-8
    fused = {k: v / Z for k, v in fused.items()}
    lab = max(fused, key=fused.get)
    return lab, fused[lab], fused, {"cnn": w_c, "stats": w_s}


# ---------- Main ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load models
    if not Path(CKPT_PATH).exists(): raise FileNotFoundError(CKPT_PATH)
    if not Path(STATS_MODEL_PATH).exists(): raise FileNotFoundError(STATS_MODEL_PATH)

    model, img_size, cnn_classes = build_cnn_from_ckpt(CKPT_PATH, device)
    stats_bundle = joblib.load(STATS_MODEL_PATH)
    stats_clf = stats_bundle["model"]
    stats_classes = stats_bundle.get("class_names", CLASSES)
    classes = CLASSES or cnn_classes

    # Open stream
    cap = cv2.VideoCapture(URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {URL}")
    print("Connected. Press ESC to quit.")

    i = 0
    last = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            i += 1
            if i % FRAME_SAMPLE_EVERY != 0:
                cv2.imshow("Pi + CNN+Stats (fused)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # CNN
            c_lab, c_conf, c_probs = predict_cnn_from_bgr(frame, model, img_size, classes, device)
            # Stats
            s_lab, s_conf, s_probs, feats = predict_stats_from_bgr(frame, stats_clf, stats_classes)
            # Fuse
            f_lab, f_conf, f_probs, w = fuse_probs(c_probs, s_probs, classes, alpha=None)

            # Overlay
            vis = frame.copy()
            cv2.putText(vis, f"FUSED: {f_lab} {f_conf:.2f}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if f_lab == "non_drowsy" else (0, 0, 255), 2)
            cv2.putText(vis, f"CNN: {c_lab} {c_conf:.2f}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(vis, f"STATS: {s_lab} {s_conf:.2f}", (16,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            now = time.time()
            if now - last > 0.2:
                print(f"FUSED {f_lab}({f_conf:.2f}) | CNN {c_lab}({c_conf:.2f}) | STATS {s_lab}({s_conf:.2f})")
                last = now

            # MQTT publish
            if mqtt_client is not None:
                payload = {
                    "ts": int(now),
                    "pi_id": "pi-01",
                    "fused": {"label": f_lab, "p": float(f_conf)},
                    "cnn":   {"label": c_lab, "p": float(c_conf)},
                    "stats": {"label": s_lab, "p": float(s_conf)}
                }
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=False)

            cv2.imshow("Pi + CNN+Stats (fused)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client is not None:
            mqtt_client.loop_stop()

if __name__ == "__main__":
    main()
