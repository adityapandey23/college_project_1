# pc_step2_cnn_infer.py
# Pull MJPEG from Pi, run your CNN every Nth frame, overlay result.
# pip install opencv-python torch torchvision pillow

import time
from pathlib import Path
import cv2
import torch
from PIL import Image

# ---------- CONFIG ----------
PI_IP = "192.168.43.231"                 # <-- change to your Pi's IP
URL = f"http://{PI_IP}:5000/video_feed"

CKPT_PATH = "cnn_model/best.pt"                # <-- your saved CNN checkpoint
FRAME_SAMPLE_EVERY = 2                         # run inference every Nth frame
USE_FACE_DET = False                           # True if you already have process_img() to crop faces
# ----------------------------

def build_cnn_from_ckpt(ckpt_path, device):
    from torchvision import models
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "resnet18")
    img_size = ckpt.get("img_size", 224)
    class_names = ckpt.get("class_names", ["drowsy","non_drowsy"])

    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet34":
        m = models.resnet34(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    in_f = m.fc.in_features
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_f, len(class_names)))
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(device)
    return m, img_size, class_names

def center_crop_resize(pil_img, out_size):
    w, h = pil_img.size; s = min(w, h)
    left, top = (w - s)//2, (h - s)//2
    return pil_img.crop((left, top, left + s, top + s)).resize((out_size, out_size), Image.Resampling.LANCZOS)

def predict_cnn_from_bgr(bgr, model, img_size, class_names, device):
    from torchvision import transforms
    # If you already have a process_img(path_or_PIL) that face-crops, plug it here.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = center_crop_resize(pil, img_size)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    out = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    label = max(out, key=out.get)
    return label, out[label], out  # label, confidence, probs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model, img_size, class_names = build_cnn_from_ckpt(CKPT_PATH, device)
    print("Loaded CNN:", {"arch": type(model).__name__, "img_size": img_size, "classes": class_names})

    cap = cv2.VideoCapture(URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {URL}")

    print("Connected. Press ESC to quit.")
    i = 0
    last_log = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        i += 1
        if i % FRAME_SAMPLE_EVERY != 0:
            # still show the stream, only skip inference
            cv2.imshow("Pi Stream + CNN", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        label, conf, _ = predict_cnn_from_bgr(frame, model, img_size, class_names, device)

        # overlay
        vis = frame.copy()
        cv2.putText(vis, f"{label}  {conf:.2f}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if label=="non_drowsy" else (0,0,255), 2)

        # light logging (~5 Hz)
        now = time.time()
        if now - last_log > 0.2:
            print(f"CNN: {label} ({conf:.2f})")
            last_log = now

        cv2.imshow("Pi Stream + CNN", vis)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
