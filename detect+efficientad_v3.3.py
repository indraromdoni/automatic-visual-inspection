import threading
import cv2
import time
import av  # PyAV
import numpy as np
import json
import os
import datetime
import base64
from psycopg2 import pool, connect
from collections import deque
from ultralytics import YOLO, SAM, solutions
from flask import Flask, Response, render_template, request, redirect, url_for, send_file, jsonify
from flask_socketio import SocketIO, emit
from pyModbusTCP.client import ModbusClient
import io
import torch
from PIL import Image
from torchvision import transforms
from v2_efficientAD import predict
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor


# Load YOLOv8 model
model = YOLO("ingot-detectorV9n.pt")  # Ubah path sesuai model Anda
model.to('cuda')  # Hapus atau komentari jika tidak pakai GPU
#model_sam = SAM(model='sam_b.pt')
model_sam = SAM(model='mobile_sam.pt')
model_sam.to('cuda')

on_gpu = torch.cuda.is_available()
teacher = student = autoencoder = teacher_mean = teacher_std = mapnorm = None
image_size = 256
out_channels = 384

# === Preprocess image ===
effad_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# RTSP URL
#RTSP_URL = "rtsp://admin:adm12345@192.168.24.100/ISAPI/Streaming/channels/102/preview"
#RTSP_URL = "rtsp://admin:adm12345@192.168.200.109/stream1"
RTSP_URL = "rtsp://admin:P4ssword@192.168.200.110/ISAPI/Streaming/channels/101/preview"

# Shared resources
frame_lock = threading.Lock()
shared_frame = deque(maxlen=3)
annotated_frame = None
_exit = False
detection_lock = threading.Lock()
status = ["-", (0, 255, 255)]  # Status awal

ingot_buffer = deque(maxlen=5)
recent_results = deque(maxlen=8)  # Buffer (img, status) 8 produk terakhir

# Garis diagonal (bisa diubah dari web)
LINE1_P1 = [390, 480]
LINE1_P2 = [550, 120]
CONFIDENCE = 0.25
THRESHOLD_EFFAD = 1.6

app = Flask(__name__)
socketio = SocketIO(app)

getimg_thread_started = False
inference_thread_started = False
production_start = False

LINES_FILE = "lines2.json"

cropped_rect_image = None
original_frame_with_lines = None

CHARGING_NO = ''
PRODUCT_NO = ''
GROUP = ''
product_list = []

analyze_executor = ThreadPoolExecutor(max_workers=1)

def model_effad(p):
    path = f'output/{p}/trainings/'
    teacher_path = f'{path}teacher_final.pth'
    teacher_mean_path = f'{path}teacher_mean.pth'
    teacher_std_path = f'{path}teacher_std.pth'
    student_path = f'{path}student_final.pth'
    autoencoder_path = f'{path}autoencoder_final.pth'
    mapnorm_path = f'{path}mapnorm.json'

    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    autoencoder = torch.load(autoencoder_path, map_location='cpu', weights_only=False)
    teacher_mean = torch.load(teacher_mean_path, map_location='cpu')
    teacher_std = torch.load(teacher_std_path, map_location='cpu')

    teacher.eval()
    student.eval()
    autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        teacher_mean = teacher_mean.cuda()
        teacher_std = teacher_std.cuda()

    mapnorm = dict()
    with open(mapnorm_path, 'r') as f:
        mapnorm = json.load(f)
    
    return teacher, student, autoencoder, teacher_mean, teacher_std, mapnorm

def send_signal(coil, message):
    client = ModbusClient(host="192.168.200.37", port=8010, auto_open=True)
    try:
        if client.open():
            if client.write_single_coil(coil, True):
                print(f"signal sent: {message}")
            else:
                print(f"Failed to send {message}")
    except Exception as e:
        print(f"Modbus protocol error: {e}")
    finally:
        if client.open():
            client.write_single_coil(coil, False)
            client.close()
            print("Modbus connection closed")

def db_insert(threshold, score_max, score_mean, judgement, detected_image, cropped_image, original_image):
    global CHARGING_NO, PRODUCT_NO, GROUP
    def thread_insert():
        conn = connect(
            host="192.168.25.208",
            port="5436",
            database="ingot_visual_inspection",
            user="postgres",
            password="Postgre@sql1"
        )
        conn.autocommit = True
        cur = conn.cursor()
        byte_detect = cv2.imencode('.jpg', detected_image)[1].tobytes()
        byte_cropped = cv2.imencode('.jpg', cropped_image)[1].tobytes()
        byte_original = cv2.imencode('.jpg', original_image)[1].tobytes()

        cur.execute("INSERT INTO public.tbl_inspection_data(threshold, score_max, score_mean, judgement, detected_image, cropped_image, original_image, charging_no, product_no, \"group\") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", (threshold, score_max, score_mean, judgement, byte_detect, byte_cropped, byte_original, CHARGING_NO, PRODUCT_NO, GROUP))
    threading.Thread(target=thread_insert, daemon=True).start()

def yolo_seg_crop(image, mask, box, output_size=(800, 400), margin=40):
    print("segmenting with YOLO mask")
    x, y, w, h = box
    segmented = None
    cropped = None
    if mask is None or mask.size == 0:
        print("Empty mask, using SAM inference")
        #return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        # Bounding box
        #x = int(x - (w / 2))
        #y = int(y - (h / 2))

        # Crop
        #cropped = image[y:y+h, x:x+w]
        #results = model_sam.predict(cropped, points=[(w//2), (h//2)])
        results = model_sam.predict(image, points=[int(x), int(y)])
        #results = model_sam.predict(image, bboxes=[x1, y1, x2, y2])
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        segmented = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_, y_, w_, h_ = cv2.boundingRect(cnt)
            cropped = segmented[y_:y_+h_, x_:x_+w_]
        #cropped = segmented
    else:   
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
    
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Terapkan mask dan ganti background
        segmented = cv2.bitwise_and(image, image, mask=mask)

        # Bounding box
        x = int(x - (w / 2))
        y = int(y - (h / 2))

        # Crop
        cropped = segmented[y:y+h, x:x+w]

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Temukan kontur terbesar
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)

        # Ambil semua titik kontur dan reshape untuk PCA
        data = cnt.reshape(-1, 2)
        pca = PCA(n_components=2)
        pca.fit(data)

        # Komponen utama pertama menentukan arah utama
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)

        # Rotasi agar komponen utama sejajar horizontal
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    else:
        rotated = cropped

    # Hitung ukuran area crop yang akan ditempel (background dikurangi margin kiri-kanan-atas-bawah)
    bg_w, bg_h = output_size
    max_w = bg_w - 2 * margin
    max_h = bg_h - 2 * margin

    # Resize segmented agar muat di area crop (dengan margin)
    seg_h, seg_w = rotated.shape[:2]
    scale = min(max_w / seg_w, max_h / seg_h)
    new_w, new_h = int(seg_w * scale), int(seg_h * scale)
    resized = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Siapkan background hitam
    background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)

    # Center-kan hasil crop di background dengan margin
    x_offset = margin + (max_w - new_w) // 2
    y_offset = margin + (max_h - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    print("image cropped")
    return background

def sam_crop(image, box, output_size=(800, 400), margin=40):
    print("segmenting with SAM")
    start_time = time.time()
    x, y, w, h = box    
    x1 = int(x - (w / 2))-20
    y1 = int(y - (h / 2))-20
    x2 = int(x1 + w)+20
    y2 = int(y1 + h)+20
    
    #results = model_sam.predict(image, points=[int(x), int(y)])
    '''mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    
    mask_area = np.sum(mask > 0)
    image_area = image.shape[0] * image.shape[1]
    mask_ratio = mask_area / image_area
    
    if mask_ratio < 0.1:
        print("Points inference failed, falling back to bounding box")
        results = model_sam.predict(image, bboxes=[[x1, y1, x2, y2]])'''
    results = model_sam.predict(image, bboxes=[[x1, y1, x2, y2]])
    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    segmented = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        data = cnt.reshape(-1, 2)
        pca = PCA(n_components=2)
        pca.fit(data)

        # Komponen utama pertama menentukan arah utama
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)

        # Rotasi agar komponen utama sejajar horizontal
        center = (segmented.shape[1] / 2, segmented.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # Hitung ukuran canvas baru agar tidak terpotong
        h, w = segmented.shape[:2]
        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        # Update matriks rotasi agar gambar tetap di tengah
        M[0, 2] += bound_w/2 - center[0]
        M[1, 2] += bound_h/2 - center[1]

        rotated = cv2.warpAffine(segmented, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Hitung ukuran area crop yang akan ditempel (background dikurangi margin kiri-kanan-atas-bawah)
    bg_w, bg_h = output_size

    # Resize agar hasil rotasi fit ke frame output_size (tanpa margin)
    max_w = bg_w - 2 * margin
    max_h = bg_h - 2 * margin
    seg_h, seg_w = rotated.shape[:2]
    scale = min(max_w / seg_w, max_h / seg_h)
    new_w, new_h = int(seg_w * scale), int(seg_h * scale)
    resized = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Siapkan background hitam
    background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)

    # Crop area non-hitam (area objek) dari hasil rotasi
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x2, y2, w2, h2 = cv2.boundingRect(max(contours, key=cv2.contourArea))
        rotated_cropped = rotated[y2:y2+h2, x2:x2+w2]
    else:
        rotated_cropped = rotated

    # Lanjutkan resize dan penempatan ke background seperti biasa
    seg_h, seg_w = rotated_cropped.shape[:2]
    scale = min(max_w / seg_w, max_h / seg_h)
    new_w, new_h = int(seg_w * scale), int(seg_h * scale)
    resized = cv2.resize(rotated_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center-kan hasil crop di background dengan margin
    x_offset = margin + (bg_w - 2 * margin - new_w) // 2
    y_offset = margin + (bg_h - 2 * margin - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    finish_time = time.time()
    total_time = round(finish_time-start_time, 2)
    print(total_time)
    return background

def effad(image):
    start_time = time.time()
    global teacher, student, autoencoder, teacher_mean, teacher_std, mapnorm
    print("Analyze image")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_img = Image.fromarray(img)
    orig_width, orig_height = orig_img.size
    input_tensor = effad_transform(orig_img).unsqueeze(0)

    if on_gpu:
        input_tensor = input_tensor.cuda()

    '''map_combined, map_st, map_ae = predict(
        image=input_tensor,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_st_start=mapnorm['q_st_start'],
        q_st_end=mapnorm['q_st_end'],
        q_ae_start=mapnorm['q_ae_start'],
        q_ae_end=mapnorm['q_ae_end'],
    )'''

    map_combined, map_st, map_ae = predict(
        image=input_tensor,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std
    )

    # === Resize ke ukuran asli ===
    map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
    #map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
    heatmap_np = map_combined[0, 0].cpu().numpy()
    heatmap_np = cv2.resize(heatmap_np, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
    score_max = np.max(heatmap_np)
    score_mean = np.mean(heatmap_np)
    print(score_max, score_mean)

    # === Normalisasi heatmap ke 0-255 dan warnai ===  
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # === Gabungkan dengan gambar asli ===
    orig_img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_img_cv, 0.5, colored_heatmap, 0.5, 0)
    finish_time = time.time()
    total_time = round(finish_time-start_time, 2)
    print(total_time)
    return overlay, score_max, score_mean


def save_lines():
    global LINE1_P1, LINE1_P2, CONFIDENCE, THRESHOLD_EFFAD
    print("Save configuration")
    data = {
        "LINE1_P1": LINE1_P1,
        "LINE1_P2": LINE1_P2,
        "CONFIDENCE": CONFIDENCE,
        "THRESHOLD_EFFAD": THRESHOLD_EFFAD
    }
    with open(LINES_FILE, "w") as f:
        json.dump(data, f)

def load_lines():
    global LINE1_P1, LINE1_P2, CONFIDENCE, THRESHOLD_EFFAD
    if os.path.exists(LINES_FILE):
        with open(LINES_FILE, "r") as f:
            data = json.load(f)
            LINE1_P1 = data.get("LINE1_P1", LINE1_P1)
            LINE1_P2 = data.get("LINE1_P2", LINE1_P2)
            CONFIDENCE = data.get("CONFIDENCE", CONFIDENCE)
            THRESHOLD_EFFAD = data.get("THRESHOLD_EFFAD", THRESHOLD_EFFAD)

def analyze_image(image, mask, box):
    global status
    threading.Thread(target=send_signal, args=(14,"object detected"), daemon=True).start()
    cropped = None
    print("Starting analysis")
    if mask is not None:
        print("Using YOLO mask")
        cropped = yolo_seg_crop(image, mask, box)
    else:
        print("Using SAM mask")
        cropped = sam_crop(image=image, box=box)
    overlay_img, score_max, score_mean = effad(image=cropped)
    produk_ok = score_max < THRESHOLD_EFFAD
    warped = overlay_img
    with detection_lock:        
        status = [f"OK max:{round(float(score_max), 4)}", (0, 255, 0)] if produk_ok else [f"NG max:{round(float(score_max), 4)}", (0, 0, 255)]
    cv2.putText(warped, f'Judgement: {status[0]}', (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, status[1], 2)
    # Simpan ke buffer recent_results
    recent_results.appendleft((
        warped.copy(),
        status[0]
    ))

    socketio.start_background_task(socketio.emit, 'recent_images', [
        {
            'image': f'data:image/jpeg;base64,{base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")}',
            'status': status[:2]
        }
        for img, status in list(recent_results)
    ])

    scale_factor = 0.5 #scale down 50%
    frame_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    db_insert(THRESHOLD_EFFAD, round(float(score_max), 4), round(float(score_mean), 4), status[0][:2], warped, cropped, frame_resized)
    if not produk_ok:
        threading.Thread(target=send_signal, args=(12,"NG"), daemon=True).start()
        '''cv2.imwrite(f"detected ingot\\annotated\\annotated_ng{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}.jpg", warped)
        cv2.imwrite(f"detected ingot\\raw\\raw_ng{datetime.datetime.now().strftime('%d%m%Y %H%M%S')}.jpg", image)
        cv2.imwrite(f"detected ingot\\raw\\raw_ng_{datetime.datetime.now().strftime('%d%m%Y %H%M%S')}.jpg", cropped)'''
    else:
        threading.Thread(target=send_signal, args=(15,"OK"), daemon=True).start()
        '''cv2.imwrite(f"detected ingot\\annotated\\annotated_ok{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}.jpg", warped)
        cv2.imwrite(f"detected ingot\\raw\\raw_ok{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}.jpg", image)
        cv2.imwrite(f"detected ingot\\raw\\raw_ok_{datetime.datetime.now().strftime('%d%m%Y %H%M%S')}.jpg", cropped)'''
    #return status, warped

def product_name():
    global product_list
    conn = connect(
        host="192.168.25.208",
        port="5436",
        database="ingot_visual_inspection",
        user="postgres",
        password="Postgre@sql1"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT nama_produk FROM tbl_nama_produk ORDER BY nomor")
    products = cur.fetchall()
    cur.close()
    conn.close()
    product_list = [row[0] for row in products if row[0] is not None]

# Use pyAV
def getImg():
    global _exit, getimg_thread_started
    getimg_thread_started = True
    container = av.open(RTSP_URL)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    while not _exit:
        try:
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='bgr24')
                if img is None or img.size == 0 or len(img.shape) != 3 or img.shape[2] != 3:
                    print("Skip corrupt frame")
                    continue
                with frame_lock:
                    shared_frame.append(img)
                if _exit:
                    break
                time.sleep(0.01)
            container = av.open(RTSP_URL)
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
        except Exception as e:
            print(f"RTSP/PyAV error: {e}")
            container = av.open(RTSP_URL)
            time.sleep(1)
            continue
    print("Get live image function finished")

# Use opencv
'''def getImg():
    global _exit, getimg_thread_started
    getimg_thread_started = True
    cap = cv2.VideoCapture(RTSP_URL)
    while cap.isOpened:
        try:
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    shared_frame.append(frame)
            if _exit:
                break
            time.sleep(0.01)
        except Exception as e:
            print(f"RTSP error: {e}")
            cap = cv2.VideoCapture(RTSP_URL)
            time.sleep(1)
            continue
    print("Get live image function finished")'''

def inference_thread():
    global _exit, annotated_frame, LINE1_P1, LINE1_P2, CONFIDENCE, THRESHOLD_EFFAD, inference_thread_started, production_start, status
    inference_thread_started = True
    new_time = 0
    prev_time = 0
    prev_cy_dict = {}
    annotated = None
    while not _exit:
        with frame_lock:
            if len(shared_frame) == 0:
                frame = None
            else:
                frame = shared_frame[-1].copy()
                shared_frame.clear()
        if frame is None:
            time.sleep(0.1)
            continue

        results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE, imgsz=640, tracker='botsort_light.yaml')
        #results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE, imgsz=640, tracker='.myvenv\\Lib\\site-packages\\ultralytics\\cfg\\trackers\\bytetrack.yaml')
        annotated = results[0].plot()
        capture_line = LINE1_P1[1]
        cv2.line(annotated, (0, capture_line), (annotated.shape[1], capture_line), (0, 255, 0), 2)

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        for i in range(len(boxes)):
            track_id = ids[i] if ids is not None else None
            if track_id is None:
                continue
            cls_id = int(boxes.cls[i].cpu().numpy())
            label = model.names[cls_id].lower()
            if "ingot" in label:
                color_ = None
                box = boxes.xywh[i].cpu().numpy().astype(int)
                x, y, w, h = box
                cy = int(y)
                mask = results[0].masks
                if mask is not None and mask.data is not None:
                    mask = mask.data[i].cpu().numpy().astype(np.uint8) * 255
                prev_cy = prev_cy_dict.get(track_id, 0)
                if prev_cy > capture_line:
                    color_ = (0,255,0)
                else:
                    color_ = (0,0,255)
                if (cy - capture_line) >= 0 and prev_cy < capture_line and production_start:
                    analyze_executor.submit(analyze_image, image=frame, mask=None, box=box)
                prev_cy_dict[track_id] = cy
                cv2.putText(annotated, f"{(cy - capture_line) >= 10} {prev_cy < capture_line}", (int(x-(w//2)), int(y-(h//2))+50), cv2.FONT_HERSHEY_SIMPLEX, 2, color_, 3)

        new_time = time.time()
        fps = int(1/(new_time-prev_time))
        prev_time = new_time

        keys_to_remove = [k for k, v in prev_cy_dict.items() if v > annotated.shape[0] + 100]
        for k in keys_to_remove:
            prev_cy_dict.pop(k)

        cv2.putText(annotated, f"{fps} FPS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        with detection_lock:
            cv2.putText(annotated, f'Judgement: {status[0]}', (270,50), cv2.FONT_HERSHEY_SIMPLEX, 2, status[1], 3)
        if annotated is not None:
            annotated_frame = annotated.copy()
        time.sleep(0.03)
    print("Inference thread finished")

@app.route('/')
def index():
    if CHARGING_NO == '':
        return redirect(url_for('login'))
    return render_template("index1.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    global CHARGING_NO, PRODUCT_NO, GROUP, teacher, student, autoencoder, teacher_mean, teacher_std, mapnorm, product_list, production_start
    if CHARGING_NO != '':
        return redirect('/')
    elif request.method == 'POST':
        try:
            GROUP = request.form['grup']
            CHARGING_NO = request.form['charging']
            PRODUCT_NO = request.form['produk']

            conn = connect(
                host="192.168.25.208",
                port="5436",
                database="ingot_visual_inspection",
                user="postgres",
                password="Postgre@sql1"
            )
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("INSERT INTO tbl_trace(\"group\", charging_no, product_no) VALUES (%s, %s, %s);", (GROUP, CHARGING_NO, PRODUCT_NO))
            cur.execute("SELECT model FROM tbl_nama_produk WHERE nama_produk = %s;", (PRODUCT_NO,))
            model_path = cur.fetchone()[0]
            teacher, student, autoencoder, teacher_mean, teacher_std, mapnorm = model_effad(model_path)
            production_start = True
        except Exception as e:
            print("Invalid input:", e)
        return redirect('/')
    else:
        return render_template("login.html")

@app.route('/production_end')
def production_end():
    global CHARGING_NO, PRODUCT_NO, GROUP, production_start
    CHARGING_NO = PRODUCT_NO = GROUP = ''
    production_start = False
    return redirect(url_for('login'))

@app.route('/show_data')
def show_data():
    return render_template('table.html')

@app.route('/get_products', methods=['GET'])
def get_products():
    global product_list
    return jsonify(product_list)

@app.route('/api/data', methods=['POST'])
def get_data_paginated():
    conn = connect(
        host="192.168.25.208",
        port="5436",
        database="ingot_visual_inspection",
        user="postgres",
        password="Postgre@sql1"
    )
    conn.autocommit = True
    cur = conn.cursor()

    draw = request.form.get('draw', type=int)
    start = request.form.get('start', type=int)
    length = request.form.get('length', type=int)
    search_value = request.form.get('search[value]', default='')
    order_column_index = request.form.get('order[0][column]', type=int)
    order_dir = request.form.get('order[0][dir]', default='desc')

    # Daftar nama kolom sesuai urutan DataTables
    columns = [
        "nomor", "date", "time", "threshold", "score_max", "score_mean", "judgement",
        "detected_image", "cropped_image", "original_image", "charging_no", "product_no", "\"group\""
    ]

    # Tentukan kolom untuk sorting
    order_column = columns[order_column_index] if order_column_index is not None and order_column_index < len(columns) else "nomor"

    # Query filter/search
    where_clause = ""
    params = []
    if search_value:
        where_clause = """WHERE 
            CAST(nomor AS TEXT) ILIKE %s OR
            CAST(date AS TEXT) ILIKE %s OR
            CAST(time AS TEXT) ILIKE %s OR
            CAST(threshold AS TEXT) ILIKE %s OR
            CAST(score_max AS TEXT) ILIKE %s OR
            CAST(judgement AS TEXT) ILIKE %s OR
            CAST(charging_no AS TEXT) ILIKE %s OR
            CAST(product_no AS TEXT) ILIKE %s OR
            CAST("group" AS TEXT) ILIKE %s
        """
        for _ in range(9):
            params.append(f"%{search_value}%")

    # Hitung total data (tanpa filter)
    cur.execute("SELECT COUNT(*) FROM tbl_inspection_data")
    records_total = cur.fetchone()[0]

    # Hitung total data setelah filter
    if where_clause:
        cur.execute(f"SELECT COUNT(*) FROM tbl_inspection_data {where_clause}", params)
        records_filtered = cur.fetchone()[0]
    else:
        records_filtered = records_total

    # Ambil data sesuai pagination, filter, dan sort
    query = f"""
        SELECT * FROM tbl_inspection_data
        {where_clause}
        ORDER BY {order_column} {order_dir.upper()}
        LIMIT %s OFFSET %s
    """
    params += [length, start]
    cur.execute(query, params)
    rows = cur.fetchall()

    data = []
    for r in rows:
        detected_base64 = base64.b64encode(r[7]).decode('utf-8') if r[7] else ''
        orig_base64 = base64.b64encode(r[9]).decode('utf-8') if r[9] else ''
        time_str = r[2].strftime('%H:%M:%S') if r[2] else ''
        data.append({
            "nomor": r[0],
            "date": str(r[1]),
            "time": time_str,
            "threshold": r[3],
            "score_max": r[4],
            "judgement": r[6],
            "detected_image": f"data:image/jpeg;base64,{detected_base64}",
            "original_image": f"data:image/jpeg;base64,{orig_base64}",
            "charging_no": r[10],
            "product_no": r[11],
            "group": r[12]
        })

    cur.close()
    conn.close()

    return jsonify({
        "draw": draw,
        "recordsTotal": records_total,
        "recordsFiltered": records_filtered,
        "data": data
    })

@app.route('/adjust', methods=['GET', 'POST'])
def adjust():
    global LINE1_P1, LINE1_P2, CONFIDENCE, THRESHOLD_EFFAD
    if request.method == 'POST':
        try:
            LINE1_P1 = [int(request.form['l1x1']), int(request.form['l1y1'])]
            LINE1_P2 = [int(request.form['l1x2']), int(request.form['l1y2'])]
            CONFIDENCE = float(request.form['confidence'])
            THRESHOLD_EFFAD = float(request.form['threshold'])
            save_lines()  # Simpan garis ke file
        except Exception as e:
            print("Invalid input:", e)
        return redirect(url_for('adjust'))  # Tetap di halaman adjust agar bisa lihat hasilnya

    # Form HTML + live stream
    return render_template("adjust1.html", 
                           l1x1=LINE1_P1[0], l1y1=LINE1_P1[1], l1x2=LINE1_P2[0], l1y2=LINE1_P2[1], 
                           confidence=CONFIDENCE, threshold=THRESHOLD_EFFAD)

'''@app.route('/download_crop')
def download_crop():
    global cropped_rect_image
    if cropped_rect_image is not None:
        _, buffer = cv2.imencode('.jpg', cropped_rect_image)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'cropped_rect_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        )
    else:
        return "No cropped image available", 404'''

@app.route('/recent_image/<int:idx>')
def recent_image(idx):
    # Ukuran frame hasil crop SAM
    crop_w, crop_h = 800, 400  # Sesuaikan dengan output_size di sam_inference
    if 0 <= idx < len(recent_results):
        img, _ = list(recent_results)[idx]
        _, buffer = cv2.imencode('.jpg', img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    else:
        # Placeholder: gambar abu-abu dengan tulisan "No Image"
        placeholder = np.full((crop_h, crop_w, 3), 180, dtype=np.uint8)
        cv2.putText(placeholder, "No Image", (crop_w//2 - 120, crop_h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 80), 4)
        _, buffer = cv2.imencode('.jpg', placeholder)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/recent_status')
def recent_status():
    statuses = []
    for i in range(8):
        if i < len(recent_results):
            _, status = list(recent_results)[i]
            statuses.append(status[:2])
        else:
            statuses.append('-')
    return {"statuses": statuses}

def gen():
    while True:
        img = recent_results[-1][0] if recent_results else None
        if img is not None:
            ret, jpeg = cv2.imencode('.jpg', img)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_original():
    global annotated_frame
    while True:
        if annotated_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed_original')
def video_feed_original():
    return Response(gen_original(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_lines()  # Muat garis dari file saat aplikasi dimulai
    product_name() # Muat daftar nama produk
    if not getimg_thread_started:
        t1 = threading.Thread(target=getImg, daemon=True)
        t1.start()
    if not inference_thread_started:
        t2 = threading.Thread(target=inference_thread, daemon=True)
        t2.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    _exit = True
    t1.join()
    t2.join()