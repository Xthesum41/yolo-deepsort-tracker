import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from collections import deque
import argparse
import numpy as np

# Argumentos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')
parser.add_argument('--resize-width', type=int, default=None, help='Resize frame width for processing')
parser.add_argument('--resize-height', type=int, default=None, help='Resize frame height for processing')
parser.add_argument('--filter-classes', nargs='+', type=int, default=None, help='Filter specific class IDs')
args = parser.parse_args()

# Dispositivo CUDA ou CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Auto-detecta classes de veículos e ROI se for vídeo de exemplo
if 'sample_video2.mp4' in args.video:
    if args.filter_classes is None:
        args.filter_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        print("Auto-detected sample_video2.mp4 - filtering vehicle classes: [2, 3, 5, 7]")
    
    # ROI (região da pista) - apenas para sample_video2.mp4
    ROI_X_MIN, ROI_X_MAX = 50, 1250
    ROI_Y_MIN, ROI_Y_MAX = 0, 700
    USE_ROI = True
    print("Auto-detected sample_video2.mp4 - using ROI filter")
elif 'sample_video.mp4' in args.video:
    if args.filter_classes is None:
        args.filter_classes = [0]  # person
        print("Auto-detected sample_video.mp4 - filtering person class: [0]")
    USE_ROI = False
else:
    USE_ROI = False

# Inicializa modelo e tracker
model = YOLO(args.model)
model.to(device)
tracker = DeepSort(max_age=30)

# Abre o vídeo
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video file '{args.video}'")
    print("Please check if the file exists and is a valid video format.")
    exit(1)

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if original_width == 0 or original_height == 0:
    print(f"Error: Invalid video dimensions ({original_width}x{original_height})")
    print("The video file may be corrupted or in an unsupported format.")
    cap.release()
    exit(1)

resize_width = args.resize_width or original_width
resize_height = args.resize_height or original_height
scale_x = original_width / resize_width
scale_y = original_height / resize_height

# Pre-allocate variables and cache values for performance
frame_count = 0
detections = []
filter_classes_set = set(args.filter_classes) if args.filter_classes else None
needs_resize = resize_width != original_width or resize_height != original_height
resize_dims = (resize_width, resize_height)

# Cache ROI bounds for faster comparison
if USE_ROI:
    roi_bounds = (ROI_X_MIN, ROI_Y_MIN, ROI_X_MAX, ROI_Y_MAX)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Redimensiona se necessário (usando variável cached)
    if needs_resize:
        processing_frame = cv2.resize(frame, resize_dims)
    else:
        processing_frame = frame

    if frame_count % args.skip_frames == 0:
        results = model.predict(processing_frame, verbose=False, device=device, conf=args.conf)[0]
        detections.clear()

        # Otimizada: acesso direto aos dados sem conversão desnecessária
        boxes_data = results.boxes.data
        if len(boxes_data) > 0:
            # Convert CUDA tensors to CPU numpy array for processing
            boxes_cpu = boxes_data.cpu().numpy()
            for det in boxes_cpu:
                x1, y1, x2, y2, score, cls = det
                cls_id = int(cls)

                # Filtra por classe usando set (O(1) lookup)
                if filter_classes_set and cls_id not in filter_classes_set:
                    continue

                # Converte coords para o frame original (operações em batch)
                x1_scaled, x2_scaled = x1 * scale_x, x2 * scale_x
                y1_scaled, y2_scaled = y1 * scale_y, y2 * scale_y

                # Filtra por ROI usando variáveis cached
                if USE_ROI:
                    roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
                    if x1_scaled < roi_x_min or x2_scaled > roi_x_max or y1_scaled < roi_y_min or y2_scaled > roi_y_max:
                        continue

                detections.append(([x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled], float(score), str(cls_id)))

    # Atualiza o tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Otimizada: evita conversões desnecessárias
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Desenha a ROI apenas para sample_video2.mp4
    if USE_ROI:
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
        cv2.rectangle(frame, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 0, 0), 2)
        cv2.putText(frame, "ROI - pista", (roi_x_min, roi_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if needs_resize:
        cv2.putText(frame, f"Resize: {resize_width}x{resize_height}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mostra o frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()