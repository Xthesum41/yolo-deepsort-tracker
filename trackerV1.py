import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

# Carregar modelo YOLO
model = YOLO(args.model)
tracker = DeepSort(max_age=15)

# Abrir vídeo
cap = cv2.VideoCapture(args.video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)[0]
    detections = []

    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = det
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, str(int(cls))))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
