import cv2
import numpy as np
import math
import heapq
import time  # IMPORT TIME FOR FPS CALCULATION
from ultralytics import YOLO

# --- INITIALIZATION ---
model = YOLO("yolov8n.pt")
TARGET_CLASSES = ["box", "pallet", "carton", "book", "cell phone"]
ANOMALY_CLASSES = ["cell phone", "damaged"]

cap = cv2.VideoCapture(0)

# Variables for the Gate
prev_frame = None
SENSITIVITY_THRESHOLD = 25
PIXEL_COUNT_THRESHOLD = 500

# Variables for Tracking
tracked_objects = {}
next_object_id = 0
total_count = 0
MAX_DISTANCE = 50

# Variables for Anomalies
anomaly_heap = []
flagged_anomalies = set()

# Variables for FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions for our UI panel
    h, w, _ = frame.shape

    # --- FPS CALCULATION ---
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    # --- STEP 2: PRE-PROCESSING & THE GATE ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray_blurred
        continue
#software layer
    frame_delta = cv2.absdiff(prev_frame, gray_blurred)
    thresh = cv2.threshold(frame_delta, SENSITIVITY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = np.sum(thresh == 255)

    # --- STEP 3, 4 & 5: INFERENCE, TRACKING & ANOMALIES ---
    if changed_pixels < PIXEL_COUNT_THRESHOLD:
        status = "Static - AI Idle"
        status_color = (0, 0, 255)
        prev_frame = gray_blurred
    else:
        status = "Motion - AI Active"
        status_color = (0, 255, 0)
        prev_frame = gray_blurred

        results = model.predict(source=frame, conf=0.25, verbose=False)
        current_frame_centroids = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # if class_name in TARGET_CLASSES:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                current_frame_centroids.append((cx, cy, x1, y1, x2, y2, class_name, conf))

        new_tracked_objects = {}

        for (cx, cy, x1, y1, x2, y2, class_name, conf) in current_frame_centroids:
            matched_id = None
            min_dist = float('inf')

            for obj_id, (prev_cx, prev_cy) in tracked_objects.items():
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist < MAX_DISTANCE and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                new_tracked_objects[matched_id] = (cx, cy)
                del tracked_objects[matched_id]
            else:
                matched_id = next_object_id
                new_tracked_objects[matched_id] = (cx, cy)
                next_object_id += 1
                total_count += 1

                print(f"[INVENTORY UPDATE] Logged new {class_name.upper()} with ID: {matched_id}. Total count: {total_count}")

            if class_name in ANOMALY_CLASSES and matched_id not in flagged_anomalies:
                heapq.heappush(anomaly_heap, (-conf, matched_id, class_name))
                flagged_anomalies.add(matched_id)
                print(f">>> [URGENT ALERT] {class_name.upper()} detected! (ID: {matched_id}, Confidence: {conf:.2f}) <<<")
            # Draw bounding boxes on the main feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {matched_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        tracked_objects = new_tracked_objects

    # --- STEP 6: VISUALIZATION & UI DASHBOARD ---

    # 1. Create a black sidebar image (same height as video, 350 pixels wide)
    sidebar = np.zeros((h, 350, 3), dtype=np.uint8)

    # 2. Add System Stats to the Sidebar
    cv2.putText(sidebar, "INSPECTOR DASHBOARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(sidebar, (10, 45), (340, 45), (255, 255, 255), 1)

    cv2.putText(sidebar, f"FPS: {int(fps)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(sidebar, f"Status: {status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(sidebar, f"Total Count: {total_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 3. Add Top Anomalies (Heap Visualization)
    cv2.line(sidebar, (10, 180), (340, 180), (255, 255, 255), 1)
    cv2.putText(sidebar, "TOP ANOMALIES (HEAPSORT)", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if anomaly_heap:
        # Use nsmallest to safely view the top 5 without popping them from the heap
        top_5_anomalies = heapq.nsmallest(5, anomaly_heap)

        y_offset = 250
        for i, anomaly in enumerate(top_5_anomalies):
            conf = -anomaly[0]
            a_id = anomaly[1]
            a_class = anomaly[2]

            text = f"{i + 1}. {a_class.upper()} (ID:{a_id}) - {conf:.2f}"
            cv2.putText(sidebar, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
    else:
        cv2.putText(sidebar, "No anomalies detected.", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4. Stitch the video frame and sidebar together
    final_display = np.hstack((frame, sidebar))

    # Show the combined window
    cv2.imshow("Warehouse Intelligence System", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()