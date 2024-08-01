import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLOv8 model
    model = YOLO('E:\\AiTec Internship 2024\\dataset detection through tracker\\dataset\\weights\\Ahsan Pen.pt')

    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker_initialized = False
    tracker = None
    roi = None
    class_name = None
    conf = None

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        if not tracker_initialized:
            # Perform inference
            results = model(frame)

            # Extract bounding boxes and confidence scores
            detections = results[0].boxes

            # Clear frame before drawing
            display_frame = frame.copy()

            # Check if any object is detected
            if len(detections) == 0:
                cv2.putText(display_frame, 'Object not found', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Use the first detected object for tracking
                detection = detections[0]
                x1, y1, x2, y2 = detection.xyxy[0]
                conf = detection.conf[0]
                cls = detection.cls[0]

                if conf > 0.5:  # Only consider detections with confidence > 0.5
                    roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                    # Get class name (Assuming you have a way to map class index to class name)
                    class_name = model.names[int(cls)]

                    # Initialize the tracker with the selected ROI
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, roi)
                    tracker_initialized = True
        else:
            # Update the tracker
            ret, roi = tracker.update(frame)

            # Draw bounding box and display confidence and class
            if ret:
                p1 = (int(roi[0]), int(roi[1]))
                p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker_initialized = False

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
