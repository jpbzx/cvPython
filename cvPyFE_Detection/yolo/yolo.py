import cv2
import ultralytics as YOLO

class HandDetectorYOLO:
    def __init__(self):
        
        self.model = YOLO.YOLO("yolov8n.pt")
        self.confidence_threshold = 0.5

    def detect_hands(self, frame):

        results = self.model(frame, conf = self.confidence_threshold, verbose = False)

        hands = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int()
                confidence = box.conf[0].item()

                x = int(x1)
                y = int(y1)
                h = int(x2 - x1)
                w = int(y2 - y1)

                hands.append((x,y,h,w, confidence))

            return hands
        
detector = HandDetectorYOLO()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hands = detector.detect_hands(frame)

    for (x,y,w,h, conf) in hands:
        cv2.rectangle(
            frame,
            (x,y),
            (x + w, y + h),
            (255,255,255),
            2
        )
        cv2.putText(
            frame,
            f"Object {conf:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            1    
        )

    cv2.imshow("Hand detection - Yolov8", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()