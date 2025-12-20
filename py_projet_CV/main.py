import cv2
from gaze import get_gaze_direction
from attention import AttentionTracker

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_eye.xml"
)
side_face_cascade = cv2.CascadeClassifier (
    "haar/haarcascade_sideface_default.xml"
)

# Webcam
webcam = cv2.VideoCapture(0)

# Attention tracker
attention_tracker = AttentionTracker(timeout=2)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7
    )

    gaze_text = "UNKNOWN"

    # --- FACE LOOP ---
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # --- EYE LOOP ---
        for (ex, ey, ew, eh) in eyes[:2]:  # limit to 2 eyes
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

            eye_img = roi_color[ey:ey + eh, ex:ex + ew]
            gaze_text = get_gaze_direction(eye_img)
            break  # use first detected eye

        break  # use first detected face

    # --- ATTENTION STATE ---
    attention_state = attention_tracker.update(len(faces) > 0)

    # --- UI OVERLAY ---
    cv2.putText(
        frame,
        f"Gaze: {gaze_text}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    color = (0, 255, 0) if attention_state == "ATTENTIVE" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Status: {attention_state}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face & Eye Attention Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
