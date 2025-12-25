
import cv2
from gaze import get_gaze_direction
from attention import AttentionTracker
from smile import SmileTracker

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_eye.xml"
)

smile_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_smile.xml"
)



cap = cv2.VideoCapture(0)

roi_color = None
eye_img = None

# Attention tracker
attention_tracker = AttentionTracker()

# Hand 
smile_tracker = SmileTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=15,
        minSize=(50,50)
    )

    gaze_text = "UNKNOWN"

    # --- Hand LOOP ---
    for (hx, hy, hw, hh) in smiles:
        cv2.rectangle(
            frame,
            (hx, hy),
            (hx + hw, hy + hh),
            (0, 0, 255),
            2
        )

        gray_roi_smile = gray[hy:hy + hh, hx:hx + hw]
        color_roi_smile = frame[hy:hy + hh, hx:hx + hw]

        break


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
            '''
             Representação visual 
                    (x, y)
                    ┌──────────────────┐
                    │                  │
                    │                  │  h
                    │                  │
                    └──────────────────┘
                            w

                    (x + w, y + h)
            '''

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

    # --- Hand State ---
    smile_state = smile_tracker.update(len(smiles) > 0)

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

    # Hand text in frame
    cv2.putText(
        frame,
        f"Volume: {smile_state}",
        (30, 120),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255,255,255),
        1
    )

    cv2.imshow("Face & Eye Attention Tracker", frame)
    if roi_color is not None:
        cv2.imshow("FACE ROI", roi_color)
    if eye_img is not None:
        cv2.imshow("EYE ROI", eye_img)
    #if color_roi_hand is not None:
    #    cv2.imshow("Hand Roi", color_roi_hand) 

    #press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
