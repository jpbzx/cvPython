import cv2
import numpy as np
import mediapipe as mp
import time
import alsaaudio
from distanceFinder import DistanceFinder

cap = cv2.VideoCapture(0)



BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

detection_result = None

# TODO: understand whats tf is realy goin on in here!!!!
def update_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detection_result
    detection_result = result
    print(f"handlandmarkResult: {result}")

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = update_result
)

landmarker = HandLandmarker.create_from_options(options)

distance_finder = DistanceFinder()

mixer = alsaaudio.Mixer()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #conversão de bgr to rgb (mediapipe specs)
    dataPrep = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dataPrep)

    #deteção de maos
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if detection_result and detection_result.hand_landmarks:
        if len(frame.shape) == 3:
            h, w, chanel = frame.shape
        else:
            h, w = frame.shape
            
        print(f"nmr of hand detected: {len(detection_result.hand_landmarks)}")

        for hand_landmarks in detection_result.hand_landmarks:
            #dots para cada landmark

            #identificação das cordenadas das pontas do polegar e do indicador 
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y *h)
            index_x, index_y = int(index_tip.x *w), int(index_tip.y *h)

            distance = distance_finder.calculate_distance(thumb_x, thumb_y, index_x, index_y) 

            mixer.setvolume(int(distance))

            cv2.putText(
                frame,
                str(int(distance)),
                (20,30),
                5,
                2,
                (0,255,0),
                1,
                1
            )
            
            for landmark in hand_landmarks:
                
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0,255,0),
                    -1
                )
            
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
            ]

            for start, end in connections:
                x1 = int(hand_landmarks[start].x *w)
                y1 = int(hand_landmarks[start].y *h)
                x2 = int(hand_landmarks[end].x *w)
                y2 = int(hand_landmarks[end].y *h)
                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

    cv2.imshow("Frame in gray", frame)
    if cv2.waitKey(1) == ord('q'):
        break

