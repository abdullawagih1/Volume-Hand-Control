import cv2
import math
import time
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Define detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get device audio parameter
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

# Active camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        success, image = cap.read()

        # Horizontal flipping
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB for accurate results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mark the image as not writeable to pass by reference for accurate results
        image.flags.writeable = False

        # Process the image and detect the pose
        results = hands.process(image)

        # remark the image as writeable
        image.flags.writeable = True

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks list
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        if lmList:
            # print(lmList[4], lmList[8])

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            # Hand range 50 - 230
            # Volume Range -65 - 0
            vol = np.interp(length, [50, 230], [minVol, maxVol])
            volBar = np.interp(length, [50, 230], [400, 150])
            volPer = np.interp(length, [50, 230], [0, 100])
            print(int(length), int(vol))
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(image, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(image, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Get current time
        current_time = str(time.ctime())
        cv2.putText(image, current_time, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        # Show results
        cv2.imshow('Live', image)

        # To exit from live, press Esc key
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the Esc Key
            break

# release camera and close windows
cap.release()
cv2.destroyAllWindows()

