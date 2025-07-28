# ... (all imports same as before)
import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
import pygetwindow as gw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# System audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()

# Mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_locked = False
last_play_state = None
play_pause_cooldown = 1.5
last_action_time = 0
allow_volume = False

# Double-tap trackers
tap_count_right = 0
last_tap_time_right = 0
last_skip_time_right = 0

tap_count_left = 0
last_tap_time_left = 0
last_skip_time_left = 0

double_tap_cooldown = 0.5

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def fingers_up_list(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
    for i in range(1, 5):
        fingers.append(
            1 if hand_landmarks.landmark[tips[i]].y <
                 hand_landmarks.landmark[tips[i] - 2].y else 0
        )
    return fingers

def focus_media_window(title="VLC"):
    for win in gw.getWindowsWithTitle(title):
        if not win.isActive:
            try:
                win.activate()
                time.sleep(0.3)
            except:
                pass

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    right_lm, left_lm = None, None
    right_fingers, left_fingers = None, None
    allow_volume = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            lm = hand_landmarks.landmark
            fingers = fingers_up_list(hand_landmarks)

            if label == "Left":
                left_lm = lm
                left_fingers = fingers

                if sum(fingers) == 5:
                    gesture_locked = True
                elif sum(fingers) == 0:
                    gesture_locked = False

                if (fingers[1] == 1 and sum(fingers) == 1 and
                    not gesture_locked and
                    (last_play_state is not True) and
                    time.time() - last_action_time > play_pause_cooldown):

                    pyautogui.press("playpause")
                    last_play_state = True
                    last_action_time = time.time()
                    print("[▶️] Play")

                if (fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2 and
                    not gesture_locked and
                    (last_play_state is not False) and
                    time.time() - last_action_time > play_pause_cooldown):

                    pyautogui.press("playpause")
                    last_play_state = False
                    last_action_time = time.time()
                    print("[⏸️] Pause")

            elif label == "Right":
                right_lm = lm
                right_fingers = fingers
                allow_volume = (right_fingers[4] == 1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if right_lm and not gesture_locked and allow_volume:
        thumb_pt = (int(right_lm[4].x * w), int(right_lm[4].y * h))
        index_pt = (int(right_lm[8].x * w), int(right_lm[8].y * h))
        distance = get_distance(thumb_pt, index_pt)

        if 15 < distance < 200:
            vol_pct = np.interp(distance, [20, 150], [0, 100])
            vol_db = np.interp(distance, [20, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol_db, None)

            bar_h = int(np.interp(distance, [20, 150], [400, 150]))
            cv2.rectangle(frame, (50, 150), (85, 400), (255, 255, 255), 2)
            cv2.rectangle(frame, (50, bar_h), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'Vol {int(vol_pct)}%', (40, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if right_lm and not gesture_locked:
        thumb_pt = (int(right_lm[4].x * w), int(right_lm[4].y * h))
        index_pt = (int(right_lm[8].x * w), int(right_lm[8].y * h))
        if get_distance(thumb_pt, index_pt) < 30:
            if time.time() - last_tap_time_right < 0.4:
                if tap_count_right == 1 and time.time() - last_skip_time_right > double_tap_cooldown:
                    focus_media_window("VLC")
                    pyautogui.press("right")
                    print("[⏩] Right double-tap — Skip +10s")
                    last_skip_time_right = time.time()
                    tap_count_right = 0
            else:
                tap_count_right = 0
            tap_count_right += 1
            last_tap_time_right = time.time()

    if left_lm and not gesture_locked:
        thumb_pt = (int(left_lm[4].x * w), int(left_lm[4].y * h))
        index_pt = (int(left_lm[8].x * w), int(left_lm[8].y * h))
        if get_distance(thumb_pt, index_pt) < 30:
            if time.time() - last_tap_time_left < 0.4:
                if tap_count_left == 1 and time.time() - last_skip_time_left > double_tap_cooldown:
                    focus_media_window("VLC")
                    pyautogui.press("left")
                    print("[⏪] Left double-tap — Skip -10s")
                    last_skip_time_left = time.time()
                    tap_count_left = 0
            else:
                tap_count_left = 0
            tap_count_left += 1
            last_tap_time_left = time.time()

    lock_text = "LOCKED 🔒" if gesture_locked else "UNLOCKED 🔓"
    cv2.putText(frame, lock_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255) if gesture_locked else (0, 255, 0), 3)

    vol_text = "VOL ENABLED" if allow_volume and not gesture_locked else "VOL DISABLED"
    vol_color = (0, 255, 255) if allow_volume else (100, 100, 100)
    cv2.putText(frame, vol_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vol_color, 2)

    cv2.imshow("🖐️ Gesture Media Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
