import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smoothening = 5
dragging = False
scrolling = False
scroll_start_y = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmarks: 4=thumb_tip, 8=index_tip, 12=middle_tip
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            # Webcam coordinates
            ix, iy = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            mx, my = int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)
            tx, ty = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)

            # Map index finger to screen coordinates with smoothing
            screen_x = np.interp(ix, [0, frame_width], [0, screen_width])
            screen_y = np.interp(iy, [0, frame_height], [0, screen_height])
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw gesture lines
            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 2)  # thumb-index
            cv2.line(frame, (tx, ty), (mx, my), (255, 0, 0), 2)  # thumb-middle

            # Gesture distances
            dist_index_thumb = math.hypot(tx - ix, ty - iy)
            dist_middle_thumb = math.hypot(tx - mx, ty - my)

            # LEFT CLICK / DRAG
            if dist_index_thumb < 30:
                cv2.circle(frame, ((tx + ix)//2, (ty + iy)//2), 15, (0, 255, 0), cv2.FILLED)
                if not dragging:
                    pyautogui.mouseDown()  # click or start drag
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()  # release drag
                    dragging = False

            # RIGHT CLICK
            if dist_middle_thumb < 30:
                cv2.circle(frame, ((tx + mx)//2, (ty + my)//2), 15, (255, 0, 0), cv2.FILLED)
                pyautogui.click(button='right')
                time.sleep(0.3)  # debounce

            # SCROLL: if thumb + index are pinched AND middle finger extended
            if dist_index_thumb < 40 and dist_middle_thumb > 80:
                if scroll_start_y is None:
                    scroll_start_y = iy
                else:
                    delta = iy - scroll_start_y
                    if abs(delta) > 10:  # scroll threshold
                        scroll_amount = int(delta / 5)  # adjust scroll speed
                        pyautogui.scroll(-scroll_amount)
                        scroll_start_y = iy  # reset reference point
                        scrolling = True
            else:
                scroll_start_y = None
                scrolling = False

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
