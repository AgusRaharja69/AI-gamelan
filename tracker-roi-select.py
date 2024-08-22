import cv2
import numpy as np
import pygame

# Inisialisasi pygame untuk memutar suara
pygame.init()

# Load suara gamelan
sound1 = pygame.mixer.Sound('asset/Bonang Penerus/BP_1.wav')
sound2 = pygame.mixer.Sound('asset/Bonang Penerus/BP_5.wav')

# Fungsi untuk memutar suara berdasarkan posisi
def play_sound(x, y, rois):
    for roi in rois:
        if roi['x1'] < x < roi['x2'] and roi['y1'] < y < roi['y2']:
            roi['sound'].play()

# Daftar ROI dengan posisi dan suara yang sesuai
rois = [
    {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'sound': sound1},
    {'x1': 300, 'y1': 100, 'x2': 400, 'y2': 200, 'sound': sound2},
]

# Mengakses kamera
cap = cv2.VideoCapture(1)
cv2.namedWindow('Frame')

# Fungsi untuk deteksi objek menggunakan mouse
def select_object(event, x, y, flags, param):
    global bbox, tracker, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x, y, 0, 0)
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            bbox = (bbox[0], bbox[1], x - bbox[0], y - bbox[1])
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)

cv2.setMouseCallback('Frame', select_object)

bbox = None
tracker = None
selecting = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracker is not None:
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            play_sound(p1[0] + (p2[0] - p1[0]) // 2, p1[1] + (p2[1] - p1[1]) // 2, rois)

    for roi in rois:
        cv2.rectangle(frame, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (0, 255, 0), 2)

    if selecting and bbox is not None:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                      (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()






##################################################

# import cv2
# import numpy as np
# import pygame

# # Inisialisasi pygame untuk memutar suara
# pygame.init()

# # Load suara gamelan
# sound1 = pygame.mixer.Sound('Bonang Penerus/BP_1.wav')
# sound2 = pygame.mixer.Sound('Bonang Penerus/BP_5.wav')

# # Fungsi untuk mendeteksi warna yang dipilih
# def detect_color(event, x, y, flags, param):
#     global lower_color, upper_color, tracking

#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Dapatkan nilai HSV dari piksel yang diklik
#         hsv_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
#         color_range = 20  # Rentang warna yang diperbolehkan
#         lower_color = np.array([hsv_value[0] - color_range, 50, 50])
#         upper_color = np.array([hsv_value[0] + color_range, 255, 255])
#         tracking = True

# # Fungsi untuk memutar suara berdasarkan posisi
# def play_sound(x, y, rois):
#     for roi in rois:
#         if roi['x1'] < x < roi['x2'] and roi['y1'] < y < roi['y2']:
#             roi['sound'].play()

# # Daftar ROI dengan posisi dan suara yang sesuai
# rois = [
#     {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'sound': sound1},
#     {'x1': 300, 'y1': 100, 'x2': 400, 'y2': 200, 'sound': sound2},
# ]

# # Mengakses kamera
# cap = cv2.VideoCapture(1)
# cv2.namedWindow('Frame')
# cv2.setMouseCallback('Frame', detect_color)

# lower_color = None
# upper_color = None
# tracking = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if tracking and lower_color is not None and upper_color is not None:
#         # Konversi frame ke HSV
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower_color, upper_color)
#         mask = cv2.erode(mask, None, iterations=2)
#         mask = cv2.dilate(mask, None, iterations=2)

#         # Temukan kontur
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             (x, y, w, h) = cv2.boundingRect(largest_contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             play_sound(x + w // 2, y + h // 2, rois)

#     for roi in rois:
#         cv2.rectangle(frame, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (255, 0, 0), 2)

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# pygame.quit()
