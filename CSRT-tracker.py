# import cv2

# # Inisialisasi tracker CSRT
# tracker = cv2.TrackerCSRT_create()
# video = cv2.VideoCapture(1)  # Menggunakan kamera bawaan

# # Fungsi untuk seleksi objek (ROI)
# def select_object(event, x, y, flags, param):
#     global roi_selected, roi, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         roi_selected = True
#         roi = (x, y, 10, 10)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if roi_selected:
#             roi = (roi[0], roi[1], x - roi[0], y - roi[1])
#     elif event == cv2.EVENT_LBUTTONUP:
#         roi_selected = False
#         tracker.init(frame, roi)

# cv2.namedWindow('Tracking')
# cv2.setMouseCallback('Tracking', select_object)

# roi_selected = False
# roi = (0, 0, 0, 0)

# while True:
#     ok, frame = video.read()
#     if not ok:
#         break

#     if roi_selected:
#         p1 = (roi[0], roi[1])
#         p2 = (roi[0] + roi[2], roi[1] + roi[3])
#         cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

#     if tracker is not None:
#         ok, bbox = tracker.update(frame)
#         if ok:
#             p1 = (int(bbox[0]), int(bbox[1]))
#             p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#             cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
#         else:
#             cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

#     cv2.imshow('Tracking', frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
#         break

# video.release()
# cv2.destroyAllWindows()

import cv2

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture(1)

while True:
    k,frame = video.read()
    cv2.imshow("Tracking",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)
cv2.destroyWindow("ROI selector")

while True:
    ok, frame = video.read()
    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]),
              int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 2)

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break