import time
import cv2
import numpy as np
import pygame
import sys
import getopt

# Inisialisasi pygame untuk memutar suara
pygame.init()
pygame.mixer.init()

# Daftar suara untuk setiap tuts
sounds = [
    pygame.mixer.Sound('asset/Peking/Peking_11.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_12.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_13.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_14.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_15.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_16.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_17.wav'),
    pygame.mixer.Sound('asset/Peking/Peking_18.wav'),
    pygame.mixer.Sound('asset/Bonang Barung/BB_21.wav')
]

NOTES = [60, 62, 64, 65, 67, 69, 71, 72, 74]  # Daftar NOTEs sesuai dengan daftar sound
NOTE_VELOCITY = 127
FPS_SHOW = False
WINDOW_NAME = "VirtualGamelan"

DEBUG = False

CONSTANT_BACKGROUND = True
MINIMUM_DISPLAY_WIDTH = 800
MAXIMUM_DISPLAY_WIDTH = 1900
RECOGNIZER_WIDTH = 500
KERNEL_SIZE = 0.042
KEY_HEIGHT = 0.25
RESET_TIME = 5
SAVE_CHECK_TIME = 1
THRESHOLD = 25
COMPARISON_VALUE = 128

numKeys = len(NOTES)
playing = numKeys * [False]

videoNumber = 1
flip = True

optlist, _ = getopt.getopt(sys.argv[1:], "v:m:w:W:nf")
for o, a in optlist:
    if o == '-v':
        videoNumber = int(a)
    elif o == '-m':
        portNumber = int(a)
    elif o == '-w':
        MINIMUM_DISPLAY_WIDTH = int(a)
    elif o == '-W':
        MAXIMUM_DISPLAY_WIDTH = int(a)
    elif o == '-n':
        flip = False
    elif o == '-f':
        FPS_SHOW = True

def play_sound(index):
    sounds[index].play()

def stop_sound(index):
    sounds[index].stop()

video = cv2.VideoCapture(videoNumber)

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frameWidth,frameHeight)

if RECOGNIZER_WIDTH >= frameWidth:
    scaledWidth = frameWidth
    scaledHeight = frameHeight
else:
    aspect = frameWidth / frameHeight
    scaledWidth = RECOGNIZER_WIDTH
    scaledHeight = int(RECOGNIZER_WIDTH / aspect)

kernelSize = 2 * int(KERNEL_SIZE * scaledWidth / 2) + 1

if frameWidth < MINIMUM_DISPLAY_WIDTH:
    displayWidth = MINIMUM_DISPLAY_WIDTH
    displayHeight = int(displayWidth / aspect)
elif frameWidth > MAXIMUM_DISPLAY_WIDTH:
    displayWidth = MAXIMUM_DISPLAY_WIDTH
    displayHeight = int(displayWidth / aspect)
else:
    displayWidth = frameWidth
    displayHeight = frameHeight

blankOverlay = np.zeros((displayHeight, displayWidth, 3), dtype=np.uint8)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(WINDOW_NAME, displayWidth, displayHeight)

displayRects = []
scaledRects = []
frameRects = []

for i in range(numKeys):
    x0 = scaledWidth * i // numKeys
    x1 = scaledWidth * (i + 1) // numKeys - 1

    r = [(x0, 0), (x1, int(KEY_HEIGHT * scaledHeight))]
    scaledRects.append(r)

    x0 = frameWidth * i // numKeys
    x1 = frameWidth * (i + 1) // numKeys - 1

    r = [(x0, 0), (x1, int(KEY_HEIGHT * frameHeight))]
    frameRects.append(r)

    x0 = displayWidth * i // numKeys
    x1 = displayWidth * (i + 1) // numKeys - 1

    r = [(x0, 0), (x1, int(KEY_HEIGHT * displayHeight))]
    displayRects.append(r)

keysTopLeftFrame = (min(r[0][0] for r in frameRects), min(r[0][1] for r in frameRects))
keysBottomRightFrame = (max(r[1][0] for r in frameRects), max(r[1][1] for r in frameRects))

keysTopLeftScaled = (min(r[0][0] for r in scaledRects), min(r[0][1] for r in scaledRects))
keysBottomRightScaled = (max(r[1][0] for r in scaledRects), max(r[1][1] for r in scaledRects))
keysWidthScaled = keysBottomRightScaled[0] - keysTopLeftScaled[0]
keysHeightScaled = keysBottomRightScaled[1] - keysTopLeftScaled[1]

keys = np.zeros((keysHeightScaled, keysWidthScaled), dtype=np.uint8)

def adjustToKeys(xy):
    return (xy[0] - keysTopLeftScaled[0], xy[1] - keysTopLeftScaled[1])

for i in range(numKeys):
    r = scaledRects[i]
    cv2.rectangle(keys, adjustToKeys(r[0]), adjustToKeys(r[1]), i + 1, cv2.FILLED)

savedFrame = None
comparisonFrame = None
savedTime = 0
lastCheckTime = 0

def compare(a, b):
    return cv2.threshold(cv2.absdiff(a, b), THRESHOLD, COMPARISON_VALUE, cv2.THRESH_BINARY)[1]

if FPS_SHOW:
    readTime = 0

while True:
    if FPS_SHOW:
        t = time.time()
    ok, frame = video.read()
    if FPS_SHOW:
        readTime += time.time() - t
    if not ok:
        time.sleep(0.05)
        continue
    if flip:
        frame = cv2.flip(frame, 1) 

    keysFrame = frame[keysTopLeftFrame[1]:keysBottomRightFrame[1], keysTopLeftFrame[0]:keysBottomRightFrame[0]]
    if scaledWidth != frameWidth:
        keysFrame = cv2.resize(keysFrame, (keysWidthScaled, keysHeightScaled))
    keysFrame = cv2.cvtColor(keysFrame, cv2.COLOR_BGR2GRAY)
    if DEBUG: 
        cv2.imshow("gray", keysFrame)
    blurred = cv2.GaussianBlur(keysFrame, (kernelSize, kernelSize), 0)
    if DEBUG: 
        cv2.imshow("blurred", blurred)

    if CONSTANT_BACKGROUND:
        t = time.time()
        save = False
        if savedFrame is None:
            save = True
            lastCheckTime = t
        else:
            if t >= lastCheckTime + SAVE_CHECK_TIME:
                if COMPARISON_VALUE in compare(savedFrame, blurred):
                    save = True
                lastCheckTime = t
            if t >= savedTime + RESET_TIME:
                print("resetting")
                comparisonFrame = blurred
                save = True
        if save:
            savedFrame = blurred
            savedTime = t

    if comparisonFrame is None:
        comparisonFrame = blurred
        if FPS_SHOW:
            frameCount = 0
            startTime = time.time()
            readTime = 0
        continue

    delta = compare(comparisonFrame, blurred)
    sum = keys + delta
    if DEBUG: cv2.imshow("sum", sum)

    overlay = blankOverlay.copy()

    for i in range(numKeys):
        r = displayRects[i]
        if 1 + i + COMPARISON_VALUE in sum:
            cv2.rectangle(overlay, r[0], r[1], (255, 255, 255), cv2.FILLED)
            if not playing[i]:
                play_sound(i)
                playing[i] = True
        else:
            if playing[i]:
                stop_sound(i)
                playing[i] = False
        cv2.rectangle(overlay, r[0], r[1], (0, 255, 0), 2)

    display = cv2.resize(frame, (displayWidth, displayHeight)) if frameWidth != displayWidth else frame
    cv2.imshow(WINDOW_NAME, cv2.addWeighted(display, 1, overlay, 0.25, 1.0))

    if (cv2.waitKey(1) & 0xFF) == 27 or cv2.getWindowProperty(WINDOW_NAME, 0) == -1:
        break

    if not CONSTANT_BACKGROUND:
        comparisonFrame = blurred

    if FPS_SHOW:
        frameCount += 1
        print(frameCount / (time.time() - startTime), frameCount / (time.time() - startTime - readTime))

video.release()
cv2.destroyAllWindows()
pygame.quit()

