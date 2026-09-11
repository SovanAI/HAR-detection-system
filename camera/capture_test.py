import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera opened.")

for i in range(30):
    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break

    if i == 29:
        cv2.imwrite("input/live_test.jpg", frame)
        print("Saved: input/live_test.jpg")

cap.release()
