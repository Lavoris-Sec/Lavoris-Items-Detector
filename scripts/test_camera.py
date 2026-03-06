import cv2

def test_cameras():
    print("Searching for available cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera found at index: {i}")
            ret, frame = cap.read()
            if ret:
                print(f"   - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"❌ No camera at index: {i}")

if __name__ == "__main__":
    test_cameras()