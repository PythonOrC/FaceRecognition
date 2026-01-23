"""
Webcam Test Script
Tests camera access and lists available cameras
"""

import cv2


def test_cameras():
    """Test available camera indices"""
    print("Testing available cameras...")
    print("-" * 40)

    available_cameras = []

    # Test camera indices 0-4 with DirectShow backend (Windows)
    for i in range(5):
        # Try DirectShow first (more reliable on Windows)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        backend = "DSHOW"

        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
            backend = "DEFAULT"

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Camera {i}: AVAILABLE ({width}x{height}) [{backend}]")
                available_cameras.append(i)
            else:
                print(f"✗ Camera {i}: Opens but can't read frames")
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")

    print("-" * 40)

    if not available_cameras:
        print("\n⚠ No cameras detected!")
        print("\nTroubleshooting tips:")
        print("1. Make sure your webcam is plugged in")
        print("2. Check if another app is using the camera")
        print("3. Check Windows camera privacy settings:")
        print("   Settings > Privacy > Camera > Allow apps to access camera")
        print("4. Try running as Administrator")
        print("5. Check Device Manager for camera drivers")
        return None

    print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
    return available_cameras[0]


def show_webcam(camera_index=0):
    """Show live webcam feed"""
    print(f"\nOpening camera {camera_index}...")
    print("Press 'q' to quit\n")

    # Try DirectShow first on Windows
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Try setting resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break

        # Add text overlay
        cv2.putText(
            frame,
            f"Camera {camera_index} - Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    camera = test_cameras()

    if camera is not None:
        response = (
            input(f"\nShow live feed from camera {camera}? (y/n): ")
            .strip()
            .lower()
        )
        if response == "y":
            show_webcam(camera)
