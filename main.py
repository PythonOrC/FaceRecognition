"""
Face Recognition Door Unlock System

Main application that orchestrates the complete door unlock workflow:
1. Capture frames from webcam
2. Check image quality
3. Detect faces and extract embeddings
4. Match against database
5. Perform anti-replay checks
6. Grant or deny access
"""

import cv2
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Optional

import config
from image_quality import assess_quality, check_face_quality
from face_processor import process_frame, draw_face_box
from database import FaceDatabase
from anti_replay import AntiReplayChecker
from debug_panel import get_debug_panel, debug_enabled


class DoorUnlockSystem:
    """
    Main door unlock system controller.

    Manages the webcam capture loop and coordinates all
    subsystems for face recognition-based door unlock.
    """

    def __init__(self, camera_index: int = None):
        """
        Initialize the door unlock system.

        Args:
            camera_index: Index of camera to use
        """
        self.camera_index = (
            camera_index if camera_index is not None else config.CAMERA_INDEX
        )
        self.cap = None

        # Initialize subsystems
        print("Initializing Face Recognition Door Unlock System...")
        self.database = FaceDatabase()
        self.anti_replay = AntiReplayChecker()

        # State tracking
        self.consecutive_good_frames = 0
        self.last_unlock_time = None
        self.running = False

        # Display settings
        self.show_debug = True

        print(
            f"Database has {len(self.database.get_all_persons())} registered persons"
        )

    def _init_camera(self) -> bool:
        """Initialize the webcam."""
        print(f"Attempting to open camera {self.camera_index}...")

        # Try DirectShow backend on Windows first (often more reliable)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"DirectShow failed, trying default backend...")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            print("\nTroubleshooting tips:")
            print("  1. Run 'python test_webcam.py' to check available cameras")
            print(
                "  2. Try a different camera index: python main.py --camera 1"
            )
            print("  3. Check Windows Privacy Settings > Camera")
            print("  4. Make sure no other app is using the camera")
            return False

        # Try to read a test frame to confirm camera works
        ret, _ = self.cap.read()
        if not ret:
            print("Error: Camera opened but cannot read frames")
            self.cap.release()
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # Get actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera initialized: {actual_w}x{actual_h}")
        return True

    def _release_camera(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _draw_status(
        self, frame: np.ndarray, status: str, color: tuple = (255, 255, 255)
    ):
        """Draw status text on frame."""
        cv2.putText(
            frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    def _draw_debug_info(self, frame: np.ndarray, info: dict):
        """Draw debug information overlay."""
        y_offset = 60
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            y_offset += 20

    def process_single_frame(
        self, frame: np.ndarray
    ) -> tuple[bool, str, Optional[str]]:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: BGR image from webcam

        Returns:
            Tuple of (unlock_granted, status_message, person_name)
        """
        debug = get_debug_panel()

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Check Image Quality
        # ═══════════════════════════════════════════════════════════════
        quality = assess_quality(frame)

        # Debug: capture quality overlay
        if debug_enabled():
            debug.set_quality_overlay(
                frame,
                quality.brightness_value,
                quality.sharpness_value,
                quality.is_acceptable,
                quality.issues,
            )

        if not quality.is_acceptable:
            self.consecutive_good_frames = 0
            return False, f"Quality issue: {', '.join(quality.issues)}", None

        self.consecutive_good_frames += 1

        # Wait for consecutive good frames before processing
        if self.consecutive_good_frames < config.CONSECUTIVE_GOOD_FRAMES:
            return (
                False,
                f"Stabilizing... ({self.consecutive_good_frames}/{config.CONSECUTIVE_GOOD_FRAMES})",
                None,
            )

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Detect Face and Extract Embedding
        # ═══════════════════════════════════════════════════════════════
        detection = process_frame(frame)

        if not detection.success:
            return False, detection.message, None

        # Check face-specific quality
        face_ok, face_issues = check_face_quality(
            frame, detection.face_data.location
        )
        if not face_ok:
            return False, f"Face quality: {', '.join(face_issues)}", None

        embedding = detection.face_data.embedding

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Search Database for Match
        # ═══════════════════════════════════════════════════════════════
        match = self.database.search(embedding)

        if not match.found:
            # Debug: show no match
            if debug_enabled():
                debug.set_match_result(
                    frame, detection.face_data.location, None, None, False
                )
            return False, match.message, None

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Verify Match Quality (Threshold + Margin)
        # ═══════════════════════════════════════════════════════════════
        if match.distance > config.MATCH_THRESHOLD:
            if debug_enabled():
                debug.set_match_result(
                    frame,
                    detection.face_data.location,
                    match.person_name,
                    match.distance,
                    False,
                )
            return (
                False,
                f"Match below threshold ({match.distance:.3f})",
                match.person_name,
            )

        if match.margin is not None and match.margin < config.MATCH_MARGIN:
            if debug_enabled():
                debug.set_match_result(
                    frame,
                    detection.face_data.location,
                    match.person_name,
                    match.distance,
                    False,
                )
            return (
                False,
                f"Match margin too small ({match.margin:.3f})",
                match.person_name,
            )

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Anti-Replay / Anti-Reuse Check
        # ═══════════════════════════════════════════════════════════════
        anti_result = self.anti_replay.check(
            frame, embedding, match.person_name
        )

        if not anti_result.passed:
            return (
                False,
                f"Security check failed: {anti_result.message}",
                match.person_name,
            )

        # Debug: show successful match
        if debug_enabled():
            debug.set_match_result(
                frame,
                detection.face_data.location,
                match.person_name,
                match.distance,
                True,
            )

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Grant Access!
        # ═══════════════════════════════════════════════════════════════

        # Add this embedding to improve future matching
        self.database.add_embedding_to_matched_person(
            match.person_name, embedding
        )

        # Record the unlock for anti-reuse tracking
        self.anti_replay.record_unlock(match.person_name, match.confidence)

        self.last_unlock_time = datetime.now()

        return True, f"ACCESS GRANTED: {match.person_name}", match.person_name

    def run(self):
        """
        Main loop - capture frames and process for unlock.

        Press 'q' to quit
        Press 'r' to register a new face
        Press 'd' to toggle debug display
        Press 'p' to toggle debug panel (shows intermediate images)
        """
        if not self._init_camera():
            return

        self.running = True
        debug_panel = get_debug_panel()

        print("\n" + "=" * 60)
        print("Face Recognition Door Unlock System Running")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Register new face")
        print("  'd' - Toggle debug display")
        print("  'p' - Toggle debug panel (intermediate images)")
        print("=" * 60 + "\n")

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # Clear debug panel for new frame
                if debug_enabled():
                    debug_panel.clear()

                # Process the frame
                start_time = time.time()
                unlocked, status, person_name = self.process_single_frame(frame)
                process_time = (time.time() - start_time) * 1000

                # Determine status color
                if unlocked:
                    color = (0, 255, 0)  # Green
                    print("\n" + "█" * 60)
                    print(f"█  🔓 DOOR UNLOCKED - Welcome, {person_name}!")
                    print("█" * 60 + "\n")
                elif "GRANTED" in status or person_name:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red

                # Draw status on frame
                display_frame = frame.copy()
                self._draw_status(display_frame, status, color)

                # Draw debug info if enabled
                if self.show_debug:
                    debug_info = {
                        "FPS": f"{1000/max(process_time, 1):.1f}",
                        "Process Time": f"{process_time:.1f}ms",
                        "Good Frames": self.consecutive_good_frames,
                        "Persons in DB": len(self.database.get_all_persons()),
                    }
                    self._draw_debug_info(display_frame, debug_info)

                # Show the frame
                cv2.imshow("Face Recognition Door Unlock", display_frame)

                # Update debug panel if enabled
                if debug_enabled():
                    debug_panel.show()

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quitting...")
                    self.running = False

                elif key == ord("r"):
                    self._register_face(frame)

                elif key == ord("d"):
                    self.show_debug = not self.show_debug
                    print(
                        f"Debug display: {'ON' if self.show_debug else 'OFF'}"
                    )

                elif key == ord("p"):
                    enabled = debug_panel.toggle()
                    print(f"Debug panel: {'ON' if enabled else 'OFF'}")

        finally:
            self._release_camera()
            debug_panel.disable()
            cv2.destroyAllWindows()

    def _register_face(self, frame: np.ndarray):
        """Interactive face registration."""
        print("\n" + "-" * 40)
        print("Face Registration")
        print("-" * 40)

        # Process frame to get embedding
        detection = process_frame(frame)

        if not detection.success:
            print(f"Registration failed: {detection.message}")
            return

        # Get name from user
        name = input("Enter name for this face: ").strip()

        if not name:
            print("Registration cancelled - no name provided")
            return

        # Register the face
        if self.database.register_person(name, detection.face_data.embedding):
            print(f"✓ Successfully registered: {name}")
        else:
            print(f"✗ Failed to register: {name}")

        print("-" * 40 + "\n")


def main():
    """Entry point for the door unlock system."""
    parser = argparse.ArgumentParser(
        description="Face Recognition Door Unlock System"
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Camera index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--list-persons",
        "-l",
        action="store_true",
        help="List all registered persons and exit",
    )
    parser.add_argument(
        "--register",
        "-r",
        type=str,
        metavar="NAME",
        help="Register a new face with given name",
    )
    parser.add_argument(
        "--remove",
        type=str,
        metavar="NAME",
        help="Remove a person from the database",
    )

    args = parser.parse_args()

    # Handle database management commands
    if args.list_persons:
        db = FaceDatabase()
        persons = db.get_all_persons()
        if persons:
            print("\nRegistered Persons:")
            print("-" * 40)
            for name in sorted(persons):
                info = db.get_person_info(name)
                print(f"  {name}")
                print(f"    Embeddings: {len(info.embeddings)}")
                print(
                    f"    Registered: {info.registered_at.strftime('%Y-%m-%d %H:%M')}"
                )
                if info.last_seen:
                    print(
                        f"    Last seen: {info.last_seen.strftime('%Y-%m-%d %H:%M')}"
                    )
                print(f"    Access count: {info.access_count}")
        else:
            print("\nNo persons registered in database.")
        return

    if args.remove:
        db = FaceDatabase()
        if db.remove_person(args.remove):
            print(f"Removed '{args.remove}' from database")
        else:
            print(f"Person '{args.remove}' not found in database")
        return

    if args.register:
        # Quick registration mode
        system = DoorUnlockSystem(camera_index=args.camera)
        if not system._init_camera():
            print(
                "\nTip: Run 'python test_webcam.py' to diagnose camera issues"
            )
            return

        print(f"\nRegistering face for: {args.register}")
        print("Position your face in the camera and press SPACE to capture...")

        try:
            while True:
                ret, frame = system.cap.read()
                if not ret:
                    break

                # Show preview
                detection = process_frame(frame)
                display = frame.copy()

                if detection.success:
                    draw_face_box(
                        display, detection.face_data.location, "Ready"
                    )
                    cv2.putText(
                        display,
                        "Press SPACE to register",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        display,
                        detection.message,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Register Face", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and detection.success:
                    system.database.register_person(
                        args.register, detection.face_data.embedding
                    )
                    print(f"✓ Successfully registered: {args.register}")
                    break
                elif key == ord("q"):
                    print("Registration cancelled")
                    break

        finally:
            system._release_camera()
            cv2.destroyAllWindows()
        return

    # Normal operation - run the door unlock system
    system = DoorUnlockSystem(camera_index=args.camera)
    system.run()


if __name__ == "__main__":
    main()
