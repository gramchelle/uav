import cv2
import numpy as np
from pozisyon_kestirimi import DronePositionEstimator, DetectedObject
import time

def test_with_webcam():
    """Test the position estimation with webcam"""
    print("Testing Drone Position Estimation with Webcam")
    print("Press 'q' to quit, 'r' to reset position")
    
    # Initialize position estimator
    estimator = DronePositionEstimator(
        drone_height=50.0,  # Height in meters
        ground_resolution=0.1  # Meters per pixel
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully. Starting position estimation...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Estimate position using visual odometry
        position = estimator.estimate_visual_odometry(frame)
        
        # Display results
        if position:
            # Draw position info on frame
            pos_text = f"Position: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})"
            conf_text = f"Confidence: {position.confidence:.2f}"
            method_text = f"Method: {position.method.value}"
            
            cv2.putText(frame, pos_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, method_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No position estimate", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Drone Position Estimation', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            estimator.reset_position()
            print("Position reset")
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_video_file(video_path):
    """Test with a video file"""
    print(f"Testing with video file: {video_path}")
    
    estimator = DronePositionEstimator(
        drone_height=50.0,
        ground_resolution=0.1
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Estimate position
        position = estimator.estimate_visual_odometry(frame)
        
        if position:
            print(f"Frame {frame_count}: Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
        
        frame_count += 1
        
        # Process only first 100 frames for testing
        if frame_count >= 100:
            break
    
    cap.release()
    print(f"Processed {frame_count} frames")

def test_object_based_positioning():
    """Test object-based positioning with mock detections"""
    print("Testing Object-Based Positioning")
    
    estimator = DronePositionEstimator(
        drone_height=50.0,
        ground_resolution=0.1
    )
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock detected objects (simulating YOLO detections)
    detections = [
        DetectedObject(
            bbox=[100, 100, 200, 200],  # x1, y1, x2, y2
            confidence=0.9,
            class_id=0,
            class_name="building",
            center_point=(150, 150)
        ),
        DetectedObject(
            bbox=[400, 300, 500, 400],
            confidence=0.8,
            class_id=1,
            class_name="tree",
            center_point=(450, 350)
        )
    ]
    
    # Reference objects with known world coordinates
    reference_objects = {
        "building": (100, 200, 0),  # x, y, z in meters
        "tree": (150, 300, 0)
    }
    
    # Estimate position using objects
    position = estimator.estimate_position_from_objects(frame, detections, reference_objects)
    
    if position:
        print(f"Object-based position: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
        print(f"Confidence: {position.confidence:.2f}")
    else:
        print("Could not estimate position from objects")

if __name__ == "__main__":
    print("Drone Position Estimation Test")
    print("=" * 30)
    
    # Choose test method
    print("Choose test method:")
    print("1. Test with webcam")
    print("2. Test with video file")
    print("3. Test object-based positioning")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_with_webcam()
    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        test_with_video_file(video_path)
    elif choice == "3":
        test_object_based_positioning()
    else:
        print("Invalid choice. Running webcam test...")
        test_with_webcam() 