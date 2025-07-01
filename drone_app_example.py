import cv2
import numpy as np
import torch
from pozisyon_kestirimi import DronePositionEstimator, DetectedObject, PositionEstimationMethod
import time
import json
from typing import List, Dict, Tuple

class DroneApp:
    def __init__(self, model_path: str = None):
        """
        Initialize the drone application
        
        Args:
            model_path: Path to your pretrained YOLO model (.pt file)
        """
        self.model_path = model_path
        self.position_estimator = DronePositionEstimator(
            drone_height=50.0,  # Height in meters
            ground_resolution=0.1  # Meters per pixel
        )
        
        # Load YOLO model if provided
        self.model = None
        if model_path:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Continuing without object detection...")
        
        # Reference objects with known world coordinates
        # Format: {object_name: (x, y, z) in meters}
        self.reference_objects = {
            "building": (100, 200, 0),
            "tree": (150, 300, 0),
            "car": (50, 100, 0),
            "person": (200, 150, 0),
            # Add more reference objects as needed
        }
        
        # Position history for visualization
        self.position_history = []
    
    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in the frame using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected objects
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame)
        
        detections = []
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            if conf > 0.5:  # Confidence threshold
                class_id = int(cls)
                class_name = self.model.names[class_id]
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detection = DetectedObject(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=class_name,
                    center_point=(center_x, center_y)
                )
                detections.append(detection)
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and estimate position
        
        Args:
            frame: Input frame from drone camera
            
        Returns:
            Dictionary with position estimate and detections
        """
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Estimate position using hybrid method
        position_estimate = self.position_estimator.estimate_hybrid_position(
            frame, detections, self.reference_objects
        )
        
        # If hybrid fails, try visual odometry only
        if position_estimate is None:
            position_estimate = self.position_estimator.estimate_visual_odometry(frame)
        
        # Store position history
        if position_estimate:
            self.position_history.append(position_estimate)
            # Keep only last 100 positions
            if len(self.position_history) > 100:
                self.position_history.pop(0)
        
        return {
            "position": position_estimate,
            "detections": detections,
            "frame_timestamp": time.time()
        }
    
    def visualize_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Visualize position estimate and detections on frame
        
        Args:
            frame: Input frame
            results: Results from process_frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw detections
        for detection in results["detections"]:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw position estimate
        if results["position"]:
            pos = results["position"]
            
            # Display position info
            pos_text = f"Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})"
            conf_text = f"Confidence: {pos.confidence:.2f}"
            method_text = f"Method: {pos.method.value}"
            
            cv2.putText(annotated_frame, pos_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, method_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video_file(self, video_path: str, output_path: str = None):
        """
        Process a video file and save results
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Visualize results
            annotated_frame = self.visualize_results(frame, results)
            
            # Save frame if writer is available
            if writer:
                writer.write(annotated_frame)
            
            # Display frame (optional)
            cv2.imshow('Drone Position Estimation', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.1f}s")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Processed {frame_count} frames.")
    
    def process_live_stream(self, stream_url: str = None):
        """
        Process live video stream from drone
        
        Args:
            stream_url: URL or device index for video stream
        """
        if stream_url is None:
            # Use default camera (0) or RTSP stream
            stream_url = 0
        
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open video stream {stream_url}")
            return
        
        print("Processing live stream... Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Visualize results
            annotated_frame = self.visualize_results(frame, results)
            
            # Display frame
            cv2.imshow('Drone Position Estimation - Live', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_position_data(self, filename: str):
        """
        Save position history to JSON file
        
        Args:
            filename: Output filename
        """
        data = []
        for pos in self.position_history:
            data.append({
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "confidence": pos.confidence,
                "timestamp": pos.timestamp,
                "method": pos.method.value,
                "metadata": pos.metadata
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Position data saved to {filename}")

def main():
    """
    Main function to demonstrate the drone position estimation system
    """
    print("Drone Position Estimation System")
    print("=" * 40)
    
    # Initialize the drone app
    # Replace 'your_model.pt' with the path to your pretrained YOLO model
    app = DroneApp(model_path='your_model.pt')  # Set to None if no model available
    
    # Example 1: Process a video file
    # app.process_video_file('drone_video.mp4', 'output_video.mp4')
    
    # Example 2: Process live stream
    # app.process_live_stream('rtsp://your_drone_ip:port/stream')
    
    # Example 3: Process webcam (for testing)
    app.process_live_stream(0)  # Use webcam
    
    # Save position data
    app.save_position_data('position_history.json')

if __name__ == "__main__":
    main() 