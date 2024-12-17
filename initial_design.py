import cv2
import pytesseract
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

# Tesseract OCR path (set this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize YOLOv8 Model
model = YOLO("yolov8n.pt")  # Replace with your trained YOLO model

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3)

# API Endpoint
API_URL = "https://8541-206-132-235-3.ngrok-free.app/updateProductPlacement"

# Sample mapping for product IDs
PRODUCT_ID_MAPPING = {
    "Milk Carton": 1001,
    "Orange Juice Carton": 1002,
    "Fruit Punch Carton": 1003
}

# Function to recognize expiry date using Tesseract OCR
def extract_expiry_date(cropped_image):
    try:
        text = pytesseract.image_to_string(cropped_image, config="--psm 6")
        print(f"OCR Text: {text}")
        # Extract date from text (simplistic example, assumes format YYYY-MM-DD)
        for word in text.split():
            try:
                date = datetime.strptime(word, "%Y-%m-%d")
                return date.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception as e:
        print(f"Error in OCR: {e}")
    return None

# Function to send POST request to API
def send_api_request(item_id, product_id, position, expiry_date):
    payload = {
        "itemId": item_id,
        "productId": product_id,
        "aisle": position.get("aisle", "Unknown"),
        "shelfNumber": position.get("shelf", 0),
        "rackNumber": position.get("rack", 0),
        "rowNumber": position.get("row", 0),
        "columnNumber": position.get("column", 0),
        "action": 0,
        "expiryDate": expiry_date
    }
    try:
        response = requests.post(API_URL, json=payload)
        print(f"API Response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Failed to send API request: {e}")

# Main Function
def main(video_source=0):
    cap = cv2.VideoCapture(video_source)  # Webcam or video file
    item_counter = 0  # Unique item counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        detections = model(frame)[0]  # Get detections
        boxes, confidences, class_ids = [], [], []

        for result in detections.boxes.data:
            x1, y1, x2, y2, conf, cls_id = result
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences.append(float(conf))
            class_ids.append(int(cls_id))

        # Update DeepSORT tracker
        tracks = tracker.update_tracks(zip(boxes, confidences), frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Crop and apply OCR for expiry date
            cropped_image = frame[y1:y2, x1:x2]
            expiry_date = extract_expiry_date(cropped_image)

            # Placeholder: Identify product name (replace this with your logic)
            product_name = "Milk Carton"  # Assume detected product
            product_id = PRODUCT_ID_MAPPING.get(product_name, 0)

            # Position data (mocked as an example)
            position = {
                "aisle": "A6",
                "shelf": 1,
                "rack": 1,
                "row": 1,
                "column": 1
            }

            # Send API request
            item_counter += 1  # Mock unique item ID
            send_api_request(
                item_id=item_counter,
                product_id=product_id,
                position=position,
                expiry_date=expiry_date or "Unknown"
            )

            # Display tracking info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}, {product_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show video with detections
        cv2.imshow("YOLO Detection and Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
