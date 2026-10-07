import cv2
import config
from face_engine import FaceEngine
import numpy as np
name= input("enter your name please  ").strip().lower()

if name=="" or not name.isascii():
    print("Error: Invalid name. It must not be empty and must contain Latin characters only.")
    exit()

engine = FaceEngine()
cap=cv2.VideoCapture(config.CAMERA_INDEX)
if not cap.isOpened():
  print("Error: Could not open camera.")
  exit()

samples=[]
while True:
  ret, frame = cap.read()
  if not ret:
    break

  # Detect faces using the original, clean frame
  faces = engine.detect(frame)

  # 2. Create a clean copy for drawing so the bounding box doesn't affect the embedding
  display_frame = frame.copy()

  # Set rectangle color: Green if exactly one face, Red otherwise
  if len(faces) == 1:
    box_color = (0, 255, 0)  # Green
  else:
    box_color = (0, 0, 255)  # Red

  # Draw bounding boxes on the display frame
  for face in faces:
    x, y, bw, bh = face[:4].astype(int)  # Adjust according to your bounding box structure
    cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), box_color, 2)

  # 3. Display the current samples counter and instructions on the screen
  counter_text = f"Samples: {len(samples)} / {config.ENROLL_SAMPLES}"
  instruction_text = "Press SPACE to capture | ESC to exit"
  
  cv2.putText(
      display_frame,
      counter_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      display_frame,
      instruction_text,
      (30, 90),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (200, 200, 200),
      2,
  )

  # Show the frame to the user
  cv2.imshow("Face Enrollment", display_frame)

  # 4. Read the key press ONCE per iteration to avoid swallowing inputs
  key = cv2.waitKey(1) & 0xFF

  # 5. Check if SPACE (ASCII 32) is pressed and exactly one face is present
  if key == 32:
    if len(faces) == 1:
      # Extract embedding from the original, clean 'frame'
      embedding = engine.get_embedding(frame, faces[0])
      samples.append(embedding)
      print(
          f"Sample captured successfully! Current count:"
          f" {len(samples)}/{config.ENROLL_SAMPLES}"
      )
    else:
      print("Warning: Exactly one face must be visible to capture a sample!")

  # 6. Exit condition: reached target samples or pressed ESC (ASCII 27)
  if len(samples) >= config.ENROLL_SAMPLES or key == 27:
    break

# Cleanup windows
cap.release()
cv2.destroyAllWindows()

if len(samples) == 0:
    print("No samples were captured. Exiting without saving.")
    exit()
    

samples = np.vstack(samples)

config.ENROLL_DIR.mkdir(exist_ok=True)
np.save(config.ENROLL_DIR / f"{name}.npy", samples)
print(f"Saved {len(samples)} samples for '{name}' in {config.ENROLL_DIR}")