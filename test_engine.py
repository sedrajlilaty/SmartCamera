import cv2
import config
from face_engine import FaceEngine

engine = FaceEngine()
cap = cv2.VideoCapture(config.CAMERA_INDEX)
embeddings = []

print("SPACE = التقاط بصمة | ESC = خروج")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = engine.detect(frame)
    for f in faces:
        x, y, w, h = f[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Test", frame)
    key = cv2.waitKey(1)

    if key == 32 and len(faces) == 1:
        emb = engine.get_embedding(frame, faces[0])
        embeddings.append(emb)
        print("بصمة رقم", len(embeddings), "شكلها:", emb.shape)
        if len(embeddings) >= 2:
            print("التشابه مع السابقة:", engine.similarity(embeddings[-2], embeddings[-1]))
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()