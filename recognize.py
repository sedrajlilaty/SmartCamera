import cv2
import numpy as np
import config
from face_engine import FaceEngine

# 1. تهيئة المحرك والكاميرا
engine = FaceEngine()
cap = cv2.VideoCapture(config.CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 2. تحميل قاعدة بيانات البصمات المسجلة من مجلد التسجيل (ENROLL_DIR)
known_faces = {}
enroll_dir = getattr(config, config.ENROLL_DIR, config.BASE_DIR / "enrolled")

if enroll_dir.exists():
    for file_path in enroll_dir.glob("*.npy"):
        name = file_path.stem  # اسم الشخص مستخرج من اسم الملف
        samples = np.load(file_path)  # شكلها (15, 128) عادةً
        known_faces[name] = samples
    print(f"Loaded known faces: {list(known_faces.keys())}")
else:
    print("Warning: Enrollment directory not found! No faces to recognize.")

print("Recognition system is running. Press 'ESC' or 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    display_frame = frame.copy()
    faces = engine.detect(frame)

    for face in faces:
        # استخراج بصمة الوجه الحالي (شكلها متوافق مع similarity)
        current_embedding = engine.get_embedding(frame, face)

        
        best_match_name = "Unknown"
        highest_similarity = 0.0

        # المقارنة مع كل الشخصيات والبصمات المسجلة
        for name, samples in known_faces.items():
            for sample in samples:
                # حل مشكلة الأبعاد: تحويل الصف الفردي إلى شكل (1, -1) لتتطابق مع دالة الـ match
                sample_reshaped = sample.reshape(1, -1)
                
                score = engine.similarity(current_embedding, sample_reshaped)
                
                # إيجاد التشابه الأعلى (Max Similarity) من بين اللقطات
                if score > highest_similarity:
                    highest_similarity = score
                    best_match_name = name

        # مقارنة الأعلى مع العتبة (MATCH_THRESHOLD)
        if highest_similarity < config.MATCH_THRESHOLD:
            best_match_name = "Unknown"

        # تحديد اللون والنص (الخطوة 2: عرض رقم التشابه جنب الاسم على الشاشة لضبط الـ threshold)
        if best_match_name != "Unknown":
            box_color = (0, 255, 0)  # أخضر للمعروف
            label = f"{best_match_name} ({highest_similarity:.2f})"
        else:
            box_color = (0, 0, 255)  # أحمر للغريب
            label = f"Unknown ({highest_similarity:.2f})"

        # رسم المستطيل والنص على الشاشة
        x, y, w, h = face[:4].astype(int)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            display_frame, 
            label, 
            (x, max(y - 10, 20)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            box_color, 
            2
        )

    # عرض الفريم النهائي
    cv2.imshow("Face Recognition", display_frame)

    # زر الخروج ESC أو 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()