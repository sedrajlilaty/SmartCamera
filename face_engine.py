import cv2
import numpy as np
import config

class FaceEngine():
    # هون عملنا استيراد للموديلز
    def __init__(self):
        self.detector=cv2.FaceDetectorYN.create(
            str(config.DETECTOR_MODEL), "", (320, 320),
            score_threshold=config.DETECTION_SCORE
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(config.RECOGNIZER_MODEL), ""
        )
# هون منشوف الحجم تبع الفريم وفي حال مافي فيسس برجع دالة فاضية 
    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []
#  الفيتشر بياخد الوجه المعدل وبرجع مصفوفة بتمثل ملامح الوش 
# بتاخد الوجه من الفريم وبتقصه وبتعدل ميلانه. بتستعمل النقاط الخمسة (العينين والأنف والفم) اللي رجعها الكاشف لتخلي الوجه مستقيم وبحجم ثابت. ليش؟ لأن موديل التعرف بيتوقع وجه بوضعية موحدة، وهيك النتائج أدق
    def get_embedding(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned)
# بقارن بصمتين وبرجع رقم اذا 1 بكون غالبا نفس الشخص واذا 0 او قريب منو اشخاص مختلفين 
    def similarity(self, emb1, emb2):
        return self.recognizer.match(
            emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE
        )