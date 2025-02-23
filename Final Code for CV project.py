#final code
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import pandas as pd
from openpyxl import workbook 

model = YOLO('yolov8n.pt')

known_face_encodings = []
known_face_names = []
known_faces_dir = "images atendance"

data = []

cap1 = cv2.VideoCapture(0)

people1_count = 0

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    results = model.predict(source=frame1, save=False, conf=0.5)

    people1_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # Detect person
                people1_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame1, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame1, f'People Count: {people1_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Person Detector', frame1)

    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()


for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(encoding)
        known_face_names.append(name)

cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
process_every_nth_frame = 30
last_face_locations = []
last_face_names = []

while True:
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    frame_count += 1

    small_frame = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if frame_count % process_every_nth_frame == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        last_face_locations = face_locations
        last_face_names = face_names

    for face_location, name in zip(last_face_locations, last_face_names):
        top, right, bottom, left = [int(coord * 2) for coord in face_location]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame2, (left, top), (right, bottom), color, 2)

        cv2.putText(frame2, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Recognition', frame2)

    if cv2.waitKey(1) == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()

cap3 = cv2.VideoCapture(0)

while True:
    ret3, frame3 = cap3.read()
    if not ret3:
        break

    results = model.predict(source=frame3, save=False, conf=0.5)

    lecture_prove = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # Detect person
                lecture_prove += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]

                if  lecture_prove < 10:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame3, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame3, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame3, f'People Count: { lecture_prove}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if  lecture_prove < 10:
        cv2.putText(frame3, 'Not a lecture', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame3, 'Lecture In Progress', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Person Detector', frame3)

    if cv2.waitKey(1) == ord('q'):
        break

data.append({
    'Lecture Hall Count': people1_count,
    'User Status': last_face_names,
    'Environment Count': lecture_prove,
    'Status': 'Lecture In Progress' if lecture_prove >= 10 else 'Not a Lecture'
})


cap2.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_excel('attendance_koko.xlsx', index=False)
print("Data saved to attendance_koko.xlsx")
