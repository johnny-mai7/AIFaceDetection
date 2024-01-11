import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)  # 0 = default camera
facedetect = cv2.CascadeClassifier('myproject/data/haarcascade_frontalface_default.xml')

name = input("Enter your name: ")

# Load existing database if available
database_file_path = 'myproject/data/face_database.pkl'
if os.path.exists(database_file_path):
    with open(database_file_path, 'rb') as f:
        face_database = pickle.load(f)
else:
    face_database = {}

faces_data = []
i = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 225), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the frame
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Add new face data to the database
if name in face_database:
    face_database[name]['faces'].append(faces_data)
else:
    face_database[name] = {'faces': [faces_data]}

# Save the updated database
with open(database_file_path, 'wb') as f:
    pickle.dump(face_database, f)
