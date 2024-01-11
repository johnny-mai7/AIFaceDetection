from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)  # 0 = default camera
facedetect = cv2.CascadeClassifier('myproject/data/haarcascade_frontalface_default.xml')

database_file_path = 'myproject/data/face_database.pkl'

if not os.path.exists(database_file_path):
    print("Error: Face database not found. Run add_faces.py to create the database.")
    exit()

with open(database_file_path, 'rb') as f:
    face_database = pickle.load(f)

# Extract face encodings and labels for training
FACES = []
LABELS = []

for label, data in face_database.items():
    for face_data in data['faces']:
        FACES.extend(face_data)
        LABELS.extend([label] * len(face_data))

FACES = np.array(FACES)
LABELS = np.array(LABELS)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

face_dict = {}  # Dictionary to store recognized faces and labels

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Update face_dict for tracking
        if output[0] not in face_dict:
            face_dict[output[0]] = {'label': output[0], 'coordinates': (x, y, w, h)}
        else:
            # Update face coordinates
            face_dict[output[0]]['coordinates'] = (x, y, w, h)

        # Draw rectangles and labels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Update the face_dict and remove faces that are no longer detected
    keys_to_remove = []
    for key, value in face_dict.items():
        if key not in output:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del face_dict[key]

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

#Johnny
#James K
#Luan
#Derrick
#Mother
#Karl
#Justin
#Kendall