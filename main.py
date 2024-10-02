import threading
import cv2
from deepface import DeepFace

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global face_match
    global face_matchLiz
    try:
        result = DeepFace.verify(frame, reference_img.copy())
        if result['verified']:
            face_match = True
        else:
            face_match = False

    except Exception as e:
        print(f"Error in face verification: {str(e)}")
        face_match = False

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Starting video capture...")
while True:
    ret, frame = cap.read()
    if ret:
        
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

        # Draw a rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception as e:
                print(f"Error starting thread: {str(e)}")
        
        counter += 1

        if face_match:
            cv2.putText(frame, "Alex Verbesey", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
