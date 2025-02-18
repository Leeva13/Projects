import face_recognition
import cv2
import numpy as np

# Знаходить камеру (якщо тільки одна)
video_capture = cv2.VideoCapture(0)

# Завантаження зображення Шевченка та створення його кодування обличчя
mask_image_1 = face_recognition.load_image_file("./Brazhnyk.jpg")
mask_face_encoding_1 = face_recognition.face_encodings(mask_image_1)[0]

# Створення масивів відомих кодувань облич та їх назв
known_face_encodings = [mask_face_encoding_1]
known_face_names = ["Brazhnyk"]

# Створення додаткових змінних
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Захват одного кадру з відео
    ret, frame = video_capture.read()

    # Змінення розміру кадру відео до 1/4 для швидшої обробки розпізнавання облич
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Перетворення зображення з кольору BGR (який використовує OpenCV) на колір RGB (який використовує розпізнавання обличчя)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Оброблення лише кожного другого кадру відео, щоб заощадити час
    if process_this_frame:
        # Знаходження усіх облич та кодування облич у поточному кадрі відео
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Перевірка, чи відповідає особа відомому обличчю
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Використання відомого обличчя з найменшою відстанню до нового обличчя
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Показ результату
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Масштабуйте резервні копії місць обличчя, оскільки кадр, який ми виявили, був зменшений до 1/4 розміру
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Малює рамку навколо обличчя
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Малює ярлик з іменем під обличчям
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Відображує отримане зображення
    cv2.imshow('Video', frame)

    # Натисніть "q" на клавіатурі, щоб вийти!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Відпустка камери
video_capture.release()
cv2.destroyAllWindows()
