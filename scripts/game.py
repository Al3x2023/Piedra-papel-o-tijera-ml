import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar modelo entrenado
model = load_model('model/model.h5')

CLASSES = ['piedra', 'papel', 'tijera', 'lagarto', 'spock']
IMG_SIZE = 224

# Reglas del juego: qué vence a qué
rules = {
    'piedra':     ['tijera', 'lagarto'],
    'papel':      ['piedra', 'spock'],
    'tijera':     ['papel', 'lagarto'],
    'lagarto':    ['papel', 'spock'],
    'spock':      ['tijera', 'piedra'],
}

def get_prediction(frame):
    x1 = frame.shape[1]//2 - 112
    y1 = frame.shape[0]//2 - 112
    x2 = frame.shape[1]//2 + 112
    y2 = frame.shape[0]//2 + 112
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return None
    image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    pred = model.predict(image, verbose=0)
    return CLASSES[np.argmax(pred)]

def decide_winner(user, cpu):
    if user == cpu:
        return 'Empate'
    elif cpu in rules[user]:
        return 'Ganaste 🎉'
    else:
        return 'Perdiste 💀'

# Iniciar cámara
cap = cv2.VideoCapture(0)
print("Coloca tu mano en el recuadro. Presiona 'ESPACIO' para jugar. 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    x1 = w//2 - 112
    y1 = h//2 - 112
    x2 = w//2 + 112
    y2 = h//2 + 112
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "Presiona ESPACIO para jugar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        user_choice = get_prediction(frame)
        if user_choice:
            cpu_choice = random.choice(CLASSES)
            result = decide_winner(user_choice, cpu_choice)

            print(f"\n🧍 Tú: {user_choice}")
            print(f"🤖 CPU: {cpu_choice}")
            print(f"📢 Resultado: {result}")

            cv2.putText(frame, f"Tú: {user_choice}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"CPU: {cpu_choice}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"{result}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif key == ord('q'):
        break

    cv2.imshow("Juego: Piedra, Papel, Tijera, Lagarto, Spock", frame)

cap.release()
cv2.destroyAllWindows()
