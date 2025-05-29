import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar el modelo entrenado
model = load_model('model/model.h5')

# Clases en el mismo orden que tus carpetas en dataset/
CLASSES = ['piedra', 'papel', 'tijera', 'lagarto', 'spock']

# Tamaño esperado por el modelo
IMG_SIZE = 224

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar un rectángulo guía en el centro
    h, w, _ = frame.shape
    x1 = w//2 - 112
    y1 = h//2 - 112
    x2 = w//2 + 112
    y2 = h//2 + 112
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Extraer la región de interés (ROI)
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Predecir
        preds = model.predict(image)
        label = CLASSES[np.argmax(preds)]

        # Mostrar predicción
        cv2.putText(frame, f"Predicción: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow("Piedra, Papel, Tijera, Lagarto, Spock", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
