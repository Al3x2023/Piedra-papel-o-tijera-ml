import cv2
import os

# 👉 Cambia este valor para capturar otros gestos
GESTURE = "papel"  # Opciones: piedra, papel, tijera, lagarto, spock

# 📁 Ruta donde se guardarán las imágenes
SAVE_DIR = f"dataset/{GESTURE}"
NUM_IMAGES = 100

# Crea la carpeta si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# Inicia la cámara
cap = cv2.VideoCapture(0)
count = 0

print(f"📸 Coloca tu mano haciendo '{GESTURE}'. Presiona 'c' para capturar, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo acceder a la cámara.")
        break

    # Muestra la imagen
    cv2.imshow("Captura de gestos", frame)

    key = cv2.waitKey(1)

    # Captura imagen
    if key == ord('c') and count < NUM_IMAGES:
        filename = os.path.join(SAVE_DIR, f"{GESTURE}_{count:03}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Imagen guardada: {filename}")
        count += 1

    # Salir
    elif key == ord('q') or count >= NUM_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Captura finalizada.")
