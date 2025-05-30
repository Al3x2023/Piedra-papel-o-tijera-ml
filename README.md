🧠 Piedra, Papel, Tijera, Lagarto, Spock con Visión por Computadora
Juego interactivo potenciado por reconocimiento de imágenes en tiempo real usando un modelo de deep learning basado en MobileNet. Reconoce cinco gestos de mano: piedra, papel, tijera, lagarto y spock.

📁 Estructura del Proyecto

rock_paper_vision/
│
├── dataset/              # Dataset de imágenes para entrenamiento
│   ├── piedra/
│   ├── papel/
│   ├── tijera/
│   ├── lagarto/
│   └── spock/
│
├── model/
│   └── model.h5          # Modelo entrenado final
│
├── scripts/
│   ├── train_model.py    # Entrenamiento del modelo
│   └── classify.py       # Reconocimiento en tiempo real
│
├── game.py               # Lógica del juego
├── requirements.txt      # Librerías necesarias
└── README.md             # Este archivo
🚀 Requisitos
Python 3.10+

TensorFlow 2.12+

OpenCV

NumPy

Matplotlib (opcional para visualizar)

Instala dependencias:


pip install -r requirements.txt
🧑‍🏫 Entrenamiento del Modelo
Asegúrate de tener al menos 100 imágenes por clase (más = mejor).

Ejecuta el entrenamiento:


python scripts/train_model.py
Esto entrenará el modelo y lo guardará como model/model.h5.

🎮 Jugar en Tiempo Real
Asegúrate de tener tu cámara conectada.

Ejecuta el reconocimiento:


python scripts/classify.py
Esto detectará el gesto y lo clasificará en tiempo real, mostrando el resultado.

🧩 Lógica del Juego
game.py implementa las reglas:

Tijera corta papel

Papel cubre piedra

Piedra aplasta lagarto

Lagarto envenena spock

Spock rompe tijera

...

Puedes integrarlo en classify.py para jugar contra la computadora.

📷 Dataset Personalizado
Si deseas mejorar el modelo:

Usa gestos claros, fondo uniforme, buena iluminación.

Puedes capturar con tu webcam y guardar automáticamente con OpenCV.#   P i e d r a - p a p e l - o - t i j e r a - m l 
 
 
