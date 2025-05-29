import random

# Opciones válidas
choices = ["piedra", "papel", "tijera", "lagarto", "spock"]

# Diccionario de quién gana a quién
win_conditions = {
    "piedra": ["tijera", "lagarto"],
    "papel": ["piedra", "spock"],
    "tijera": ["papel", "lagarto"],
    "lagarto": ["spock", "papel"],
    "spock": ["tijera", "piedra"]
}

def determinar_ganador(jugador, computadora):
    if jugador == computadora:
        return "🤝 ¡Empate!"
    elif computadora in win_conditions[jugador]:
        return "🎉 ¡Ganaste!"
    else:
        return "💻 La computadora gana..."

def jugar():
    print("🎮 Piedra, Papel, Tijera, Lagarto o Spock")
    jugador = input("Elige una opción (piedra, papel, tijera, lagarto, spock): ").lower()

    if jugador not in choices:
        print("❌ Opción no válida. Intenta de nuevo.")
        return

    computadora = random.choice(choices)
    print(f"Computadora eligió: {computadora}")
    
    resultado = determinar_ganador(jugador, computadora)
    print(resultado)

if __name__ == "__main__":
    while True:
        jugar()
        again = input("¿Quieres jugar otra vez? (s/n): ").lower()
        if again != 's':
            print("👋 ¡Gracias por jugar!")
            break
