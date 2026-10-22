# Reconocimiento de expresiones faciales

Proyecto para feria de ciencias: detecta rostros con la cámara y reconoce su expresión en tiempo real
(Feliz, Triste, Enojado, Sorpresa, Miedo, Disgusto, Desprecio y Neutral).

- Detección de rostros: YuNet (vía DeepFace)
- Emociones: HSEmotion (red EfficientNet entrenada con AffectNet); DeepFace como alternativa

## Instalación (Windows, Python 3.11 de 64 bits)

1. Doble clic en `instalar_proyecto.bat` (necesita internet la primera vez).
2. Doble clic en `ejecutar.bat`.

## Teclas

| Tecla | Acción |
|-------|--------|
| ESC / Q | Salir |
| C | Calibrar tu cara neutra (quédate serio 2 s) |
| X | Borrar calibración |
| + / - | Más / menos sensibilidad a las emociones |
| S | Guardar captura en `capturas/` |
| D | Cambiar detector de rostros |
| M | Cambiar modelo de emociones |
| E | Efecto espejo |
| P | Mostrar/ocultar panel |

Opciones: `ejecutar.bat --camara 1`, `--modelo deepface`, `--peso-neutral 0.5`.
