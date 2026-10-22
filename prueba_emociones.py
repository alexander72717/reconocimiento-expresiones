"""
Reconocimiento de expresiones faciales en tiempo real.

Deteccion de rostros con DeepFace (YuNet por defecto) y clasificacion de
emociones con HSEmotion (entrenado en AffectNet, mucho mas preciso que el
modelo FER2013 de DeepFace). El modelo de DeepFace sigue disponible con
--modelo deepface.

Uso:
    python prueba_emociones.py                     # camara 0, modelo hsemotion
    python prueba_emociones.py --camara 1
    python prueba_emociones.py --modelo deepface
    python prueba_emociones.py --peso-neutral 0.5  # mas sensible a emociones

Teclas:
    ESC / Q  salir
    C        calibrar tu cara neutra (quedate serio 2 segundos)
    X        borrar calibracion
    + / -    mas / menos sensibilidad a las emociones (peso de "Neutral")
    S        guardar captura en ./capturas
    D        cambiar detector de rostros
    M        cambiar modelo de emociones
    E        activar/desactivar efecto espejo
    P        mostrar/ocultar panel de emociones
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime

# DeepFace imprime emojis al importarse y la consola de Windows (cp1252)
# no los soporta: sin esto el programa se cae antes de empezar.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

print("Cargando librerias (puede tardar unos segundos)...")
from deepface import DeepFace  # noqa: E402

# Traduccion y color (BGR) de cada emocion.
# Sin tildes: cv2.putText no puede dibujar caracteres acentuados.
EMOCIONES = {
    "angry": ("Enojado", (40, 40, 230)),
    "contempt": ("Desprecio", (40, 110, 180)),
    "disgust": ("Disgusto", (40, 140, 40)),
    "fear": ("Miedo", (160, 60, 160)),
    "happy": ("Feliz", (40, 210, 255)),
    "sad": ("Triste", (210, 120, 40)),
    "surprise": ("Sorpresa", (230, 200, 40)),
    "neutral": ("Neutral", (190, 190, 190)),
}
ORDEN = list(EMOCIONES)
I_NEUTRAL = ORDEN.index("neutral")

# Clases de HSEmotion -> nuestras claves
HSE_A_CLAVE = {
    "Anger": "angry", "Contempt": "contempt", "Disgust": "disgust", "Fear": "fear",
    "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise",
}

DETECTORES = ["yunet", "ssd", "opencv", "mtcnn", "retinaface"]
MODELOS = ["hsemotion", "deepface"]
HSE_MODELO = "enet_b0_8_best_afew"   # rapido (~10 ms) y ajustado para video

TAU_SUAVIZADO = 0.25      # segundos; menor = reacciona mas rapido, mayor = mas estable
IOU_MISMO_ROSTRO = 0.3    # solapamiento minimo para considerar que es el mismo rostro
SEGUNDOS_OLVIDO = 1.0     # tiempo sin ver un rostro antes de quitarlo de pantalla
SEGUNDOS_CALIBRACION = 2.0
FUERZA_CALIBRACION = 0.6  # 0 = sin efecto, 1 = compensa por completo tu cara neutra


def parse_args():
    p = argparse.ArgumentParser(description="Reconocimiento de expresiones en tiempo real")
    p.add_argument("--camara", type=int, default=0, help="indice de la camara (por defecto 0)")
    p.add_argument("--modelo", default="hsemotion", choices=MODELOS,
                   help="modelo de emociones (hsemotion es el mas preciso)")
    p.add_argument("--detector", default="yunet", choices=DETECTORES,
                   help="detector de rostros (yunet es el mas rapido)")
    p.add_argument("--ancho", type=int, default=640)
    p.add_argument("--alto", type=int, default=480)
    p.add_argument("--confianza", type=float, default=0.6,
                   help="confianza minima de deteccion de rostro (0-1)")
    p.add_argument("--peso-neutral", type=float, default=0.7,
                   help="peso de 'Neutral' (menor = detecta emociones con mas facilidad)")
    p.add_argument("--sin-espejo", action="store_true", help="no voltear la imagen")
    return p.parse_args()


# ---------------------------------------------------------------- camara

def abrir_camara(indice, ancho, alto):
    # En Windows DirectShow abre mucho mas rapido que el backend por defecto (MSMF)
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


# ---------------------------------------------------------------- modelos

class ModeloHSEmotion:
    nombre = "hsemotion"

    def __init__(self):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self.red = HSEmotionRecognizer(HSE_MODELO)
        self.indices = [ORDEN.index(HSE_A_CLAVE[self.red.idx_to_class[i]])
                        for i in range(len(self.red.idx_to_class))]

    def predecir(self, caras_bgr):
        caras_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in caras_bgr]
        _, scores = self.red.predict_multi_emotions(caras_rgb, logits=False)
        salida = np.zeros((len(caras_bgr), len(ORDEN)))
        salida[:, self.indices] = scores
        return salida


class ModeloDeepFace:
    nombre = "deepface"

    def __init__(self):
        self.predecir([np.zeros((48, 48, 3), dtype=np.uint8)])  # carga el modelo

    def predecir(self, caras_bgr):
        salida = np.zeros((len(caras_bgr), len(ORDEN)))
        for i, cara in enumerate(caras_bgr):
            r = DeepFace.analyze(cara, actions=["emotion"], detector_backend="skip",
                                 enforce_detection=False, silent=True)[0]
            for clave, valor in r["emotion"].items():
                salida[i, ORDEN.index(clave)] = float(valor) / 100.0
        return salida


def cargar_modelo(nombre):
    if nombre == "hsemotion":
        try:
            return ModeloHSEmotion()
        except Exception as e:  # noqa: BLE001
            print(f"AVISO: no se pudo cargar HSEmotion ({e}).")
            print("       Instala con:  pip install hsemotion-onnx   (se usara DeepFace)")
    return ModeloDeepFace()


def preparar_detector(detector):
    """Carga el detector por adelantado; si falla, usa opencv."""
    prueba = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(prueba, detector_backend=detector, align=False,
                               enforce_detection=False)
        return detector
    except Exception as e:  # noqa: BLE001
        if detector == "opencv":
            raise
        print(f"AVISO: no se pudo cargar el detector '{detector}' ({e}).")
        print("       Se usara 'opencv' en su lugar.")
        return preparar_detector("opencv")


# ---------------------------------------------------------------- analisis

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class Rostro:
    """Rostro seguido entre analisis, con probabilidades suavizadas en el tiempo."""

    def __init__(self, caja, probs):
        self.caja = np.array(caja, dtype=float)
        self.probs = probs
        self.visto = time.time()

    def actualizar(self, caja, probs):
        ahora = time.time()
        # Suavizado exponencial por tiempo: no depende de la velocidad del analisis
        alfa = 1.0 - np.exp(-(ahora - self.visto) / TAU_SUAVIZADO)
        self.caja = 0.4 * self.caja + 0.6 * np.array(caja, dtype=float)
        self.probs = (1 - alfa) * self.probs + alfa * probs
        self.visto = ahora

    @property
    def emocion(self):
        return ORDEN[int(np.argmax(self.probs))]


class Analizador(threading.Thread):
    """
    Analiza en segundo plano el ultimo fotograma disponible, para que el video
    se vea fluido aunque el modelo tarde mas que un fotograma en responder.
    """

    def __init__(self, detector, modelo, confianza, peso_neutral):
        super().__init__(daemon=True)
        self.detector = detector
        self.modelo = modelo
        self.confianza = confianza
        self.peso_neutral = peso_neutral
        self.base_neutra = None        # probabilidades medias de tu cara en reposo
        self.rostros = []
        self.lock = threading.Lock()
        self._frame = None
        self._hay_frame = threading.Event()
        self._activo = True
        self._calibrando_hasta = 0.0
        self._muestras = []
        self.ms_analisis = 0.0
        self.error = None

    # -- control desde el hilo principal

    def enviar(self, frame):
        self._frame = frame
        self._hay_frame.set()

    def detener(self):
        self._activo = False
        self._hay_frame.set()

    def calibrar(self):
        self._muestras = []
        self._calibrando_hasta = time.time() + SEGUNDOS_CALIBRACION

    @property
    def calibrando(self):
        return time.time() < self._calibrando_hasta

    # -- trabajo en segundo plano

    def detectar(self, frame):
        caras = DeepFace.extract_faces(frame, detector_backend=self.detector,
                                       align=False, enforce_detection=False)
        alto, ancho = frame.shape[:2]
        cajas = []
        for c in caras:
            a = c["facial_area"]
            x, y, w, h = a["x"], a["y"], a["w"], a["h"]
            # Sin rostro, DeepFace devuelve la imagen completa con confianza 0:
            # hay que descartarlo para no "adivinar" emociones en el vacio.
            if c.get("confidence", 0) < self.confianza:
                continue
            if w >= ancho - 2 and h >= alto - 2:
                continue
            if w < 30 or h < 30:
                continue
            cajas.append((max(0, x), max(0, y), w, h))
        return cajas

    def ajustar(self, probs):
        """Aplica la calibracion personal y el peso de 'Neutral'."""
        p = probs.copy()
        if self.base_neutra is not None:
            # Lo que tu cara ya muestra en reposo cuenta menos
            p = p / np.power(self.base_neutra + 0.02, FUERZA_CALIBRACION)
            p = p / p.sum()
        p[I_NEUTRAL] *= self.peso_neutral
        return p / p.sum()

    def run(self):
        while self._activo:
            self._hay_frame.wait()
            self._hay_frame.clear()
            if not self._activo:
                break
            frame = self._frame
            t0 = time.time()
            try:
                cajas = self.detectar(frame)
                crudas = self.modelo.predecir([frame[y:y + h, x:x + w] for x, y, w, h in cajas]) if cajas else []
                self.error = None
            except Exception as e:  # noqa: BLE001
                cajas, crudas = [], []
                self.error = str(e).splitlines()[0][:80]
            self.ms_analisis = (time.time() - t0) * 1000

            if self.calibrando:
                if cajas:
                    mayor = max(range(len(cajas)), key=lambda i: cajas[i][2] * cajas[i][3])
                    self._muestras.append(crudas[mayor])
            elif self._muestras:
                self.base_neutra = np.mean(self._muestras, axis=0)
                self._muestras = []
                print("Calibracion lista.")

            self._fusionar([(c, self.ajustar(p)) for c, p in zip(cajas, crudas)])

    def _fusionar(self, detecciones):
        ahora = time.time()
        with self.lock:
            libres = list(self.rostros)
            nuevos = []
            for caja, probs in detecciones:
                mejor = max(libres, key=lambda r: iou(r.caja, caja), default=None)
                if mejor is not None and iou(mejor.caja, caja) >= IOU_MISMO_ROSTRO:
                    mejor.actualizar(caja, probs)
                    libres.remove(mejor)
                    nuevos.append(mejor)
                else:
                    nuevos.append(Rostro(caja, probs))
            # Conservar un momento los rostros no vistos para evitar parpadeos
            nuevos += [r for r in libres if ahora - r.visto < SEGUNDOS_OLVIDO]
            self.rostros = nuevos

    def obtener(self):
        with self.lock:
            return [(r.caja.astype(int), r.probs.copy(), r.emocion) for r in self.rostros]


# ---------------------------------------------------------------- dibujo

FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def texto(img, msg, org, escala=0.6, color=(255, 255, 255), grosor=1):
    cv2.putText(img, msg, org, FUENTE, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, msg, org, FUENTE, escala, color, grosor, cv2.LINE_AA)


def dibujar_rostro(img, caja, probs, emocion):
    x, y, w, h = caja
    nombre, color = EMOCIONES[emocion]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    etiqueta = f"{nombre} {probs[ORDEN.index(emocion)] * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(etiqueta, FUENTE, 0.65, 2)
    y_txt = y - 8 if y - th - 14 > 0 else y + h + th + 10
    cv2.rectangle(img, (x, y_txt - th - 6), (x + tw + 10, y_txt + 6), color, -1)
    cv2.putText(img, etiqueta, (x + 5, y_txt), FUENTE, 0.65, (20, 20, 20), 2, cv2.LINE_AA)


def dibujar_panel(img, probs):
    x0, y0, ancho_barra = 10, 60, 130
    capa = img.copy()
    cv2.rectangle(capa, (0, y0 - 25), (x0 + 90 + ancho_barra + 55, y0 + 22 * len(ORDEN)), (0, 0, 0), -1)
    cv2.addWeighted(capa, 0.45, img, 0.55, 0, img)
    texto(img, "Emociones", (x0, y0 - 8), 0.5, (220, 220, 220))
    for i, e in enumerate(ORDEN):
        nombre, color = EMOCIONES[e]
        y = y0 + 8 + i * 22
        texto(img, nombre, (x0, y + 5), 0.45)
        largo = int(ancho_barra * probs[i])
        cv2.rectangle(img, (x0 + 85, y - 6), (x0 + 85 + ancho_barra, y + 6), (60, 60, 60), -1)
        cv2.rectangle(img, (x0 + 85, y - 6), (x0 + 85 + largo, y + 6), color, -1)
        texto(img, f"{probs[i] * 100:3.0f}%", (x0 + 90 + ancho_barra, y + 5), 0.45)


def dibujar_estado(img, fps, analizador, espejo):
    alto, ancho = img.shape[:2]
    calib = "si" if analizador.base_neutra is not None else "no"
    estado = (f"FPS {fps:4.1f} | {analizador.ms_analisis:3.0f} ms | {analizador.modelo.nombre}"
              f" + {analizador.detector} | neutral x{analizador.peso_neutral:.1f} | calibrado {calib}")
    texto(img, estado, (10, alto - 30), 0.42, (200, 255, 200))
    texto(img, "ESC salir  C calibrar  +/- sensibilidad  S captura  D detector  M modelo",
          (10, alto - 10), 0.42, (200, 200, 200))
    if analizador.calibrando:
        restante = analizador._calibrando_hasta - time.time()
        msg = f"CALIBRANDO: quedate serio y mira a la camara ({restante:.1f}s)"
        (tw, _), _ = cv2.getTextSize(msg, FUENTE, 0.6, 2)
        texto(img, msg, ((ancho - tw) // 2, 25), 0.6, (0, 200, 255), 2)
    elif analizador.error:
        texto(img, f"Error: {analizador.error}", (10, 25), 0.5, (0, 0, 255))


# ---------------------------------------------------------------- principal

def main():
    args = parse_args()
    ventana = "Reconocimiento de expresiones"

    print("Cargando modelos...")
    modelo = cargar_modelo(args.modelo)
    detector = preparar_detector(args.detector)
    print(f"Modelo de emociones: {modelo.nombre} | detector: {detector}")

    print(f"Abriendo camara {args.camara}...")
    cap = abrir_camara(args.camara, args.ancho, args.alto)
    if cap is None:
        print(f"ERROR: No se pudo abrir la camara {args.camara}.")
        print("  - Verifica que este conectada y que ninguna otra app la este usando.")
        print("  - Prueba otro indice:  python prueba_emociones.py --camara 1")
        input("Presiona ENTER para salir...")
        sys.exit(1)

    analizador = Analizador(detector, modelo, args.confianza, args.peso_neutral)
    analizador.start()

    espejo = not args.sin_espejo
    mostrar_panel = True
    fps, t_prev = 0.0, time.time()
    fallos = 0

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, args.ancho, args.alto)
    print("Camara iniciada. Presiona ESC o Q para salir.")
    print("Consejo: pulsa C con la cara seria para calibrar y mejorar la precision.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fallos += 1
                if fallos > 30:
                    print("ERROR: Se perdio la senal de la camara.")
                    break
                continue
            fallos = 0

            if espejo:
                frame = cv2.flip(frame, 1)
            analizador.enviar(frame.copy())

            rostros = analizador.obtener()
            if rostros:
                if mostrar_panel:
                    # Panel del rostro mas grande (el mas cercano a la camara).
                    # Se dibuja antes que los rostros para no tapar sus etiquetas.
                    _, probs, _ = max(rostros, key=lambda r: r[0][2] * r[0][3])
                    dibujar_panel(frame, probs)
                for caja, probs, emocion in rostros:
                    dibujar_rostro(frame, caja, probs, emocion)
            else:
                texto(frame, "Buscando rostro...", (10, 30), 0.7, (0, 255, 255), 2)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(ahora - t_prev, 1e-6))
            t_prev = ahora
            dibujar_estado(frame, fps, analizador, espejo)

            cv2.imshow(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (27, ord("q"), ord("Q")):
                break
            if tecla in (ord("c"), ord("C")):
                print("Calibrando: quedate serio mirando a la camara...")
                analizador.calibrar()
            elif tecla in (ord("x"), ord("X")):
                analizador.base_neutra = None
                print("Calibracion borrada.")
            elif tecla in (ord("+"), ord("=")):
                analizador.peso_neutral = max(0.1, round(analizador.peso_neutral - 0.1, 1))
            elif tecla in (ord("-"), ord("_")):
                analizador.peso_neutral = min(1.5, round(analizador.peso_neutral + 0.1, 1))
            elif tecla in (ord("s"), ord("S")):
                os.makedirs("capturas", exist_ok=True)
                ruta = os.path.join("capturas", datetime.now().strftime("captura_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(ruta, frame)
                print(f"Captura guardada: {ruta}")
            elif tecla in (ord("d"), ord("D")):
                siguiente = DETECTORES[(DETECTORES.index(analizador.detector) + 1) % len(DETECTORES)]
                print(f"Cambiando a detector '{siguiente}'...")
                analizador.detector = preparar_detector(siguiente)
            elif tecla in (ord("m"), ord("M")):
                siguiente = MODELOS[(MODELOS.index(analizador.modelo.nombre) + 1) % len(MODELOS)]
                print(f"Cambiando a modelo '{siguiente}'...")
                analizador.modelo = cargar_modelo(siguiente)
                analizador.base_neutra = None  # la calibracion depende del modelo
            elif tecla in (ord("e"), ord("E")):
                espejo = not espejo
            elif tecla in (ord("p"), ord("P")):
                mostrar_panel = not mostrar_panel

            # Cerrar con la X de la ventana
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        analizador.detener()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
