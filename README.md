# Audio a Texto

**Autor:** Job Adolfo Salinas Hernández

Transcribe archivos de audio a texto en formato `.txt` o `.md` usando **Whisper** (IA de transcripción) y mejora la calidad del audio con **DeepFilterNet** (eliminación de ruido). Los segmentos se agrupan automáticamente en párrafos por pausas de silencio. Todo funciona **100% offline** después de la primera ejecución.

---

## ¿Para qué sirve?

- Transcribir grabaciones de voz, reuniones, entrevistas, podcasts, etc.
- Obtener el texto limpio, con ortografía corregida y organizado en párrafos.
- Soporta archivos `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.wma` y cualquier formato compatible con ffmpeg.
- Reduce ruido de fondo (viento, ambiente) preservando la voz con DeepFilterNet.

---

## ¿Cómo funciona?

```
Audio (.ogg / .mp3 / .wav / .m4a / .flac / .aac / .wma)
        ↓
   ffmpeg convierte a WAV 48kHz mono
        ↓
   DeepFilterNet elimina ruido de fondo (atten_lim_db=15)
        ↓
   ffmpeg convierte audio mejorado a MP3
        ↓
   Whisper (faster-whisper) transcribe el audio a texto
        ↓
   Segmentos agrupados en párrafos por pausas de silencio
        ↓
   Archivo de salida (.txt o .md)
```

---

## Requisitos previos

### 1. Python 3.10
Este proyecto requiere **Python 3.10** por compatibilidad con DeepFilterNet, torch y torchaudio. Verifica tu versión:
```bash
python --version
```

### 2. ffmpeg
Necesario para decodificar el audio. Instálalo con:
```bash
winget install --id Gyan.FFmpeg
```
Después de instalarlo, **reinicia la terminal** para que quede disponible en el PATH.

---

## Instalación

Clona o descarga el proyecto, crea un entorno virtual con Python 3.10 e instala las dependencias:
```bash
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Esto instala:
| Paquete | Para qué sirve |
|---|---|
| `faster-whisper` | Transcripción de audio a texto con el modelo Whisper de OpenAI |
| `ffmpeg-python` | Interfaz Python para ffmpeg |
| `numpy` | Procesamiento del audio en memoria |
| `deepfilternet` | Eliminación de ruido de fondo preservando la voz |
| `torch==2.1.0` | Backend de deep learning requerido por DeepFilterNet |
| `torchaudio==2.1.0` | Procesamiento de audio con torch |
| `soundfile` | Lectura/escritura de archivos WAV |

---

## Uso

Asegúrate de que el entorno virtual esté activo antes de ejecutar.

Hay dos formas de ejecutar el script:

### Opción A: Activando el entorno virtual
```bash
venv\Scripts\activate
python transcribir.py "C:\ruta\al\audio.ogg"
```

### Opción B: Sin activar el entorno virtual (todo en una línea)
```bash
venv\Scripts\python transcribir.py "C:\ruta\al\audio.ogg"
```

### Transcribir a `.md`
```bash
venv\Scripts\python transcribir.py "C:\ruta\al\audio.ogg" --formato md
```

### Usar un modelo Whisper diferente
```bash
venv\Scripts\python transcribir.py "C:\ruta\al\audio.ogg" --formato md --modelo small
```

El archivo de salida se guarda en la **misma carpeta que el audio**, con el mismo nombre y la extensión elegida.

---

## Modelos de Whisper disponibles

| Modelo | Tamaño | Precisión | Velocidad |
|--------|--------|-----------|-----------|
| `tiny` | ~75 MB | Básica | Muy rápido |
| `base` | ~150 MB | Buena | Rápido |
| `small` | ~500 MB | Mejor | Moderado (**por defecto**) |
| `medium` | ~1.5 GB | Alta | Lento |
| `large` | ~3 GB | Máxima | Muy lento |

Los modelos se descargan automáticamente la primera vez y quedan guardados en:
```
C:\Users\<tu_usuario>\.cache\huggingface\hub\
```

---

## ¿Es offline?

| Componente | Primera vez | Después |
|---|---|---|
| Whisper (modelo) | Descarga ~150MB de Hugging Face | ✅ Offline |
| ffmpeg | Instalación local | ✅ Offline |
| DeepFilterNet | Descarga ~30MB del modelo | ✅ Offline |

---

## Privacidad

- Ningún audio ni texto se envía a internet o a la nube.
- Ningún modelo de IA se entrena con tus datos.
- Whisper y DeepFilterNet corren completamente en tu máquina.
- Una vez descargados los modelos, el programa funciona 100% offline y sin conexión a ningún servidor externo.

---

## Advertencia

Este programa está diseñado para uso personal en archivos de tu propio equipo. Se recomienda no usarlo con rutas de archivo provenientes de fuentes externas o no confiables.
