# Audio a Texto

Transcribe archivos de audio a texto en formato `.txt` o `.md` usando **Whisper** (IA de transcripción) y opcionalmente corrige ortografía y organiza párrafos con **LLaMA 3.2** via Ollama. Todo funciona **100% offline** después de la primera ejecución.

---

## ¿Para qué sirve?

- Transcribir grabaciones de voz, reuniones, entrevistas, podcasts, etc.
- Obtener el texto limpio, con ortografía corregida y organizado en párrafos.
- Soporta archivos `.ogg`, `.mp3`, `.wav` y cualquier formato compatible con ffmpeg.

---

## ¿Cómo funciona?

```
Audio (.ogg / .mp3 / .wav)
        ↓
   ffmpeg decodifica el audio a formato raw
        ↓
   Whisper (faster-whisper) transcribe el audio a texto
        ↓
   LLaMA 3.2 (ollama) corrige ortografía y organiza párrafos
        ↓
   Archivo de salida (.txt o .md)
```

---

## Requisitos previos

### 1. Python 3.8 o superior
Verifica tu versión:
```bash
python --version
```

### 2. ffmpeg
Necesario para decodificar el audio. Instálalo con:
```bash
winget install --id Gyan.FFmpeg
```
Después de instalarlo, **reinicia la terminal** para que quede disponible en el PATH.

### 3. Ollama
Necesario para la corrección de texto. Descárgalo desde [ollama.com](https://ollama.com) e instálalo.

Luego descarga el modelo LLaMA 3.2:
```bash
ollama pull llama3.2:latest
```

---

## Instalación

Clona o descarga el proyecto y ejecuta:
```bash
pip install -r requirements.txt
```

Esto instala:
| Paquete | Para qué sirve |
|---|---|
| `faster-whisper` | Transcripción de audio a texto con el modelo Whisper de OpenAI |
| `ffmpeg-python` | Interfaz Python para ffmpeg |
| `numpy` | Procesamiento del audio en memoria |
| `ollama` | Cliente Python para comunicarse con Ollama localmente |

---

## Uso

Antes de ejecutar, asegúrate de que Ollama esté corriendo:
```bash
ollama serve
```

### Transcribir a `.txt` (por defecto)
```bash
python transcribir.py "C:\ruta\al\audio.ogg"
```

### Transcribir a `.md`
```bash
python transcribir.py "C:\ruta\al\audio.ogg" --formato md
```

### Usar un modelo Whisper diferente
```bash
python transcribir.py "C:\ruta\al\audio.ogg" --formato md --modelo small
```

El archivo de salida se guarda en la **misma carpeta que el audio**, con el mismo nombre y la extensión elegida.

---

## Modelos de Whisper disponibles

| Modelo | Tamaño | Precisión | Velocidad |
|--------|--------|-----------|-----------|
| `tiny` | ~75 MB | Básica | Muy rápido |
| `base` | ~150 MB | Buena | Rápido (**por defecto**) |
| `small` | ~500 MB | Mejor | Moderado |
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
| Ollama + LLaMA 3.2 | Descarga ~2GB del modelo | ✅ Offline |

---

## Advertencia

Este programa está diseñado para uso personal en archivos de tu propio equipo. Se recomienda no usarlo con rutas de archivo provenientes de fuentes externas o no confiables.
