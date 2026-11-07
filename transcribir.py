import argparse
import subprocess
from pathlib import Path
import numpy as np
import ollama
from faster_whisper import WhisperModel

FORMATOS_AUDIO = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma"}
MAX_CHARS_LLM = 8000


def validar_ruta(audio_path: str) -> Path:
    ruta = Path(audio_path).resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo no existe: {ruta}")
    if not ruta.is_file():
        raise ValueError(f"La ruta no es un archivo: {ruta}")
    if ruta.suffix.lower() not in FORMATOS_AUDIO:
        raise ValueError(f"Formato no soportado: {ruta.suffix}. Usa: {', '.join(FORMATOS_AUDIO)}")
    return ruta


FILTROS_VOZ = ",".join([
    "highpass=f=80",
    "lowpass=f=8000",
    "afftdn=nf=-25",
    "acompressor=threshold=0.05:ratio=4:attack=5:release=50",
    "dynaudnorm=p=0.9:m=100",
])


def mejorar_audio(ruta: Path) -> Path:
    destino = ruta.with_stem(ruta.stem + "_mejorado").with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(ruta),
        "-af", FILTROS_VOZ,
        "-ar", "44100", "-ac", "1", "-q:a", "2",
        str(destino),
    ]
    resultado = subprocess.run(cmd, capture_output=True, shell=False)
    if resultado.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al mejorar el audio: {resultado.stderr.decode()}")
    print(f"Audio mejorado guardado en: {destino}")
    return destino


def cargar_audio(ruta: Path, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(ruta),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-",
    ]
    resultado = subprocess.run(cmd, capture_output=True, shell=False)
    if resultado.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al decodificar el audio: {resultado.stderr.decode()}")
    return np.frombuffer(resultado.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribir(audio_path: str, formato: str = "txt", modelo: str = "base"):
    ruta = validar_ruta(audio_path)
    print("Mejorando calidad del audio...")
    ruta = mejorar_audio(ruta)
    audio = cargar_audio(ruta)
    model = WhisperModel(modelo, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio)

    print(f"Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2f})")

    texto = "\n".join(seg.text.strip() for seg in segments)

    salida = ruta.with_suffix(f".{formato}")

    if formato == "md":
        contenido = f"# Transcripción: {ruta.name}\n\n{texto}\n"
    else:
        contenido = texto

    print("Corrigiendo texto con ollama...")
    contenido = limpiar_texto(contenido)

    salida.write_text(contenido, encoding="utf-8")
    print(f"Transcripción guardada en: {salida}")


def limpiar_texto(texto: str, modelo: str = "llama3.2:latest") -> str:
    texto = texto[:MAX_CHARS_LLM]
    prompt = (
        "Eres un corrector de textos en español. "
        "Corrige la ortografía, puntuación y organiza en párrafos el siguiente texto transcrito de audio. "
        "No agregues ni quites información, solo corrige y organiza. "
        "Devuelve únicamente el texto corregido, sin comentarios ni explicaciones.\n\n"
        f"{texto}"
    )
    respuesta = ollama.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 4096},
    )
    return respuesta.message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio a texto con faster-whisper")
    parser.add_argument("audio", help="Ruta del archivo de audio (.mp3, .wav, .ogg, etc.)")
    parser.add_argument("--formato", choices=["txt", "md"], default="txt", help="Formato de salida (default: txt)")
    parser.add_argument("--modelo", default="base", help="Modelo Whisper: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    transcribir(args.audio, args.formato, args.modelo)
