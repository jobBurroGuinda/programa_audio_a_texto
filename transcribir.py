import argparse
import subprocess
from pathlib import Path
import numpy as np
import ollama
from faster_whisper import WhisperModel


def cargar_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", audio_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-",
    ]
    resultado = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(resultado.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribir(audio_path: str, formato: str = "txt", modelo: str = "base"):
    audio = cargar_audio(audio_path)
    model = WhisperModel(modelo, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio)

    print(f"Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2f})")

    texto = "\n".join(seg.text.strip() for seg in segments)

    salida = Path(audio_path).with_suffix(f".{formato}")

    if formato == "md":
        contenido = f"# Transcripción: {Path(audio_path).name}\n\n{texto}\n"
    else:
        contenido = texto

    print("Corrigiendo texto con ollama...")
    contenido = limpiar_texto(contenido)

    salida.write_text(contenido, encoding="utf-8")
    print(f"Transcripción guardada en: {salida}")


def limpiar_texto(texto: str, modelo: str = "llama3.2:latest") -> str:
    prompt = (
        "Eres un corrector de textos en español. "
        "Corrige la ortografía, puntuación y organiza en párrafos el siguiente texto transcrito de audio. "
        "No agregues ni quites información, solo corrige y organiza. "
        "Devuelve únicamente el texto corregido, sin comentarios ni explicaciones.\n\n"
        f"{texto}"
    )
    respuesta = ollama.chat(model=modelo, messages=[{"role": "user", "content": prompt}])
    return respuesta.message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio a texto con faster-whisper")
    parser.add_argument("audio", help="Ruta del archivo de audio (.mp3, .wav, .ogg, etc.)")
    parser.add_argument("--formato", choices=["txt", "md"], default="txt", help="Formato de salida (default: txt)")
    parser.add_argument("--modelo", default="base", help="Modelo Whisper: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    transcribir(args.audio, args.formato, args.modelo)
