import argparse
from pathlib import Path
from faster_whisper import WhisperModel
import ffmpeg


def convertir_ogg_a_mp3(audio_path: str) -> str:
    origen = Path(audio_path)
    destino = origen.with_suffix(".mp3")
    print(f"Convirtiendo {origen.name} a mp3...")
    ffmpeg.input(str(origen)).output(str(destino)).run(quiet=True, overwrite_output=True)
    print(f"Conversión completada: {destino.name}")
    return str(destino)


def transcribir(audio_path: str, formato: str = "txt", modelo: str = "base"):
    if Path(audio_path).suffix.lower() == ".ogg":
        audio_path = convertir_ogg_a_mp3(audio_path)

    model = WhisperModel(modelo, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path)

    print(f"Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2f})")

    texto = "\n".join(seg.text.strip() for seg in segments)

    salida = Path(audio_path).with_suffix(f".{formato}")

    if formato == "md":
        contenido = f"# Transcripción: {Path(audio_path).name}\n\n{texto}\n"
    else:
        contenido = texto

    salida.write_text(contenido, encoding="utf-8")
    print(f"Transcripción guardada en: {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio a texto con faster-whisper")
    parser.add_argument("audio", help="Ruta del archivo de audio (.mp3, .wav, .ogg, etc.)")
    parser.add_argument("--formato", choices=["txt", "md"], default="txt", help="Formato de salida (default: txt)")
    parser.add_argument("--modelo", default="base", help="Modelo Whisper: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    transcribir(args.audio, args.formato, args.modelo)
