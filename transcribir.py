import argparse
import logging
import subprocess
from pathlib import Path
import numpy as np
import ollama
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FORMATOS_AUDIO = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma"}
MAX_CHARS_LLM = 8000


def validar_ruta(audio_path: str) -> Path:
    """
    Valida que la ruta del archivo de audio sea segura y tenga un formato soportado.

    :param audio_path: Ruta al archivo de audio proporcionada por el usuario.
    :type audio_path: str
    :returns: Ruta resuelta y validada.
    :rtype: Path
    :raises FileNotFoundError: Si el archivo no existe.
    :raises ValueError: Si la ruta no es un archivo o el formato no está soportado.
    :author: Job Adolfo Salinas Hernández
    """
    log.debug("Validando ruta: %s", audio_path)
    ruta = Path(audio_path).resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo no existe: {ruta}")
    if not ruta.is_file():
        raise ValueError(f"La ruta no es un archivo: {ruta}")
    if ruta.suffix.lower() not in FORMATOS_AUDIO:
        raise ValueError(f"Formato no soportado: {ruta.suffix}. Usa: {', '.join(FORMATOS_AUDIO)}")
    log.debug("Ruta válida: %s", ruta)
    return ruta


FILTROS_VOZ = ",".join([
    "highpass=f=80",
    "lowpass=f=8000",
    "afftdn=nf=-25",
    "acompressor=threshold=0.05:ratio=4:attack=5:release=50",
    "dynaudnorm=p=0.9:m=100",
])


def mejorar_audio(ruta: Path) -> Path:
    """
    Mejora la calidad del audio aplicando filtros de voz con ffmpeg.

    Aplica en orden: filtro paso alto (80Hz), filtro paso bajo (8000Hz),
    reducción de ruido por FFT, compresor de dinámica y normalización de volumen.
    El resultado se guarda como MP3 con el sufijo '_mejorado'.

    :param ruta: Ruta al archivo de audio original.
    :type ruta: Path
    :returns: Ruta al archivo de audio mejorado en formato MP3.
    :rtype: Path
    :raises RuntimeError: Si ffmpeg falla durante el procesamiento.
    :author: Job Adolfo Salinas Hernández
    """
    log.info("Mejorando calidad del audio: %s", ruta.name)
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
    log.info("Audio mejorado guardado en: %s", destino)
    return destino


def cargar_audio(ruta: Path, sample_rate: int = 16000) -> np.ndarray:
    """
    Decodifica un archivo de audio a un array numpy usando ffmpeg.

    Convierte el audio a PCM mono de 16 bits normalizado a float32
    en el rango [-1.0, 1.0], listo para ser procesado por Whisper.

    :param ruta: Ruta al archivo de audio.
    :type ruta: Path
    :param sample_rate: Frecuencia de muestreo en Hz (default: 16000).
    :type sample_rate: int
    :returns: Audio decodificado como array float32.
    :rtype: numpy.ndarray
    :raises RuntimeError: Si ffmpeg falla durante la decodificación.
    :author: Job Adolfo Salinas Hernández
    """
    log.info("Decodificando audio: %s", ruta.name)
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(ruta),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-",
    ]
    resultado = subprocess.run(cmd, capture_output=True, shell=False)
    if resultado.returncode != 0:
        raise RuntimeError(f"ffmpeg falló al decodificar el audio: {resultado.stderr.decode()}")
    log.debug("Audio decodificado: %d muestras a %d Hz", len(resultado.stdout) // 2, sample_rate)
    return np.frombuffer(resultado.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribir(audio_path: str, formato: str = "txt", modelo: str = "base") -> None:
    """
    Orquesta el proceso completo de transcripción de audio a texto.

    Flujo de ejecución:
        1. Valida la ruta del archivo de audio.
        2. Mejora la calidad del audio con ffmpeg.
        3. Decodifica el audio a numpy array.
        4. Transcribe con Whisper (faster-whisper).
        5. Corrige ortografía y organiza párrafos con LLaMA via Ollama.
        6. Guarda el resultado en el formato indicado.

    :param audio_path: Ruta al archivo de audio de entrada.
    :type audio_path: str
    :param formato: Formato del archivo de salida, 'txt' o 'md' (default: 'txt').
    :type formato: str
    :param modelo: Modelo Whisper a usar: tiny, base, small, medium, large (default: 'base').
    :type modelo: str
    :author: Job Adolfo Salinas Hernández
    """
    log.info("Iniciando transcripción de: %s", audio_path)
    ruta = validar_ruta(audio_path)
    ruta = mejorar_audio(ruta)
    audio = cargar_audio(ruta)

    log.info("Cargando modelo Whisper: %s", modelo)
    model = WhisperModel(modelo, device="cpu", compute_type="int8")

    log.info("Transcribiendo audio...")
    segments, info = model.transcribe(audio)
    log.info("Idioma detectado: %s (probabilidad: %.2f)", info.language, info.language_probability)

    texto = "\n".join(seg.text.strip() for seg in segments)
    log.debug("Transcripción cruda: %d caracteres", len(texto))

    salida = ruta.with_suffix(f".{formato}")

    if formato == "md":
        contenido = f"# Transcripción: {ruta.name}\n\n{texto}\n"
    else:
        contenido = texto

    contenido = limpiar_texto(contenido)

    salida.write_text(contenido, encoding="utf-8")
    log.info("Transcripción guardada en: %s", salida)


def limpiar_texto(texto: str, modelo: str = "llama3.2:latest") -> str:
    """
    Corrige la ortografía y organiza en párrafos el texto transcrito usando LLaMA via Ollama.

    El texto se trunca a MAX_CHARS_LLM caracteres antes de enviarse al modelo
    para evitar consumo ilimitado de recursos.

    :param texto: Texto crudo resultado de la transcripción.
    :type texto: str
    :param modelo: Nombre del modelo Ollama a usar (default: llama3.2:latest).
    :type modelo: str
    :returns: Texto corregido y organizado en párrafos.
    :rtype: str
    :author: Job Adolfo Salinas Hernández
    """
    log.info("Corrigiendo texto con ollama (modelo: %s)...", modelo)
    texto = texto[:MAX_CHARS_LLM]
    log.debug("Longitud del texto enviado al LLM: %d caracteres", len(texto))
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
    log.info("Corrección completada.")
    return respuesta.message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio a texto con faster-whisper")
    parser.add_argument("audio", help="Ruta del archivo de audio (.mp3, .wav, .ogg, etc.)")
    parser.add_argument("--formato", choices=["txt", "md"], default="txt", help="Formato de salida (default: txt)")
    parser.add_argument("--modelo", default="base", help="Modelo Whisper: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    transcribir(args.audio, args.formato, args.modelo)
