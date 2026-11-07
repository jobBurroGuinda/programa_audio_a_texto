import argparse
import logging
import subprocess
import warnings
from pathlib import Path
import numpy as np
import ollama
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.*")
    from df.enhance import enhance, init_df, load_audio, save_audio
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FORMATOS_AUDIO = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma"}
ATEN_LIM_DB = 15
PAUSA_PARRAFO = 2.0
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



def mejorar_audio(ruta: Path) -> Path:
    """
    Mejora la calidad del audio usando DeepFilterNet para eliminar ruido de fondo
    y realzar la voz. Usa atten_lim_db para controlar la agresividad de la reducción
    de ruido y preservar la voz. El resultado se guarda como MP3 con el sufijo '_mejorado'.

    :param ruta: Ruta al archivo de audio original.
    :type ruta: Path
    :returns: Ruta al archivo de audio mejorado en formato MP3.
    :rtype: Path
    :raises RuntimeError: Si falla durante el procesamiento.
    :author: Job Adolfo Salinas Hernández
    """
    destino = ruta.with_stem(ruta.stem + "_mejorado").with_suffix(".mp3")
    wav_tmp = ruta.with_stem(ruta.stem + "_tmp").with_suffix(".wav")
    wav_mejorado = ruta.with_stem(ruta.stem + "_mejorado_tmp").with_suffix(".wav")
    log.info("Mejorando calidad del audio con DeepFilterNet: %s", ruta.name)

    try:
        cmd_to_wav = [
            "ffmpeg", "-nostdin", "-y", "-i", str(ruta),
            "-ar", "48000", "-ac", "1", str(wav_tmp),
        ]
        resultado = subprocess.run(cmd_to_wav, capture_output=True, shell=False)
        if resultado.returncode != 0:
            raise RuntimeError(f"ffmpeg falló al convertir a WAV: {resultado.stderr.decode()}")

        model, df_state, _ = init_df()
        audio, _ = load_audio(str(wav_tmp), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio, atten_lim_db=ATEN_LIM_DB)
        save_audio(str(wav_mejorado), enhanced, df_state.sr())

        cmd_to_mp3 = [
            "ffmpeg", "-nostdin", "-y", "-i", str(wav_mejorado),
            "-q:a", "2", str(destino),
        ]
        resultado = subprocess.run(cmd_to_mp3, capture_output=True, shell=False)
        if resultado.returncode != 0:
            raise RuntimeError(f"ffmpeg falló al convertir a MP3: {resultado.stderr.decode()}")
    finally:
        wav_tmp.unlink(missing_ok=True)
        wav_mejorado.unlink(missing_ok=True)

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
        2. Mejora la calidad del audio con DeepFilterNet.
        3. Decodifica el audio a numpy array.
        4. Transcribe con Whisper (faster-whisper).
        5. Agrupa segmentos en párrafos por pausas de silencio.
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

    log.info("Transcribiendo audio en español...")
    segments, info = model.transcribe(audio, language="es", task="transcribe")
    log.info("Idioma: %s (probabilidad: %.2f)", info.language, info.language_probability)

    texto_crudo = "\n".join(seg.text.strip() for seg in segments)
    log.debug("Transcripción: %d caracteres", len(texto_crudo))

    salida_txt = ruta.with_suffix(".txt")
    salida_txt.write_text(texto_crudo, encoding="utf-8")
    log.info("Transcripción original guardada en: %s", salida_txt)

    texto_organizado = organizar_parrafos(texto_crudo)
    salida = ruta.with_stem(ruta.stem + "_organizado").with_suffix(f".{formato}")
    contenido = f"# Transcripción: {ruta.name}\n\n{texto_organizado}\n" if formato == "md" else texto_organizado
    salida.write_text(contenido, encoding="utf-8")
    log.info("Transcripción organizada guardada en: %s", salida)


def organizar_parrafos(texto: str, modelo: str = "llama3.2:latest") -> str:
    """
    Organiza el texto transcrito en párrafos usando LLaMA via Ollama.
    No corrige ni agrega nada, solo organiza el texto en párrafos.

    :param texto: Texto crudo resultado de la transcripción.
    :type texto: str
    :param modelo: Nombre del modelo Ollama a usar (default: llama3.2:latest).
    :type modelo: str
    :returns: Texto organizado en párrafos.
    :rtype: str
    :author: Job Adolfo Salinas Hernández
    """
    log.info("Organizando párrafos con ollama (modelo: %s)...", modelo)
    texto = texto[:MAX_CHARS_LLM]
    prompt = (
        "Organiza el siguiente texto en párrafos. "
        "NO corrijas ortografía. NO agregues palabras. NO quites palabras. "
        "Usa EXACTAMENTE las mismas palabras del texto original, solo agrupa las oraciones en párrafos con saltos de línea. "
        "Devuelve únicamente el texto organizado, sin comentarios, sin explicaciones, sin encabezados.\n\n"
        f"{texto}"
    )
    respuesta = ollama.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 4096},
    )
    log.info("Organización completada.")
    return respuesta.message.content


    parser = argparse.ArgumentParser(description="Transcribe audio a texto con faster-whisper")
    parser.add_argument("audio", help="Ruta del archivo de audio (.mp3, .wav, .ogg, etc.)")
    parser.add_argument("--formato", choices=["txt", "md"], default="txt", help="Formato de salida (default: txt)")
    parser.add_argument("--modelo", default="base", help="Modelo Whisper: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    transcribir(args.audio, args.formato, args.modelo)
