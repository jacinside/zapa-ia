"""Arma un MP3 recopilado con los mejores tramos de varias tomas.

Dos modos:

- FIJO (`mejor_tramo`): una tirada contigua de N ventanas por toma, la de mejor
  promedio. Simple y predecible.
- DINÁMICO (`tramos_dinamicos`): el tramo dura lo que dure la racha buena. Se
  toman todas las tiradas contiguas de ventanas sobre un umbral de score, con un
  mínimo y un tope de largo, y se permiten varias por toma. Es lo que pide
  "escuchar los temas más completos": si una zapada sostiene 4 minutos buenos,
  entran los 4.

Enganche: los cortes (inicio y fin) se pegan al beat más cercano, el crossfade
es corto, y si el tempo del tramo siguiente difiere poco del anterior se ajusta
con time-stretch para que el empalme no tropiece. Dobles y mitades de tempo se
consideran iguales (86 y 172 bpm enganchan solos).
"""
import numpy as np
import librosa
from pydub import AudioSegment

from .audio import load_mono

SR_ANALISIS = 22050


# ---------------------------------------------------------------------------
# Selección de tramos
# ---------------------------------------------------------------------------

def _contiguo(g, i0, i1, hop_s):
    """True si las filas i0..i1 (inclusive) de g son audio contiguo."""
    return abs((g.loc[i1, "start"] - g.loc[i0, "start"]) - (i1 - i0) * hop_s) < 1e-3


def mejor_tramo(g, n_win, win_s, col="score", hop_s=None):
    """Tirada de `n_win` ventanas consecutivas con mejor promedio -> (inicio, fin) en seg."""
    hop_s = win_s if hop_s is None else hop_s
    g = g.sort_values("start").reset_index(drop=True)
    if len(g) <= n_win:
        return float(g["start"].min()), float(g["start"].max() + win_s)
    roll = g[col].rolling(n_win).mean()
    # En modo veto (o con ventanas silenciosas) faltan filas, así que filas
    # consecutivas NO implican audio contiguo. Solo valen las tiradas donde el
    # tiempo transcurrido coincide con la cantidad de ventanas.
    inicio = g["start"].shift(n_win - 1)
    contiguo = (g["start"] - inicio).round(3) == round((n_win - 1) * hop_s, 3)
    roll = roll.where(contiguo)
    if not roll.notna().any():
        i = int(g[col].idxmax())                # sin tirada contigua: una sola ventana
        return float(g.loc[i, "start"]), float(g.loc[i, "start"] + win_s)
    i = int(roll.idxmax())                      # índice de la ÚLTIMA ventana de la tirada
    return float(g.loc[i - n_win + 1, "start"]), float(g.loc[i, "start"] + win_s)


def tramos_dinamicos(g, win_s, umbral, min_win=2, max_win=10, max_tramos=2,
                     col="score", hop_s=None, max_silencio=0.08):
    """Rachas contiguas de ventanas con `col >= umbral` y poco silencio.

    Devuelve lista de (inicio, fin, score_medio, tempo_mediano), de mejor a peor,
    a lo sumo `max_tramos`. Una racha más larga que `max_win` se recorta a su
    mejor sub-tirada de `max_win` ventanas.

    `max_silencio`: una ventana con más de esa fracción de silencio (split a
    -30 dB del pico) corta la racha. El score solo no lo ve: una pausa de 3 s en
    una ventana de 30 s es un 10% que pasa el umbral y se escucha como un hueco.
    """
    hop_s = win_s if hop_s is None else hop_s
    g = g.sort_values("start").reset_index(drop=True)
    sil = g["silence_ratio"].fillna(0.0) if "silence_ratio" in g else 0.0 * g[col]
    ok = (g[col] >= umbral) & (sil <= max_silencio)
    rachas, i, n = [], 0, len(g)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and _contiguo(g, j, j + 1, hop_s):
            j += 1
        if j - i + 1 >= min_win:
            rachas.append((i, j))
        i = j + 1

    out = []
    for i, j in rachas:
        if j - i + 1 > max_win:
            sub = g.loc[i:j, col].rolling(max_win).mean()
            k = int(sub.idxmax())
            i, j = k - max_win + 1, k
        bloque = g.loc[i:j]
        tempo = float(np.nanmedian(bloque["tempo"])) if "tempo" in bloque else float("nan")
        out.append((float(g.loc[i, "start"]), float(g.loc[j, "start"] + win_s),
                    float(bloque[col].mean()), tempo))
    out.sort(key=lambda t: -t[2])
    return out[:max_tramos]


# ---------------------------------------------------------------------------
# Análisis de audio para el enganche
# ---------------------------------------------------------------------------

class _Decoder:
    """Cachea el decode mono de cada archivo: varios cortes por toma no deberían
    decodificar 49 minutos cuatro veces."""

    def __init__(self):
        self._y = {}

    def mono(self, path):
        if path not in self._y:
            self._y[path] = load_mono(path, SR_ANALISIS)
        return self._y[path]


def pegar_a_beat(path, t, margen=1.5, dec=None):
    """Mueve `t` al beat más cercano dentro de +/- margen. Si no hay pulso, no toca nada."""
    try:
        y = dec.mono(path) if dec else load_mono(path, SR_ANALISIS)
        ini = max(t - margen, 0.0)
        a, b = int(ini * SR_ANALISIS), int((t + margen) * SR_ANALISIS)
        seg = y[a:b]
        if len(seg) < SR_ANALISIS:
            return t
        _, beats = librosa.beat.beat_track(y=seg, sr=SR_ANALISIS, units="time")
        if not len(beats):
            return t
        cand = beats + ini
        return float(cand[int(np.argmin(np.abs(cand - t)))])
    except Exception:
        return t


def tempo_de(path, ini, fin, dec):
    """Tempo estimado sobre el tramo real (no la mediana cuantizada del caché)."""
    try:
        y = dec.mono(path)[int(ini * SR_ANALISIS):int(fin * SR_ANALISIS)]
        if len(y) < 5 * SR_ANALISIS:
            return float("nan")
        t = librosa.beat.beat_track(y=y, sr=SR_ANALISIS)[0]
        return float(np.atleast_1d(t)[0])
    except Exception:
        return float("nan")


def ratio_tempo(t_prev, t_next):
    """Cuánto acelerar (>1) o frenar (<1) el tramo siguiente para que enganche.
    Dobles y mitades cuentan como iguales."""
    if not (t_prev > 0 and t_next > 0):
        return 1.0
    # OJO: no arrancar con mejor=1.0 — tiene distancia cero y nada lo supera,
    # así que nunca se ajustaba nada. Bug encontrado al ver que 95.7 -> 99.4 bpm
    # (3.9%) salía sin ajuste.
    cands = [t_prev / (t_next * mult) for mult in (0.5, 1.0, 2.0)]
    return min(cands, key=lambda r: abs(r - 1.0))


def _stretch(seg, rate):
    """Time-stretch de un AudioSegment (rate > 1 = más rápido), sin cambiar la altura."""
    if abs(rate - 1.0) < 1e-3:
        return seg
    full = float(1 << (8 * seg.sample_width - 1))
    y = np.array(seg.get_array_of_samples()).astype(np.float32) / full
    if seg.channels > 1:
        y = y.reshape(-1, seg.channels).T
    z = librosa.effects.time_stretch(y, rate=rate)
    if seg.channels > 1:
        z = z.T.reshape(-1)
    z = np.clip(z, -1.0, 1.0)
    data = (z * (full - 1)).astype({1: np.int8, 2: np.int16, 4: np.int32}[seg.sample_width])
    return AudioSegment(data.tobytes(), frame_rate=seg.frame_rate,
                        sample_width=seg.sample_width, channels=seg.channels)


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------

def normalizar_tramo(seg, objetivo_dbfs=-19.0, techo_pico_dbfs=-1.0):
    """Lleva el tramo a un RMS común sin pasar el techo de picos.

    Medido antes de esto: 8-11 dB de diferencia entre tramos del mismo compilado
    (grabaciones distintas). Un tramo ya comprimido al tope no se puede subir sin
    saturar: se sube hasta donde el pico lo permita. Devuelve (seg, ganancia_db).
    """
    if seg.dBFS == float("-inf"):
        return seg, 0.0
    ganancia = objetivo_dbfs - seg.dBFS
    ganancia = min(ganancia, techo_pico_dbfs - seg.max_dBFS)
    out = seg.apply_gain(ganancia)
    # Lo que importa es SUBIR lo que está bajo. Si el techo de picos frenó la
    # subida (picos altos con cuerpo bajo), un compresor suave achica los picos
    # y deja levantar el resto. Solo se aplica a los tramos que lo necesitan.
    if objetivo_dbfs - out.dBFS > 1.0:
        comp = seg.compress_dynamic_range(threshold=-12.0, ratio=4.0, attack=5.0, release=80.0)
        g2 = min(objetivo_dbfs - comp.dBFS, techo_pico_dbfs - comp.max_dBFS)
        if comp.dBFS + g2 > out.dBFS + 0.3:
            out, ganancia = comp.apply_gain(g2), float(g2)
    return out, float(ganancia)


def construir(tramos, crossfade_s=3.0, fade_borde_s=2.0, snap=True,
              ajustar_tempo=False, max_stretch=0.04, snap_fin=False,
              normalizar=True, objetivo_dbfs=-19.0):
    """tramos: lista de (ruta, inicio_s, fin_s). Devuelve (AudioSegment, usados).

    usados: lista de (ruta, inicio, fin, tempo, ratio_aplicado, ganancia_db).
    """
    dec = _Decoder()
    out = None
    usados = []
    cf = int(crossfade_s * 1000)
    t_prev = float("nan")
    for ruta, ini, fin in tramos:
        if snap:
            ini = pegar_a_beat(ruta, ini, dec=dec)
        if snap_fin:
            fin = pegar_a_beat(ruta, fin, dec=dec)
        audio = AudioSegment.from_file(ruta)
        fin = min(fin + crossfade_s, len(audio) / 1000.0)   # margen para que el crossfade no coma música
        trozo = audio[int(ini * 1000):int(fin * 1000)]
        if len(trozo) < cf * 2:
            continue
        tempo = tempo_de(ruta, ini, fin, dec)
        ratio = 1.0
        if ajustar_tempo and t_prev == t_prev and tempo == tempo:   # ambos no-NaN
            r = ratio_tempo(t_prev, tempo)
            if abs(r - 1.0) <= max_stretch:
                ratio = r
                trozo = _stretch(trozo, ratio)
        ganancia = 0.0
        if normalizar:
            trozo, ganancia = normalizar_tramo(trozo, objetivo_dbfs)
        usados.append((ruta, ini, ini + len(trozo) / 1000.0, tempo, ratio, ganancia))
        t_prev = tempo * ratio if tempo == tempo else t_prev
        if out is None:
            out = trozo.fade_in(int(fade_borde_s * 1000))
        else:
            out = out.append(trozo, crossfade=cf)
    if out is not None:
        out = out.fade_out(int(fade_borde_s * 1000))
    return out, usados


# ---------------------------------------------------------------------------
# Carátula con la lista de temas, embebida en el MP3 (tag ID3 APIC). Es la
# "foto" que muestran Drive, el celular y cualquier reproductor: sin ella se ve
# el ícono genérico. Se genera desde la lista ya armada, así que no cuesta nada.
# ---------------------------------------------------------------------------

def portada(titulo, subtitulo, items, out_png, lado=1400):
    """items: lista de (tiempo_mmss, nombre). Escribe un PNG cuadrado."""
    from PIL import Image, ImageDraw, ImageFont
    fuentes = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc"]
    def font(size, bold=False):
        for f in (fuentes if bold else fuentes[1:] + fuentes[:1]):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
        return ImageFont.load_default()
    img = Image.new("RGB", (lado, lado), (21, 24, 26))
    d = ImageDraw.Draw(img)
    m = int(lado * 0.06)
    d.rectangle([m, m, m + int(lado * 0.02), m + int(lado * 0.11)], fill=(127, 182, 194))
    d.text((m + int(lado * 0.045), m), titulo, fill=(230, 233, 231), font=font(int(lado * 0.055), True))
    d.text((m + int(lado * 0.045), m + int(lado * 0.07)), subtitulo, fill=(167, 176, 171), font=font(int(lado * 0.03)))
    # Tamaño de letra según cuántas líneas hay que meter.
    n = max(len(items), 1)
    alto_disp = lado - m * 2 - int(lado * 0.16) - int(lado * 0.05)   # deja lugar al pie
    fs = max(int(min(alto_disp / n / 1.35, lado * 0.036)), int(lado * 0.018))
    f_t, f_n = font(fs), font(fs)
    y = m + int(lado * 0.16)
    ancho_t = int(fs * 3.4)
    for t, nombre in items:
        d.text((m, y), t, fill=(127, 182, 194), font=f_t)
        nombre = nombre[:int((lado - m * 2 - ancho_t) / (fs * 0.52))]
        d.text((m + ancho_t, y), nombre, fill=(230, 233, 231), font=f_n)
        y += int(fs * 1.35)
    d.text((m, lado - m - int(lado * 0.025)), "Zapa-IA", fill=(90, 100, 96), font=font(int(lado * 0.022)))
    img.save(out_png, "PNG")
    return out_png


def embeber_portada(mp3, png):
    """Agrega la carátula sin recodificar el audio (copia el stream)."""
    import os
    import subprocess
    tmp = mp3 + ".tmp.mp3"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-i", png,
                        "-map", "0:a", "-map", "1", "-c", "copy", "-id3v2_version", "3",
                        "-metadata:s:v", "title=Album cover", "-metadata:s:v", "comment=Cover (front)",
                        tmp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    os.replace(tmp, mp3)


# ---------------------------------------------------------------------------
# Versión VIDEO (MP4) del compilado: la app de Drive no muestra la carátula de
# un MP3 (ícono fijo de auriculares), pero sí reproduce video. Se arma un cuadro
# por tramo con la lista completa y el tramo actual resaltado, se encadenan con
# sus duraciones y se les pega el audio del MP3 sin recodificar.
# ---------------------------------------------------------------------------

def _fondo(imagen, ancho, alto, oscurecer=0.30, blur=6):
    """Foto de fondo recortada a 16:9, desenfocada y oscurecida para que el texto se lea."""
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    try:
        im = Image.open(imagen)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.fit(im, (ancho, alto), method=Image.LANCZOS, centering=(0.5, 0.4))
        im = im.filter(ImageFilter.GaussianBlur(blur))
        return ImageEnhance.Brightness(im).enhance(oscurecer)
    except Exception:
        return Image.new("RGB", (ancho, alto), (21, 24, 26))


def cuadro(titulo, subtitulo, items, resaltar, out_png, ancho=1280, alto=720, fondo=None):
    """items: [(mmss, nombre, anio_o_None)]; resaltar: índice del tramo actual;
    fondo: ruta de una foto (opcional)."""
    from PIL import Image, ImageDraw, ImageFont
    fuentes = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    def font(size, bold=False):
        for f in (fuentes if bold else fuentes[1:] + fuentes[:1]):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
        return ImageFont.load_default()
    img = _fondo(fondo, ancho, alto) if fondo else Image.new("RGB", (ancho, alto), (21, 24, 26))
    d = ImageDraw.Draw(img, "RGBA")
    m = int(alto * 0.06)
    d.rectangle([m, m, m + 10, m + int(alto * 0.10)], fill=(127, 182, 194))
    d.text((m + 26, m - 4), titulo, fill=(240, 242, 240), font=font(int(alto * 0.055), True))
    d.text((m + 26, m + int(alto * 0.06)), subtitulo, fill=(190, 198, 194), font=font(int(alto * 0.028)))
    d.text((ancho - m - int(alto * 0.13), m), "Zapa-IA", fill=(140, 150, 146), font=font(int(alto * 0.024)))
    n = max(len(items), 1)
    cols = 1 if n <= 11 else 2
    filas = -(-n // cols)
    alto_disp = alto - m - int(alto * 0.17) - int(alto * 0.22)      # deja lugar al visualizador
    fs = max(int(min(alto_disp / filas / 1.3, alto * 0.036)), int(alto * 0.02))
    f_n, f_b, f_a = font(fs), font(fs, True), font(int(fs * 0.8))
    col_w = (ancho - m * 2) // cols
    y0 = m + int(alto * 0.17)
    for i, it in enumerate(items):
        t, nombre, anio = (it + (None,))[:3]
        c, r = divmod(i, filas)
        x, y = m + c * col_w, y0 + r * int(fs * 1.3)
        activo = i == resaltar
        if activo:
            d.rounded_rectangle([x - 8, y - 3, x + col_w - 28, y + fs + 5], radius=6, fill=(43, 93, 107, 230))
        maxc = int((col_w - fs * 3.6 - 28 - (fs * 2.6 if anio else 0)) / (fs * 0.52))
        d.text((x, y), t, fill=(255, 255, 255) if activo else (127, 182, 194), font=f_b if activo else f_n)
        nx = x + int(fs * 3.4)
        d.text((nx, y), nombre[:maxc], fill=(255, 255, 255) if activo else (215, 220, 217),
               font=f_b if activo else f_n)
        if anio:
            w = d.textlength(nombre[:maxc], font=f_b if activo else f_n)
            d.text((nx + w + int(fs * 0.5), y + int(fs * 0.15)), str(anio),
                   fill=(200, 210, 206) if activo else (150, 160, 156), font=f_a)
    img.save(out_png, "PNG")


def video_compilado(mp3, titulo, subtitulo, posiciones, out_mp4, workdir,
                    imagenes_dir=None, visualizador=True, semilla=None):
    """posiciones: [(segundo_inicio, nombre[, anio])] en orden. Genera el MP4.

    imagenes_dir: carpeta con fotos; se elige una distinta por tramo (al azar con
    semilla fija, así el mismo compilado da el mismo video). visualizador: banda
    de onda al pie que se mueve con el audio (showwaves de ffmpeg).
    """
    import glob
    import os
    import random
    import subprocess
    from pydub.utils import mediainfo
    total = float(mediainfo(mp3)["duration"])
    items = [(_mmss_(p[0]),) + tuple(p[1:3]) for p in posiciones]
    fotos = []
    if imagenes_dir:
        fotos = sorted(f for f in glob.glob(os.path.join(imagenes_dir, "*"))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        random.Random(semilla if semilla is not None else titulo).shuffle(fotos)
    os.makedirs(workdir, exist_ok=True)
    lista = os.path.join(workdir, "cuadros.txt")
    with open(lista, "w", encoding="utf-8") as fh:
        for i, p in enumerate(posiciones):
            t = p[0]
            fin = posiciones[i + 1][0] if i + 1 < len(posiciones) else total
            png = os.path.join(workdir, f"cuadro_{i:03d}.png")
            cuadro(titulo, subtitulo, items, i, png, fondo=fotos[i % len(fotos)] if fotos else None)
            fh.write(f"file '{os.path.abspath(png)}'\nduration {max(fin - t, 0.5):.3f}\n")
        fh.write(f"file '{os.path.abspath(png)}'\n")     # el concat exige repetir el último
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lista, "-i", mp3]
    if visualizador:
        cmd += ["-filter_complex",
                "[1:a]showwaves=s=1280x140:mode=cline:colors=cadetblue:rate=24:scale=sqrt,"
                "format=rgba,colorchannelmixer=aa=0.55[w];[0:v][w]overlay=0:575:shortest=1[v]",
                "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                "-r", "24"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-tune", "stillimage",
                "-preset", "veryfast", "-r", "4"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", out_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:400])
    return out_mp4


def _mmss_(s):
    return "%d:%02d" % (int(s) // 60, int(s) % 60)
