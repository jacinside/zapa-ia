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

def construir(tramos, crossfade_s=3.0, fade_borde_s=2.0, snap=True,
              ajustar_tempo=False, max_stretch=0.04, snap_fin=False):
    """tramos: lista de (ruta, inicio_s, fin_s). Devuelve (AudioSegment, usados).

    usados: lista de (ruta, inicio, fin, tempo, ratio_aplicado).
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
        usados.append((ruta, ini, ini + len(trozo) / 1000.0, tempo, ratio))
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
