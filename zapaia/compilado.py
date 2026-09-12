"""Arma un MP3 recopilado con los mejores tramos de varias tomas.

La unidad no es la ventana suelta de 30 s sino una TIRADA CONTIGUA de ventanas:
un tramo de 90 s que se sostiene bueno vale más que un pico aislado.

Los cortes se pegan al beat más cercano (si hay pulso detectable) para que el
empalme no caiga a contratiempo, y se cruzan con crossfade.
"""
import numpy as np
import librosa
from pydub import AudioSegment

from .audio import load_mono

SR_ANALISIS = 22050


def mejor_tramo(g, n_win, win_s, col="score"):
    """Tirada de `n_win` ventanas consecutivas con mejor promedio -> (inicio, fin) en seg."""
    g = g.sort_values("start").reset_index(drop=True)
    if len(g) <= n_win:
        return float(g["start"].min()), float(g["start"].max() + win_s)
    roll = g[col].rolling(n_win).mean()
    # En modo veto (o con ventanas silenciosas) faltan filas, así que filas
    # consecutivas NO implican audio contiguo. Solo valen las tiradas donde el
    # tiempo transcurrido coincide con la cantidad de ventanas.
    inicio = g["start"].shift(n_win - 1)
    contiguo = (g["start"] - inicio).round(3) == round((n_win - 1) * win_s, 3)
    roll = roll.where(contiguo)
    if not roll.notna().any():
        i = int(g[col].idxmax())                # sin tirada contigua: una sola ventana
        return float(g.loc[i, "start"]), float(g.loc[i, "start"] + win_s)
    i = int(roll.idxmax())                      # índice de la ÚLTIMA ventana de la tirada
    return float(g.loc[i - n_win + 1, "start"]), float(g.loc[i, "start"] + win_s)


def pegar_a_beat(path, t, margen=1.5):
    """Mueve `t` al beat más cercano dentro de +/- margen. Si no hay pulso, no toca nada."""
    try:
        ini = max(t - margen, 0.0)
        y = load_mono(path, SR_ANALISIS)
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


def construir(tramos, crossfade_s=3.0, fade_borde_s=2.0, snap=True):
    """tramos: lista de (ruta, inicio_s, fin_s). Devuelve un AudioSegment."""
    out = None
    usados = []
    cf = int(crossfade_s * 1000)
    for ruta, ini, fin in tramos:
        if snap:
            ini = pegar_a_beat(ruta, ini)
        audio = AudioSegment.from_file(ruta)
        # Margen extra al final para que el crossfade no se coma música.
        trozo = audio[int(ini * 1000):int(min(fin + crossfade_s, len(audio) / 1000.0) * 1000)]
        if len(trozo) < cf * 2:
            continue
        usados.append((ruta, ini, ini + len(trozo) / 1000.0))
        if out is None:
            out = trozo.fade_in(int(fade_borde_s * 1000))
        else:
            out = out.append(trozo, crossfade=cf)
    if out is not None:
        out = out.fade_out(int(fade_borde_s * 1000))
    return out, usados
