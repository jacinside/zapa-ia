"""Features por ventana.

Todo lo que sea comparable entre tomas tiene que ser *independiente del tempo y
del nivel de grabación*. Por eso se usan coeficientes de variación y fracciones
de beat en vez de desvíos en segundos.
"""
import numpy as np
import librosa

EPS = 1e-10

# Corte de bandas para medir enganche ritmico (Hz) y resolucion mel.
F_LOW, F_HIGH, N_MELS = 250.0, 2000.0, 64


def _db(x):
    return float(20.0 * np.log10(max(float(x), EPS)))


def _tempo_scalar(t):
    a = np.atleast_1d(np.asarray(t, dtype=float))
    return float(a[0]) if a.size else 0.0


def sound_features(y, sr):
    """Calidad técnica de la grabación: nivel, ruido de fondo, clipping, brillo."""
    f = {}
    rms = librosa.feature.rms(y=y)[0]
    f["rms_db"] = _db(np.mean(rms))
    # Piso de ruido vs picos: el mejor proxy de "sala limpia" que tenemos.
    p10, p90 = np.percentile(rms, 10), np.percentile(rms, 90)
    f["noise_floor_db"] = _db(p10)
    f["snr_proxy_db"] = _db(p90) - _db(p10)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    f["crest_db"] = _db(peak) - f["rms_db"]
    f["clip_ratio"] = float(np.mean(np.abs(y) > 0.95))
    f["dc_offset"] = float(abs(np.mean(y)))

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    f["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
    f["rolloff85"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]))
    # Planitud espectral: alto = ruido/hiss, bajo = tonal.
    f["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))

    intervals = librosa.effects.split(y, top_db=30)
    voiced = int(sum(b - a for a, b in intervals))
    f["silence_ratio"] = float(1.0 - voiced / max(len(y), 1))
    f["is_silent"] = int(f["rms_db"] <= -60.0 or peak <= 1e-4)
    return f


def exec_features(y, sr):
    """Ejecución: ¿hay pulso claro y la banda está tocando junta?"""
    nan = float("nan")
    # NaN, no defaults: un 0.25 inventado entra al percentil como si fuera medición.
    f = {
        "tempo": nan, "n_beats": 0, "ibi_cv": nan, "tempo_drift": nan,
        "onset_dev_mad": nan, "pulse_clarity": nan, "band_sync": nan,
        "band_lag_ms": nan, "onset_rate": nan, "low_energy_ratio": nan,
        "dyn_range_db": nan,
    }
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, units="time")
    f["tempo"] = _tempo_scalar(tempo)
    f["n_beats"] = int(len(beats))

    if len(beats) >= 6:
        ibi = np.diff(beats)
        mean_ibi = float(np.mean(ibi))
        # Coeficiente de variación: adimensional, comparable entre tempos.
        f["ibi_cv"] = float(np.std(ibi) / max(mean_ibi, EPS))
        # Drift real = pendiente del IBI en el tiempo, como cambio relativo total.
        k = np.arange(len(ibi))
        slope = float(np.polyfit(k, ibi, 1)[0])
        f["tempo_drift"] = float(abs(slope) * len(ibi) / max(mean_ibi, EPS))

        onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv,
                                            units="time", backtrack=True)
        if len(onsets) >= 4:
            f["onset_rate"] = float(len(onsets) / (len(y) / sr))
            # Solo onsets dentro de la grilla: fuera de ella la fase no está definida
            # y daba valores > 0.5, imposibles para una distancia al beat.
            inside = onsets[(onsets >= beats[0]) & (onsets < beats[-1])]
            if len(inside) >= 4:
                j = np.clip(np.searchsorted(beats, inside) - 1, 0, len(beats) - 2)
                # Desviación en FRACCIÓN DE BEAT, no en segundos.
                phase = (inside - beats[j]) / np.maximum(ibi[j], EPS)
                phase = np.clip(phase, 0.0, 1.0)
                phase = np.minimum(phase, 1.0 - phase)  # distancia al beat más cercano
                f["onset_dev_mad"] = float(np.median(phase))

    if len(oenv) > 8:
        plp = librosa.beat.plp(onset_envelope=oenv, sr=sr)
        f["pulse_clarity"] = float(np.mean(plp))

    # Enganche ritmico: onsets de banda baja (bombo/bajo) vs alta (platillos/caja).
    # Antes esto se hacia con HPSS, que costaba el 74% del pipeline y era inestable
    # (Spearman 0.68 al cambiar el kernel de la mediana). Por bandas es 12x mas
    # rapido, estable (0.94-0.99 ante cambios de corte) y mas directo de leer.
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=sr // 2)
    hz = librosa.mel_frequencies(n_mels=N_MELS, fmax=sr // 2)
    lo_i = int(np.searchsorted(hz, F_LOW))
    hi_i = int(np.searchsorted(hz, F_HIGH))
    envs = librosa.onset.onset_strength_multi(
        S=librosa.power_to_db(M), sr=sr, channels=[0, lo_i, hi_i, N_MELS])
    a, b = envs[0], envs[2]
    if len(a) >= 32 and len(b) >= 32:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        if denom > EPS:
            xc = np.correlate(a, b, mode="full") / denom
            hop_s = 512.0 / sr
            ml = max(int(round(0.25 / hop_s)), 1)
            c = len(a) - 1
            seg = xc[max(c - ml, 0):c + ml + 1]
            if len(seg):
                k = int(np.argmax(seg))
                f["band_sync"] = float(seg[k])
                f["band_lag_ms"] = float(abs(k - ml) * hop_s * 1000.0)
    Mp = M.sum(axis=1)
    tot = float(Mp.sum())
    f["low_energy_ratio"] = float(Mp[:lo_i].sum() / tot) if tot > EPS else float("nan")

    rms = librosa.feature.rms(y=y)[0]
    f["dyn_range_db"] = _db(np.percentile(rms, 95)) - _db(np.percentile(rms, 5))
    return f


def window_features(y, sr):
    f = sound_features(y, sr)
    f.update(exec_features(y, sr))
    f.update(harmony_features(y, sr))
    f.update(pifie_features(y, sr))
    # intonacion_features NO se usa: probada sobre 3 tramos con voz desafinada /
    # pifies de viola contra 3 tomas buenas, no los separa (5-13 cents vs 8-19).
    # pYIN sigue la altura dominante de la mezcla (bajo, rítmica: afinados), no
    # la voz; y un pifie es una nota equivocada pero afinada. Requiere separar
    # la voz (Demucs) antes de volver a intentarlo. Ver CLAUDE.md.
    f.update(grilla_features(y, sr))
    f["beat_strength"] = beat_strength(y, sr)
    return f


def signature(y, sr, n_mfcc=20):
    """Huella compacta (32-d) para detectar duplicados / re-encodes."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=2048)
    c = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048)
    v = np.concatenate([m.mean(axis=1), c.mean(axis=1)]).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


# ---------------------------------------------------------------------------
# "Bien tocada": afinación y coherencia tonal.
# Una banda desafinada o zapando sin centro tonal suena mal por más que el
# tempo sea perfecto. Nada de esto lo capturaba el score de timing.
# ---------------------------------------------------------------------------

# Perfiles Krumhansl-Schmuckler, para estimar cuán clara es la tonalidad.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _key_clarity(chroma_mean):
    """Máxima correlación contra las 24 tonalidades. Alto = hay centro tonal."""
    v = chroma_mean - chroma_mean.mean()
    nv = np.linalg.norm(v)
    if nv < EPS:
        return float("nan")
    best = -1.0
    for prof in (_KS_MAJOR, _KS_MINOR):
        p = prof - prof.mean()
        p /= np.linalg.norm(p)
        for k in range(12):
            best = max(best, float(np.dot(v / nv, np.roll(p, k))))
    return best


def harmony_features(y, sr):
    """Afinación, centro tonal y movimiento armónico."""
    f = {}
    # Desviación de afinación en fracción de semitono. Guitarra desafinada = toma mala.
    try:
        f["tuning_dev"] = float(abs(librosa.estimate_tuning(y=y, sr=sr)))
    except Exception:
        f["tuning_dev"] = float("nan")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
    cm = chroma.mean(axis=1)
    total = cm.sum()
    cn = cm / total if total > EPS else cm
    for i in range(12):
        f["chroma_%d" % i] = float(cn[i])

    f["key_clarity"] = _key_clarity(cm)
    # Entropía de croma: baja = foco tonal, alta = ruido o todo a la vez.
    p = cn[cn > 0]
    f["chroma_entropy"] = float(-np.sum(p * np.log2(p)) / np.log2(12)) if len(p) else float("nan")
    # Flujo armónico: cuánto se mueve la armonía (ni estático ni caótico).
    if chroma.shape[1] > 1:
        d = np.diff(chroma, axis=1)
        f["harmonic_flux"] = float(np.mean(np.linalg.norm(d, axis=0)))
    else:
        f["harmonic_flux"] = float("nan")

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    f["spectral_contrast"] = float(np.mean(contrast))
    return f


def beat_strength(y, sr):
    """¿Los beats están realmente acentuados, o el pulso es inferido?"""
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        _, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, units="frames")
        if len(bf) < 4 or np.mean(oenv) < EPS:
            return float("nan")
        bf = bf[bf < len(oenv)]
        if len(bf) < 4:
            return float("nan")
        return float(np.mean(oenv[bf]) / np.mean(oenv))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# "Pifies": notas equivocadas.
#
# Punto ciego que ninguna otra dimensión cubría. timing mide CUÁNDO suena una
# nota, afinacion mide si el instrumento está afinado respecto a A440 — ninguna
# mira si la nota es la CORRECTA. Un pifie perfectamente a tiempo con un bajo
# bien afinado puntuaba alto.
#
# Idea: una nota equivocada es, casi por definición, energía en una clase de
# altura que no pertenece a la tonalidad que viene sonando. Se estima la
# tonalidad local (contexto de ~8 s) y se mide cuánta energía cae afuera.
# ---------------------------------------------------------------------------

_MASK_MAY = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
_MASK_MEN = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)


def _perfiles_tonalidad():
    """24 perfiles (12 mayores + 12 menores) normalizados, con su máscara diatónica."""
    P, M = [], []
    for prof, mask in ((_KS_MAJOR, _MASK_MAY), (_KS_MINOR, _MASK_MEN)):
        p = prof - prof.mean()
        p = p / np.linalg.norm(p)
        for k in range(12):
            P.append(np.roll(p, k))
            M.append(np.roll(mask, k))
    return np.array(P), np.array(M)


_KEY_P, _KEY_M = _perfiles_tonalidad()


def pifie_features(y, sr, ctx_s=8.0, hop=512):
    """Energía fuera de la tonalidad local. Vectorizado sobre todos los frames."""
    f = {"fuera_tono_med": float("nan"), "fuera_tono_p95": float("nan"),
         "fuera_tono_picos": float("nan")}
    try:
        C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        s = C.sum(axis=0, keepdims=True)
        C = C / np.maximum(s, EPS)                       # 12 x n
        n = C.shape[1]
        if n < 16:
            return f
        # Contexto: promedio móvil del croma a lo largo del tiempo.
        w = max(int(ctx_s * sr / hop), 8)
        k = np.ones(w) / w
        ctx = np.vstack([np.convolve(C[i], k, mode="same") for i in range(12)])
        v = ctx - ctx.mean(axis=0, keepdims=True)
        nv = np.linalg.norm(v, axis=0, keepdims=True)
        v = v / np.maximum(nv, EPS)
        corr = _KEY_P @ v                                 # 24 x n
        mejor = np.argmax(corr, axis=0)                   # tonalidad por frame
        fuera = np.einsum("ij,ji->i", (1.0 - _KEY_M[mejor]), C)
        f["fuera_tono_eventos"] = tonal_eventos(fuera, hop, sr)
        f["fuera_tono_med"] = float(np.mean(fuera))
        # p95 capta PICOS: un pifie aislado se diluye en la media.
        f["fuera_tono_p95"] = float(np.percentile(fuera, 95))
        thr = np.percentile(fuera, 98)
        f["fuera_tono_picos"] = float(np.mean(fuera >= thr) * len(fuera) * hop / sr)
    except Exception as e:
        print(f"pifie_features error: {e}", flush=True)
    return f


# ---------------------------------------------------------------------------
# Afinación MELÓDICA (voz / guitarra líder), a diferencia de tuning_dev, que
# es el offset global de la mezcla y lo domina lo que más energía tonal tiene
# (guitarras rítmicas, bajo). Una voz medio tono abajo sobre instrumentos
# afinados no mueve tuning_dev. Caso real: sisterborrachosa 1:00-2:31 tenía
# tuning_dev en el percentil 96 con la voz desafinada.
#
# Se sigue la altura predominante en la banda 80-1000 Hz con pYIN y se mide
# el desvío en cents a la nota más cercana (corregido por la afinación global
# de la ventana). Vibrato y bends lo inflan un poco; una voz desafinada o una
# viola pifiando lo inflan mucho y de forma sostenida.
# ---------------------------------------------------------------------------

def intonacion_features(y, sr, fmin=80.0, fmax=1000.0, hop=512):
    from scipy.signal import butter, sosfiltfilt
    f = {"inton_mad": float("nan"), "inton_p90": float("nan"), "inton_voiced": float("nan")}
    try:
        sos = butter(4, [fmin * 0.9, min(fmax * 1.5, sr / 2 - 1)], btype="band", fs=sr, output="sos")
        yb = sosfiltfilt(sos, y).astype(np.float32)
        f0, vflag, vprob = librosa.pyin(yb, fmin=fmin, fmax=fmax, sr=sr,
                                        frame_length=2048, hop_length=hop)
        ok = np.isfinite(f0) & (vprob >= 0.6)
        f["inton_voiced"] = float(np.mean(ok))
        if ok.sum() < 20:
            return f
        try:
            off = float(librosa.estimate_tuning(y=yb, sr=sr)) * 100.0   # cents
        except Exception:
            off = 0.0
        cents = 1200.0 * np.log2(f0[ok] / 440.0) - off
        dev = np.abs(((cents + 50.0) % 100.0) - 50.0)                # distancia a la nota
        f["inton_mad"] = float(np.median(dev))
        f["inton_p90"] = float(np.percentile(dev, 90))
    except Exception as e:
        print(f"intonacion_features error: {e}", flush=True)
    return f


def grilla_features(y, sr, subdiv=4):
    """Timing contra una grilla de subdivisiones, no contra el beat principal.

    onset_dev_mad castigaba la síncopa: un ataque en el "&" daba desvío 0.5.
    Acá el desvío es a la posición métrica más cercana (16avos con subdiv=4), y
    `grid_consistency` mide si los ataques caen siempre en las mismas posiciones
    (síncopa tight = histograma concentrado) o en cualquier lado (timing flojo =
    histograma plano). Ver docs/review-2026-09-12.md §1.3.
    """
    f = {"onset_dev_grid": float("nan"), "grid_consistency": float("nan"), "tempo_cont": float("nan")}
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        # Tempo continuo: mediana de la estimación por frame, sin la grilla de beat_track.
        tl = librosa.feature.tempo(onset_envelope=oenv, sr=sr, aggregate=None)
        tl = tl[np.isfinite(tl) & (tl > 0)]
        if len(tl):
            f["tempo_cont"] = float(np.median(tl))
        _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, units="time")
        if len(beats) < 6:
            return f
        onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, units="time", backtrack=True)
        inside = onsets[(onsets >= beats[0]) & (onsets < beats[-1])]
        if len(inside) < 6:
            return f
        ibi = np.diff(beats)
        j = np.clip(np.searchsorted(beats, inside) - 1, 0, len(beats) - 2)
        phase = np.clip((inside - beats[j]) / np.maximum(ibi[j], EPS), 0.0, 1.0)
        pos = phase * subdiv
        f["onset_dev_grid"] = float(np.median(np.abs(pos - np.round(pos))))   # 0..0.5 de subdivisión
        # Consistencia: entropía normalizada del histograma de fases (16 bins).
        h, _ = np.histogram(phase, bins=16, range=(0.0, 1.0))
        p = h / max(h.sum(), 1)
        p = p[p > 0]
        f["grid_consistency"] = float(1.0 - (-np.sum(p * np.log(p)) / np.log(16)))
    except Exception as e:
        print(f"grilla_features error: {e}", flush=True)
    return f


def tonal_eventos(fuera, hop, sr, umbral=0.55, min_dur_s=0.15):
    """Excursiones sobre un umbral ABSOLUTO (no el percentil 98 de la ventana,
    que marcaba el 2% de frames por construcción). Devuelve eventos por minuto."""
    if fuera is None or len(fuera) == 0:
        return float("nan")
    m = fuera >= umbral
    ev, i, n = 0, 0, len(m)
    minf = max(int(min_dur_s * sr / hop), 1)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if j - i + 1 >= minf:
                ev += 1
            i = j + 1
        else:
            i += 1
    return float(ev / (n * hop / sr / 60.0))
