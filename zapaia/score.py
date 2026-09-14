"""Normalización robusta y combinación de features en scores.

Dos niveles:
  - VENTANA (30 s): sonido, timing, groove, afinación, tonal_outlier. Se normalizan
    por percentil contra todas las ventanas del corpus.
  - ARCHIVO: mediana de las dimensiones de ventana + las que solo existen a nivel
    archivo (desarrollo, creatividad: cómo evoluciona la zapada en el tiempo).

Dos composites, y un perfil que elige cuál manda:
  - ejecucion = ¿qué tan bien está tocado?   (timing, groove, afinacion, tonal_outlier)
  - interes   = ¿hay acá una idea que vale?  (creatividad, desarrollo)
Son problemas distintos: una zapada con pifies y tempo flojo puede tener el mejor
riff del archivo. Ver docs/review-2026-09-12.md §2.

Regla de oro: NINGUNA feature entra a una suma ponderada sin normalizar antes.
La versión vieja sumaba centroide (~2000) con RMS (~0.05) y el peso 0.4 del RMS
era decorativo: el score era el centroide y nada más.
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Dimensiones de VENTANA. feature -> (dirección, peso). +1 = más alto es mejor.
# --------------------------------------------------------------------------

# Calidad técnica de captura. En este corpus casi no discrimina calidad musical
# (misma banda, misma sala), así que existe sobre todo para el modo 'veto'.
SONIDO = {
    "flatness":       (-1, 0.40),   # espectro plano = hiss / ruido
    "silence_ratio":  (-1, 0.25),   # tramos muertos
    "clip_ratio":     (-1, 0.20),   # saturación
    "crest_db":       (+1, 0.15),   # margen de picos (poco aplastado)
    # FUERA snr_proxy_db (0.35) y dyn_range_db (0.10): correlacionaban 0.979 entre
    # sí — eran la MISMA medición pesando 0.45 — y no medían ruido sino variación
    # de nivel dentro de la ventana. Resultado: las ventanas de "mejor sonido"
    # tenían 31% de silencio y estaban 14 dB más bajas. Premiaban el silencio
    # mientras silence_ratio intentaba castigarlo.
    # Medir ruido de verdad (hiss en los frames callados) pide cambiar features.py.
}

# ¿Toca preciso? Estabilidad y alineación rítmica.
TIMING = {
    "ibi_cv":         (-1, 0.35),   # estabilidad de tempo (adimensional)
    "tempo_drift":    (-1, 0.25),   # deriva sostenida
    "onset_dev_mad":  (-1, 0.40),   # desvío al beat, en fracción de beat
    # Historia de este peso, porque es una lección:
    # - P0 lo bajó de 0.40 a 0.15 por estadística: mediana del corpus 0.227 contra
    #   0.25 de fase uniforme al azar, 74% de ventanas >= 0.20 -> "casi ruido".
    # - Con etiquetas humanas (5 tomas elegidas para mezclar + 2 que el usuario
    #   descartó por "voz desafinada, pifies de viola") volver a 0.40 SUBE la
    #   mediana de los positivos (82.9 -> 89.1) y BAJA un negativo (92.6 -> 79.7):
    #   los pifies dejan ataques fuera de grilla, y esta feature los olía.
    # Criterio 5 del review (correlación con humanos) le gana al 3 (estabilidad).
    # El reemplazo por grilla de subdivisiones (grilla_features) está en
    # features.py y entra en el próximo reprocesamiento.
}

# ¿Hay pulso y la banda está enganchada?
GROOVE = {
    "pulse_clarity":  (+1, 0.40),   # ¿hay groove reconocible? (PLP)
    "band_sync":      (+1, 0.35),   # bombo/bajo enganchados con platillos/caja
    "beat_strength":  (+1, 0.25),   # los beats acentuados de verdad, no inferidos
}

# ¿Suena afinado y con centro tonal? Una banda desafinada suena mal aunque el
# tempo sea perfecto, y eso el score de timing no lo ve.
AFINACION = {
    "tuning_dev":     (-1, 0.55),   # desviación de afinación (fracción de semitono)
    "key_clarity":    (+1, 0.45),   # hay tonalidad, no ruido atonal
    # chroma_entropy salió de acá: correlaciona 0.69 con harmonic_flux, o sea que
    # medía movimiento armónico y no foco tonal. Quedó como diagnóstico.
    # tuning_dev es estable e independiente pero NO está validada semánticamente
    # (estimate_tuning sobre una mezcla). Secundaria hasta tener pares humanos.
}

# Energía de croma FUERA de la tonalidad local. Antes se llamaba NOTAS ("pifies"),
# y el nombre era una hipótesis, no una medición: correlaciona +0.50 con flatness
# y -0.56 con spectral_contrast, o sea que mide contenido de banda ancha
# (distorsión, platillos, ruido) tanto como notas equivocadas; y "fuera de escala"
# también son cromatismos, blue notes y disonancia intencional. El usuario
# validó por oído que los picos coinciden con pifies reales, así que sirve como
# señal de EJECUCIÓN secundaria. En 'interes' pesa cero: penalizaría justo lo que
# se quiere rescatar.
TONAL_OUTLIER = {
    "fuera_tono_med":   (-1, 0.55),  # energía fuera de la tonalidad local
    "fuera_tono_p95":   (-1, 0.45),  # picos: un pifie aislado se diluye en la media
    # fuera_tono_picos salió: era el 2% de frames por construcción (percentil 98
    # dentro de la misma ventana) -> 23 valores únicos en 11.205 ventanas.
}

# --------------------------------------------------------------------------
# Dimensiones de ARCHIVO: solo tienen sentido comparando ventanas entre sí.
# --------------------------------------------------------------------------

# ¿La zapada se sostiene? Nivel archivo.
DESARROLLO = {
    "tempo_consistency": (+1, 0.55),   # sostener el tempo 10 min es mérito real
    "arco_dinamico":     (+1, 0.45),   # partes suaves y partes fuertes
    # OJO: tempo viene cuantizado (26 valores únicos en 11.205 ventanas), así que
    # tempo_consistency es gruesa. Tempo continuo requiere FEATURE_VERSION 5.
}

# ¿Pasa algo interesante, o es un riff en loop? Lo que separa una zapada que va
# a algún lado de una que repite la misma idea 10 minutos. SIN VALIDAR: hay un
# A/B pendiente (muestra_A vs muestra_B).
CREATIVIDAD = {
    "novedad_armonica":  (+1, 0.40),   # la armonía se mueve entre secciones
    "harmonic_flux":     (+1, 0.35),   # movimiento armónico dentro de las ventanas
    "variedad_textura":  (+1, 0.25),   # la densidad de eventos cambia a lo largo
}

VENTANA_DIMS = {"sonido": SONIDO, "timing": TIMING, "groove": GROOVE,
                "afinacion": AFINACION, "tonal_outlier": TONAL_OUTLIER}
ARCHIVO_DIMS = {"desarrollo": DESARROLLO, "creatividad": CREATIVIDAD}

# Qué dimensiones forman cada composite.
DIMS_EJECUCION = ("timing", "groove", "afinacion", "tonal_outlier")
DIMS_EJEC_LIMPIA = ("timing", "groove", "afinacion")      # sin castigar cromatismos
DIMS_INTERES = ("creatividad", "desarrollo")
# balance = la mezcla VALIDADA. Medido sobre los 3 positivos held-out (tomas con
# proyecto de REAPER): timing+groove+afinacion+desarrollo da mediana 83.3;
# agregar tonal_outlier 82.6; agregar creatividad 79.1; las seis 74.2. Cada
# dimensión sin validar que entra, baja las referencias humanas. Por eso
# tonal_outlier y creatividad solo viven en 'performances' / 'ideas' / 'gems'
# hasta que el feedback pareado (P1) diga otra cosa.
DIMS_BALANCE = ("timing", "groove", "afinacion", "desarrollo")

PERFILES = ("balance", "performances", "ideas", "gems")

# Se calculan y reportan, pero NO puntúan: describen el carácter del audio, no su
# calidad. El centroide acá era el que hundía todo lo grave.
DIAGNOSTICO = ["centroid", "rolloff85", "rms_db", "noise_floor_db", "tempo", "n_beats",
               "onset_rate", "band_lag_ms", "low_energy_ratio", "dc_offset",
               "harmonic_flux", "spectral_contrast", "chroma_entropy",
               "snr_proxy_db", "dyn_range_db", "fuera_tono_picos"]

# Un solo juego de pesos sobre las seis dimensiones puntuables. Cada composite
# usa el subconjunto que le corresponde, renormalizado.
PESOS_DEFAULT = {"timing": 0.35, "groove": 0.30, "afinacion": 0.15, "desarrollo": 0.20,
                 "tonal_outlier": 0.10, "creatividad": 0.20}


def rank_norm(s):
    """Percentil [0,1]. Robusto a outliers, a diferencia de min-max.

    NaN -> 0.5 (neutral): una medición que falló no debe premiar ni castigar.
    """
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=s.index)
    return x.rank(method="average", pct=True).fillna(0.5)


def _group_score(df, spec):
    total = sum(w for _, w in spec.values())
    acc = pd.Series(0.0, index=df.index)
    for feat, (direction, w) in spec.items():
        if feat not in df.columns:
            continue
        n = rank_norm(df[feat])
        acc += w * (n if direction > 0 else (1.0 - n))
    return acc / total


def _combo(d, pesos, dims):
    """Suma ponderada de dimensiones ya en [0,1], con los pesos renormalizados
    al subconjunto. Si el subconjunto pesa cero, promedio simple."""
    dims = [k for k in dims if k in d.columns]
    if not dims:
        return pd.Series(0.5, index=d.index)
    w = {k: float(pesos.get(k, 0.0)) for k in dims}
    tot = sum(w.values())
    if tot <= 0:
        w = {k: 1.0 for k in dims}
        tot = float(len(dims))
    return sum(d[k] * (w[k] / tot) for k in dims)


def drop_degenerate(df):
    """Saca ventanas sin señal: silencio o audio muerto no debe puntuar."""
    if "is_silent" in df.columns:
        keep = df["is_silent"].fillna(0) == 0
    else:
        keep = pd.to_numeric(df.get("rms_db"), errors="coerce").fillna(0) > -60.0
    return df[keep].copy()


def score_windows(df, peso_sonido=0.35, peso_ejec=0.65, modo="mixto", sonido_min=0.15,
                  pesos=None, perfil="balance"):
    """Puntúa cada ventana en las dimensiones de ventana y arma su `score`.

    modo='mixto': suma ponderada de sonido y ejecución.
    modo='veto' : el sonido NO suma, solo descalifica. Todas las tomas son de la
                  misma banda en la misma sala, así que la varianza de sonido es
                  más accidente de micrófono que calidad de la toma; una zapada
                  bien tocada y mal grabada se arregla mezclando, una mal tocada no.

    El `score` de ventana decide qué TRAMO se elige dentro de un archivo. Con
    perfil 'ideas' o 'gems' usa la ejecución LIMPIA (sin tonal_outlier): el mejor
    tramo de una idea es el que se sostiene tocando, no el que evita cromatismos.
    """
    pesos = dict(pesos or PESOS_DEFAULT)
    df = drop_degenerate(df)
    for name, spec in VENTANA_DIMS.items():
        df[name] = _group_score(df, spec)

    if modo == "veto":
        df = df[df["sonido"] >= df["sonido"].quantile(sonido_min)].copy()

    df["ejec_ventana"] = _combo(df, pesos, DIMS_EJECUCION)
    df["ejec_limpia"] = _combo(df, pesos, DIMS_EJEC_LIMPIA)
    base = df["ejec_limpia"] if perfil in ("ideas", "gems") else df["ejec_ventana"]

    if modo == "veto":
        df["score"] = base
    else:
        df["score"] = peso_sonido * df["sonido"] + peso_ejec * base
    return df


def _file_level_features(g):
    """Features que solo existen comparando ventanas entre sí."""
    out = {}
    tempo = pd.to_numeric(g.get("tempo"), errors="coerce").dropna()
    tempo = tempo[tempo > 0]
    if len(tempo) >= 3 and tempo.mean() > 0:
        out["tempo_consistency"] = float(1.0 - tempo.std() / tempo.mean())
    else:
        out["tempo_consistency"] = float("nan")

    rms = pd.to_numeric(g.get("rms_db"), errors="coerce").dropna()
    out["arco_dinamico"] = (float(np.percentile(rms, 90) - np.percentile(rms, 10))
                            if len(rms) >= 3 else float("nan"))

    cols = ["chroma_%d" % i for i in range(12)]
    if all(c in g.columns for c in cols) and len(g) >= 3:
        C = g[cols].astype(float).to_numpy()
        n = np.linalg.norm(C, axis=1, keepdims=True)
        C = C / np.maximum(n, 1e-10)
        S = C @ C.T
        iu = np.triu_indices(len(C), k=1)
        out["novedad_armonica"] = float(1.0 - np.mean(S[iu]))
    else:
        out["novedad_armonica"] = float("nan")

    orate = pd.to_numeric(g.get("onset_rate"), errors="coerce").dropna()
    out["variedad_textura"] = (float(orate.std() / max(orate.mean(), 1e-9))
                               if len(orate) >= 3 and orate.mean() > 0 else float("nan"))
    hf = pd.to_numeric(g.get("harmonic_flux"), errors="coerce").dropna()
    out["harmonic_flux"] = float(hf.median()) if len(hf) else float("nan")
    return out


def aggregate_files(df_win, files, best_q=0.95, shrink_k=4.0, modo="mixto",
                    peso_sonido=0.35, peso_ejec=0.65, pesos=None, perfil="balance"):
    """Un archivo -> dimensiones, composites (ejecucion / interes / balance / gems),
    score según perfil, y mejor tramo.

    `shrink_k` corrige la inflación de varianza de los archivos cortos: con 2
    ventanas la mediana se va a los extremos por azar, con 30 no. Se encoge hacia
    0.5 con peso n/(n+k), así una toma de 1 minuto tiene que ser MUY buena para
    ganarle a una de 10.
    """
    pesos = dict(pesos or PESOS_DEFAULT)
    if perfil not in PERFILES:
        raise ValueError(f"perfil desconocido: {perfil}")
    out = []
    for path, g in df_win.groupby("path"):
        g = g.sort_values("start")
        best = g.loc[g["score"].idxmax()]
        row = {
            "Ruta": path,
            "Toma": path.rsplit("/", 1)[-1],
            "Ensayo": path.rsplit("/", 2)[-2] if "/" in path else "",
            "dur_min": round(files.get(path, 0.0) / 60.0, 2),
            "n_win": len(g),
            "best_start_s": float(best["start"]),
        }
        for dim in VENTANA_DIMS:
            row[dim] = float(g[dim].median())
        row["_w_med"] = float(g["score"].median())
        row["_w_best"] = float(g["score"].quantile(best_q))
        row.update(_file_level_features(g))
        for c in DIAGNOSTICO:
            if c in g.columns:
                row[c] = float(pd.to_numeric(g[c], errors="coerce").median())
        out.append(row)

    d = pd.DataFrame(out)
    if d.empty:
        return d

    # Las de archivo se normalizan entre archivos, no entre ventanas.
    for name, spec in ARCHIVO_DIMS.items():
        d[name] = _group_score(d, spec)

    # Composites, todos en [0,1].
    d["ejecucion"] = _combo(d, pesos, DIMS_EJECUCION)
    d["interes"] = _combo(d, pesos, DIMS_INTERES)
    d["balance"] = _combo(d, pesos, DIMS_BALANCE)
    # gems: interés alto que el ranking de ejecución no habría mostrado. A
    # ejecución 0 vale el interés completo, a ejecución 1 la mitad. Heurística
    # hasta tener feedback humano; ver review §2.
    d["gems"] = d["interes"] * (1.0 - 0.5 * d["ejecucion"])

    base = {"balance": d["balance"], "performances": d["ejecucion"],
            "ideas": d["interes"], "gems": d["gems"]}[perfil]
    if modo != "veto" and perfil in ("balance", "performances"):
        base = peso_sonido * d["sonido"] + peso_ejec * base

    # El mejor tramo se juzga con el score de ventana (las de archivo no aplican
    # a 30 s). En los perfiles de idea pesa menos: el archivo importa más que
    # cuán bien se toca su mejor minuto y medio.
    w_win = 0.2 if perfil in ("ideas", "gems") else 0.5
    d["score_med_raw"] = (1.0 - w_win) * base + w_win * d["_w_med"]
    d["score_best_raw"] = d["_w_best"]

    w = d["n_win"] / (d["n_win"] + shrink_k)
    for c in ("score_med", "score_best"):
        d[c] = w * d[c + "_raw"] + (1.0 - w) * 0.5
    d["best_start"] = d["best_start_s"].map(lambda s: "%d:%02d" % (int(s) // 60, int(s) % 60))
    return d.drop(columns=["_w_med", "_w_best"]).sort_values(
        "score_med", ascending=False).reset_index(drop=True)


def diagnose(df_win):
    """Detecta features degeneradas: las que no discriminan nada."""
    feats = []
    for spec in VENTANA_DIMS.values():
        feats += list(spec)
    for spec in ARCHIVO_DIMS.values():
        feats += list(spec)
    feats += DIAGNOSTICO
    rows = []
    for feat in feats:
        if feat not in df_win.columns:
            continue
        x = pd.to_numeric(df_win[feat], errors="coerce").dropna()
        if x.empty:
            continue
        grupo = next((n for n, s in VENTANA_DIMS.items() if feat in s), None) or \
            next((n for n, s in ARCHIVO_DIMS.items() if feat in s), "diag")
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        rows.append({
            "feature": feat, "grupo": grupo,
            "min": x.min(), "p10": p10, "p50": p50, "p90": p90, "max": x.max(),
            "iqr/|p50|": iqr / max(abs(p50), 1e-9),
            "n_nan": int(pd.to_numeric(df_win[feat], errors="coerce").isna().sum()),
            "n_unicos": int(x.nunique()),
        })
    return pd.DataFrame(rows)
