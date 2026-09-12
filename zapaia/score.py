"""Normalización robusta y combinación de features en scores.

Dos niveles:
  - VENTANA (30 s): sonido, timing, groove, afinación. Se normalizan por percentil
    contra todas las ventanas del corpus.
  - ARCHIVO: mediana de las dimensiones de ventana + desarrollo (que solo existe a
    nivel archivo: cómo evoluciona la zapada a lo largo del tiempo).

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
}

# ¿Toca las notas CORRECTAS? Punto ciego de todo lo anterior: timing mide cuándo
# suena una nota y afinacion si el instrumento está afinado, pero un pifie a
# tiempo con un bajo bien afinado puntuaba alto. Un pifie es energía en una clase
# de altura ajena a la tonalidad que viene sonando.
NOTAS = {
    "fuera_tono_med":   (-1, 0.45),  # energía que cae fuera de la tonalidad local
    "fuera_tono_p95":   (-1, 0.40),  # picos: un pifie aislado se diluye en la media
    "fuera_tono_picos": (-1, 0.15),  # cuánto tiempo se pasa en esos picos
}

# --------------------------------------------------------------------------
# Dimensión de ARCHIVO: ¿la zapada va a algún lado, o es un loop de 20 minutos?
# Solo tiene sentido comparando ventanas entre sí, no dentro de una.
# --------------------------------------------------------------------------
DESARROLLO = {
    "tempo_consistency": (+1, 0.55),   # sostener el tempo 10 min es mérito real
    "arco_dinamico":     (+1, 0.45),   # partes suaves y partes fuertes
}

# ¿Pasa algo interesante, o es un riff en loop? Lo que separa una zapada que va
# a algún lado de una que repite la misma idea 10 minutos. También de archivo.
CREATIVIDAD = {
    "novedad_armonica":  (+1, 0.40),   # la armonía se mueve entre secciones
    "harmonic_flux":     (+1, 0.35),   # movimiento armónico dentro de las ventanas
    "variedad_textura":  (+1, 0.25),   # la densidad de eventos cambia a lo largo
}

VENTANA_DIMS = {"sonido": SONIDO, "timing": TIMING, "groove": GROOVE,
                "afinacion": AFINACION, "notas": NOTAS}

# Se calculan y reportan, pero NO puntúan: describen el carácter del audio, no su
# calidad. El centroide acá era el que hundía todo lo grave.
DIAGNOSTICO = ["centroid", "rolloff85", "rms_db", "noise_floor_db", "tempo", "n_beats",
               "onset_rate", "band_lag_ms", "low_energy_ratio", "dc_offset",
               "harmonic_flux", "spectral_contrast", "chroma_entropy",
               "snr_proxy_db", "dyn_range_db"]

PESOS_EJEC_DEFAULT = {"timing": 0.25, "groove": 0.20, "notas": 0.15,
                      "afinacion": 0.10, "desarrollo": 0.10, "creatividad": 0.20}


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


def drop_degenerate(df):
    """Saca ventanas sin señal: silencio o audio muerto no debe puntuar."""
    if "is_silent" in df.columns:
        keep = df["is_silent"].fillna(0) == 0
    else:
        keep = pd.to_numeric(df.get("rms_db"), errors="coerce").fillna(0) > -60.0
    return df[keep].copy()


def score_windows(df, peso_sonido=0.35, peso_ejec=0.65, modo="mixto", sonido_min=0.15,
                  pesos_ejec=None):
    """Puntúa cada ventana en las 4 dimensiones y arma el score de ventana.

    modo='mixto': suma ponderada de sonido y ejecución.
    modo='veto' : el sonido NO suma, solo descalifica. Todas las tomas son de la
                  misma banda en la misma sala, así que la varianza de sonido es
                  más accidente de micrófono que calidad de la toma; una zapada
                  bien tocada y mal grabada se arregla mezclando, una mal tocada no.
    """
    pesos_ejec = dict(pesos_ejec or PESOS_EJEC_DEFAULT)
    df = drop_degenerate(df)
    for name, spec in VENTANA_DIMS.items():
        df[name] = _group_score(df, spec)

    if modo == "veto":
        df = df[df["sonido"] >= df["sonido"].quantile(sonido_min)].copy()

    # 'desarrollo' es de archivo: acá se reparte su peso entre las que sí son de ventana.
    wv = {k: pesos_ejec.get(k, 0.0) for k in ("timing", "groove", "afinacion", "notas")}
    tot = sum(wv.values()) or 1.0
    df["ejec_ventana"] = sum(df[k] * (w / tot) for k, w in wv.items())

    if modo == "veto":
        df["score"] = df["ejec_ventana"]
    else:
        df["score"] = peso_sonido * df["sonido"] + peso_ejec * df["ejec_ventana"]
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
                    peso_sonido=0.35, peso_ejec=0.65, pesos_ejec=None):
    """Un archivo -> sub-scores por dimensión, score global y mejor tramo.

    `shrink_k` corrige la inflación de varianza de los archivos cortos: con 2
    ventanas la mediana se va a los extremos por azar, con 30 no. Se encoge hacia
    0.5 con peso n/(n+k), así una toma de 1 minuto tiene que ser MUY buena para
    ganarle a una de 10.
    """
    pesos_ejec = dict(pesos_ejec or PESOS_EJEC_DEFAULT)
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

    # 'desarrollo' se normaliza entre archivos, no entre ventanas.
    d["desarrollo"] = _group_score(d, DESARROLLO)
    d["creatividad"] = _group_score(d, CREATIVIDAD)

    # Ejecución = las 3 dimensiones de ventana + desarrollo, todo ya en [0,1].
    tot = sum(pesos_ejec.values()) or 1.0
    d["ejecucion"] = sum(d[k] * (w / tot) for k, w in pesos_ejec.items())

    if modo == "veto":
        base_med, base_best = d["ejecucion"], d["ejecucion"]
    else:
        base_med = peso_sonido * d["sonido"] + peso_ejec * d["ejecucion"]
        base_best = base_med
    # El mejor tramo se juzga con el score de ventana (desarrollo no aplica a 30 s).
    d["score_med_raw"] = 0.5 * base_med + 0.5 * d["_w_med"]
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
    feats += list(DESARROLLO) + list(CREATIVIDAD) + DIAGNOSTICO
    rows = []
    for feat in feats:
        if feat not in df_win.columns:
            continue
        x = pd.to_numeric(df_win[feat], errors="coerce").dropna()
        if x.empty:
            continue
        grupo = next((n for n, s in VENTANA_DIMS.items() if feat in s),
                     "desarrollo" if feat in DESARROLLO else
                     ("creatividad" if feat in CREATIVIDAD else "diag"))
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
