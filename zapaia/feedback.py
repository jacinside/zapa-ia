"""Feedback humano pareado: la única fuente de etiquetas que funcionó.

Preguntar "¿qué tan creativo es esto, del 1 al 10?" no produjo respuestas.
Mostrar A contra B y preguntar "¿cuál rescatarías?" sí. Este módulo arma los
pares, exporta los clips, guarda las respuestas y las convierte en un score
latente por segmento (Bradley–Terry) que se puede correlar con cada dimensión.

Ver docs/review-2026-09-12.md §5.
"""
import os
import random
import re
import subprocess
import time

import numpy as np
import pandas as pd
from pydub import AudioSegment

from . import FEATURE_VERSION, compilado, dedupe

ELECCIONES = ("A", "B", "ambos", "ninguno")


# ---------------------------------------------------------------------------
# Segmentos
# ---------------------------------------------------------------------------

def segmento_id(con, path, start, end, origen="feedback"):
    """Devuelve el id del segmento (path, start, end), creándolo si no existe."""
    row = con.execute("SELECT id FROM segmentos WHERE path=? AND start=? AND end=?",
                      (path, start, end)).fetchone()
    if row:
        return int(row[0])
    cur = con.execute(
        "INSERT INTO segmentos (path, start, end, origen, fver, creado) VALUES (?,?,?,?,?,?)",
        (path, start, end, origen, FEATURE_VERSION, time.time()))
    con.commit()
    return int(cur.lastrowid)


def candidatos(dfw, d, seg_win, win_s, col="score"):
    """Un segmento por archivo: su mejor tirada contigua de `seg_win` ventanas.

    Devuelve un DataFrame con las dimensiones del archivo más start/end del tramo.
    """
    d = d[d["n_win"] >= seg_win].copy()
    filas = []
    for _, r in d.iterrows():
        g = dfw[dfw["path"] == r["Ruta"]]
        if g.empty:
            continue
        ini, fin = compilado.mejor_tramo(g, seg_win, win_s, col=col)
        row = r.to_dict()
        row["start"], row["end"] = float(ini), float(fin)
        filas.append(row)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Muestreo de pares
# ---------------------------------------------------------------------------

def _distintos(c, i, j, temas, grupos):
    """Dos candidatos sirven como par si no son el mismo tema ni el mismo audio."""
    if i == j:
        return False
    ti, tj = temas.get(c.at[i, "Toma"]), temas.get(c.at[j, "Toma"])
    if ti is not None and ti == tj:
        return False
    gi, gj = grupos.get(c.at[i, "Ruta"], -1), grupos.get(c.at[j, "Ruta"], -2)
    return gi != gj


def muestrear_pares(c, n, temas, grupos, seed=None, ya_vistos=()):
    """Devuelve lista de (idx_a, idx_b, tipo).

    50% 'parejos'  : score compuesto parecido, temas distintos -> máxima información
    30% 'azar'     : calibra
    20% 'gem_vs_perf': interés alto/ejecución baja contra ejecución alta/interés bajo
    """
    rnd = random.Random(seed)
    c = c.reset_index(drop=True)
    idx = list(c.index)
    vistos = set(ya_vistos)
    pares = []

    def agregar(i, j, tipo):
        key = (c.at[i, "Ruta"], c.at[j, "Ruta"])
        if key in vistos or key[::-1] in vistos:
            return False
        vistos.add(key)
        pares.append((i, j, tipo))
        return True

    cuotas = {"parejos": round(n * 0.5), "azar": round(n * 0.3)}
    cuotas["gem_vs_perf"] = n - cuotas["parejos"] - cuotas["azar"]

    # gem vs performance: colas opuestas de interes y ejecucion
    if "interes" in c.columns and "ejecucion" in c.columns:
        gems = c[(c["interes"] >= c["interes"].quantile(0.7)) &
                 (c["ejecucion"] <= c["ejecucion"].quantile(0.5))].index.tolist()
        perfs = c[(c["ejecucion"] >= c["ejecucion"].quantile(0.7)) &
                  (c["interes"] <= c["interes"].quantile(0.5))].index.tolist()
        intentos = 0
        while cuotas["gem_vs_perf"] > 0 and gems and perfs and intentos < 200:
            intentos += 1
            i, j = rnd.choice(gems), rnd.choice(perfs)
            if _distintos(c, i, j, temas, grupos) and agregar(*(rnd.sample([i, j], 2)), "gem_vs_perf"):
                cuotas["gem_vs_perf"] -= 1
        cuotas["azar"] += cuotas["gem_vs_perf"]      # lo que no se pudo, va al azar

    # parejos: score parecido, temas distintos
    col = "score_med" if "score_med" in c.columns else "balance"
    orden = c.sort_values(col).index.tolist()
    intentos = 0
    while cuotas["parejos"] > 0 and intentos < 500 and len(orden) > 3:
        intentos += 1
        k = rnd.randrange(len(orden))
        i = orden[k]
        vecinos = [orden[m] for m in range(max(0, k - 6), min(len(orden), k + 7)) if m != k]
        rnd.shuffle(vecinos)
        for j in vecinos:
            if _distintos(c, i, j, temas, grupos) and agregar(*(rnd.sample([i, j], 2)), "parejos"):
                cuotas["parejos"] -= 1
                break
    cuotas["azar"] += cuotas["parejos"]

    intentos = 0
    while cuotas["azar"] > 0 and intentos < 500:
        intentos += 1
        i, j = rnd.sample(idx, 2)
        if _distintos(c, i, j, temas, grupos) and agregar(i, j, "azar"):
            cuotas["azar"] -= 1
    return pares


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def exportar_clip(path, start, dur_s, out, fade_s=1.0, bitrate="128k", snap=True):
    """Corta `dur_s` segundos desde `start` (pegado al beat) y exporta MP3."""
    if snap:
        start = compilado.pegar_a_beat(path, start)
    audio = AudioSegment.from_file(path)
    a, b = int(start * 1000), int(min(start + dur_s, len(audio) / 1000.0) * 1000)
    clip = audio[a:b].fade_in(int(fade_s * 1000)).fade_out(int(fade_s * 1000))
    clip.export(out, format="mp3", bitrate=bitrate)
    return start, (b - a) / 1000.0


def armar_lote_mp3(clips, out, gap_clip_s=1.0, gap_par_s=2.5, tono_s=0.35, bitrate="128k"):
    """Un solo MP3 con todos los pares seguidos, para escuchar en el celular.

    clips: lista de (num, ruta_A, ruta_B). Antes de cada clip suena un tono corto:
    agudo (A) o grave (B), así se sabe dónde se está sin mirar. Devuelve la lista
    de (num, lado, segundo_de_inicio) para armar el índice.
    """
    from pydub.generators import Sine
    tono = {"A": Sine(880).to_audio_segment(duration=int(tono_s * 1000)).apply_gain(-12),
            "B": Sine(440).to_audio_segment(duration=int(tono_s * 1000)).apply_gain(-12)}
    for k in tono:
        tono[k] = tono[k].fade_in(20).fade_out(80)
    gap_clip = AudioSegment.silent(duration=int(gap_clip_s * 1000))
    gap_par = AudioSegment.silent(duration=int(gap_par_s * 1000))
    audio = AudioSegment.silent(duration=500)
    indice = []
    for num, ra, rb in clips:
        for lado, ruta in (("A", ra), ("B", rb)):
            audio += tono[lado] + AudioSegment.silent(duration=250)
            indice.append((num, lado, len(audio) / 1000.0))
            audio += AudioSegment.from_file(ruta)
            audio += gap_clip
        audio += gap_par
    audio.export(out, format="mp3", bitrate=bitrate)
    return indice


def reproducir(path):
    """macOS: afplay bloquea hasta que termina. Ctrl-C corta el clip, no el programa."""
    try:
        subprocess.run(["afplay", path], check=False)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Persistencia de pares y respuestas
# ---------------------------------------------------------------------------

def nuevo_lote(con):
    row = con.execute("SELECT COALESCE(MAX(lote), 0) + 1 FROM pares").fetchone()
    return int(row[0])


def guardar_par(con, lote, num, seg_a, seg_b, tipo):
    con.execute("INSERT INTO pares (lote, num, seg_a, seg_b, tipo, fver) VALUES (?,?,?,?,?,?)",
                (lote, num, seg_a, seg_b, tipo, FEATURE_VERSION))
    con.commit()


def responder(con, lote, num, eleccion, tags=None):
    if eleccion not in ELECCIONES:
        raise ValueError(f"elección inválida: {eleccion!r} (A, B, ambos, ninguno)")
    n = con.execute("UPDATE pares SET eleccion=?, tags=?, ts=? WHERE lote=? AND num=?",
                    (eleccion, tags, time.time(), lote, num)).rowcount
    con.commit()
    return n


def parsear_respuestas(texto):
    """'001 A, 002 B, 3 ninguno; 4:ambos' -> [(1,'A'), (2,'B'), (3,'ninguno'), (4,'ambos')].

    Tolerante: acepta lo que se escribe por WhatsApp.
    """
    # El player manda "LOTE 1 (nombre): 1 A, 2 B, ..." — el prefijo se descarta.
    texto = re.sub(r"^\s*lote\s*\d+\s*(\([^)]*\))?\s*:\s*", "", texto, flags=re.IGNORECASE)
    out = []
    for trozo in re.split(r"[,;\n]+", texto):
        trozo = trozo.strip()
        if not trozo:
            continue
        m = re.match(r"^(\d+)\s*[:\-=]?\s*(a|b|ambos|ambas|ninguno|ninguna|los dos|las dos)\s*$",
                     trozo, re.IGNORECASE)
        if not m:
            raise ValueError(f"no entiendo: {trozo!r}")
        e = m.group(2).lower()
        e = {"a": "A", "b": "B", "ambas": "ambos", "los dos": "ambos", "las dos": "ambos",
             "ninguna": "ninguno"}.get(e, e)
        out.append((int(m.group(1)), e))
    return out


def pares_vistos(con):
    return [(a, b) for a, b in con.execute(
        "SELECT sa.path, sb.path FROM pares p JOIN segmentos sa ON sa.id=p.seg_a "
        "JOIN segmentos sb ON sb.id=p.seg_b")]


# ---------------------------------------------------------------------------
# Análisis: Bradley–Terry
# ---------------------------------------------------------------------------

def bradley_terry(comparaciones, iters=200):
    """comparaciones: lista (ganador, perdedor, peso). Empates: dos entradas con 0.5.

    Devuelve {item: score latente en log-escala, centrado en 0}. Algoritmo MM de
    Hunter (2004) con un prior débil para que un ítem invicto no diverja.
    """
    items = sorted({i for c in comparaciones for i in c[:2]})
    if not items:
        return {}
    ix = {it: k for k, it in enumerate(items)}
    n = len(items)
    W = np.zeros((n, n))
    for g, p, w in comparaciones:
        W[ix[g], ix[p]] += w
    W += 0.05 / n                          # prior débil: cada uno le ganó un poquito a todos
    np.fill_diagonal(W, 0.0)
    wins = W.sum(axis=1)
    theta = np.ones(n)
    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    denom[i] += (W[i, j] + W[j, i]) / (theta[i] + theta[j])
        theta = wins / np.maximum(denom, 1e-12)
        theta /= np.exp(np.mean(np.log(theta)))
    return {it: float(np.log(theta[ix[it]])) for it in items}


def comparaciones_desde_db(con):
    """Lee las respuestas y las vuelve (ganador, perdedor, peso) sobre ids de segmento."""
    out = []
    for a, b, e in con.execute("SELECT seg_a, seg_b, eleccion FROM pares WHERE eleccion IS NOT NULL"):
        if e == "A":
            out.append((a, b, 1.0))
        elif e == "B":
            out.append((b, a, 1.0))
        else:                               # ambos / ninguno: empate
            out.append((a, b, 0.5))
            out.append((b, a, 0.5))
    return out
