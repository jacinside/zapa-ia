"""Detección de duplicados y re-encodes por huella de audio.

El corpus tiene copias ('Bossa.mp3' / 'Bossa (1).mp3', 'Copia de ...'), que hoy
ocupan dos lugares del top-N con el mismo material.
"""
import numpy as np


def find_groups(sigs, sim_thr=0.9995, dur_tol=2.0):
    """sigs: lista (path, duration, vector). Devuelve {path: id_de_grupo}."""
    items = [(p, d, v) for p, d, v in sigs if v is not None and v.size]
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    group = {}
    gid = 0
    for ii, i in enumerate(order):
        pi, di, vi = items[i]
        if pi in group:
            continue
        group[pi] = gid
        for j in order[ii + 1:]:
            pj, dj, vj = items[j]
            if dj - di > dur_tol:
                break                      # ordenado por duración: cortamos
            if pj in group or vi.shape != vj.shape:
                continue
            if float(np.dot(vi, vj)) >= sim_thr:
                group[pj] = gid
        gid += 1
    return group


def centrar(V):
    """Resta el promedio del corpus y normaliza.

    CRÍTICO: en crudo, dos zapadas cualesquiera dan similitud ~0.98 porque
    comparten sala, banda e instrumentos — el fondo común tapa la diferencia
    musical. Centrado, mismo tema da +0.40 y distinto tema -0.03.
    """
    V = np.asarray(V, dtype=float)
    if len(V) < 2:
        return V
    V = V - V.mean(axis=0)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def seleccion_diversa(rutas, sigs, n, umbral=0.20):
    """Recorre `rutas` (ya ordenadas por score) y acepta solo lo suficientemente
    distinto de lo ya aceptado. Evita que 4 tomas del mismo tema copen la lista.
    """
    disp = {p: v for p, v in sigs.items() if v is not None and v.size}
    orden = [p for p in rutas if p in disp]
    if not orden:
        return list(rutas)[:n]
    M = centrar([disp[p] for p in orden])
    pos = {p: i for i, p in enumerate(orden)}
    elegidos, descartados = [], []
    for p in orden:
        if len(elegidos) >= n:
            break
        i = pos[p]
        if all(float(np.dot(M[i], M[pos[q]])) < umbral for q in elegidos):
            elegidos.append(p)
        else:
            descartados.append(p)
    # Si la diversidad dejó pocos, completar con los mejores descartados.
    for p in descartados:
        if len(elegidos) >= n:
            break
        elegidos.append(p)
    return elegidos


# --- Identidad de tema por NOMBRE de archivo ---------------------------------
# La huella de audio no distingue temas: dos zapadas cualesquiera dan ~0.98 de
# similitud porque comparten sala, banda e instrumentos, y aun centrada apenas
# separa. Los nombres, informales como son, identifican el tema mucho mejor.
_GENERICOS = {"nebu", "nebulosa", "zapa", "zapada", "zapadas", "demo", "test", "mix",
              "take", "final", "version", "parte", "con", "del", "para", "por",
              "una", "las", "los", "que", "mas", "muy", "video", "audio"}


def _tokens(nombre):
    import re
    n = re.sub(r"\.mp3$", "", nombre.lower())
    n = re.sub(r"[^a-z0-9áéíóúñ ]", " ", n)
    return [t for t in n.split() if len(t) > 2]


def temas_por_nombre(nombres, min_df=3, max_frac=0.12):
    """nombre -> token de tema (o None). El tema es el token recurrente más
    frecuente que no sea genérico ni demasiado común (banda, formato, versión).
    """
    from collections import Counter
    df = Counter()
    for n in nombres:
        df.update(set(_tokens(n)))
    tope = max(int(len(nombres) * max_frac), min_df)
    out = {}
    for n in nombres:
        cand = [(df[t], t) for t in _tokens(n)
                if min_df <= df[t] <= tope and t not in _GENERICOS]
        out[n] = max(cand)[1] if cand else None
    return out
