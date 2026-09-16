"""CLI: extract / rank / diag / dupes / eval."""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment

from . import cache, compilado, dedupe, score as scoring
from .audio import load_mono, windows
from .features import window_features, signature

DEFAULTS = dict(sr=22050, win=30.0, hop=30.0, db=".zapaia_cache.db")


# ---------- worker ----------
def _process(task):
    path, sr, win, hop = task
    t0 = time.time()
    try:
        y = load_mono(path, sr)
        dur = len(y) / sr
        wins = [(i, st, window_features(seg, sr)) for i, st, seg in windows(y, sr, win, hop)]
        sig = signature(y, sr)
        return path, dur, wins, sig, "ok", None, time.time() - t0
    except Exception as e:
        return path, 0.0, [], None, "error", f"{type(e).__name__}: {e}", time.time() - t0


def _mp3s(root):
    return sorted(str(p.resolve()) for p in Path(root).rglob("*.mp3"))


def cmd_extract(a):
    if getattr(a, "files_from", None):
        files = [l.strip() for l in open(a.files_from, encoding="utf-8") if l.strip()]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = _mp3s(a.root)
    if not files:
        sys.exit(f"No hay MP3 en {a.root}")
    if a.include:
        keep = [f for f in files if any(s.lower() in os.path.basename(f).lower() for s in a.include)]
    else:
        keep = []
    if a.sample and a.sample < len(files):
        rnd = random.Random(a.seed)
        pool = [f for f in files if f not in keep]
        keep = sorted(set(keep) | set(rnd.sample(pool, min(a.sample, len(pool)))))
        files = keep
    elif keep:
        files = keep

    con = cache.connect(a.db)
    todo = [f for f in files if a.force or not cache.is_fresh(
        con, f, a.sr, a.win, a.hop, reintentar_errores=a.reintentar_errores)]
    print(f"{len(files)} archivos objetivo | {len(files)-len(todo)} en caché | {len(todo)} a procesar")
    if not todo:
        return

    tasks = [(f, a.sr, a.win, a.hop) for f in todo]
    t0, done, errs = time.time(), 0, 0

    def emit(res):
        nonlocal done, errs
        path, dur, wins, sig, status, err, dt = res
        cache.store(con, path, a.sr, a.win, a.hop, dur, wins, sig, status, err)
        done += 1
        if status != "ok":
            errs += 1
            print(f"  ERROR {os.path.basename(path)}: {err}", flush=True)
        el = time.time() - t0
        eta = el / done * (len(tasks) - done)
        print(f"[{done}/{len(tasks)}] {dur/60:5.1f}min {len(wins):3d}win {dt:5.1f}s "
              f"| ETA {eta/60:.1f}min | {os.path.basename(path)[:46]}", flush=True)

    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for res in pool.imap_unordered(_process, tasks):
                emit(res)
    else:
        for t in tasks:
            emit(_process(t))
    print(f"\nListo en {(time.time()-t0)/60:.1f}min | errores: {errs} | caché: {a.db}")


def _pesos(a):
    return {"timing": a.w_timing, "groove": a.w_groove, "tonal_outlier": a.w_tonal,
            "afinacion": a.w_afinacion, "desarrollo": a.w_desarrollo,
            "creatividad": a.w_creatividad}


def _load(a):
    if getattr(a, "sync", False):
        _sync_previo(a)
    con = cache.connect(a.db)
    prefix = str(Path(a.root).resolve())
    rows, files = cache.load_windows(con, a.sr, a.win, a.hop, prefix)
    if not rows:
        sys.exit("Caché vacío para estos parámetros. Corré 'extract' primero.")
    df = scoring.score_windows(pd.DataFrame(rows), a.peso_sonido, a.peso_ejec,
                               a.modo, a.sonido_min, _pesos(a), a.perfil)
    return con, df, files


def _agg(a, dfw, files):
    return scoring.aggregate_files(dfw, files, shrink_k=a.shrink, modo=a.modo,
                                   peso_sonido=a.peso_sonido, peso_ejec=a.peso_ejec,
                                   pesos=_pesos(a), perfil=a.perfil)


def cmd_rank(a):
    con, dfw, files = _load(a)
    d = _agg(a, dfw, files)
    if a.min_win:
        before = len(d)
        d = d[d["n_win"] >= a.min_win]
        print(f"Filtro min_win={a.min_win}: {before} -> {len(d)} tomas")
    d = _filtrar_fechas(a, con, d)

    groups = dedupe.find_groups(cache.load_signatures(con, str(Path(a.root).resolve())))
    d["dup_grupo"] = d["Ruta"].map(groups).fillna(-1).astype(int)
    if a.dedupe:
        before = len(d)
        d = d.sort_values(a.sort_by, ascending=False).drop_duplicates("dup_grupo", keep="first")
        print(f"Dedupe: {before} -> {len(d)} archivos")
    d = d.sort_values(a.sort_by, ascending=False).reset_index(drop=True)
    d.insert(0, "rank", d.index + 1)

    cols = ["rank", "Toma", "dur_min", "n_win", a.sort_by, "ejecucion", "interes",
            "timing", "groove", "afinacion", "tonal_outlier", "creatividad",
            "desarrollo", "sonido", "best_start"]
    cols = list(dict.fromkeys(c for c in cols if c in d.columns))
    modo = f"modo={a.modo}" + (f" sonido_min={a.sonido_min}" if a.modo == "veto" else
                               f" p_son={a.peso_sonido} p_ejec={a.peso_ejec}")
    pe = _pesos(a)
    print(f"\nTop {a.top} de {len(d)} tomas | perfil: {a.perfil} | orden: {a.sort_by} | {modo}")
    print("pesos: " + " ".join(f"{k}={v}" for k, v in pe.items()) + "\n")
    print(d.head(a.top)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    d.to_csv(a.out, index=False, float_format="%.5f")
    print(f"\nCSV -> {a.out}")

    if a.windows_out:
        dfw.sort_values(["path", "start"]).to_csv(a.windows_out, index=False, float_format="%.5f")
        print(f"Ventanas -> {a.windows_out}")


def cmd_diag(a):
    _, dfw, _ = _load(a)
    t = scoring.diagnose(dfw)
    print(f"\nDistribución de features sobre {len(dfw)} ventanas")
    print("(iqr/|p50| bajo = la feature casi no discrimina -> su peso es decorativo)\n")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    flat = t[t["iqr/|p50|"] < 0.05]
    if len(flat):
        print("\nSOSPECHOSAS (casi constantes):", ", ".join(flat["feature"]))


def cmd_dupes(a):
    con = cache.connect(a.db)
    sigs = cache.load_signatures(con, str(Path(a.root).resolve()))
    groups = dedupe.find_groups(sigs, a.sim, a.dur_tol)
    inv = {}
    for p, g in groups.items():
        inv.setdefault(g, []).append(p)
    dup = {g: v for g, v in inv.items() if len(v) > 1}
    print(f"{len(dup)} grupos duplicados sobre {len(sigs)} archivos")
    for g, v in sorted(dup.items()):
        print(f"\ngrupo {g}:")
        for p in sorted(v):
            print(f"   {os.path.basename(p)}")


def cmd_eval(a):
    """Mide el ranking contra el ground truth de refs.txt.

    Con positivos y negativos calcula AUC y precision@k. Con solo positivos,
    reporta percentiles: sugestivo, no concluyente.
    """
    _, dfw, files = _load(a)
    d = _agg(a, dfw, files)
    if a.min_win:
        d = d[d["n_win"] >= a.min_win]
    d = d.sort_values(a.sort_by, ascending=False).reset_index(drop=True)
    d["pct"] = 100.0 * (1.0 - (d.index + 1) / len(d))

    pos, neg = [], []
    for line in open(a.refs, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (neg if line.startswith("-") else pos).append(line.lstrip("-").strip())

    def find(name):
        exact = d[d["Toma"].str.lower() == name.lower()]
        if len(exact):
            return exact
        return d[d["Toma"].str.lower().str.contains(
            name.lower().replace(".mp3", ""), regex=False)]

    print(f"\nGround truth: {len(pos)} positivos, {len(neg)} negativos "
          f"| {len(d)} tomas rankeadas por '{a.sort_by}'\n")
    idx = {"pos": [], "neg": []}
    for etiqueta, nombres in (("pos", pos), ("neg", neg)):
        for nombre in nombres:
            hit = find(nombre)
            if hit.empty:
                print(f"  (no encontrada/no extraída) {nombre}")
                continue
            for _, row in hit.iterrows():
                idx[etiqueta].append(int(row.name))
                marca = "+" if etiqueta == "pos" else "-"
                print(f"  {marca} #{int(row.name)+1:4d}/{len(d)} pct={row['pct']:5.1f} "
                      f"{a.sort_by}={row[a.sort_by]:.3f} | eje={row['ejecucion']:.2f} "
                      f"int={row['interes']:.2f} tim={row['timing']:.2f} "
                      f"gro={row['groove']:.2f} afi={row['afinacion']:.2f} "
                      f"ton={row['tonal_outlier']:.2f} cre={row['creatividad']:.2f} "
                      f"des={row['desarrollo']:.2f}  {row['Toma']}")

    P, N = idx["pos"], idx["neg"]
    if P:
        pcts = [d.loc[i, "pct"] for i in P]
        print(f"\n  percentil mediano de los positivos: {np.median(pcts):.1f} (objetivo >85)")
        for k in (10, 25, 50):
            if k <= len(d):
                print(f"  precision@{k}: {sum(1 for i in P if i < k)}/{min(k, len(P))} "
                      f"positivos conocidos en el top-{k}")
    if P and N:
        # AUC = probabilidad de que un positivo al azar quede sobre un negativo al azar.
        wins = sum(1 for i in P for j in N if i < j) + 0.5 * sum(
            1 for i in P for j in N if i == j)
        auc = wins / (len(P) * len(N))
        print(f"\n  AUC = {auc:.3f}  (0.5 = azar, 1.0 = separación perfecta)")
    elif P:
        print("\n  Sin negativos en refs.txt no se puede calcular AUC: agregá tomas malas.")
    if len(P) < 10:
        print(f"  ATENCIÓN: solo {len(P)} positivos. Ajustar pesos con esto es sobreajuste;\n"
              "  hacen falta 10-20 para que el número signifique algo.")


def cmd_compilado(a):
    """Arma un solo MP3 con los mejores tramos de varias tomas."""
    con, dfw, files = _load(a)
    d = _agg(a, dfw, files)
    if a.min_win:
        d = d[d["n_win"] >= a.min_win]
    d = _filtrar_fechas(a, con, d)
    grupos = dedupe.find_groups(cache.load_signatures(con, str(Path(a.root).resolve())))
    d["dup_grupo"] = d["Ruta"].map(grupos).fillna(-1).astype(int)
    d = d.sort_values(a.sort_by, ascending=False).drop_duplicates("dup_grupo", keep="first")
    # Solo tomas con suficiente material para un tramo entero.
    d = d[d["n_win"] >= a.seg_win]
    if a.max_por_tema > 0:
        # Las frecuencias de token se calculan sobre el CORPUS COMPLETO: sobre la
        # lista ya filtrada salen inestables y el tema detectado cambia.
        todos = [os.path.basename(p) for p in _mp3s(a.root)]
        temas = dedupe.temas_por_nombre(todos)
        d["_tema"] = d["Toma"].map(temas)
        vistos, filas = {}, []
        for i, r in d.iterrows():
            t = r["_tema"]
            if t is None or vistos.get(t, 0) < a.max_por_tema:
                filas.append(i)
                if t is not None:
                    vistos[t] = vistos.get(t, 0) + 1
        antes = len(d)
        d = d.loc[filas]
        print(f"Tope por tema ({a.max_por_tema}): {antes} -> {len(d)} tomas")
    if a.diversidad > 0:
        sigs = dict((p, v) for p, _, v in cache.load_signatures(con, str(Path(a.root).resolve())))
        elegidos = dedupe.seleccion_diversa(list(d["Ruta"]), sigs, a.top, a.diversidad)
        d = d[d["Ruta"].isin(elegidos)]
        d = d.set_index("Ruta").loc[elegidos].reset_index()
        print(f"Diversidad (umbral {a.diversidad}): evita repetir el mismo tema")
    else:
        d = d.head(a.top)
    if d.empty:
        sys.exit("No hay tomas con suficientes ventanas. Bajá --seg-win o --min-win.")

    if a.orden == "tempo":
        # Ordenar por tempo hace que los empalmes no salten de 90 a 170 BPM.
        d = d.sort_values("tempo", na_position="last")

    # Las dimensiones de archivo no se le pueden preguntar a 90 s. Si se pide una,
    # el tramo se elige con el score de ventana del perfil.
    col = a.seg_by
    if col not in ("timing", "groove", "afinacion", "tonal_outlier", "sonido",
                   "ejec_ventana", "ejec_limpia"):
        col = "score"
    # (ruta, ini, fin, toma, tempo_cache, score_tramo)
    tramos = []
    if a.dinamico:
        # El umbral es un cuantil del score de ventana de TODO el corpus: un tramo
        # sigue mientras sus ventanas estén en el top (1 - umbral_q) del archivo.
        umbral = float(dfw[col].quantile(a.umbral_q))
        for _, r in d.iterrows():
            g = dfw[dfw["path"] == r["Ruta"]]
            for ini, fin, sc_, tempo in compilado.tramos_dinamicos(
                    g, a.win, umbral, min_win=a.seg_win, max_win=a.max_seg_win,
                    max_tramos=a.tramos_por_toma, col=col, hop_s=a.hop,
                    max_silencio=a.max_silencio):
                tramos.append((r["Ruta"], ini, fin, r["Toma"], tempo, sc_))
        if not tramos:
            sys.exit("Ninguna toma tiene una racha sobre el umbral. Bajá --umbral-q.")
    else:
        for _, r in d.iterrows():
            g = dfw[dfw["path"] == r["Ruta"]]
            ini, fin = compilado.mejor_tramo(g, a.seg_win, a.win, col=col, hop_s=a.hop)
            tramos.append((r["Ruta"], ini, fin, r["Toma"], r.get("tempo", float("nan")),
                           float(r[a.sort_by])))

    if a.orden == "tempo":
        # Ordenar por tempo hace que los empalmes no salten de 90 a 170 BPM, y deja
        # juntos los tramos de una misma toma (mismo tempo -> enganchan solos).
        # Cadena por vecino más cercano en tempo, contando dobles y mitades como
        # iguales. Plegar a una octava fija no alcanzaba: 161.5 caía como "80.75"
        # y abría el compilado para saltar a 89 en el tramo siguiente.
        import math

        def _dist(ta, tb):
            if ta != ta or tb != tb:
                return 1.0
            # Un doble/mitad engancha, pero se siente distinto: se penaliza para que
            # gane un vecino de la misma octava cuando lo hay. Sin esto, 161.5 se
            # consideraba vecino de 89 (≈ 2×80) y el compilado saltaba de uno a otro.
            return min(abs(math.log(ta / (tb * m))) + (0.0 if m == 1.0 else 0.15)
                       for m in (0.5, 1.0, 2.0))

        pend = list(tramos)
        # Arranca por el tempo REAL más lento, para que el compilado vaya subiendo.
        pend.sort(key=lambda t: t[4] if t[4] == t[4] else 1e9)
        orden, act = [pend.pop(0)], None
        while pend:
            act = orden[-1]
            # Misma toma = mismo tempo real: va pegada aunque el número difiera un poco.
            j = min(range(len(pend)),
                    key=lambda k: (_dist(act[4], pend[k][4]) - (0.05 if pend[k][0] == act[0] else 0.0),
                                   pend[k][1]))
            orden.append(pend.pop(j))
        tramos = orden
    else:
        tramos.sort(key=lambda t: -t[5])

    if a.duracion_max:
        acum, corte = 0.0, []
        for t in tramos:
            if acum >= a.duracion_max * 60:
                break
            corte.append(t)
            acum += t[2] - t[1]
        tramos = corte

    crossfade = a.crossfade if a.crossfade is not None else (1.0 if a.dinamico else 3.0)
    modo_txt = (f"dinámico (umbral q{a.umbral_q}, {a.seg_win}-{a.max_seg_win} ventanas, "
                f"hasta {a.tramos_por_toma} por toma)" if a.dinamico
                else f"fijo ({a.seg_win} ventanas)")
    print(f"\nArmando {len(tramos)} tramos, modo {modo_txt} | perfil {a.perfil} "
          f"| tomas por '{a.sort_by}' | tramos por '{col}' | orden {a.orden} "
          f"| crossfade {crossfade}s"
          + (f" | ajuste de tempo ≤{a.max_stretch*100:.0f}%" if a.ajustar_tempo else "") + "\n")
    if a.solo_lista:
        # Sin renderizar audio: misma selección, tempo del caché, sin ajuste.
        audio = None
        usados = [(t[0], t[1], t[2], t[4], 1.0, 0.0) for t in tramos]
    else:
        audio, usados = compilado.construir(
            [(t[0], t[1], t[2]) for t in tramos], crossfade, a.fade, not a.sin_snap,
            ajustar_tempo=a.ajustar_tempo, max_stretch=a.max_stretch, snap_fin=a.dinamico,
            normalizar=not a.sin_normalizar, objetivo_dbfs=a.nivel)
        if audio is None:
            sys.exit("No se pudo construir el compilado.")

    # El filtro y los parámetros quedan en el nombre, en un .txt al lado y en los
    # tags ID3: hay que poder saber qué es cada compilado sin volver a la terminal.
    filtro = getattr(a, "_filtro", "todo")
    codigo = f"{filtro}_{a.perfil}_{'dinamico' if a.dinamico else 'fijo'}"
    if a.out == "compilado.mp3":
        a.out = f"compilado_{codigo}.mp3"
    params = (f"filtro={filtro} perfil={a.perfil} modo={a.modo} sort={a.sort_by} "
              f"{'dinamico q' + str(a.umbral_q) if a.dinamico else 'fijo ' + str(a.seg_win) + 'win'} "
              f"crossfade={crossfade}s{' tempo<=' + str(a.max_stretch) if a.ajustar_tempo else ''} "
              f"pesos=" + ",".join(f"{k}:{v}" for k, v in _pesos(a).items()))

    # Scores del tramo: mediana de las ventanas que lo componen (dimensiones de
    # ventana) y las de archivo desde la tabla agregada. Todo en percentil 0-1 del
    # corpus completo: 0.90 = mejor que el 90% de las ventanas de todo el archivo.
    dfile = d.set_index("Ruta")
    dims_v = ["timing", "groove", "afinacion", "tonal_outlier", "sonido"]
    dims_f = ["interes", "creatividad", "desarrollo", "ejecucion"]
    lineas = [f"COMPILADO {codigo}", params, "",
              "Scores en percentil del corpus completo (0-1). ejec=ejecución, int=interés, "
              "tim=timing, gro=groove, afi=afinación, ton=tonal_outlier, cre=creatividad, "
              "des=desarrollo, son=sonido. 'tramo' = ese tramo; 'toma' = el archivo entero.", ""]
    t0 = 0.0
    _posiciones = []            # (segundo en el compilado, toma, año) para la carátula
    from . import sync as sy
    _fechas = sy.fechas_locales(con, [t[0] for t in tramos])
    _anios = {r: f.year for r, f in _fechas.items()}
    for (ruta, _, _, toma, _, sc_), (_, ini, fin, tempo, ratio, gan) in zip(tramos, usados):
        dur = fin - ini
        aj = f" x{ratio:.3f}" if abs(ratio - 1.0) > 1e-3 else ""
        aj += f" {gan:+.1f}dB" if abs(gan) > 0.05 else ""
        _posiciones.append((t0, toma, _anios.get(ruta)))
        g = dfw[(dfw["path"] == ruta) & (dfw["start"] >= ini - 1) & (dfw["start"] < fin - 1)]
        w = {k: float(g[k].median()) if len(g) and k in g else float("nan") for k in dims_v}
        w["score"] = float(g["score"].median()) if len(g) else sc_
        fr = dfile.loc[ruta] if ruta in dfile.index else None
        f_ = {k: (float(fr[k]) if fr is not None and k in fr else float("nan")) for k in dims_f}
        lineas.append(f"{_mmss(t0):>6s}  {toma}")
        lineas.append(f"        {_mmss(ini)}-{_mmss(fin)}  {dur:.0f}s  {tempo:.1f}bpm{aj}")
        lineas.append(f"        tramo: score {w['score']:.2f} | tim {w['timing']:.2f} "
                      f"gro {w['groove']:.2f} afi {w['afinacion']:.2f} ton {w['tonal_outlier']:.2f} "
                      f"son {w['sonido']:.2f}")
        lineas.append(f"        toma:  ejec {f_['ejecucion']:.2f} | int {f_['interes']:.2f} "
                      f"cre {f_['creatividad']:.2f} des {f_['desarrollo']:.2f}")
        t0 += dur - crossfade
    print("\n".join("  " + l for l in lineas[5:]))

    from . import config as _cfg
    if audio is not None:
        audio.export(a.out, format="mp3", bitrate=a.bitrate,
                     tags={"title": f"Zapa-IA {codigo}", "artist": _cfg.banda()["nombre"],
                           "album": "Zapa-IA compilados", "comment": params})
        lineas.append(f"\nDuración total: {len(audio)/60000:.1f} min")
        # Carátula con la lista: es la "foto" que muestran Drive y el celular.
        try:
            items = [(_mmss(t), toma.replace(".mp3", "") + (f"  ·  {anio}" if anio else ""))
                     for t, toma, anio in _posiciones]
            png = str(Path(a.out).with_suffix(".png"))
            compilado.portada(f"{a.perfil.upper()} · {filtro.replace('_', ' ')}",
                              f"Zapa-IA · {_cfg.banda()['nombre']} · {len(audio)/60000:.0f} min · "
                              f"{'dinámico' if a.dinamico else 'fijo'}", items, png)
            compilado.embeber_portada(a.out, png)
            # Versión MP4: la app de Drive no muestra la carátula del MP3, pero
            # reproduce video con la lista y el tema actual resaltado.
            if not a.sin_video:
                mp4 = str(Path(a.out).with_suffix(".mp4"))
                pos_v = [(t, toma.replace(".mp3", ""), anio) for t, toma, anio in _posiciones]
                compilado.video_compilado(
                    a.out, f"{a.perfil.upper()} · {filtro.replace('_', ' ')}",
                    f"Zapa-IA · {_cfg.banda()['nombre']} · {len(audio)/60000:.0f} min", pos_v, mp4,
                    str(Path(a.out).with_suffix("")) + "_cuadros",
                    imagenes_dir=(a.imagenes if os.path.isdir(a.imagenes or "") else None),
                    visualizador=not a.sin_visualizador, semilla=codigo)
                print(f"  video -> {mp4}")
        except Exception as e:
            print(f"  (sin carátula/video: {e})")
    Path(a.out).with_suffix(".txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    # manifest.json: lo que una app necesita para mostrar capítulos y recibir "me gusta"
    # por tramo (ver docs/review-2026-09-12.md §5). Mismo nombre que el MP3.
    import json as _json
    from . import config as _cfg
    _banda = _cfg.banda()
    manifest = {"banda": _banda["nombre"], "banda_clave": _banda["clave"],
                "compilado": codigo, "perfil": a.perfil, "filtro": filtro, "params": params,
                "fver": __import__("zapaia").FEATURE_VERSION, "tramos": []}
    tt = 0.0
    for (ruta, _, _, toma, _, sc_), (_, ini, fin, tempo, ratio, gan) in zip(tramos, usados):
        dur = fin - ini
        g = dfw[(dfw["path"] == ruta) & (dfw["start"] >= ini - 1) & (dfw["start"] < fin - 1)]
        manifest["tramos"].append({
            "ganancia_db": round(gan, 1),
            "pos_s": round(tt, 2), "dur_s": round(dur, 2), "toma": toma,
            "origen_ini_s": round(ini, 2), "origen_fin_s": round(fin, 2),
            "tempo": None if tempo != tempo else round(tempo, 1), "anio": _anios.get(ruta),
            "score": round(float(g["score"].median()), 3) if len(g) else None,
            **{k: round(float(g[k].median()), 3) for k in dims_v if len(g) and k in g},
        })
        tt += dur - crossfade
    Path(a.out).with_suffix(".json").write_text(_json.dumps(manifest, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    if audio is not None:
        print(f"\nDuración total: {len(audio)/60000:.1f} min  ->  {a.out}  (+ .txt con la lista)")
    else:
        print(f"\nLista -> {Path(a.out).with_suffix('.txt')}  (sin audio: --solo-lista)")


# La carpeta de salida en Drive se direcciona por ID, no por nombre (Drive admite
# nombres duplicados). El ID vive en zapaia_local.json, NO en el repo público.
def _drive_salida():
    from . import config
    return config.remote("salida")


def _drive_feedback():
    return _drive_salida() + "feedback"


def _filtrar_fechas(a, con, d):
    """Aplica --meses / --desde / --ultima-sesion al DataFrame de archivos, si se pidieron."""
    from . import sync as sy
    desde = sy.desde_argumentos(getattr(a, "meses", None), getattr(a, "desde", None))
    ultima = getattr(a, "ultima_sesion", False)
    if desde is None and not ultima:
        return d
    fechas = sy.fechas_locales(con, list(d["Ruta"]))
    antes = len(d)
    d = sy.filtrar_por_fecha(d, fechas, desde=desde, ultima_sesion=ultima)
    if d.empty:
        sys.exit("Ningún archivo en ese rango de fechas. ¿Corriste 'zapaia sync'?")
    a._filtro = sy.etiqueta_filtro(d, desde=desde, ultima_sesion=ultima,
                                   meses=getattr(a, "meses", None))
    rango = f"{sy.dia_local(d['_fecha'].min())} a {sy.dia_local(d['_fecha'].max())}"
    print(f"Filtro de fechas [{a._filtro}]: {antes} -> {len(d)} tomas ({rango})")
    return d


def cmd_sync(a):
    """Baja de Drive los MP3 nuevos de los últimos N meses y (opcional) los procesa."""
    from . import sync as sy
    if a.instalar_launchd is not None:
        plist, log, ok, err = instalar_launchd(a.root, a.instalar_launchd, a.meses, a.jobs,
                                               sys.executable)
        print(f"LaunchAgent {'instalado' if ok else 'ERROR: ' + err}: {plist}\n"
              f"corre al iniciar sesión y cada {a.instalar_launchd} h con la máquina prendida; "
              f"log en {log}\n"
              f"para sacarlo: launchctl unload -w {plist} && rm {plist}")
        return
    if a.destino is None:
        a.destino = str(Path(a.root) / "drive")
    con = cache.connect(a.db)
    desde = sy.desde_argumentos(a.meses, a.desde)
    excluir = tuple(a.excluir) if a.excluir else sy.EXCLUIR_DEFAULT
    bajados, sl, sm, errores = sy.sincronizar(
        con, a.root, a.destino, desde, origen=a.origen, excluir=excluir, dry_run=a.dry_run)
    print(f"\n{'Bajaría' if a.dry_run else 'Bajados'}: {len(bajados)} | ya locales (registrados): {sl} "
          f"| ya en manifest: {sm} | errores: {len(errores)}")
    if a.dry_run or not bajados:
        return
    if a.extraer:
        print("\nProcesando lo nuevo...")
        ns = argparse.Namespace(root=a.root, db=a.db, sr=a.sr, win=a.win, hop=a.hop,
                                jobs=a.jobs, sample=None, seed=0, include=[], files_from=None,
                                force=False, reintentar_errores=False)
        cmd_extract(ns)


def _sync_previo(a):
    """--sync: antes de rankear/compilar, traer lo nuevo de Drive y procesarlo."""
    ns = argparse.Namespace(root=a.root, db=a.db, sr=a.sr, win=a.win, hop=a.hop,
                            meses=a.sync_meses, desde=None, origen=None,
                            destino=None, excluir=None, dry_run=False, extraer=True,
                            jobs=max(os.cpu_count() // 2, 1), instalar_launchd=None)
    print("── sync previo ──")
    cmd_sync(ns)
    print("──")


def instalar_launchd(root, cada_horas, meses, jobs, python):
    """LaunchAgent que corre 'zapaia sync --extraer' al iniciar sesión y cada N horas
    mientras la máquina esté prendida. No depende de una hora fija: si estuvo apagada,
    corre en cuanto vuelve. Sin novedades tarda segundos."""
    proyecto = str(Path(".").resolve())
    label = "com.zapaia.sync"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    log = Path(proyecto) / "sync.log"
    args = [python, "-m", "zapaia", "sync", root, "--meses", str(meses), "--extraer",
            "--jobs", str(jobs)]
    xml_args = "\n".join(f"        <string>{x}</string>" for x in args)
    contenido = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key><array>
{xml_args}
    </array>
    <key>WorkingDirectory</key><string>{proyecto}</string>
    <key>RunAtLoad</key><true/>
    <key>StartInterval</key><integer>{int(cada_horas * 3600)}</integer>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
    <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
</dict></plist>
"""
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(contenido, encoding="utf-8")
    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    r = subprocess.run(["launchctl", "load", "-w", str(plist)], capture_output=True, text=True)
    return plist, log, r.returncode == 0, r.stderr.strip()


def _mmss(s):
    return "%d:%02d" % (int(s) // 60, int(s) % 60)


def cmd_comparar(a):
    """Arma un lote de pares A/B y lo hace escuchar (afplay) o lo sube a Drive."""
    from . import feedback as fb
    con, dfw, files = _load(a)
    d = _agg(a, dfw, files)
    if a.min_win:
        d = d[d["n_win"] >= a.min_win]
    root = str(Path(a.root).resolve())
    temas = dedupe.temas_por_nombre([os.path.basename(p) for p in _mp3s(a.root)])
    grupos = dedupe.find_groups(cache.load_signatures(con, root))
    # El tramo de cada archivo se elige por ejecución LIMPIA: queremos comparar
    # ideas, no castigar cromatismos antes de que el humano opine.
    c = fb.candidatos(dfw, d, a.seg_win, a.win, col="ejec_limpia")
    if len(c) < 4:
        sys.exit("Muy pocos candidatos. Bajá --seg-win o --min-win.")
    pares = fb.muestrear_pares(c, a.n, temas, grupos, seed=a.seed,
                               ya_vistos=fb.pares_vistos(con))
    if not pares:
        sys.exit("No quedan pares nuevos con estos criterios.")

    lote = fb.nuevo_lote(con)
    outdir = Path(a.out_dir) / f"lote_{lote:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Lote {lote}: {len(pares)} pares, clips de {a.dur:.0f}s -> {outdir}\n")

    lineas = [f"LOTE {lote} — ¿cuál rescatarías? Respondé: '{'{'}num{'}'} A' / B / ambos / ninguno",
              "(por ejemplo: 1 A, 2 B, 3 ninguno, 4 ambos)", ""]
    registro = []
    for num, (i, j, tipo) in enumerate(pares, start=1):
        fa, fb_ = c.loc[i], c.loc[j]
        clips = {}
        for lado, r in (("A", fa), ("B", fb_)):
            seg = fb.segmento_id(con, r["Ruta"], r["start"], r["end"])
            out = outdir / f"par_{num:03d}_{lado}.mp3"
            ini, dur = fb.exportar_clip(r["Ruta"], r["start"], a.dur, str(out),
                                        snap=not a.sin_snap)
            clips[lado] = (seg, str(out), r["Toma"], ini)
        fb.guardar_par(con, lote, num, clips["A"][0], clips["B"][0], tipo)
        lineas.append(f"{num:03d}  A: {clips['A'][2]} ({_mmss(clips['A'][3])})   "
                      f"B: {clips['B'][2]} ({_mmss(clips['B'][3])})")
        registro.append((num, clips))
        print(f"  {num:03d}  A: {clips['A'][2][:40]:42s} B: {clips['B'][2][:40]}")
    (outdir / "lote.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    if a.un_mp3 or a.drive:
        # Un solo archivo con todo el lote: en el celular es mucho más cómodo que 2N clips.
        todo = outdir / f"lote_{lote:03d}_completo.mp3"
        indice = fb.armar_lote_mp3([(num, cl["A"][1], cl["B"][1]) for num, cl in registro],
                                   str(todo))
        por_par = {}
        for num, lado, seg in indice:
            por_par.setdefault(num, {})[lado] = seg
        lineas_idx = [f"LOTE {lote} — un solo MP3. Tono agudo = A, grave = B.", ""]
        for num, cl in registro:
            lineas_idx.append(f"{num:03d}  {_mmss(por_par[num]['A'])} A: {cl['A'][2]}"
                              f"   |   {_mmss(por_par[num]['B'])} B: {cl['B'][2]}")
        (outdir / f"lote_{lote:03d}_completo.txt").write_text("\n".join(lineas_idx) + "\n",
                                                               encoding="utf-8")
        print(f"\nUn solo MP3: {todo} ({len(AudioSegment.from_file(str(todo)))/60000:.1f} min)")

    if a.drive:
        dest = f"{_drive_feedback()}/lote_{lote:03d}/"
        print(f"\nSubiendo a {dest} ...")
        r = subprocess.run(["rclone", "copy", str(outdir), dest], capture_output=True, text=True)
        print("  listo" if r.returncode == 0 else f"  ERROR rclone: {r.stderr.strip()[:300]}")
        print(f"\nCuando escuches, respondé con:\n  zapaia feedback importar \"1 A, 2 B, ...\" --lote {lote}")
        return

    if a.lote:
        print(f"\nRespondé después con:  zapaia feedback importar \"1 A, 2 B, ...\" --lote {lote}")
        return

    print("\nEscuchá cada par y contestá. [a] [b] [ambos] [ninguno] [r]epetir [s]altar [q] salir\n")
    for num, clips in registro:
        while True:
            print(f"--- par {num:03d} ---  A: {clips['A'][2]}  |  B: {clips['B'][2]}")
            fb.reproducir(clips["A"][1])
            fb.reproducir(clips["B"][1])
            try:
                resp = input("¿cuál rescatarías? > ").strip().lower()
            except EOFError:
                resp = "q"
            if resp in ("a", "b", "ambos", "ninguno"):
                fb.responder(con, lote, num, resp.upper() if resp in ("a", "b") else resp)
                break
            if resp == "s":
                break
            if resp == "q":
                print("Guardado lo respondido hasta acá.")
                return
            # 'r' o cualquier otra cosa: repetir
    print("\nLote completo. Mirá el resultado con:  zapaia feedback resumen")


def cmd_feedback(a):
    from . import feedback as fb
    from scipy.stats import spearmanr
    con = cache.connect(a.db)

    if a.accion == "pendientes":
        rows = con.execute(
            "SELECT p.lote, p.num, sa.path, sb.path FROM pares p "
            "JOIN segmentos sa ON sa.id=p.seg_a JOIN segmentos sb ON sb.id=p.seg_b "
            "WHERE p.eleccion IS NULL ORDER BY p.lote, p.num").fetchall()
        print(f"{len(rows)} pares sin responder")
        for lote, num, pa, pb in rows:
            print(f"  lote {lote} par {num:03d}  A: {os.path.basename(pa)[:40]:42s} B: {os.path.basename(pb)[:40]}")
        return

    if a.accion == "importar":
        if a.lote is None:
            row = con.execute("SELECT MAX(lote) FROM pares WHERE eleccion IS NULL").fetchone()
            if not row or row[0] is None:
                sys.exit("No hay lotes con pares pendientes; indicá --lote.")
            a.lote = int(row[0])
        respuestas = fb.parsear_respuestas(a.texto)
        ok = 0
        for num, e in respuestas:
            n = fb.responder(con, a.lote, num, e)
            if n == 0:
                print(f"  par {num:03d}: no existe en el lote {a.lote}")
            else:
                ok += 1
        print(f"{ok} respuestas guardadas en el lote {a.lote}")
        return

    # resumen
    comps = fb.comparaciones_desde_db(con)
    n_resp = con.execute("SELECT COUNT(*) FROM pares WHERE eleccion IS NOT NULL").fetchone()[0]
    if not comps:
        sys.exit("Todavía no hay respuestas. Corré 'zapaia comparar'.")
    bt = fb.bradley_terry(comps)
    segs = {sid: (p, s, e) for sid, p, s, e in con.execute("SELECT id, path, start, end FROM segmentos")}

    # dimensiones del archivo de cada segmento (un segmento por archivo, hoy)
    _, dfw, files = _load(a)
    d = _agg(a, dfw, files).set_index("Ruta")
    filas = []
    for sid, score in bt.items():
        p = segs[sid][0]
        if p in d.index:
            r = d.loc[p]
            filas.append({"seg": sid, "Toma": os.path.basename(p), "bt": score,
                          **{k: float(r[k]) for k in ("ejecucion", "interes", "balance", "timing",
                                                      "groove", "afinacion", "tonal_outlier",
                                                      "creatividad", "desarrollo", "sonido")}})
    t = pd.DataFrame(filas)
    print(f"\n{n_resp} respuestas | {len(t)} segmentos con score latente (Bradley–Terry)\n")

    # ¿qué gana cuando se enfrenta una gema con una performance?
    gp = con.execute("SELECT eleccion, seg_a, seg_b FROM pares WHERE tipo='gem_vs_perf' "
                     "AND eleccion IS NOT NULL").fetchall()
    if gp:
        gana_gem = 0
        for e, sa, sb in gp:
            ia, ib = d.loc[segs[sa][0]], d.loc[segs[sb][0]]
            gem_es_a = ia["interes"] - ia["ejecucion"] > ib["interes"] - ib["ejecucion"]
            if (e == "A" and gem_es_a) or (e == "B" and not gem_es_a):
                gana_gem += 1
        print(f"  gema vs performance: la gema ganó {gana_gem}/{len(gp)} veces "
              f"(ambos/ninguno cuentan como no-gana)\n")

    print("  ¿qué dimensión predice tu criterio?  (Spearman entre score latente y dimensión)")
    res = []
    for k in ("ejecucion", "interes", "balance", "timing", "groove", "afinacion",
              "tonal_outlier", "creatividad", "desarrollo", "sonido"):
        if t[k].nunique() > 1 and len(t) >= 5:
            rho, pv = spearmanr(t["bt"], t[k])
            res.append((k, rho, pv))
    for k, rho, pv in sorted(res, key=lambda x: -abs(x[1])):
        flag = " ***" if pv < 0.01 else " *" if pv < 0.05 else ""
        print(f"    {k:14s} rho={rho:+.2f}  p={pv:.3f}{flag}")
    if len(t) < 30:
        print(f"\n  Con {len(t)} segmentos esto es orientativo. A partir de ~30-50 respuestas "
              "las correlaciones empiezan a significar algo; a partir de 150 se puede entrenar.")
    print("\n  top por tu criterio:")
    for _, r in t.sort_values("bt", ascending=False).head(a.top).iterrows():
        print(f"    {r['bt']:+.2f}  eje={r['ejecucion']:.2f} int={r['interes']:.2f}  {r['Toma']}")
    if a.out:
        t.sort_values("bt", ascending=False).to_csv(a.out, index=False, float_format="%.4f")
        print(f"\n  CSV -> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="zapaia", description="Ranking de tomas de ensayo")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("root")
        sp.add_argument("--db", default=DEFAULTS["db"])
        sp.add_argument("--sr", type=int, default=DEFAULTS["sr"])
        sp.add_argument("--win", type=float, default=DEFAULTS["win"])
        sp.add_argument("--hop", type=float, default=DEFAULTS["hop"])

    def weights(sp):
        sp.add_argument("--peso-sonido", type=float, default=0.35)
        sp.add_argument("--peso-ejec", type=float, default=0.65)
        sp.add_argument("--shrink", type=float, default=4.0,
                        help="fuerza del shrinkage por cantidad de ventanas (0 = apagado)")
        sp.add_argument("--min-win", type=int, default=0,
                        help="descartar tomas con menos de N ventanas (N*30s de audio)")
        sp.add_argument("--modo", choices=["mixto", "veto"], default="mixto",
                        help="veto: el sonido solo descalifica, no suma al puntaje")
        sp.add_argument("--sonido-min", type=float, default=0.15,
                        help="percentil de sonido bajo el cual se descarta (modo veto)")
        sp.add_argument("--sync", action="store_true",
                        help="antes de empezar, bajar de Drive lo nuevo (últimos --sync-meses) y procesarlo")
        sp.add_argument("--sync-meses", type=int, default=3)
        sp.add_argument("--meses", type=int, default=None,
                        help="solo tomas de los últimos N meses (fecha de Drive o mtime)")
        sp.add_argument("--desde", default=None, help="solo tomas desde AAAA-MM-DD")
        sp.add_argument("--ultima-sesion", action="store_true",
                        help="solo la última zapada: los archivos a <36 h del más reciente")
        sp.add_argument("--perfil", choices=list(scoring.PERFILES), default="balance",
                        help="balance: mezcla de todo (default). performances: solo "
                             "ejecución. ideas: solo interés musical, sin castigar "
                             "cromatismos. gems: interés alto con ejecución baja/media")
        sp.add_argument("--w-timing", type=float, default=0.35)
        sp.add_argument("--w-groove", type=float, default=0.30)
        sp.add_argument("--w-tonal", "--w-notas", dest="w_tonal", type=float, default=0.10,
                        help="peso de tonal_outlier (energía fuera de la tonalidad local; "
                             "antes 'notas'). Solo cuenta en ejecución, nunca en interés")
        sp.add_argument("--w-afinacion", type=float, default=0.15)
        sp.add_argument("--w-desarrollo", type=float, default=0.20)
        sp.add_argument("--w-creatividad", type=float, default=0.20,
                        help="peso de 'pasa algo interesante' (anti riff en loop)")

    e = sub.add_parser("extract", help="extraer features al caché")
    common(e)
    e.add_argument("--jobs", type=int, default=max(os.cpu_count() // 2, 1))
    e.add_argument("--sample", type=int, help="procesar solo N archivos al azar")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--include", nargs="*", default=[], help="subcadenas que sí entran al sample")
    e.add_argument("--files-from", help="archivo de texto con una ruta por línea")
    e.add_argument("--force", action="store_true")
    e.add_argument("--reintentar-errores", action="store_true",
                   help="volver a procesar los archivos que fallaron aunque no hayan cambiado")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("rank", help="rankear desde el caché")
    common(r); weights(r)
    r.add_argument("--out", default="ranking.csv")
    r.add_argument("--windows-out")
    r.add_argument("--top", type=int, default=25)
    r.add_argument("--dedupe", action="store_true")
    r.set_defaults(func=cmd_rank)

    g = sub.add_parser("diag", help="distribución de features (detecta pesos decorativos)")
    common(g); weights(g)
    g.set_defaults(func=cmd_diag)

    u = sub.add_parser("dupes", help="listar duplicados por huella de audio")
    common(u)
    u.add_argument("--sim", type=float, default=0.997,
                   help="similitud CENTRADA mínima: duplicados exactos dan 1.000, remaster de la misma toma 0.998, tomas distintas del mismo tema <= 0.994")
    u.add_argument("--dur-tol", type=float, default=2.0)
    u.set_defaults(func=cmd_dupes)

    m = sub.add_parser("compilado", help="un MP3 con los mejores tramos pegados")
    common(m); weights(m)
    m.add_argument("--out", default="compilado.mp3")
    m.add_argument("--top", type=int, default=10, help="cuántas tomas entran")
    m.add_argument("--seg-win", type=int, default=3,
                   help="ventanas por tramo (3 = 90s de cada toma)")
    m.add_argument("--crossfade", type=float, default=None,
                   help="segundos de crossfade (default 3, o 1 en modo dinámico)")
    m.add_argument("--fade", type=float, default=2.0, help="fade de entrada y salida")
    m.add_argument("--orden", choices=["tempo", "score"], default="tempo",
                   help="tempo = empalmes más suaves; score = de mejor a peor")
    m.add_argument("--bitrate", default="192k")
    m.add_argument("--max-por-tema", type=int, default=1,
                   help="maximo de tomas del mismo tema (0 = sin tope)")
    m.add_argument("--diversidad", type=float, default=0.15,
                   help="similitud maxima permitida entre tomas elegidas "
                        "(0 = apagado; 0.20 evita repetir tema)")
    m.add_argument("--seg-by", default="score",
                   choices=["score", "ejec_ventana", "ejec_limpia", "timing", "groove",
                            "afinacion", "tonal_outlier", "sonido"],
                   help="criterio para elegir el tramo DENTRO de cada toma "
                        "(score = el del perfil; las dimensiones de archivo no aplican a 90s)")
    m.add_argument("--sin-snap", action="store_true",
                   help="no pegar los cortes al beat más cercano")
    m.add_argument("--dinamico", action="store_true",
                   help="tramos de largo variable: siguen mientras la racha se mantenga "
                        "sobre el umbral; varios por toma")
    m.add_argument("--umbral-q", type=float, default=0.60,
                   help="cuantil del score de ventana que define 'racha buena' (dinámico)")
    m.add_argument("--max-seg-win", type=int, default=10,
                   help="tope de ventanas por tramo (10 = 5 min) en modo dinámico")
    m.add_argument("--tramos-por-toma", type=int, default=2)
    m.add_argument("--max-silencio", type=float, default=0.08,
                   help="una ventana con más de esta fracción de silencio corta la racha "
                        "(dinámico). 0.08 = 2.4 s en 30 s")
    m.add_argument("--duracion-max", type=float, default=None,
                   help="minutos totales aproximados del compilado")
    m.add_argument("--ajustar-tempo", action="store_true",
                   help="time-stretch leve para que el tempo enganche con el tramo anterior")
    m.add_argument("--max-stretch", type=float, default=0.04,
                   help="ajuste máximo de tempo (0.04 = 4%%)")
    m.add_argument("--imagenes", default="imagenes",
                   help="carpeta con fotos para el fondo del video (una distinta por tramo)")
    m.add_argument("--sin-visualizador", action="store_true",
                   help="video estático (sin la onda que se mueve con la música)")
    m.add_argument("--sin-normalizar", action="store_true",
                   help="no igualar el volumen entre tramos")
    m.add_argument("--nivel", type=float, default=-16.0,
                   help="tope del nivel objetivo (dBFS). El objetivo real es el tramo más fuerte "
                        "del compilado; solo se SUBEN los que están por debajo, nunca se baja")
    m.add_argument("--sin-video", action="store_true",
                   help="no generar el MP4 con la lista (Drive no muestra la carátula del MP3)")
    m.add_argument("--solo-lista", action="store_true",
                   help="no renderizar audio: solo escribir el .txt con la lista y los scores")
    m.set_defaults(func=cmd_compilado)

    v = sub.add_parser("eval", help="medir el ranking contra refs.txt")
    common(v); weights(v)
    v.add_argument("--refs", default="refs.txt")
    v.set_defaults(func=cmd_eval)

    c = sub.add_parser("comparar", help="pares A/B para escuchar y decir cuál rescatarías")
    common(c); weights(c)
    c.add_argument("--n", type=int, default=10, help="cantidad de pares del lote")
    c.add_argument("--dur", type=float, default=40.0, help="segundos por clip")
    c.add_argument("--seg-win", type=int, default=2,
                   help="ventanas contiguas para elegir el tramo (2 = 60 s)")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--out-dir", default="feedback")
    c.add_argument("--lote", action="store_true",
                   help="solo generar los clips y lote.txt, sin reproducir")
    c.add_argument("--drive", action="store_true",
                   help="generar y subir el lote a Drive con rclone (para escuchar en el celular)")
    c.add_argument("--sin-snap", action="store_true")
    c.add_argument("--un-mp3", action="store_true",
                   help="además de los clips, un solo MP3 con todo el lote (implícito con --drive)")
    c.set_defaults(func=cmd_comparar)

    f = sub.add_parser("feedback", help="importar respuestas y ver qué dimensión predice tu criterio")
    f.add_argument("accion", choices=["importar", "resumen", "pendientes"])
    f.add_argument("texto", nargs="?", default="", help='importar: "1 A, 2 B, 3 ninguno"')
    # Acá root es opcional: el cwd del proyecto es prefijo de todas las rutas del caché.
    f.add_argument("--root", default=".")
    f.add_argument("--db", default=DEFAULTS["db"])
    f.add_argument("--sr", type=int, default=DEFAULTS["sr"])
    f.add_argument("--win", type=float, default=DEFAULTS["win"])
    f.add_argument("--hop", type=float, default=DEFAULTS["hop"])
    weights(f)
    f.add_argument("--lote", type=int, default=None)
    f.add_argument("--top", type=int, default=10)
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_feedback)

    s = sub.add_parser("sync", help="bajar de Drive los ensayos nuevos (por fecha) y procesarlos")
    common(s)
    s.add_argument("--meses", type=int, default=3, help="últimos N meses (default 3)")
    s.add_argument("--desde", default=None, help="o desde AAAA-MM-DD")
    s.add_argument("--origen", default=None, help="remote rclone de origen (default: Nebulosa por ID)")
    s.add_argument("--destino", default=None,
                   help="carpeta local (default <root>/drive/AAAA-MM/)")
    s.add_argument("--excluir", nargs="*", default=None,
                   help="subcarpetas de Drive que no son ensayos crudos")
    s.add_argument("--dry-run", action="store_true", help="mostrar qué bajaría, sin bajar")
    s.add_argument("--extraer", action="store_true", help="procesar lo bajado al terminar")
    s.add_argument("--jobs", type=int, default=max(os.cpu_count() // 2, 1))
    s.add_argument("--instalar-launchd", type=float, metavar="HORAS", default=None,
                   help="instalar un LaunchAgent que corra sync --extraer al iniciar sesión y "
                        "cada N horas mientras la máquina esté prendida")
    s.set_defaults(func=cmd_sync)

    for sp in (r, v, m, c, f):
        sp.add_argument("--sort-by", default="score_med",
                        choices=["score_med", "score_best", "ejecucion", "interes",
                                 "balance", "gems", "timing", "groove", "afinacion",
                                 "tonal_outlier", "desarrollo", "creatividad", "sonido"],
                        help="dimensión o composite por el que ordenar el ranking")

    a = p.parse_args(argv)
    a.func(a)
