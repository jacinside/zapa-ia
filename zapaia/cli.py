"""CLI: extract / rank / diag / dupes / eval."""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    tramos, info = [], []
    for _, r in d.iterrows():
        g = dfw[dfw["path"] == r["Ruta"]]
        ini, fin = compilado.mejor_tramo(g, a.seg_win, a.win, col=col)
        tramos.append((r["Ruta"], ini, fin))
        info.append((r["Toma"], r.get("tempo", float("nan")), r[a.sort_by]))

    print(f"\nArmando con {len(tramos)} tramos de ~{a.seg_win * a.win / 60:.1f} min "
          f"| perfil {a.perfil} | tomas por '{a.sort_by}' | tramos por '{col}' "
          f"| orden {a.orden}\n")
    audio, usados = compilado.construir(tramos, a.crossfade, a.fade, not a.sin_snap)
    if audio is None:
        sys.exit("No se pudo construir el compilado.")

    t = 0.0
    for (toma, tempo, sc_), (_, ini, fin) in zip(info, usados):
        dur = fin - ini
        print(f"  {int(t)//60:3d}:{int(t)%60:02d}  {toma[:44]:46s} "
              f"desde {int(ini)//60:d}:{int(ini)%60:02d}  "
              f"({dur:4.0f}s, {tempo:5.1f}bpm, {a.sort_by}={sc_:.2f})")
        t += dur - a.crossfade

    audio.export(a.out, format="mp3", bitrate=a.bitrate)
    print(f"\nDuración total: {len(audio)/60000:.1f} min  ->  {a.out}")


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
    m.add_argument("--crossfade", type=float, default=3.0)
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
    m.set_defaults(func=cmd_compilado)

    v = sub.add_parser("eval", help="medir el ranking contra refs.txt")
    common(v); weights(v)
    v.add_argument("--refs", default="refs.txt")
    v.set_defaults(func=cmd_eval)

    for sp in (r, v, m):
        sp.add_argument("--sort-by", default="score_med",
                        choices=["score_med", "score_best", "ejecucion", "interes",
                                 "balance", "gems", "timing", "groove", "afinacion",
                                 "tonal_outlier", "desarrollo", "creatividad", "sonido"],
                        help="dimensión o composite por el que ordenar el ranking")

    a = p.parse_args(argv)
    a.func(a)
