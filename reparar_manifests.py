#!/usr/bin/env python
"""Recalcula los scores de manifests YA publicados, sin tocar el audio.

Por qué existe: el `score` de cada tramo se calcula contra el corpus completo
(normalización por percentil), así que cuando el corpus cambia —por ejemplo al
sacar 25 stems de multipista que eran el 19% de las horas— los manifests viejos
quedan con scores medidos contra otra escala. Rearmar el compilado entero sería
carísimo y, peor, cambiaría la selección: otros tramos, otras posiciones, y los
votos ya emitidos apuntarían a otra música.

Lo que SÍ se toca: score, las dimensiones y score_dist.
Lo que NO se toca: pos_s, dur_s, toma, drive_id, tempo, anio, ganancia_db. Las
posiciones sólo existen si se renderiza el audio (ver el bug de --solo-lista del
17/9), así que se copian tal cual del manifest original.

Uso:  ./reparar_manifests.py <carpeta-corpus> [--aplicar]
      Sin --aplicar sólo muestra qué cambiaría.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zapaia import cli, config, score as scoring

DIMS_V = ["timing", "groove", "afinacion", "tonal_outlier", "sonido"]
DIMS_F = ["interes", "creatividad", "desarrollo", "ejecucion"]


def dataframes(corpus, perfil):
    """(ventanas, archivos) puntuados con ESTE perfil, sobre el corpus actual."""
    # Mismos parámetros con los que se arman los compilados (ver compilar_*.sh):
    # modo veto, sonido fuera del puntaje. Si cambian allá, cambian acá.
    a = argparse.Namespace(
        root=corpus, db=".zapaia_cache.db", perfil=perfil, modo="veto", sonido_min=0.15,
        peso_sonido=0.0, peso_ejec=1.0, shrink=4.0, sr=22050, win=30.0, hop=30.0,
        **{"w_" + ("tonal" if k == "tonal_outlier" else k): v
           for k, v in scoring.PESOS_DEFAULT.items()})
    con, dfw, files = cli._load(a)
    return con, dfw, cli._agg(a, dfw, files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--aplicar", action="store_true", help="escribir y subir a Drive")
    args = ap.parse_args()

    dest = config.remote("salida")
    # --include también deja pasar carpetas (feedback/), así que se filtran.
    listado = [x for x in subprocess.run(["rclone", "lsf", dest, "--include", "compilado_*.json"],
                                         capture_output=True, text=True).stdout.split()
               if x.endswith(".json") and "/" not in x]
    if not listado:
        sys.exit("no encontré manifests en Drive")

    cache = {}
    tmp = Path(tempfile.mkdtemp())
    for nombre in sorted(listado):
        subprocess.run(["rclone", "copyto", dest + nombre, str(tmp / nombre)], check=True)
        m = json.loads((tmp / nombre).read_text(encoding="utf-8"))
        perfil = m.get("perfil")
        if not perfil:
            print(f"  {nombre}: sin perfil, salteo"); continue
        if perfil not in cache:
            cache[perfil] = dataframes(args.corpus, perfil)
        _, dfw, d = cache[perfil]
        dfile = d.set_index("Ruta")
        from zapaia import sync as _sy
        _fechas = _sy.fechas_locales(cache[perfil][0], list(d["Ruta"]))
        # toma -> ruta: los nombres de archivo son únicos en este corpus (se
        # verifica abajo); si alguno se repite, se salta para no adivinar.
        por_toma = {}
        for r in d["Ruta"]:
            n = Path(r).name
            por_toma.setdefault(n, []).append(r)

        cambios, sin_datos = [], 0
        for t in m["tramos"]:
            rutas = por_toma.get(t["toma"], [])
            if len(rutas) != 1:
                sin_datos += 1; continue
            ruta = rutas[0]
            g = dfw[(dfw["path"] == ruta) & (dfw["start"] >= t["origen_ini_s"] - 1)
                    & (dfw["start"] < t["origen_fin_s"] - 1)]
            if not len(g):
                sin_datos += 1; continue
            viejo = t.get("score")
            t["score"] = round(float(g["score"].median()), 3)
            for k in DIMS_V:
                if k in g: t[k] = round(float(g[k].median()), 3)
            if not t.get("fecha"):
                f = _fechas.get(ruta)
                if f is not None: t["fecha"] = f.date().isoformat()
            fr = dfile.loc[ruta] if ruta in dfile.index else None
            if fr is not None:
                for k in DIMS_F:
                    if k in fr and fr[k] == fr[k]: t[k] = round(float(fr[k]), 3)
            if viejo is not None:
                cambios.append(t["score"] - viejo)
        m["score_dist"] = {k: round(float(dfw["score"].quantile(q)), 4)
                           for k, q in [("p10", .10), ("p25", .25), ("p50", .50),
                                        ("p75", .75), ("p90", .90), ("p99", .99)]}
        m["score_dist"]["max"] = round(float(dfw["score"].max()), 4)
        m["score_dist"]["min"] = round(float(dfw["score"].min()), 4)

        med = sorted(abs(c) for c in cambios)
        print("  %-52s perfil=%-12s tramos=%2d  sin datos=%d  |Δscore| mediano=%.3f máx=%.3f"
              % (nombre, perfil, len(m["tramos"]), sin_datos,
                 med[len(med) // 2] if med else 0, max(med) if med else 0))
        if args.aplicar:
            (tmp / nombre).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
            subprocess.run(["rclone", "copyto", str(tmp / nombre), dest + nombre], check=True)
    print("\n(dry-run: no se escribió nada)" if not args.aplicar else "\nSubidos.")


if __name__ == "__main__":
    main()
