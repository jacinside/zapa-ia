#!/usr/bin/env python
"""Reapunta votos y notas cuando se regenera un compilado con otra selección.

Por qué hace falta: el feedback se guarda por (compilado, tramo), pero al
regenerar un compilado el tramo N pasa a ser OTRA música. Sin migrar, un "me
gusta" queda señalando algo que nadie votó — peor que perderlo.

Lo que lo hace posible: cada evento guarda `toma` (el archivo original), y cada
manifest guarda `origen_ini_s`/`origen_fin_s` (qué pedazo de ese archivo es el
tramo). Con eso se puede encontrar el MISMO pedazo de la MISMA zapada en otro
compilado, que es lo que realmente le gustó a la persona; el índice de tramo es
sólo dónde cayó esa vez.

Orden de preferencia para reubicar un voto:
  1. el mismo compilado regenerado, si la toma sigue estando
  2. otro compilado que la tenga, priorizando el que cubra la misma parte
Si no hay ninguno, se anula con `undo`: dejarlo apuntando a otra música es peor.

Los .jsonl son append-only y se respeta: no se reescribe nada, se AGREGA el
`undo` del viejo y el evento corregido, ambos con un campo `migracion`.

Uso:  ./migrar_votos.py <dir-nuevos> <dir-publicados> [--aplicar]
"""
import argparse, json, subprocess, sys, tempfile, datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zapaia import config

VOTOS = ("like", "star", "dislike")


def indexar(dirs):
    """toma -> [(compilado, tramo, ini, fin)] de todos los manifests dados."""
    idx, mans = {}, {}
    for d in dirs:
        for j in Path(d).glob("*.json"):
            try: m = json.loads(j.read_text(encoding="utf-8"))
            except Exception: continue
            if "compilado" not in m or "tramos" not in m: continue
            mans[m["compilado"]] = m          # los últimos pisan: pasar nuevos al final
    for comp, m in mans.items():
        for i, t in enumerate(m["tramos"]):
            idx.setdefault(t["toma"], []).append(
                (comp, i, t.get("origen_ini_s", 0), t.get("origen_fin_s", 0)))
    return idx, mans


def solape(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nuevos"); ap.add_argument("publicados")
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()

    # Los nuevos van al final para que pisen a los publicados con la misma clave.
    idx, mans = indexar([a.publicados, a.nuevos])
    regenerados = {json.loads(j.read_text(encoding="utf-8"))["compilado"]
                   for j in Path(a.nuevos).glob("*.json")}
    viejos, _ = indexar([a.publicados])          # para saber QUÉ pedazo se votó
    _, mans_viejos = indexar([a.publicados])
    print("regenerados: %s\n" % ", ".join(sorted(regenerados)))

    dest = config.remote("salida"); fb = dest + "feedback/"
    tmp = Path(tempfile.mkdtemp())
    archivos = [x for x in subprocess.run(["rclone", "lsf", fb, "--include", "*.jsonl"],
                                          capture_output=True, text=True).stdout.split()
                if x.endswith(".jsonl")]
    ahora = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    cuenta = {"mismo": 0, "mudado": 0, "igual": 0, "perdido": 0}
    detalle = []

    for nombre in sorted(archivos):
        subprocess.run(["rclone", "copyto", fb + nombre, str(tmp / nombre)], check=True)
        evs = []
        for l in (tmp / nombre).read_text(encoding="utf-8").splitlines():
            if l.strip():
                try: evs.append(json.loads(l))
                except Exception: pass

        vigentes = {}
        for e in evs:
            if e.get("compilado") not in regenerados: continue
            k = (e.get("compilado"), e.get("tramo"))
            ev = e.get("evento")
            if ev == "undo":
                if e.get("deshace") == "nota": vigentes.pop(("nota",) + k, None)
                else: vigentes.pop(k, None)
            elif ev in VOTOS: vigentes[k] = e
            elif ev == "nota": vigentes[("nota",) + k] = e

        salida = []
        for e in vigentes.values():
            comp, toma = e["compilado"], e.get("toma")
            # Qué pedazo del original se votó, según el manifest VIEJO.
            mv = mans_viejos.get(comp)
            tv = mv["tramos"][e["tramo"]] if mv and e.get("tramo", -1) < len(mv["tramos"]) else {}
            o0, o1 = tv.get("origen_ini_s", 0), tv.get("origen_fin_s", 0)

            cands = idx.get(toma, [])
            mismo = [c for c in cands if c[0] == comp]
            otros = [c for c in cands if c[0] != comp and c[0] not in regenerados]
            # El mismo compilado gana; si no, el que más solape con lo votado.
            pool = mismo or otros
            if not pool:
                cuenta["perdido"] += 1
                detalle.append(("PERDIDO", nombre, comp, toma, e["evento"], ""))
                salida.append({**{x: e[x] for x in ("usuario","banda","compilado","tramo","toma") if x in e},
                               "ts": ahora, "evento": "undo",
                               **({"deshace":"nota","nota_ts":e["ts"]} if e["evento"]=="nota" else {}),
                               "migracion": "la toma no quedó en ningún compilado"})
                continue
            mejor = max(pool, key=lambda c: (solape(o0, o1, c[2], c[3]), -abs(c[1] - e.get("tramo", 0))))
            if mejor[0] == comp and mejor[1] == e.get("tramo"):
                cuenta["igual"] += 1
                continue
            cuenta["mismo" if mejor[0] == comp else "mudado"] += 1
            detalle.append(("mismo" if mejor[0] == comp else "MUDADO", nombre, comp, toma,
                            e["evento"], "-> %s t=%d" % (mejor[0], mejor[1])))
            salida.append({**{x: e[x] for x in ("usuario","banda","compilado","tramo","toma") if x in e},
                           "ts": ahora, "evento": "undo",
                           **({"deshace":"nota","nota_ts":e["ts"]} if e["evento"]=="nota" else {}),
                           "migracion": "reubicado en %s tramo %d" % (mejor[0], mejor[1])})
            nuevo = dict(e); nuevo["compilado"] = mejor[0]; nuevo["tramo"] = mejor[1]
            nuevo["ts"] = ahora; nuevo["migracion"] = "venía de %s tramo %s" % (comp, e.get("tramo"))
            salida.append(nuevo)

        if salida:
            print("  %-38s %d eventos nuevos" % (nombre, len(salida)))
            if a.aplicar:
                txt = (tmp / nombre).read_text(encoding="utf-8").rstrip("\n") + "\n" + \
                      "\n".join(json.dumps(x, ensure_ascii=False) for x in salida) + "\n"
                (tmp / nombre).write_text(txt, encoding="utf-8")
                subprocess.run(["rclone", "copyto", str(tmp / nombre), fb + nombre], check=True)

    print("\n  en el mismo compilado: %(mismo)d | mudados a otro: %(mudado)d | "
          "sin cambio: %(igual)d | perdidos: %(perdido)d" % cuenta)
    if detalle:
        print("\n  detalle:")
        for tipo, f, c, t, ev, dst in detalle:
            print("     %-8s %-20s %-26s %-5s %-34s %s" %
                  (tipo, f.split("_")[0][:20], c[:26], ev, str(t).replace(".mp3","")[:34], dst))
    print("\n(dry-run: no se escribió nada)" if not a.aplicar else "\nAplicado.")


if __name__ == "__main__":
    main()
