#!/usr/bin/env python
"""¿El algoritmo rescata lo que la banda marca como bueno?

Es la única validación que escala: cada 👍/⭐ dice que a alguien le importó esa
zapada. Si el sistema no la pone en ningún compilado, falló para esa persona.

Uso:  ./test_favoritos.py [dir-con-manifests]   (default: los de Drive)
"""
import json, subprocess, sys, tempfile, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zapaia import config

def main():
    tmp = Path(tempfile.mkdtemp())
    dest = config.remote("salida")
    mans = Path(sys.argv[1]) if len(sys.argv) > 1 else tmp / "mans"
    if len(sys.argv) <= 1:
        mans.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rclone", "copy", dest, str(mans), "--include", "compilado_*.json"],
                       capture_output=True)
    fb = tmp / "fb"; fb.mkdir()
    subprocess.run(["rclone", "copy", dest + "feedback", str(fb), "--include", "*.jsonl"],
                   capture_output=True)

    evs = []
    for f in fb.glob("*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try: evs.append(json.loads(l))
                except Exception: pass
    evs.sort(key=lambda e: e.get("ts", ""))
    # Voto vigente por persona+compilado+tramo (undo borra).
    ult = {}
    for e in evs:
        k = (e.get("usuario"), e.get("compilado"), e.get("tramo"))
        ev = e.get("evento")
        if ev == "undo":
            if e.get("deshace") != "nota": ult.pop(k, None)
        elif ev in ("like", "star"): ult[k] = e

    votadas = collections.Counter(e["toma"] for e in ult.values())
    quien = collections.defaultdict(set)
    for e in ult.values(): quien[e["toma"]].add(e["usuario"])

    presentes = collections.defaultdict(set)
    for j in mans.glob("compilado_*.json"):
        try: m = json.loads(j.read_text(encoding="utf-8"))
        except Exception: continue
        for t in m["tramos"]: presentes[t["toma"]].add(m["compilado"])

    dentro = [t for t in votadas if t in presentes]
    fuera = [t for t in votadas if t not in presentes]
    print("zapadas votadas (👍/⭐ vigentes): %d" % len(votadas))
    print("  en algún compilado : %d  (%.0f%%)" % (len(dentro), 100*len(dentro)/max(1,len(votadas))))
    print("  en ninguno         : %d" % len(fuera))
    if fuera:
        print("\n  las que el sistema NO rescata:")
        for t in sorted(fuera, key=lambda x: -len(quien[x])):
            print("     %-40s la votaron %d" % (t.replace(".mp3","")[:40], len(quien[t])))
    # Las votadas por MÁS gente son las que más importa no perder.
    print("\n  cobertura entre las más votadas:")
    for n in (2, 3):
        sub = [t for t in votadas if len(quien[t]) >= n]
        ok = [t for t in sub if t in presentes]
        if sub: print("     votadas por >=%d personas: %d de %d rescatadas" % (n, len(ok), len(sub)))

if __name__ == "__main__":
    main()
