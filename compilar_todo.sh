#!/bin/zsh
# "Todos los tiempos" sobre el corpus completo, con etiqueta propia.
#
# Uso:  ./compilar_todo.sh <carpeta-corpus> [etiqueta]
#   ./compilar_todo.sh mibanda              # etiqueta todo-AAAA-MM (mes actual)
#   ./compilar_todo.sh mibanda todo-2026-09
#
# La etiqueta lleva FECHA y no un número de versión: el corpus crece, así que
# cada tanda dice sola cuánto material cubría. Y va por --etiqueta, no sólo en
# el nombre del archivo, porque esa es la clave con la que el player asocia
# votos y notas (campo "compilado" del manifest): dos tandas con la misma clave
# mezclarían su feedback.
set -u
cd "$(dirname "$0")"
CORPUS="${1:?uso: $0 <carpeta-corpus> [etiqueta]}"
ETIQ="${2:-todo-$(date +%Y-%m)}"
PY=.venv/bin/python
# Una carpeta por banda: los años se repiten entre bandas, así que
# "compilado_2020_ideas_dinamico.mp3" colisionaba. El guardia de "ya existe"
# convertía eso en un salteo silencioso: ninjam no generaba nada para los años
# que nebulosa ya tenía.
SALIDA="salidas/$CORPUS"
mkdir -p "$SALIDA"
DEST="$($PY -c 'from zapaia import config; print(config.remote("salida"))')"
# Sin --duracion-max ni --max-seg-win: ningún límite de minutos decide qué
# entra. El corte por duración se aplicaba DESPUÉS de ordenar por tempo, así que
# lo que se perdía era lo de tempo más alto, no lo peor puntuado; y el tope por
# tramo recortaba justo las rachas largas, que son las de mejor score.
COMUN=(--modo veto --sonido-min 0.15 --min-win 4 --top 16
       --dinamico --umbral-q 0.60 --seg-win 2 --tramos-por-toma 2
       --ajustar-tempo --max-por-tema 1 --diversidad 0.15
       --sin-video)
for P in ideas gems balance; do
  OUT="$SALIDA/compilado_${ETIQ}_${P}_dinamico.mp3"
  [[ -f "$OUT" ]] && { echo "== $OUT ya existe, salteo"; continue; }
  echo "=== $ETIQ / $P ==="
  if $PY -m zapaia compilado "$CORPUS" --perfil $P --etiqueta "$ETIQ" $COMUN --out "$OUT"; then
    rclone copy "$OUT" "$DEST" && rclone copy "${OUT%.mp3}.json" "$DEST" && echo "   subido"
  fi
done
echo "=== verificacion manifest vs audio (tiene que dar +0.00) ==="
for P in ideas gems balance; do
  B="$SALIDA/compilado_${ETIQ}_${P}_dinamico"
  [[ -f "$B.json" ]] || continue
  $PY -c "
import json, subprocess, sys
b = sys.argv[1]
m = json.load(open(b + '.json')); t = m['tramos']
real = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0', b+'.mp3'], capture_output=True, text=True).stdout)
fin = t[-1]['pos_s'] + t[-1]['dur_s']
print('  %-44s manifest %8.2f  audio %8.2f  delta %+.2f' % (b, fin, real, real - fin))" "$B"
done
