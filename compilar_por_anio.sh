#!/bin/zsh
# Un compilado por AÑO y por PERFIL, como registro histórico de una banda:
#   ideas   -> "¿hay una idea que vale la pena recuperar?"
#   gems    -> interés alto CON ejecución baja/media: la idea buena mal tocada
#   balance -> la mezcla validada (timing + groove + afinacion + desarrollo)
#
# Uso:  ./compilar_por_anio.sh <carpeta-corpus> [año...]
#   ./compilar_por_anio.sh mibanda              # todos los años con material
#   ./compilar_por_anio.sh mibanda 2019 2020    # sólo esos
#
# Sin MP4: con muchas salidas el video pesa 30-50 MB cada una y el player web
# sólo necesita el .mp3 y su .json.
# Reanudable: si el .mp3 ya existe, saltea. Cada par se sube apenas sale, así
# que si se corta a mitad lo hecho ya está disponible.
set -u
cd "$(dirname "$0")"
CORPUS="${1:?uso: $0 <carpeta-corpus> [año...]}"; shift
PY=.venv/bin/python
# Una carpeta por banda: los años se repiten entre bandas, así que
# "compilado_2020_ideas_dinamico.mp3" colisionaba. El guardia de "ya existe"
# convertía eso en un salteo silencioso: ninjam no generaba nada para los años
# que nebulosa ya tenía.
SALIDA="salidas/$CORPUS"
mkdir -p "$SALIDA"
DEST="$($PY -c 'from zapaia import config; print(config.remote("salida"))')"
if (( $# )); then ANIOS=("$@"); else ANIOS=(2017 2018 2019 2020 2021 2022 2023 2024 2025 2026); fi
# Sin --duracion-max ni --max-seg-win: ningún límite de minutos decide qué
# entra. El corte por duración se aplicaba DESPUÉS de ordenar por tempo, así que
# lo que se perdía era lo de tempo más alto, no lo peor puntuado; y el tope por
# tramo recortaba justo las rachas largas, que son las de mejor score.
COMUN=(--modo veto --sonido-min 0.15 --min-win 4 --top 16
       --dinamico --umbral-q 0.60 --seg-win 2 --tramos-por-toma 2
       --ajustar-tempo --max-por-tema 1 --diversidad 0.15
       --sin-video)
for A in $ANIOS; do
  for P in ideas gems balance; do
    OUT="$SALIDA/compilado_${A}_${P}_dinamico.mp3"
    [[ -f "$OUT" ]] && { echo "== $OUT ya existe, salteo"; continue; }
    echo "=== $A / $P ==="
    if $PY -m zapaia compilado "$CORPUS" --anio $A --perfil $P $COMUN --out "$OUT"; then
      rclone copy "$OUT" "$DEST" && rclone copy "${OUT%.mp3}.json" "$DEST" && echo "   subido"
    else
      echo "   (sin material para $A, salteo)"
    fi
  done
done
echo "=== LISTO: $(ls "$SALIDA"/compilado_????_*_dinamico.mp3 2>/dev/null | wc -l | tr -d ' ') compilados por año ==="
