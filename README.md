# zapa-ia

Analiza cientos de MP3 de ensayos de banda (zapadas/improvisaciones) y los rankea
automáticamente para encontrar las mejores tomas —y los mejores **tramos** dentro de cada
toma— sin escuchar todo a mano. Además arma un MP3 recopilado con los mejores fragmentos
enganchados con crossfade.

Pensado para material crudo: grabaciones de celular, handy o multipista casera, de 1 a 50
minutos, tempo libre, y muchas tomas del mismo tema.

## Instalación

Requiere Python 3.9+ y `ffmpeg` en el PATH.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
# 1. Extraer features al caché (paralelo, reanudable, saltea lo ya procesado)
.venv/bin/python -m zapaia extract ensayos/ --jobs 12

# 2. Rankear (instantáneo: lee del caché, no vuelve a tocar el audio)
.venv/bin/python -m zapaia rank ensayos/ --top 30 --dedupe --out ranking.csv

#    elegir QUÉ pregunta responde el ranking:
.venv/bin/python -m zapaia rank ensayos/ --perfil performances # ¿qué tan bien está tocado?
.venv/bin/python -m zapaia rank ensayos/ --perfil ideas        # ¿hay una idea que vale?
.venv/bin/python -m zapaia rank ensayos/ --perfil gems         # ideas buenas mal tocadas

#    rankear por una dimensión sola:
.venv/bin/python -m zapaia rank ensayos/ --sort-by timing      # las más precisas
.venv/bin/python -m zapaia rank ensayos/ --sort-by creatividad # las que más pasan cosas

# 3. Un solo MP3 con los mejores tramos pegados
.venv/bin/python -m zapaia compilado ensayos/ --top 10 --seg-win 3 --out compilado.mp3
#    tramos de largo variable (siguen mientras la racha sea buena), varios por toma,
#    cortes al beat y tempo ajustado para que enganchen:
.venv/bin/python -m zapaia compilado ensayos/ --dinamico --ajustar-tempo --duracion-max 30

# 4. Validar
.venv/bin/python -m zapaia diag ensayos/    # ¿alguna feature no discrimina?
.venv/bin/python -m zapaia eval ensayos/    # ¿dónde caen las refs de refs.txt?
.venv/bin/python -m zapaia dupes ensayos/   # duplicados por huella de audio

# 0. (opcional) Traer de Google Drive solo lo nuevo, con rclone configurado
.venv/bin/python -m zapaia sync ensayos/ --meses 3 --extraer
.venv/bin/python -m zapaia compilado ensayos/ --ultima-sesion --dinamico --ajustar-tempo
#    -> compilado_sesion-AAAA-MM-DD_balance_dinamico.mp3 (+ .txt con la lista)

# 5. Enseñarle tu criterio: pares A/B ("¿cuál rescatarías?")
.venv/bin/python -m zapaia comparar ensayos/ --n 10            # reproduce con afplay y pregunta
.venv/bin/python -m zapaia comparar ensayos/ --n 20 --lote     # solo genera clips + lote.txt
.venv/bin/python -m zapaia feedback importar "1 A, 2 B, 3 ninguno"
.venv/bin/python -m zapaia feedback resumen                    # qué dimensión predice tu criterio
```

Las respuestas se guardan en el mismo caché SQLite (`segmentos`, `pares`) con la versión de
features, y `feedback resumen` ajusta un modelo Bradley–Terry para obtener un score latente por
segmento y correlarlo con cada dimensión. Con suficientes comparaciones, eso reemplaza el ajuste
manual de pesos.

Cambiar pesos NO requiere reprocesar: para eso está el caché. Cambiar `features.py` sí —
subir `FEATURE_VERSION` en `zapaia/__init__.py` lo invalida automáticamente.

## Cómo puntúa

Dos niveles: **ventana de 30 s** (normalizada por percentil contra todas las ventanas del
corpus) y **archivo** (mediana de sus ventanas + dimensiones que solo existen a nivel archivo).

| Dimensión | Qué mide |
|---|---|
| `timing` | precisión rítmica: `ibi_cv`, deriva real del tempo, desvío al beat en fracción de beat |
| `groove` | pulso y enganche: PLP, sincronía entre banda baja y alta, acento real de los beats |
| `notas` | notas correctas: energía de croma fuera de la tonalidad local (detecta pifies) |
| `afinacion` | afinación del instrumento: desviación en fracción de semitono, claridad tonal |
| `creatividad` | que pase algo: movimiento armónico, variedad de textura, novedad entre secciones |
| `desarrollo` | consistencia de tempo sostenida y arco dinámico (nivel archivo) |
| `sonido` | calidad de captura. Existe para **vetar** lo inescuchable, no para puntuar |

Por archivo se reporta `score_med` (consistencia), `score_best` (mejor momento) y
`best_start` con el timestamp de ese momento.

## Lecciones de diseño

Las trampas que este proyecto pisó, por si sirven para otro parecido:

1. **Ninguna feature entra a una suma ponderada sin normalizar primero.** La primera versión
   hacía `rms*0.4 - beat_var*0.2 + centroid*0.2 - ...`, y medido sobre audio real el centroide
   valía 175–421 mientras el resto valía <0.2: el score *era* el centroide y los demás pesos
   eran decorativos. Resultado: rankeaba por brillo, hundía todo lo grave, y un ringtone de
   0.8 MB entraba en el puesto 6 de 848.
2. **Normalizar por percentil, no min–max.** Min–max se rompe con un outlier.
3. **Métricas adimensionales.** Un desvío en segundos premia los tempos lentos; hay que usar
   coeficiente de variación o fracción de beat.
4. **Fallback = `NaN`, nunca un número inventado.** Un default de `0.25` entra al percentil
   como si fuera una medición real.
5. **Verificar que una feature sea estable ante sus propios parámetros.** Una métrica de
   sincronía basada en HPSS daba Spearman 0.68 al cambiar el kernel de la mediana: medía el
   parámetro tanto como el audio. Se reemplazó por correlación entre bandas de frecuencia:
   0.94–0.99 de estabilidad y 12x más rápida.
6. **Cuidado con features redundantes.** Dos de ellas correlacionaban 0.979 —eran la misma
   medición pesando 0.45 juntas— y encima medían variación de nivel, no ruido: el resultado
   era que "mejor sonido" premiaba las ventanas con más silencio adentro.
7. **Corregir por comparaciones múltiples.** Testear 7 dimensiones y festejar el único
   p=0.046 que apareció es engañarse: con Bonferroni es p≈0.32.
8. **La varianza de las muestras chicas infla los extremos.** Sin shrinkage, los archivos de
   1 minuto copaban el top-20 por azar y no por calidad.
9. **Sin ground truth, ajustar pesos es adivinar.** `refs.txt` guarda las tomas con juicio
   humano; `eval` reporta percentiles, precision@k y AUC. Y **verificar tamaño o hash** antes
   de usar un archivo como etiqueta: dos archivos con el mismo nombre pueden ser tomas
   distintas.
10. **Para diversificar, centrar las huellas.** En crudo, dos zapadas de la misma banda en la
    misma sala dan ~0.98 de similitud: el fondo común tapa la diferencia musical.

## Licencia

MIT
