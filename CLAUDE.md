# zapa-ia — Ranking automático de tomas de ensayo

Proyecto personal para analizar **muchos MP3 de ensayos/zapadas** (grabaciones de banda) y
rankearlos automáticamente para encontrar las mejores tomas sin escuchar 850 archivos a mano.

## Contexto del dominio

- El material son **zapadas/ensayos crudos**, no mezclas terminadas: celular, handy o multipista
  casera. Nivel, ruido y duración varían muchísimo (de 0.9 a 49 minutos).
- Son improvisaciones: el tempo no es estable ni pretende serlo, y muchas tomas no tienen batería.
  Cualquier métrica que asuma grilla fija o presencia de percusión va a fallar.
- Nombres de archivo descriptivos-informales (`"fluyendo despavorido.mp3"`, `"nebu side - estoy de
  pasada.mp3"`), con metadata de facto (nombre de zapada, versión, `demo`/`test`/`render`, año).
  No hay convención formal.
- Objetivo real: shortlist de candidatos para mezclar/publicar. **Precisión en el top-N importa
  mucho más que exactitud en la cola del ranking.**
- La unidad de decisión NO es el archivo: en una zapada de 49 minutos lo que se publica es un tramo
  de 3–5 minutos. Por eso el pipeline puntúa **ventanas de 30 s** y reporta el mejor tramo.

## Layout

```
zapa-ia/
├── zapaia/                  # PIPELINE ACTUAL — usar esto
│   ├── audio.py             # carga mono + ventaneo
│   ├── features.py          # features por ventana (sonido + ejecución) y huella de audio
│   ├── cache.py             # caché SQLite, resume-safe, clave (path, mtime, params)
│   ├── score.py             # normalización robusta + pesos + diagnóstico
│   ├── dedupe.py            # duplicados / re-encodes por huella
│   └── cli.py               # extract / rank / diag / dupes / eval
├── refs.txt                 # GROUND TRUTH manual (tomas que se sabe que son buenas)
├── nebulosa/                # 850 MP3 (~9.1 GB) — carpeta de ensayo, NO commitear
├── .zapaia_cache.db         # caché de features (gitignored)
├── requirements.txt
├── .venv/                   # Python 3.9.6 — la única que corre tensorflow-macos
└── legacy/                  # notebooks v1–v3, ver "Historia" abajo
```

Usar siempre `.venv/bin/python`: el Python del sistema es 3.14 y no corre `tensorflow-macos`.
`ffmpeg` en `/opt/homebrew/bin/ffmpeg` (lo necesita `pydub`).

## Uso

```bash
# 1. Extraer features al caché (paralelo, reanudable, saltea lo ya procesado)
.venv/bin/python -m zapaia extract nebulosa --jobs 12
#    iterar rápido sobre un subconjunto:
.venv/bin/python -m zapaia extract nebulosa --sample 40 --include "epilepsia orgasmica"

# 2. Rankear (instantáneo: lee del caché, no toca el audio)
.venv/bin/python -m zapaia rank nebulosa --top 30 --dedupe --out ranking.csv

#    rankear por UNA dimensión de "bien tocada":
.venv/bin/python -m zapaia rank nebulosa --sort-by timing      # las más precisas
.venv/bin/python -m zapaia rank nebulosa --sort-by groove      # las más enganchadas
.venv/bin/python -m zapaia rank nebulosa --sort-by afinacion   # las mejor afinadas
.venv/bin/python -m zapaia rank nebulosa --sort-by desarrollo  # las que más evolucionan

#    ignorar la calidad de grabación, solo vetar lo inescuchable:
.venv/bin/python -m zapaia rank nebulosa --modo veto --sonido-min 0.15

#    repesar las dimensiones de ejecución:
.venv/bin/python -m zapaia rank nebulosa --w-timing 0.5 --w-groove 0.3 --w-afinacion 0.2 --w-desarrollo 0

#    filtrar fragmentos cortos (N ventanas = N*30s):
.venv/bin/python -m zapaia rank nebulosa --min-win 6

# 2b. COMPILADO: un solo MP3 con los mejores tramos pegados con crossfade
.venv/bin/python -m zapaia compilado nebulosa --modo veto --min-win 4 \
    --sort-by timing --top 10 --seg-win 3 --crossfade 4 --orden tempo \
    --out compilado_prolijo.mp3

# 3. Validar
.venv/bin/python -m zapaia diag nebulosa    # ¿alguna feature no discrimina?
.venv/bin/python -m zapaia eval nebulosa    # ¿dónde caen las refs de refs.txt?
.venv/bin/python -m zapaia dupes nebulosa   # duplicados
```

**Cambiar pesos, modo u orden NO requiere reprocesar** — para eso existe el caché, `rank` es
instantáneo. Cambiar `features.py` sí: hay que subir `FEATURE_VERSION` en `zapaia/__init__.py`,
que invalida el caché automáticamente.

## Diseño del score

Dos niveles. **Ventana de 30 s** (normalizada por percentil contra todas las ventanas del
corpus) y **archivo** (mediana de las ventanas + dimensiones que solo existen a nivel archivo).

### Las cuatro dimensiones de "bien tocada"

Se reportan **por separado** y se puede rankear por cada una con `--sort-by`:

| Dimensión | Qué mide | Features |
|---|---|---|
| `timing` | ¿toca preciso? | `ibi_cv` (estabilidad adimensional), `tempo_drift` (pendiente real del IBI), `onset_dev_mad` (desvío al beat en fracción de beat) |
| `groove` | ¿hay pulso y están enganchados? | `pulse_clarity` (PLP), `band_sync` (onsets de banda baja vs alta), `beat_strength` (beats acentuados vs pulso inferido) |
| `afinacion` | ¿afinado y con centro tonal? | `tuning_dev` (fracción de semitono), `key_clarity` (Krumhansl sobre 24 tonalidades), `chroma_entropy` |
| `desarrollo` | ¿la zapada va a algún lado? | `tempo_consistency`, `arco_dinamico`, `novedad_armonica` — **nivel archivo** |

`desarrollo` no puede medirse en una ventana de 30 s (no se le pregunta a medio minuto si
evoluciona), así que se calcula comparando ventanas entre sí y se normaliza entre archivos. Por lo
mismo, el **mejor tramo** se elige sin desarrollo.

`ejecucion` = combinación pesada de las cuatro (`--w-timing --w-groove --w-afinacion
--w-desarrollo`, default 0.35/0.30/0.15/0.20).

### ¿El pipeline nuevo es mejor que el viejo? Sí, y por mucho

Head-to-head sobre las 4 tomas con proyecto de REAPER en Drive (ground truth: alguien las mezcló):

| toma | algoritmo viejo (`log.txt`) | pipeline nuevo |
|---|---|---|
| `kikikiiiiki.mp3` | #786/848 — pct **7** | #90/777 — pct 88 |
| `kikikiikii take 2.mp3` | #703/848 — pct 17 | #2/777 — pct **100** |
| `zapadita fluyendo.mp3` | #706/848 — pct 17 | #111/777 — pct 86 |
| `zapadita fluyendo remaster.mp3` | #714/848 — pct 16 | #51/777 — pct 93 |
| **percentil mediano** | **16.3** | **90.9** |

El viejo mandaba las cuatro al fondo: estaba prácticamente anti-correlacionado con la calidad,
consecuencia directa de que su score era el centroide espectral. Nuevo: 4/4 sobre pct 75, p=0.004.

### Pifies: era una capacidad ausente, no un problema de pesos

Feedback humano: "hay pifies de bajo e instrumentos en la mayoría de los temas". Correcto, y el
sistema era **ciego** a eso: `timing` mide CUÁNDO suena una nota, `afinacion` si el instrumento
está afinado respecto a A440 — ninguna mira si la nota es la CORRECTA. Un pifie a tiempo con un
bajo afinado puntuaba alto.

Se verificó que no era cuestión de reponderar: predecir `notas` con TODAS las demás dimensiones da
**R² = 0.22**, o sea que el 78% de su información no existía en ninguna feature previa. Ninguna
combinación de pesos podía detectarlo.

La dimensión `NOTAS` (`fuera_tono_med/p95/picos`) estima la tonalidad local (contexto 8 s, perfiles
Krumhansl vectorizados) y mide la energía de croma que cae fuera de ella. El `p95` importa tanto
como la media: un pifie aislado se diluye en el promedio de 30 s.

Efecto medido al penalizarlo en el compilado: **−12% de fuera-de-tonalidad promedio, −15% en el
peor tramo**, cambiando 7 de 10 tramos. Mejora real pero moderada: el piso es alto porque son
zapadas de ensayo y los pifies están en todos lados (mediana del corpus 0.335, mejor selección
alcanzable ~0.276). **No esperar tomas impecables, sí menos pifies.**

Validado por oído: los timestamps que marcó el detector coincidieron con lo que el usuario escuchó.

ATENCIÓN: solo los 150 mejores candidatos tienen features v4 (`--files-from`). Para el corpus
completo hay que re-extraer (~2 h, +86 ms/ventana).

### Diversidad: por qué los compilados sonaban "siempre el mismo tema"

Feedback humano: los tramos extraídos sonaban todos parecidos. Diagnóstico: los 10 tramos
elegidos eran **más similares entre sí (0.982) que el promedio del corpus (0.975)** — la selección
concentraba material en vez de diversificar. 4 de 10 eran del tema "fluyendo". Causa: si un tema es
bueno, TODAS sus tomas puntúan bien y suben juntas.

**La huella de audio no sirve para identidad de tema.** Dos zapadas cualesquiera dan ~0.98 porque
comparten sala, banda e instrumentos. Centrando (restando el promedio del corpus) aparece algo de
señal — mismo tema +0.40, distinto tema -0.03 — pero al filtrar apenas mejora: 6 temas sin filtro
vs 7-8 con. `centrar()` es obligatorio para cualquier comparación de firmas, pero no alcanza.

**Los nombres identifican el tema mucho mejor.** `temas_por_nombre()` toma el token recurrente más
frecuente que no sea genérico (`nebu`, `zapada`, `demo`, `take`...) ni demasiado común. Detecta 108
temas sobre 850 archivos, 37% queda sin tema (son piezas únicas, inherentemente diversas).

**Las frecuencias de token se calculan sobre el CORPUS COMPLETO**, no sobre la lista ya filtrada:
con la lista filtrada las frecuencias cambian y el tema detectado sale inestable (con `--max-por-tema 1`
seguían entrando 2 "fluyendo").

Resultado: `--max-por-tema 1 --diversidad 0.15` pasa de 6 temas distintos en 10 tramos a **10 de 10**.

### Creatividad

`CREATIVIDAD` (nivel archivo): `novedad_armonica` (la armonía se mueve entre secciones),
`harmonic_flux` (movimiento dentro de las ventanas), `variedad_textura` (la densidad de eventos
cambia a lo largo). Apunta a separar una zapada que va a algún lado de un riff en loop de 10 min.
`novedad_armonica` se movió de `DESARROLLO` a acá para no contarla dos veces.

**Sin validar**: es un proxy razonable, no una medición de creatividad. Necesita oído humano.

### Qué criterio usar para elegir partes

Las cuatro dimensiones son **casi ortogonales a nivel ventana** (correlación máxima 0.19, entre
timing y groove). Eso significa que cada una aporta información que las otras no tienen, y que
**mezclarlas suma de verdad** en vez de contar lo mismo tres veces.

Ninguna sola alcanza:
- `timing` mide precisión métrica y nada más. Es limpia (|r| ≤ 0.12 contra densidad, tempo y
  nivel — se sospechó que premiaba pasajes escasos y **resultó falso**), pero un tramo puede ser
  milimétrico y aburrido.
- `groove` mide enganche, que es lo que engancha al escuchar.
- `afinacion` es ortogonal a las otras dos (r ≈ 0.00 con timing): agrega un eje entero.
- `desarrollo` **no aplica a un tramo de 90 s** — es de archivo. En `compilado` se ignora.

Combinación recomendada para extraer partes — **sonido en 0, solo como veto**:

```bash
--modo veto --sonido-min 0.15 \
--w-timing 0.40 --w-groove 0.35 --w-afinacion 0.25 --w-desarrollo 0
```

Por qué 0 y no un peso chico: se midió y darle 0.10 deja el ranking **prácticamente igual**
(Spearman 0.989, 9 de 10 del top-10 idénticos). Una métrica sin validar, con sesgo residual de
−0.36 contra el nivel, que además no cambia el resultado, no tiene ningún argumento a favor.
El veto igual cumple lo único que importa del sonido: sacar del medio lo inescuchable.

`--sort-by` elige las TOMAS y `--seg-by` elige el TRAMO dentro de cada una; son independientes y
conviene no confundirlos (`--sort-by timing --seg-by score` elige las tomas más precisas pero el
mejor tramo general de cada una).

### Sonido: existe para vetar, no para puntuar

`SONIDO` (`flatness`, `silence_ratio`, `clip_ratio`, `crest_db`) mide calidad de captura.

**Tuvo un bug grave, ya corregido.** `snr_proxy_db` (p90−p10 del RMS) y `dyn_range_db` (p95−p5)
correlacionaban **0.979: eran la misma medición**, y juntas pesaban 0.45 de `SONIDO`. Y no medían
ruido sino variación de nivel dentro de la ventana, así que una ventana mitad callada y mitad
fuerte daba "SNR" altísimo. Resultado medido: el top-300 de `sonido` tenía **31% de silencio** y
estaba 14 dB más bajo que el corpus. `sonido` premiaba el silencio mientras `silence_ratio`
(peso −0.10) intentaba castigarlo. Las dos salieron del score; quedaron como diagnóstico.
Tras el arreglo la correlación con silencio bajó de fuertemente positiva a −0.14.

Medir ruido de verdad (hiss en los frames callados, no variación de nivel) pide cambiar
`features.py` y re-extraer. Pendiente, y de baja prioridad: el sonido importa poco acá. **En este corpus casi no informa calidad musical**: es la misma banda en
la misma sala, así que la varianza de sonido es más accidente de micrófono que mérito de la toma.

Evidencia: `epilepsia orgasmica.mp3` (referencia humana de toma buena) tiene ejecución en buen
percentil pero sonido hundido por hiss y poca dinámica. Con `peso_sonido=0.35` caía al percentil
56; con 0 sube al 79. Una zapada bien tocada y mal grabada se arregla mezclando; una mal tocada no.

Por eso existe `--modo veto`: el sonido descarta las ventanas bajo `--sonido-min` (percentil) y
después **no suma al puntaje**. Rankeo por ejecución pura.

### Agregación por archivo

- `score_med` — consistencia de toda la toma (mediana de ventanas).
- `score_best` — mejor momento (p95), con `best_start` marcando el timestamp.
- **Shrinkage** (`--shrink`, default 4): con 2 ventanas la mediana se va a los extremos por azar,
  con 30 no. Se encoge hacia 0.5 con peso `n/(n+k)`. Sin esto los archivos de 1 minuto copaban el
  top-20 por varianza, no por calidad (eran el 8% del top siendo el 9% del corpus, mientras los
  largos eran 0%).

### Reglas de oro (aprendidas a los golpes, ver Historia)

1. **Ninguna feature entra a una suma ponderada sin normalizar primero.** Si las escalas difieren,
   los pesos son decorativos y el score es, en la práctica, la feature de mayor magnitud.
2. **Normalizar por percentil, no min–max.** Min–max se rompe con un outlier y devuelve 0.5 si hay
   un solo elemento.
3. **Métricas adimensionales o normalizadas por tempo.** Un desvío en segundos premia los tempos
   lentos; hay que usar coeficiente de variación o fracción de beat.
4. **El centroide espectral NO es calidad.** Es brillo. Usarlo como score condena todo lo grave
   (zapadas de bajo y viola) y premia cualquier cosa con platillos.
5. **Fallback = `NaN`, nunca un número inventado.** Un `0.25` de default entra al percentil como si
   fuera medición real. `rank_norm` manda los `NaN` a 0.5 (neutral: no premia ni castiga).
6. **Verificar que una feature sea ESTABLE ante sus propios parámetros.** `ph_sync` (vía HPSS) daba
   Spearman 0.68 al cambiar el kernel de la mediana: la feature medía el parámetro tanto como el
   audio. `band_sync` da 0.94–0.99 ante cambios de corte de banda, y encima es 12x más rápida.
   Validar con muestra amplia: con 4 ventanas de un solo archivo parecía estable y no lo era.
7. **Correr `diag` después de tocar features.** Si `iqr/|p50|` es bajo, o el rango absoluto es
   inaudible, esa feature no aporta y su peso está desperdiciado.
8. **Correr `eval` después de tocar pesos.** Sin ground truth, ajustar pesos es adivinar. Y ajustar
   contra UNA sola referencia es sobreajuste: hacen falta 10–20, incluyendo tomas malas.
9. **Chequear que una dimensión no sea proxy de otra cosa.** `chroma_entropy` correlacionaba 0.69
   con `harmonic_flux`: medía movimiento armónico, no foco tonal, así que salió de `AFINACION`.
   `tuning_dev` en cambio es limpísima (|r| ≤ 0.06 contra todo descriptor de textura).
   Al sospechar que `afinacion` premiaba material escaso, la sospecha resultó FALSA: `violabajo.mp3`
   tiene `tuning_dev` 0.020 contra 0.060 de mediana, está genuinamente mejor afinado. Verificar
   antes de "arreglar".
10. **Corregir por comparaciones múltiples.** Testear 7 dimensiones y festejar el p=0.046 que
   apareció es engañarse: con Bonferroni eso es p≈0.32.

## Compilado (`zapaia compilado`)

Arma un solo MP3 con los mejores tramos de varias tomas. Decisiones de diseño:

- **La unidad es una tirada contigua de ventanas**, no una ventana suelta: `--seg-win 3` = 90 s
  sostenidos. Un tramo que aguanta bueno vale más que un pico aislado de 30 s.
- **La contigüidad se verifica en el tiempo, no por filas consecutivas.** En modo veto (y con
  ventanas silenciosas descartadas) faltan filas, así que tres filas seguidas de la tabla pueden
  abarcar un hueco de varios minutos. Se exige `start[i] - start[i-n+1] == (n-1)*win`. Sin este
  chequeo salían tramos de 154 s donde se pidieron 90.
- **Los cortes se pegan al beat más cercano** (±1.5 s) para que el empalme no caiga a contratiempo.
  Se puede desactivar con `--sin-snap`.
- **`--orden tempo`** (default) ordena los tramos por BPM para que los empalmes no salten de 90 a
  170. `--orden score` los pone de mejor a peor.
- `--sort-by` elige con qué criterio se seleccionan las tomas Y los tramos: `timing` para
  prolijidad de ejecución, `groove` para enganche, etc.

## Ground truth: de dónde salen las etiquetas

`refs.txt` es el ground truth. Formato: un nombre por línea, prefijo `-` = toma mala.
`eval` calcula percentiles, precision@k y (si hay negativos) AUC.

### Google Drive: `Zapadas New/Nebulosa`

El MCP "claude.ai Google Drive" está conectado, pero **NO puede leer favoritos**: los términos
de query soportados son `title`, `fullText`, `mimeType`, `modifiedTime`/`createdTime`/
`viewedByMeTime`, `parentId`, `owner`, `sharedWithMe`. No hay `starred`, y `get_file_metadata`
tampoco lo devuelve. Para favoritos hace falta un script con OAuth propio (la API de Drive sí
soporta `starred = true`).

Lo que sí hay, y es mejor:

- **`Canciones/`** — subcarpetas por tema que llegó a canción: `Psicorgasmo (epilepsia)`,
  `Fluyendo`, `Pobre`, `kikikiki`, `Cósmico`, `tunana nanan`, `Antiguo epec`.
- **Archivos `.RPP`** (proyectos de REAPER) — **la señal más fuerte disponible**. Que exista un
  proyecto de mezcla significa que alguien se sentó a trabajar esa toma: acción costosa y
  deliberada, no una marca pasiva.
- **`Zapadas New/Para masterizar`** — selección explícita.

### Dos trampas de etiquetado, ya pagadas

**1. La etiqueta a nivel tema no sirve.** Marcar los 177 archivos del corpus cuyo nombre matchea
un tema-canción y testear contra el scorer dio AUC 0.465–0.509 y p>0.2 en todas las dimensiones de
ejecución. Único efecto: `sonido` con AUC 0.555 y p=0.046 — pero eran 7 dimensiones testeadas, con
Bonferroni p≈0.32, **no significativo**. Causa: "fluyendo" tiene 59 tomas en el corpus; que el
tema llegara a canción dice que *el tema* servía, no que esas 59 tomas de ensayo sirvan.
**Una etiqueta de identidad-de-tema no es una etiqueta de calidad-de-toma.**

**2. Matchear por nombre entre Drive y disco NO alcanza.** `pobre.mp3` existía en los dos lados,
pero el de Drive (el que tiene el `.RPP`) pesa 24.383.737 bytes y el local 13.716.374: 44% de
diferencia, **no es la misma toma**. Se descartó por tamaño, criterio objetivo e independiente del
resultado. **Verificar tamaño o hash antes de usar cualquier cosa como ground truth.**

### Estado actual y resultado

4 positivos verificados byte a byte o por referencia directa, más 1 sin verificar:

| toma | origen | percentil |
|---|---|---|
| `epilepsia orgasmica.mp3` | referencia directa del usuario | 95.3 |
| `kikikiikii take 2.mp3` | `.RPP` en Drive, tamaño idéntico | 93.6 |
| `zapadita fluyendo remaster.mp3` | `.RPP` en Drive, tamaño idéntico | 92.5 |
| `kikikiiiiki.mp3` | `.RPP` en Drive, tamaño idéntico | 72.6 |
| `zapadita fluyendo.mp3` | `.RPP` en Drive, mp3 no comparable | 85.5 |

Sobre los **3 verificados que no se usaron para diseñar nada** (held-out real): 2 de 3 sobre el
percentil 90, binomial **p=0.028**. Evidencia positiva pero con n=3: el efecto existe, el tamaño
del efecto no está estimado con precisión.

`epilepsia orgasmica.mp3` NO cuenta como held-out: se usó para argumentar el modo veto, así que
está contaminado por el diseño.

**Lo que falta**: negativos. Sin tomas malas etiquetadas no hay AUC. Y 10–20 positivos para que
ajustar pesos deje de ser sobreajuste.

### Costo

~243 ms de CPU por ventana de 30 s en caliente (sound 21, exec 41, harmony 134). El corpus completo
son ~50 min con `--jobs 12`, dominado por el decodificado de 9 GB de MP3.
**Ojo al medir:** la primera llamada a `window_features` tarda ~2.7 s por el JIT de numba; hay que
hacer warmup antes de cronometrar.

## Historia — qué falló antes (no repetir)

`legacy/` tiene tres iteraciones previas, todas en notebooks con el código duplicado:

- `Mejor_toma_YAMNet.ipynb` (v1, Colab + Drive). **JSON corrupto**, no abre (strings sin escapar).
- `Zapadas-IA.ipynb` (v2): port local, un solo score, top-5 por carpeta. Generó `log.txt`.
- `Zapadas-IA copy.ipynb` (v3): separó Sonido/Ejecución y agregó min–max por carpeta.

El bug central de v2/v3, medido sobre audio real:

```
score = avg_rms*0.4 - beat_var*0.2 + avg_centroid*0.2 - silence_ratio*0.2 - penalty*5
         ~0.05           ~0.15           175 a 421           ~0.003          ~0
```

`Score_Sonido == avg_centroid * 0.2` con 3 decimales de exactitud. Todo el resto era ruido
numérico. Resultado en `log.txt` (848 archivos): el top lo ocupaban cosas brillantes y percusivas,
el fondo era todo lo grave (`violas duo`, `violabajo`, `bajo pinfloyiano`), y un **ringtone de
0.8 MB** (`macarena rimgtone.mp3`) entraba en el puesto 6.

Otros problemas de v3, ya resueltos en `zapaia/`: `beat_var` en frames mezclado con segundos;
YAMNet promediado sobre tracks de 49 min (señal aplastada a cero) con `penalty_labels`
(`Hum/Buzz/Static/Noise`) que casi nunca activan en música; `yamnet_class_map.csv` releído por
archivo; min–max por carpeta devolviendo 0.5; `np.correlate(mode='valid')` sobre vectores de igual
largo (devolvía un escalar, el `np.max` era cosmético); `DECIMAL_COMA=True` con separador `,`;
cero caché, cero paralelismo, cero resume sobre un corpus de 9 GB; duplicados sin filtrar.

**YAMNet quedó fuera del score** a propósito: los scores de clase no aportaban y era el componente
más caro. Si vuelve, que sea por los **embeddings** (1024-d) para clustering por estilo y
similitud, no por clasificación.

**HPSS también salió** (v4). Separar armónico/percusivo costaba el 74% del pipeline (722 ms de 980
por ventana) y `ph_sync`, la feature que lo justificaba, era inestable ante su propio kernel. Se
reemplazó por `band_sync`: correlación cruzada entre envolventes de onset de banda baja (<250 Hz,
bombo y bajo) y alta (>2 kHz, platillos y caja), reusando el melspectrograma. Más rápida, más
estable y más directa de interpretar. `harmonic_ratio` se fue con ella; en su lugar quedó
`low_energy_ratio` como descriptor de carácter.

## Convenciones

- Todo en español: comentarios, prints, nombres de columnas del CSV.
- Los audios **nunca** se commitean. El repo todavía no es un git repo; si se inicializa, el
  `.gitignore` ya está listo.
