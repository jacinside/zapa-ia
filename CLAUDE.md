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

#    elegir qué pregunta responde el ranking (ver "Diseño del score"):
.venv/bin/python -m zapaia rank nebulosa --perfil performances   # ¿qué tan bien está tocado?
.venv/bin/python -m zapaia rank nebulosa --perfil ideas          # ¿hay una idea que vale?
.venv/bin/python -m zapaia rank nebulosa --perfil gems           # ideas buenas mal tocadas

#    repesar dimensiones (un solo juego de pesos, cada composite renormaliza su subconjunto):
.venv/bin/python -m zapaia rank nebulosa --w-timing 0.5 --w-groove 0.3 --w-afinacion 0.2 --w-desarrollo 0

#    reprocesar solo los archivos que fallaron (p. ej. uoho.mp3, MP3 corrupto):
.venv/bin/python -m zapaia extract nebulosa --reintentar-errores

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

# 0. SYNC desde Drive: baja solo los MP3 nuevos de los últimos N meses y los procesa
.venv/bin/python -m zapaia sync nebulosa --meses 3 --dry-run     # qué bajaría
.venv/bin/python -m zapaia sync nebulosa --meses 3 --extraer     # bajar + procesar lo nuevo
.venv/bin/python -m zapaia sync nebulosa --instalar-launchd 4    # todos los días a las 4:00

#    y después, solo lo reciente:
.venv/bin/python -m zapaia compilado nebulosa --ultima-sesion --dinamico --ajustar-tempo
.venv/bin/python -m zapaia compilado nebulosa --meses 3 --dinamico --ajustar-tempo
#    -> compilado_sesion-2026-09-09_balance_dinamico.mp3 + .txt con la lista y los parámetros

# 4. FEEDBACK HUMANO PAREADO — la fuente de etiquetas que funciona
.venv/bin/python -m zapaia comparar nebulosa --modo veto --min-win 4 --n 20 --drive   # lote al celular
.venv/bin/python -m zapaia comparar nebulosa --modo veto --min-win 4 --n 10           # escuchar acá (afplay)
.venv/bin/python -m zapaia feedback importar "1 A, 2 B, 3 ninguno, 4 ambos" --lote 1  # tal cual llega por WhatsApp
.venv/bin/python -m zapaia feedback pendientes
.venv/bin/python -m zapaia feedback resumen --modo veto --min-win 4   # Bradley–Terry + qué dimensión predice el criterio
```

**Cambiar pesos, modo u orden NO requiere reprocesar** — para eso existe el caché, `rank` es
instantáneo. Cambiar `features.py` sí: hay que subir `FEATURE_VERSION` en `zapaia/__init__.py`,
que invalida el caché automáticamente.

## Diseño del score

Dos niveles. **Ventana de 30 s** (normalizada por percentil contra todas las ventanas del
corpus) y **archivo** (mediana de las ventanas + dimensiones que solo existen a nivel archivo).

### Seis dimensiones, dos composites, cuatro perfiles

Se reportan **por separado** y se puede rankear por cada una con `--sort-by`:

| Dimensión | Nivel | Qué mide | Features |
|---|---|---|---|
| `timing` | ventana | ¿toca preciso? | `ibi_cv`, `tempo_drift`, `onset_dev_mad` (peso 0.15: es casi ruido acá, ver review §1.3) |
| `groove` | ventana | ¿hay pulso y están enganchados? | `pulse_clarity`, `band_sync`, `beat_strength` |
| `afinacion` | ventana | ¿afinado y con centro tonal? | `tuning_dev` (sin validar semánticamente), `key_clarity` |
| `tonal_outlier` | ventana | energía fuera de la tonalidad local | `fuera_tono_med`, `fuera_tono_p95`. Antes "notas"/"pifies": el nombre era una hipótesis. Correlaciona +0.50 con `flatness`: mide banda ancha tanto como notas equivocadas |
| `desarrollo` | archivo | ¿se sostiene? | `tempo_consistency` (gruesa: tempo cuantizado a 26 valores), `arco_dinamico` |
| `creatividad` | archivo | ¿pasa algo? | `novedad_armonica`, `harmonic_flux`, `variedad_textura`. **Sin validar**; premia la duración |

Las de archivo no se le pueden preguntar a 30 s, así que se calculan comparando ventanas y se
normalizan entre archivos. El **mejor tramo** siempre se elige con el score de ventana.

Dos composites — son preguntas distintas, no un solo número:

- **`ejecucion`** = timing, groove, afinacion, tonal_outlier → *¿qué tan bien está tocado?*
- **`interes`** = creatividad, desarrollo → *¿hay una idea que vale la pena recuperar?*

Un solo juego de pesos (`--w-timing --w-groove --w-afinacion --w-tonal --w-desarrollo
--w-creatividad`, default 0.35/0.30/0.15/0.10/0.20/0.20); cada composite renormaliza su subconjunto.

`--perfil` decide qué composite es el `score`:

| Perfil | `score` | Tramo dentro del archivo |
|---|---|---|
| `balance` (default) | timing + groove + afinacion + desarrollo — **la mezcla validada** | ejecución completa |
| `performances` | `ejecucion` | ejecución completa |
| `ideas` | `interes` | ejecución **limpia** (sin tonal_outlier: no castigar cromatismos) |
| `gems` | `interes × (1 − 0.5·ejecucion)` — interés alto, ejecución baja/media | ejecución limpia |

**Por qué `balance` no incluye las seis.** Medido sobre los 3 positivos held-out (tomas con
proyecto de REAPER), percentil mediano: las cuatro validadas **83.3**; + tonal_outlier 82.6;
+ creatividad 79.1; las seis 74.2. Cada dimensión sin validar que entra, baja las referencias.
`tonal_outlier` y `creatividad` se habían agregado sin re-correr `eval` en el corpus completo —
violando la regla 8 — y el default había quedado en 74.2. Bajo `ideas` las mismas referencias
caen al percentil 40: consistente con que "elegida para mezclar" mide ejecución, no interés.
Es la pregunta que el feedback pareado (P1) tiene que responder.

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

### Modo dinámico (`--dinamico`): tramos de largo variable, enganchados

Pedido del usuario después de escuchar los compilados fijos: "sin sonido entre temas, enganchados,
secciones que no sean fijas, más completas, y del mismo tema sacar varias partes".

- **`tramos_dinamicos`**: rachas contiguas de ventanas con `score >= cuantil --umbral-q` (default
  0.60) **y** `silence_ratio <= --max-silencio` (0.08). Mínimo `--seg-win` ventanas, tope
  `--max-seg-win` (10 = 5 min; una racha más larga se recorta a su mejor sub-tirada), hasta
  `--tramos-por-toma` (2) por archivo. El filtro de silencio existe porque el score no ve una
  pausa de 3 s en 30 s: sin él salían 6 huecos en un compilado, con él 1.
- **Enganche**: inicio y fin pegados al beat (`snap_fin`), crossfade default 1 s (en fijo 3 s),
  y `--ajustar-tempo`: time-stretch (`librosa.effects.time_stretch`, sin cambiar altura) del
  tramo siguiente para igualar el anterior si la diferencia es `<= --max-stretch` (4%). Dobles y
  mitades de tempo cuentan como iguales. El tempo se mide sobre el tramo real (`tempo_de`), no
  con la mediana cuantizada del caché. El ajuste se **encadena**: una racha a 99.4 después de
  una a 95.7 queda toda a 95.7.
- **`_Decoder`** cachea el decode mono por archivo: con varios cortes por toma, `pegar_a_beat`
  decodificaba 49 minutos cuatro veces.
- `--duracion-max` corta la lista al llegar a los minutos pedidos; `--orden tempo` deja juntos
  los tramos de una misma toma.
- Bug encontrado y corregido: `ratio_tempo` arrancaba con `mejor = 1.0`, que tiene distancia
  cero a 1.0 y nada lo superaba → **nunca ajustaba**. Se detectó porque 95.7 → 99.4 (3.9%) salía
  sin ajuste. Lección: un valor centinela no puede ser el óptimo trivial.

```bash
.venv/bin/python -m zapaia compilado nebulosa --modo veto --min-win 4 --top 16 \
    --dinamico --umbral-q 0.60 --seg-win 2 --max-seg-win 10 --tramos-por-toma 2 \
    --ajustar-tempo --duracion-max 30 --out compilado_continuo.mp3
```

## Preferencias del usuario (validadas por oído, no inferidas)

**Prioriza variedad y creatividad por encima de prolijidad técnica.** Dijo textualmente:
"ya sé que son ensayos y tienen pifies, y esa es la razón principal" — no quiere perseguir
perfección técnica, quiere partes con creatividad. Confirmó escuchando que los compilados con
diversidad de temas (`--max-por-tema 1 --diversidad 0.15`) son claramente mejores que los
seleccionados solo por ejecución.

Por eso esos dos flags son **default**. No subir el peso de `notas`/`afinacion` buscando tomas
impecables: no existen en este corpus (mediana de fuera-de-tonalidad 0.335, mejor selección
alcanzable ~0.276).

**La calidad de audio le importa poco.** De ahí que `sonido` tenga peso 0 y solo actúe como veto.

**Validar pidiendo comparaciones, no introspección.** Dijo "es difícil validar lo que decís" ante
la pregunta de si `creatividad` elegía bien. Funciona mucho mejor darle un A/B controlado —dos
muestras idénticas salvo una variable— y que solo diga cuál prefiere. `muestra_A` (sin
creatividad) vs `muestra_B` (creatividad 0.40) comparten 6 de 10 tramos justamente para eso.

**Entrega:** los MP3 van a Google Drive para escucharlos desde el celular, en
`Zapadas New/Nebulosa/Seleccion IA - compilados`. Se sube con `rclone copy <archivo>
"gdrive:Zapadas New/Nebulosa/Seleccion IA - compilados/"`. El remote `gdrive` ya está
configurado. **El MCP de Drive no sirve para esto**: sube pasando el contenido en base64 por la
conversación y un clip de 20 s ya cuesta ~53 mil tokens. Ojo que rclone usa un client_id
compartido que deja de funcionar durante 2026; habrá que crear uno propio.

## Sync desde Drive (`zapaia sync`) y filtros de fecha

Drive es **source**, no parte del algoritmo: el core sigue 100% local. `sync.py` usa `rclone
lsjson` sobre `gdrive:Zapadas New/Nebulosa` (recursivo), filtra `*.mp3` por `ModTime` de Drive
(`--meses N` o `--desde AAAA-MM-DD`) y **excluye** subcarpetas que no son ensayos crudos:
`Seleccion IA - compilados` (nuestras propias salidas: 46 de los 73 archivos "recientes" eran
eso), `Canciones`, `REAPER`, `NINJAMsessions`. Lo bajado va a `nebulosa/drive/AAAA-MM/`.

- **Manifest** `drive_files(drive_id, size, modtime, local_path)`: identidad por id de Drive, no
  por ruta. No vuelve a bajar lo que ya está; si un archivo de Drive ya existe local con el mismo
  nombre y tamaño (el corpus viejo se bajó a mano), lo registra sin bajar.
- rclone preserva el `ModTime` de Drive como mtime local, así `is_fresh` no se confunde.
- **Fecha de un archivo** (`fechas_locales`): la del manifest si existe, si no el mtime local. Para
  el corpus viejo el mtime es de copias y poco confiable; para lo sincronizado es la fecha de
  subida a Drive, que es lo más cercano a "fecha de ensayo" que hay.
- **`--ultima-sesion`** = todos los archivos del **día calendario local** más reciente con
  subidas (una subida a las 23:30 no cae en el día siguiente por estar en UTC). `--meses` /
  `--desde` filtran por rango. Aplican a `rank` y `compilado`; la normalización por percentil
  sigue siendo sobre TODO el corpus, el filtro solo decide qué tomas son elegibles: la sesión
  nueva se puntúa contra toda la historia.
- El compilado con filtro se nombra solo: `compilado_<filtro>_<perfil>_<fijo|dinamico>.mp3`
  (ej. `compilado_sesion-2026-09-09_balance_dinamico.mp3`), escribe un `.txt` con parámetros y
  lista, y mete lo mismo en los tags ID3 (title/comment) para verlo en el reproductor.
- **Automatización: NO instalar nada en la máquina del usuario.** Decisión explícita: la
  máquina no está prendida a horas fijas, y tampoco quiere que procese al encenderla porque en
  ese momento la necesita para otra cosa ("no quiero que procese cuando la necesito"). Existe
  `--instalar-launchd HORAS` (LaunchAgent con `RunAtLoad` + `StartInterval`) pero **no se usa**.
  El flujo es manual y a demanda: `--sync` en `rank` / `compilado` / `eval` / `comparar` baja lo
  nuevo de Drive (`--sync-meses`, default 3), lo procesa y recién después arma. "Dame el compilado
  de la última zapada" = `compilado --sync --ultima-sesion --dinamico --ajustar-tempo`.

En Drive/Nebulosa hay 957 MP3 en la raíz contra 849 locales: hay ~100 viejos que nunca se
bajaron. `sync --meses 120` los traería.

## Feedback pareado (`zapaia comparar` / `zapaia feedback`)

Preguntar "¿qué tan creativo es esto?" no produjo respuestas; mostrar A contra B y preguntar
"¿cuál rescatarías?" sí. `feedback.py` convierte eso en datos:

- **Candidatos**: un segmento por archivo, la mejor tirada contigua de `--seg-win` ventanas
  (default 2 = 60 s) elegida por **ejecución limpia** (sin `tonal_outlier`): queremos comparar
  ideas, no castigar cromatismos antes de que el humano opine. Se exportan `--dur` s (default 40)
  desde ahí, con el corte pegado al beat.
- **Muestreo** (`muestrear_pares`): 50% *parejos* (score compuesto parecido, temas distintos por
  `temas_por_nombre` y distinto grupo de dedupe → máxima información por respuesta), 30% al azar
  (calibra), 20% *gem_vs_perf* (interés alto/ejecución baja contra lo inverso → responde la
  pregunta estratégica del review). Nunca repite un par ya mostrado.
- **Dos modos**: `afplay` en la máquina, o `--drive` que sube `lote_NNN/par_NNN_{A,B}.mp3` +
  `lote.txt` a `Seleccion IA - compilados/feedback/` con rclone. Las respuestas vuelven por
  WhatsApp y `feedback importar "1 A, 2 b; 3 ninguno"` las parsea tal cual (tolera `:`, `-`,
  `ambas`, `los dos`).
- **Tablas** (aditivas, `PRAGMA user_version = 1`): `segmentos(path, start, end)` y
  `pares(lote, num, seg_a, seg_b, tipo, eleccion, fver)`. `eleccion NULL` = pendiente. Cada
  respuesta guarda `fver` para poder reentrenar sin ambigüedad cuando cambien las features.
- **`feedback resumen`**: Bradley–Terry (MM de Hunter, prior débil) → score latente por
  segmento; Spearman contra cada dimensión y composite; tasa de victoria de la gema en los pares
  `gem_vs_perf`. **Es lo que reemplaza el ajuste manual de pesos.** Con <30 respuestas es
  orientativo; con 150–300 se puede entrenar una regresión logística sobre diferencias de features
  (P6). `ambos`/`ninguno` cuentan como empate.

El `lote.txt` no muestra scores ni el tipo de par, para no sesgar la escucha.

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

### Primeros negativos (2026-09-14) y lo que enseñaron

El usuario escuchó el compilado de todos los tiempos y marcó `sisterborrachosa 1:00-2:31`
("la voz desafinada, varios pifies de viola") y los dos tramos de `bativiola gera joguito
nebuzap 1` ("igual estos dos"; "la bata está bien, es lo mejor del tema, pero el resto pifia
bastante"). El score los tenía en 0.70 y 0.77, `bativiola` era **#5 de 733**. Con esos dos
negativos contra los 5 positivos, **AUC = 0.00**: los dos descartados quedaban por encima de
todos los elegidos.

Diagnóstico por feature: lo que el score premiaba era **la batería** — `ibi_cv` pct 0.91,
`band_sync` 0.94–0.98, `beat_strength` 0.99, `tuning_dev` 0.96 (afinación *global*, dominada por
bajo y rítmica). No existe ninguna feature que escuche voz ni melodía. La regla del usuario:
**"la bata importa, pero solo si el resto no pifia tanto"** → la limpieza del resto debería
actuar como condición, no como sumando que se promedia.

Se probaron variantes contra las 7 etiquetas: devolver `onset_dev_mad` a 0.40 fue lo único que
mejoró ambos lados (AUC 0.00 → 0.40, positivos 82.9 → 89.1, `sisterborrachosa` 92.6 → 79.7).
`bativiola` no baja con ninguna combinación de features existentes.

**Intento fallido, documentado para no repetirlo**: `intonacion_features` (pYIN sobre la banda
80–1000 Hz, con y sin separación armónica, desvío en cents a la nota más cercana). Sobre los 3
tramos malos vs 3 buenos: malos 5–13 cents, buenos 8–19. **No separa.** Causas: pYIN sigue la
altura dominante de la mezcla (bajo, rítmica: afinados), no la voz; la fracción "cantada" es
6–32% aun separando; y un pifie de viola es una nota **equivocada pero afinada**, invisible en
cents. Queda en `features.py` sin conectar. Detectar voz desafinada requiere separar la voz
(Demucs/PyTorch) primero; detectar pifies requiere contexto armónico, no afinación.

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

## Reviews y decisiones pendientes

- `docs/review-2026-09-12.md` — contraste de un review externo contra el código real. Contiene el
  roadmap vigente (P0–P8), los bugs verificados con datos (`fuera_tono_picos` constante,
  `onset_dev_mad` ≈ ruido, `find_groups` no transitivo y sin centrar, `is_fresh` ignora `size`) y
  el diseño del experimento de embeddings y de la herramienta de feedback pareado. **Leerlo antes
  de tocar `score.py` o `features.py`.**

## Convenciones

- Todo en español: comentarios, prints, nombres de columnas del CSV.
- Los audios **nunca** se commitean. El repo todavía no es un git repo; si se inicializa, el
  `.gitignore` ya está listo.
