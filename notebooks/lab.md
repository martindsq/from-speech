# Cuaderno de laboratorio

## Arquitectura general

CoWaver mapea una imagen de una palabra a un espectrograma Mel. El flujo común es:

```text
imagen
-> codificador visual
-> secuencia visual horizontal
-> adaptador temporal
-> decodificador Mel
-> espectrograma Mel predicho
```

El codificador visual usa CORnet-Z como extractor de características inspirado en la vía ventral. Luego, en vez de colapsar la imagen a un vector, el modelo preserva el eje horizontal y convierte (usando el módulo `ImageToHorizontalFeatures`) el mapa visual de características en una secuencia de `width_steps` posiciones. Este es el sesgo inductivo principal para lectura: la organización espacial de izquierda a derecha del estímulo escrito se vuelve la secuencia de entrada para las etapas que generan audio.

El adaptador temporal transforma esta secuencia visual horizontal antes de decodificar. Las variantes arquitectónicas actuales difieren sobre todo en este adaptador:

| Arquitectura | Adaptador | Hipótesis principal |
| :-- | :-- | :-- |
| unconditioned + adapter pointwise | proyección independiente por posición | el decoder puede resolver la composición sin contexto compartido |
| unconditioned + adapter convolutional | convolución temporal 1D local | compartir contexto local beneficia a las rutas de salida |
| conditioned | adaptador convolucional + embedding de tarea | la misma imagen puede mapearse a modos de salida distintos cuando el contexto de tarea es explícito |
| dual-route | adaptador convolucional + decodificador por tarea | letras y palabras se benefician de rutas de salida separadas |

Finalmente, el decodificador mapea la secuencia temporal latente a un espectrograma Mel de tamaño fijo. Las variantes de decodificador se tratan como un eje experimental separado:

| Decodificador | Mecanismo | Hipótesis principal |
| :-- | :-- | :-- |
| convolutional | interpolación fija más una convolución 1D | un refinamiento local suave es suficiente |
| recurrent | interpolación seguida de una GRU | la dinámica temporal acústica importa |

## Entrenamiento

El entrenamiento está organizado como *curriculum learning*. En el curriculum principal, la fase 1 entrena con letras, la fase 2 mezcla letras y palabras fonetizadas, y la fase 3 mezcla letras, palabras fonetizadas y palabras de MSWC. Para las arquitecturas no condicionadas, phones y MSWC son targets incompatibles para el mismo estímulo visual, así que es esperable que la red olvide palabras fonetizadas en la fase 3. En los modelos conditioned y dual-route, `task_id` desambigua el modo de salida esperado: las letras y las palabras fonetizadas usan la tarea 1, y las palabras de MSWC usan la tarea 2.

## Curriculums de lectura

Con la configuración `dual-route`, decoder/adaptador convolucional, `ws7-hb7`, `max_epochs=8` y `max_classes=50`, se compararon varios curriculums. Las métricas reportadas son `top1/top3/top5` al final de la fase 3:

| Curriculum | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `baseline` | 22.58 / 38.71 / 61.29 | 12 / 42 / 54 | 4 / 20 / 24 |
| `phonics-first` | 19.35 / 41.94 / 58.06 | 18 / 42 / 54 | 4 / 16 / 22 |
| `decoding-bridge` | 16.13 / 41.94 / 61.29 | 10 / 34 / 48 | 6 / 10 / 20 |
| `word-heavy` | 16.13 / 45.16 / 51.61 | 10 / 24 / 44 | **12 / 36 / 44** |
| `word-heavy-soft` | 19.35 / 38.71 / 54.84 | 16 / 40 / 52 | 12 / 28 / 34 |
| `word-heavy-balanced` | **22.58 / 45.16 / 64.52** | 20 / 46 / 60 | 12 / 26 / 30 |
| `word-heavy-phonics-retain` | 20.13 / 43.62 / 60.40 | **24 / 56 / 60** | 8 / 20 / 24 |

Los curriculums evaluados fueron:

```text
baseline
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.25 letters, 0.75 phones, 0.00 words
  p3 = 0.10 letters, 0.15 phones, 0.75 words

phonics-first
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.50 letters, 0.50 phones, 0.00 words
  p3 = 0.05 letters, 0.55 phones, 0.40 words

decoding-bridge
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.60 letters, 0.40 phones, 0.00 words
  p3 = 0.10 letters, 0.50 phones, 0.40 words

word-heavy
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.20 letters, 0.60 phones, 0.20 words
  p3 = 0.05 letters, 0.10 phones, 0.85 words

word-heavy-soft
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.15 letters, 0.60 phones, 0.25 words
  p3 = 0.05 letters, 0.15 phones, 0.80 words

word-heavy-balanced
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.20 letters, 0.50 phones, 0.30 words
  p3 = 0.05 letters, 0.20 phones, 0.75 words

word-heavy-phonics-retain
  p1 = 1.00 letters, 0.00 phones, 0.00 words
  p2 = 0.15 letters, 0.70 phones, 0.15 words
  p3 = 0.05 letters, 0.25 phones, 0.70 words
```

La primera lectura es que `word-heavy` maximiza MSWC, mientras que `word-heavy-balanced` ofrece el mejor compromiso global en la corrida corta: conserva letras, mejora phones y mantiene el mismo `top1` en MSWC que `word-heavy`, aunque pierde `top3/top5` en palabras. `word-heavy-phonics-retain` confirma el costo de poner demasiado peso en phones: mejora la tarea fonetizada, pero cae en MSWC.

También se probó limitar el largo de palabras en fase 2 (`<=4`) y volver a vocabulario completo en fase 3. Las variantes `phonics-first-len4-full` y `word-heavy-len4-full` no mejoraron las métricas finales: en particular `word-heavy-len4-full` bajó MSWC de `12/36/44` a `10/22/32`. Por ahora, limitar el largo de las palabras no parece una dirección prometedora.

Luego se escaló la comparación principal a `max_epochs=15` y `max_classes=100`, manteniendo `ws7-hb7`:

| Curriculum | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `word-heavy` | 16.78 / 36.91 / 57.05 | **15 / 37 / 45** | **15 / 25 / 32** |
| `word-heavy-balanced` | 16.78 / 36.91 / 57.05 | 13 / 27 / 42 | 15 / 22 / 27 |

Con más clases y más epochs, `word-heavy-balanced` pierde la ventaja global observada en la corrida corta. Ambos empatan en letras y en `top1` de MSWC, pero `word-heavy` queda mejor tanto en phones como en `top3/top5` de MSWC. La conclusión actual es usar `word-heavy` como candidato principal para escalar, manteniendo `word-heavy-balanced` como control más equilibrado si el objetivo explícito es preservar una representación fonológica más fuerte.

## Configuracion del encoder

El encoder visual convierte la salida espacial de CORnet-Z en una secuencia horizontal. Las imágenes de entrada tienen tamaño `224x224`, pero la pila convolucional de CORnet-Z reduce la resolución espacial hasta mapas de `7x7`. Una configuración como `width_steps=7` y `height_bands=7` usa casi directamente la grilla nativa del extractor visual, mientras que otras configuraciones interpolan esa grilla para obtener más o menos pasos horizontales y bandas verticales.

Las configuraciones con más estructura vertical, por ej. `height_bands=4`, mejora consistentemente frente a `height_bands=1` en los anchos comparables:

| Configuracion | avg@1 | avg@3 | avg@5 |
| :------------ | :---: | :---: | :---: |
| `ws12-hb1`    | 13.53 | 31.83 | 44.95 |
| `ws12-hb4`    | 15.27 | 32.24 | 46.02 |
| `ws20-hb1`    | 14.60 | 33.05 | 42.28 |
| `ws20-hb4`    | 15.53 | 32.24 | 44.69 |
| `ws24-hb1`    | 10.04 | 27.57 | 39.10 |
| `ws24-hb4`    | 12.19 | 29.98 | 40.69 |

En la ronda intermedia (`max_epochs=15`, `max_classes=100`) se compararon los mejores candidatos del barrido rápido:

| Configuracion | Fase 3 avg@1 | Fase 3 avg@3 | Fase 3 avg@5 |
| :-- | --: | --: | --: |
| `ws7-hb7` | 18.75 | 31.13 | 41.43 |
| `ws20-hb4` | 18.01 | 33.95 | 47.47 |
| `ws12-hb4` | 17.34 | 32.87 | 43.10 |

Luego se ajustó `ImageToHorizontalFeatures` para evitar interpolaciones cuando la salida de CORnet-Z ya coincide con la configuración pedida. En particular, para `ws7-hb7` esto evita reescalar una grilla que ya es nativamente cercana a `7x7`. Al repetir `ws7-hb7` con `max_epochs=15` y `max_classes=100`, la métrica cambió poco en avg@1 y mejoró en top-k:

| Configuracion | Fase 3 avg@1 | Fase 3 avg@3 | Fase 3 avg@5 |
| :-- | --: | --: | --: |
| `ws7-hb7` antes | 18.75 | 31.13 | 41.43 |
| `ws7-hb7` después | 18.01 | 35.28 | 43.51 |

La interpretación actual es que preservar algo de estructura vertical ayuda, y que el encoder no debería forzar interpolaciones innecesarias. Los candidatos principales son entonces `ws7-hb7` y `ws20-hb4`: el primero explota la grilla nativa del encoder y el segundo parece más robusto en métricas top-k.

## CTC y pooling vertical

Para inducir composicionalidad se agregó una cabeza CTC auxiliar sobre la secuencia horizontal producida por el encoder visual. La pérdida total de entrenamiento queda:

```text
loss = mel_loss + ctc_weight * ctc_loss
```

La cabeza CTC predice la secuencia de caracteres normalizados de la palabra renderizada. Los acentos se normalizan a la vocal base, mientras que `ñ` se conserva como símbolo propio. En los experimentos siguientes se usó `ctc_weight=0.1`, arquitectura `dual-route`, adapter y decoder convolucionales, curriculum `word-heavy`, y el adapter convolucional local con kernels `3,3,3` sin dilatación.

Primero se evaluó si CTC ayudaba sobre la configuración nativa `ws7-hb7` con corridas cortas (`max_epochs=8`, `max_classes=50`). Las métricas son las de la fase 3:

| Configuracion | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `ws7-hb7`, sin CTC, kernel previo | 23.49 / 43.62 / 63.76 | 24 / 52 / 64 | 6 / 20 / 28 |
| `ws7-hb7`, sin CTC, kernel 3 | 20.13 / 36.91 / 53.69 | **30 / 64 / 72** | 6 / 16 / 24 |
| `ws7-hb7`, CTC 0.1, kernel 3 | 19.35 / 41.94 / 67.74 | 10 / 26 / 44 | 20 / 36 / 40 |

Reducir el kernel del adapter convolucional de la configuración previa a `3,3,3` mejora la retención en phones cuando no hay CTC, lo cual sugiere que mezclar demasiado rápido toda la palabra perjudica la ruta fonológica. Al activar CTC, el efecto principal es distinto: sube mucho MSWC (`6` a `20` top-1 en la corrida corta), aunque phones vuelve a caer en fase 3.

Luego se compararon resoluciones horizontales y verticales manteniendo CTC y kernel 3, todavía en corridas cortas (`max_epochs=8`, `max_classes=50`):

| Configuracion | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `ws7-hb7` | 19.35 / 41.94 / 67.74 | 10 / 26 / 44 | 20 / 36 / 40 |
| `ws7-hb4` | 22.58 / 38.71 / 64.52 | 10 / 34 / 52 | 18 / 32 / 38 |
| `ws12-hb7` | 25.81 / 38.71 / 58.06 | 10 / 26 / 44 | 16 / 28 / 36 |
| `ws12-hb4` | 22.58 / 38.71 / 54.84 | 10 / 26 / 48 | **22** / 30 / 36 |
| `ws16-hb7` | 25.81 / 45.16 / 61.29 | **12** / 28 / 46 | 18 / 32 / **44** |

Las resoluciones interpoladas mejoran algunas métricas, pero agregan un hiperparámetro difícil de justificar: el mapa IT nativo de CORnet-Z ya es `7x7`. Por eso se probó una alternativa más interpretable: preservar los 7 pasos horizontales nativos y promediar la altura con `AdaptiveAvgPool2d((1, 7))`. Esta variante produce la misma forma de salida que `ws7-hb7`, `[B, 7, feature_dim]`, pero introduce invariancia vertical explícita.

En corridas cortas (`max_epochs=8`, `max_classes=50`), el pooling vertical mejora letras pero pierde algo en MSWC:

| Encoder | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `ws7-hb7` full-height | 19.35 / 41.94 / 67.74 | 10 / 26 / 44 | **20 / 36 / 40** |
| pooled vertical | **22.58** / 41.94 / **70.97** | 10 / 32 / 52 | 16 / 28 / 38 |

Con más entrenamiento y más clases (`max_epochs=15`, `max_classes=100`), el pooling vertical se recupera y pasa a ser el mejor compromiso global:

| Configuracion | Letras | Phones | MSWC |
| :-- | :--: | :--: | :--: |
| `ws7-hb7` full-height | 32.26 / 58.06 / 67.74 | 18 / 36 / **45** | **16** / **26** / 32 |
| pooled vertical | **38.71** / 58.06 / **74.19** | 18 / 36 / 43 | **16** / 24 / **33** |
| `ws12-hb4` interpolado | 32.26 / **61.29** / **74.19** | **20** / **38** / **45** | 11 / 25 / **34** |
| `ws20-hb4` interpolado | 32.26 / 58.06 / **74.19** | 15 / 35 / 44 | 15 / 25 / 31 |

La lectura actual es que CTC aporta una presión composicional útil y que el pooling vertical es una forma limpia de obtener una secuencia ortográfica horizontal invariante a la posición vertical. Frente a las resoluciones interpoladas, pooled no maximiza todas las métricas individuales, pero ofrece el mejor balance: gana claramente en letras, empata o queda muy cerca en phones y MSWC, y evita introducir pasos horizontales artificiales. Por ahora, la configuración principal recomendada es `AvgPooledITEncoder` con CTC `0.1`, kernel convolucional `3,3,3`, `dual-route` y curriculum `word-heavy`.

### Split de generalización en palabras y LR global

Para medir composicionalidad de manera más directa, `train.py` pasó a reservar un 10% de las clases de `tiny-mswc-200` como palabras de test. Las letras no se dividen: `tiny-letter-30` funciona como alfabeto compartido. Phones tampoco se dividen por ahora. Con `max_classes=100`, el protocolo queda:

```text
letters: 30 clases, todas disponibles
phones: 100 clases, todas disponibles
words: 90 clases train + 10 clases test
```

En la corrida actual, las palabras reservadas para test fueron:

```text
['cada', 'cinco', 'dio', 'dos', 'fecha', 'figura', 'idea', 'lista', 'local', 'lugar']
```

La evaluación de `words test` usa solo ejemplos de esas 10 clases, pero los rankea contra prototipos de las 100 clases seleccionadas. Por lo tanto, el azar vuelve a ser bajo:

```text
Top-1 chance: 1%
Top-3 chance: 3%
Top-5 chance: 5%
```

También se simplificó el entrenamiento: se reemplazaron tasas de aprendizaje específicas por submódulo por un único schedule global lineal. Esto reduce grados de libertad experimentales sin degradar el fenómeno principal. Se compararon tres schedules usando `AvgPooledITEncoder`, CTC `0.1`, `max_classes=100`, `theta_max=45`, decoder y adapter convolucionales. Las métricas son las de fase 3:

| Schedule | Letras | Phones | MSWC train | MSWC test |
| :-- | :--: | :--: | :--: | :--: |
| `3e-4`, `theta=30` | 29.03 / **54.84** / **67.74** | 21 / 38 / 46 | **16.67 / 33.33 / 38.89** | 0 / 10 / 10 |
| `1e-3`, `theta=30` | 22.58 / 41.94 / 64.52 | 21 / 40 / 46 | 14.44 / 30.00 / 37.78 | 0 / 0 / 20 |
| `3e-4`, `theta=45` | **32.26** / **54.84** / 64.52 | **26 / 43 / 47** | 14.44 / 24.44 / 34.44 | 0 / 10 / 20 |

El schedule `3e-4`, `theta=45`, `epsilon_theta=3e-5` queda como baseline principal porque conserva mejor letras y phones, que son las señales subléxicas más relevantes si el objetivo es composición. `3e-4`, `theta=30` maximiza `MSWC train`, pero no mejora `MSWC test`. El LR alto (`1e-3`) no aporta una ventaja clara.

La lectura conceptual cambió con la métrica dura: la aparente mejora previa en heldout era demasiado optimista porque rankeaba solo contra las 10 clases reservadas. Con test contra las 100 clases, `MSWC test` queda cerca del azar. El modelo aprende palabras entrenadas y conserva información útil para letras/phones, pero todavía no aprendió una función robusta de composición de palabra visual nueva a prototipo acústico.

Se probó luego aumentar la diversidad léxica a `max_classes=200`. El split pasa a ser 180 clases train y 20 clases test. En este régimen, el azar en `words test` baja aún más:

```text
Top-1 chance: 0.5%
Top-3 chance: 1.5%
Top-5 chance: 2.5%
```

También se comparó dónde aplicar la pérdida CTC: sobre `h`, la salida del encoder visual, o sobre `z`, la representación después del adapter que alimenta al decoder. Las métricas son fase 3, con `3e-4`, `theta=45`, `theta_max=45`, CTC `0.1`, decoder y adapter convolucionales:

| Clases | CTC | Letras | Phones | MSWC train | MSWC test |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 100 | `z` | 22.58 / 51.61 / 61.29 | **23 / 41 / 48** | 14.44 / **28.89 / 36.67** | 0 / 10 / 20 |
| 200 | `z` | **25.81** / 48.39 / **74.19** | **9.5 / 20.5 / 25.5** | 14.44 / 21.11 / 25.00 | 10 / 15 / 15 |
| 200 | `h` | 19.35 / 48.39 / 70.97 | 9.0 / 18.5 / 24.5 | **15.56** / 21.11 / **27.22** | **15 / 20 / 35** |

Este es el primer resultado fuerte de generalización composicional: con 200 clases y CTC sobre `h`, `MSWC test` queda claramente por encima del azar (`15 / 20 / 35` contra `0.5 / 1.5 / 2.5`). Además, el control `h` vs `z` favorece supervisar la representación visual. Aunque `z` conserva algo mejor letras/phones top-k, CTC sobre `h` generaliza mejor a palabras reservadas y es más interpretable neurobiológicamente: la presión ortográfica se aplica sobre una representación visual/IT-like antes de la transformación hacia el espacio acústico.

Por ahora, el baseline principal pasa a ser `AvgPooledITEncoder`, adapter y decoder convolucionales, CTC `0.1` sobre `h`, `max_classes=200`, `theta_max=45`, `epsilon_zero=3e-4`, `theta=45` y `epsilon_theta=3e-5`.

## Configuración del modelo convolucional

La fase 1 entrena con letras renderizadas aisladas emparejadas con sus representaciones de audio, usando posición aleatoria del texto pero sin aumentación de escena acústica.
La fase 2 continúa desde el mismo modelo en la tarea de letras, agregando aumentación de waveform con alineamiento aleatorio (el waveform se ubica al principio, al centro o al final de la ventana de 1 segundo, en vez de estar siempre centrado como en la fase 1) y escenas acústicas aleatorias. La fase 3 transfiere el modelo a palabras reales del microset de MSWC (con apenas 17 palabras en español), donde el modelo debe mapear palabras renderizadas completas a sus targets de espectrograma Mel correspondientes.

La siguiente tabla lista las mejores pérdidas de validación para todos los modelos en las fases 1, 2 y 3:

| LD[^1] | HS[^2] | MB[^3] | WS[^4] | P-1 | P-2 | P-3 |
| :-: | :-: | :-: | :-: | :--: | :--: | :--: |
| 256 | 256 | 80 | 24 | .229 | .258 | .357 |
| 256 | 256 | 80 | 32 | .232 | .258 | .357 |
| 256 | 256 | 80 | 40 | .231 | .253 | .357 |
| 256 | 256 | 80 | 48 | .239 | .260 | .357 |
| 256 | 256 | 80 | 56 | .236 | .259 | .357 |
| 256 | 384 | 80 | 48 | .231 | .260 | .356 |
| 320 | 320 | 80 | 40 | .233 | .257 | .356 |
| 320 | 320 | 80 | 48 | .232 | .260 | .357 |
| 384 | 256 | 80 | 48 | .237 | .264 | .356 |
| 384 | 384 | 80 | 32 | .231 | .255 | .356 |
| 512 | 512 | 80 | 32 | .234 | .256 | .356 |
| 256 | 256 | 64 | 24 | .246 | .272 | .379 |
| 128 | 128 | 40 | 16 | .296 | .349 | .428 |
| 128 | 256 | 40 | 24 | .291 | .332 | .425 |
| 256 | 256 | 40 | 24 | .284 | .329 | .425 |
| 256 | 256 | 40 | 32 | .290 | .332 | .425 |
| 256 | 512 | 40 | 32 | .290 | .329 | .425 |

Usar 80 bins Mel en vez de 40 o 64 es la variable más relevante. Los modelos de 40 bins quedan en una meseta bastante más alta en fase 3, alrededor de .425-.428, mientras que el modelo de 64 bins mejora hasta .379 y los modelos de 80 bins se agrupan alrededor de .356-.357. Una vez que los bins Mel están en 80, aumentar la dimensión latente, el hidden size o los width steps produce diferencias chicas: configuraciones razonables de 80 bins alcanzan casi la misma pérdida de validación en fase 3.

El mejor valor observado en fase 3 (384/256/80/48) es apenas mejor que el del modelo más chico (256/256/80/24); esta diferencia puede ser ruido de semilla más que una ventaja arquitectónica significativa. Por lo tanto, 256/256/80/24 es un buen modelo para seguir adelante y, en caso de que haga falta, 384/256/80/48 puede venir bien si la tarea se vuelve más exigente en el futuro.

[^1]: Dimensiones latentes
[^2]: Tamaño de la capa oculta
[^3]: Bines del espectrograma mel
[^4]: *Width steps*
