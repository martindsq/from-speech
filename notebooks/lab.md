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
| unconditioned + adapter convolutional | convoluciones temporales 1D residuales | la composición local de izquierda a derecha alcanza |
| unconditioned + adapter recurrent | GRU bidireccional | el contexto secuencial ayuda al mapeo grafema-sonido |
| unconditioned + adapter transformer | codificador Transformer | las interacciones globales entre posiciones visuales ayudan a la composición |
| conditioned | adaptador convolucional + embedding de tarea | la misma imagen puede mapearse a modos de salida distintos cuando el contexto de tarea es explícito |
| dual-route | adaptador convolucional + decodificador por tarea | letras y palabras se benefician de rutas de salida separadas |

Finalmente, el decodificador mapea la secuencia temporal latente a un espectrograma Mel de tamaño fijo. Las variantes de decodificador se tratan como un eje experimental separado:

| Decodificador | Mecanismo | Hipótesis principal |
| :-- | :-- | :-- |
| convolutional | interpolación fija más convoluciones 1D residuales | un refinamiento local suave es suficiente |
| recurrent | interpolación seguida de una GRU | la dinámica temporal acústica importa |
| transformer | consultas aprendidas de frames Mel que atienden a la secuencia latente | el alineamiento flexible visual-acústico importa |

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
