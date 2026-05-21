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
