# CoWaver

CoWaver entrena modelos que reciben una imagen de texto y predicen una
representacion Mel del audio correspondiente. El flujo principal usa tres
fases de entrenamiento sobre letras, palabras fonetizadas y palabras.

## Requisitos

El proyecto usa Python 3.10 y PyTorch. Hay dos archivos de entorno:

- `environment.yml`: entorno base, util para CPU o Apple Silicon/MPS.
- `environment-cuda.yml`: entorno con CUDA 11.8 para correr en GPU.

Para crear el entorno local:

```bash
conda env create -f environment.yml
conda activate cowaver
```

En un nodo con CUDA:

```bash
conda env create -f environment-cuda.yml
conda activate cowaver
```

## Datos

Los scripts esperan encontrar estos archivos comprimidos en la raiz del repo:

- `tiny-letter-30.tar.xz`
- `tiny-phones-200.tar.xz`
- `tiny-mswc-200.tar.xz`

`train.py` y `eval.py` los descomprimen automaticamente dentro de la carpeta
indicada con `--data` y borran esas carpetas temporales al terminar.

## Entrenamiento local

Ejemplo con la arquitectura y decoder por defecto:

```bash
python train.py
```

Ejemplo indicando arquitectura, decoder, carpeta de datos y checkpoints:

```bash
python train.py \
  --data data \
  --checkpoints checkpoints \
  --architecture dual-route \
  --adapter transformer \
  --decoder convolutional
```

Arquitecturas disponibles:

- `unconditioned`
- `conditioned`
- `dual-route`

Usa `unconditioned` con `--adapter` para elegir el procesamiento temporal.

Decoders disponibles:

- `convolutional`
- `recurrent`
- `transformer`

Adapters temporales disponibles:

- `convolutional`
- `recurrent`
- `transformer`

Las proporciones de cada fase se pasan como triples en el orden:
`letters phones words`.

```bash
python train.py \
  --phase1-proportions 1 0 0 \
  --phase2-proportions 0.25 0.75 0 \
  --phase3-proportions 0.10 0.15 0.75
```

Para comparaciones rapidas se puede acortar el entrenamiento y reducir la
cantidad de clases usadas en `tiny-phones-200` y `tiny-mswc-200`:

```bash
python train.py \
  --max-epochs 8 \
  --max-classes 50
```

`--max-classes` no afecta `tiny-letter-30`; solo recorta los datasets de
palabras a las primeras clases ordenadas alfabeticamente. Por defecto vale
`200`.

## Evaluacion local

`eval.py` carga los checkpoints disponibles para cada fase y evalua el modelo
en letras, palabras fonetizadas y palabras.

```bash
python eval.py \
  --data data \
  --checkpoints checkpoints \
  --architecture dual-route \
  --adapter transformer \
  --decoder convolutional
```

## Uso en ClusterUY/Slurm

El repo incluye plantillas para instalar y evaluar con `sbatch`, y un
`train.sh` para enviar las corridas de entrenamiento.

Primero crear los scripts locales de instalacion y evaluacion:

```bash
cp install.example.batch install.batch
cp eval.example.batch eval.batch
```

Configurar el email para las notificaciones de Slurm en el shell. Por ejemplo,
agregar esto a `~/.bashrc`:

```bash
export USER_MAIL=tu_email@example.com
```

Luego recargar la configuracion o abrir una nueva terminal:

```bash
source ~/.bashrc
```

Instalar el entorno CUDA:

```bash
sbatch --mail-user="$USER_MAIL" install.batch
```

Entrenar las corridas definidas en `train.sh`:

```bash
./train.sh
```

`train.sh` es una lista explicita de llamados `sbatch`. Deja activo el barrido
rapido recomendado:

```text
ws12-hb1
ws16-hb1
ws20-hb1
ws24-hb1
ws28-hb1
```

Todos usan `max_epochs=8` y `max_classes=50`. Mas abajo en el mismo archivo quedan
comentados algunos controles `hb4` y finalistas completos para descomentar
cuando haga falta.

`train.sh` requiere `USER_MAIL`.

Cada job llama a `train.batch`, que recibe los parametros del modelo en este
orden:

```text
architecture decoder adapter latent_dim hidden_size mel_bins width_steps height_bands seq_len max_epochs max_classes
```

`max_epochs` y `max_classes` son opcionales y por defecto valen `30` y `200`.

`train.batch` usa una carpeta temporal unica en scratch para cada job:

```text
/scratch/$USER/$SLURM_JOB_ID
```

Los checkpoints se guardan en:

```text
checkpoints/<architecture>/<adapter>/<decoder>/
```

Evaluar checkpoints en Slurm:

```bash
sbatch --mail-user="$USER_MAIL" eval.batch dual-route convolutional transformer
```

## Checkpoints y salidas

Los checkpoints se guardan por fase con nombres como:

```text
<model_name>_1st_phase.pt
<model_name>_2nd_phase.pt
<model_name>_3rd_phase.pt
```
