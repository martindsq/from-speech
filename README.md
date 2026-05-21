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
  --decoder convolutional
```

Arquitecturas disponibles:

- `convolutional`
- `recurrent`
- `transformer`
- `conditioned`
- `dual-route`

Decoders disponibles:

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

## Evaluacion local

`eval.py` carga los checkpoints disponibles para cada fase y evalua el modelo
en letras, palabras fonetizadas y palabras.

```bash
python eval.py \
  --data data \
  --checkpoints checkpoints \
  --architecture dual-route \
  --decoder convolutional
```

## Uso en ClusterUY/Slurm

El repo incluye plantillas para instalar, entrenar y evaluar con `sbatch`.

Primero crear los scripts locales ignorados por git:

```bash
cp install.example.batch install.batch
cp eval.example.batch eval.batch
cp train.example.sh train.sh
```

Editar esos archivos y reemplazar `YOUR_EMAIL_HERE` por tu email.

Instalar el entorno CUDA:

```bash
sbatch install.batch
```

Entrenar una combinacion de arquitectura y decoder:

```bash
./train.sh dual-route convolutional
```

`train.sh` envia `train.batch` con un nombre de job basado en la arquitectura y
el decoder. `train.batch` guarda checkpoints en:

```text
checkpoints/C/<architecture>/<decoder>/
```

Evaluar checkpoints en Slurm:

```bash
sbatch eval.batch
```

## Checkpoints y salidas

Los checkpoints se guardan por fase con nombres como:

```text
<model_name>_1st_phase.pt
<model_name>_2nd_phase.pt
<model_name>_3rd_phase.pt
```
