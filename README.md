# From Speech to Print: A Neurocomputational Model of the Predictors of Reading Based on Learned Speech Representations

CoWaver entrena modelos que reciben una imagen de texto y predicen una
representacion Mel del audio correspondiente.

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

## Entrenamiento local

```bash
python train.py
```

## Evaluacion local

`eval.py` carga los checkpoints disponibles y evalua el modelo.

```bash
python eval.py --data data --checkpoints checkpoints
```

## Uso en ClusterUY/Slurm

El repo incluye plantillas para instalar y evaluar con `sbatch`.

Instalar el entorno CUDA:

```bash
sbatch install.batch
```

Luego ya es posible entrenar:

```bash
sbatch train.batch
```

Para evaluar checkpoints en Slurm:

```bash
sbatch eval.batch
```
