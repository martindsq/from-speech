```bash
cp install.example.batch install.batch
cp train.example.batch train.batch
```

Editar estos nuevos archivos y reemplazar `YOUR_EMAIL_HERE` por tu email (para que te avise cuando la tarea termina).

Luego, para instalar el entorno correr:

```batch
sbatch install.batch
```

y para entrenar los modelos:

```batch
sbatch train.batch
```
