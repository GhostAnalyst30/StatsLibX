# StatsLibX v2.6

![](https://raw.githubusercontent.com/GhostAnalyst30/StatsLibX/main/StatsLibX.png)

StatsLibX es un paquete de Python diseñado para proporcionar una solución sencilla, eficiente y flexible para manejar volumenes de datos.

Este proyecto surge con la idea de ofrecer una alternativa moderna, intuitiva y ligera que permita a desarrolladores y entusiastas integrar la **estadistica descriptiva, inferencial y computacional** sin complicaciones, con multiples funcionalidades y utilidades pensadas para el futuro.


| **Documentacion:** | **GitHub del Proyecto:** |
|-------------------|--------------------------|
[Documentacion StatsLibX](https://ghostanalyst30.github.io/StatsLibX/Documentation_Page/index.html) |  [Github/StatsLibX](https://github.com/GhostAnalyst30/StatsLibX)
|**Version:** 0.2.6 | **Autor:** Emmanuel Ascendra |


## Características principales

- Rápido y eficiente: optimizado para ofrecer un rendimiento suave incluso en tareas exigentes.

- Fácil de usar: una API limpia para que empieces en segundos.

- Altamente extensible: personalízalo según tus necesidades.

- Documentación clara: ejemplos simples y prácticos.

- Diseñado con visión a futuro: construido para escalar y adaptarse.

## Ejemplo rápido
```python
from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, UtilsStats
from statslibx.datasets import load_iris()

data = load_iris()

stats = DescriptiveStats(data) 
# InferentialStats(data), ComputationalStats(data), UtilsStats()

stats.summary()
```
Para ver mas funciones: [StatslibX](https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb)

## Instalación
```bash
pip install statslibx
```

## Implementacion con ViewX

```python
from statslibx.viewx import HTML, Slides, Report, DataMatrix
```
![ViewX](https://raw.githubusercontent.com/GhostAnalyst30/ViewX/main/images_for_git/DashBoard_Example.png)

Para saber mas: [ViewX](https://ghostanalyst30.github.io/ViewX/Documentation_Page/index.html)

## 👩‍💻 ¡Usalo en la terminal!
```bash
# Data
statslibx data iris.csv

statslibx data mi_archivo.csv --summary --types --missing

# Info
statslibx info iris.csv

statslibx info iris.csv --detailed

# Describe
statslibx describe iris.csv --numeric

statslibx describe iris.csv --categorical

statslibx describe iris.csv

# Quality
statslibx quality iris.csv

statslibx quality iris.csv --verbose

# Preview
statslibx preview iris.csv -n 10

statslibx preview iris.csv -n 5 --sample
```

🤝 Contribuciones

¡Todas las mejoras e ideas son bienvenidas! 

E-mail: ascendraemmanuel@gmail.com