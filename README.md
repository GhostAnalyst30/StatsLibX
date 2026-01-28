# 📦 StatsLibX

StatsLibX es un paquete de Python diseñado para proporcionar una solución sencilla, eficiente y flexible para manejar volumenes de datos.

Este proyecto surge con la idea de ofrecer una alternativa moderna, intuitiva y ligera que permita a desarrolladores y entusiastas integrar la **estadistica descriptiva, inferencial y computacional (En desarrollo)** sin complicaciones, con multiples funcionalidades y utilidades pensadas para el futuro.

GitHub del Proyecto: [https://github.com/GhostAnalyst30/StatsLibX](https://github.com/GhostAnalyst30/StatsLibX)

## ✨ Características principales

- ⚡ Rápido y eficiente: optimizado para ofrecer un rendimiento suave incluso en tareas exigentes.

- 🧩 Fácil de usar: una API limpia para que empieces en segundos.

- 🔧 Altamente extensible: personalízalo según tus necesidades.

- 📚 Documentación clara: ejemplos simples y prácticos.

- 🔮 Diseñado con visión a futuro: construido para escalar y adaptarse.

## 🚀 Ejemplo rápido
```python
from statslibx import DescriptiveStats, InferentialStats, UtilsStats
from statslibx.datasets import load_iris()

data = load_iris()

stats = DescriptiveStats(data) # InferentialStats(data), UtilsStats()

stats.summary()
```
Para ver mas funciones: [https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb](https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb)

##  📦 Instalación
```bash
pip install statslibx
```

## 👩‍💻 ¡Usalo en la terminal! (De forma preliminar)
```bash
statslibx describe .\archive.csv # Devuelve una descripcion de la data
statslibx quality .\archive.csv # Devuelve la calidad de los datos
statslibx preview .\archive.csv # Devuelve una visualizacion de los datos
```

🤝 Contribuciones

¡Todas las mejoras e ideas son bienvenidas! 

E-mail: ascendraemmanuel@gmail.com