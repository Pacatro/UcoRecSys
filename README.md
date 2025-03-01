# EduRecSys

Este repositorio forma parte de mi Trabajo de Fin de Grado (TFG) en la Universidad de Córdoba (UCO):  
***Aplicación de sistemas de recomendación en entornos educativos.***

El objetivo de este TFG es desarrollar un sistema de recomendación para e-learning basado en un conjunto de datos de referencia, lo que permitirá evaluar su rendimiento en comparación con otros modelos previos.

## Puesta en marcha

> [!NOTE]
> Para ejecutar este proyecto es necesario tener instalado el manajador de paquetes [`uv`](https://docs.astral.sh/uv/)

Sigue estos pasos para ejecutar el proyecto:

1. **Clona el repositorio**

    ```bash
    git clone https://github.com/Pacatro/EduRecSys.git
    cd EduRecSys
    ```

2. **Ejecuta el proyecto**

    El siguiente comando ejecutará el proyecto creando el entorno virtual e instalando todas las dependencias necesarias:

    ```bash
    uv run main.py
    ```

## Estructura del proyecto

> [!WARNING]
> La estructura de directorios y archivos puede variar con el paso del tiempo.

La estructura del repositorio es la siguiente:

```terminal
├── data
│   ├── explicit_ratings_en.csv
│   ├── explicit_ratings_fr.csv
│   ├── ...
├── database
│   ├── tfg_db.db
│   └── tfg_db.sql
├── db.py
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

### Descripción de directorios y archivos principales

- `data/`: Contiene los datasets en formato CSV utilizados para entrenar y validar el modelo.
- `database/`: Incluye la base de datos SQLite y el script SQL para su creación.
- `db.py`: Funciones para cargar y crear la base de datos a partir de los archivos CSV.
- `main.py`: Lógica principal del proyecto, encargada de cargar la base de datos y gestionar la generación de recomendaciones.

## Autor  

**Francisco de Paula Algar Muñoz**  

## Tutores  

- **Amelia Zafra Gómez**  
- **Cristóbal Romero Morales**
