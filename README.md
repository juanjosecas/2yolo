# Scripts de Conversión de Anotaciones a YOLO

Este repositorio contiene scripts de Python robustos y mejorados que te permiten convertir anotaciones en formato COCO (Common Objects in Context), Labelbox JSON, e INFOLKS JSON a formato YOLO (You Only Look Once), lo que facilita su uso en tareas de detección de objetos.

## Características

✅ **Control de errores completo**: Manejo robusto de errores en todas las operaciones  
✅ **Validación de datos**: Validación exhaustiva de estructuras JSON, dimensiones de imágenes y coordenadas de bboxes  
✅ **Logging detallado**: Sistema de logging configurable para depuración y seguimiento  
✅ **Argumentos CLI**: Interfaz de línea de comandos intuitiva con argparse  
✅ **Códigos de salida**: Códigos de retorno apropiados para integración en pipelines  
✅ **Manejo de red**: Descarga de imágenes con timeout y manejo de errores  
✅ **Normalización de coordenadas**: Validación y clamping de coordenadas normalizadas

## Instalación

### Dependencias

Los scripts requieren las siguientes bibliotecas de Python:

```bash
pip install pillow requests pyyaml tqdm numpy opencv-python
```

## Uso

### 1. coco2yolo.py

Convierte anotaciones de formato COCO a formato YOLO.

```bash
python coco2yolo.py -j <ruta_al_json_coco> -o <directorio_salida>
```

**Opciones:**
- `-j, --json`: Ruta al archivo JSON de COCO (requerido)
- `-o, --output`: Directorio de salida para las etiquetas YOLO (requerido)
- `-v, --verbose`: Habilita logging detallado

**Ejemplo:**
```bash
python coco2yolo.py -j instances_train2017.json -o output_labels/ -v
```

### 2. labelbox2yolo.py

Convierte anotaciones de Labelbox JSON a formato YOLO.

```bash
python labelbox2yolo.py -j <ruta_al_json_labelbox> -o <directorio_salida> [--zip]
```

**Opciones:**
- `-j, --json`: Ruta al archivo JSON de Labelbox (requerido)
- `-o, --output`: Directorio de salida (requerido)
- `--zip`: Crear archivo zip de los resultados
- `-v, --verbose`: Habilita logging detallado

**Ejemplo:**
```bash
python labelbox2yolo.py -j export.json -o output_directory --zip -v
```

### 3. infolks_to_yolo.py

Convierte anotaciones de INFOLKS JSON a formato YOLO.

```bash
python infolks_to_yolo.py -n <nombre_dataset> -f <patron_archivos_json> [-i <ruta_imagenes>]
```

**Opciones:**
- `-n, --name`: Nombre base para archivos de salida (requerido)
- `-f, --files`: Patrón glob para archivos JSON, ej: "data/*.json" (requerido)
- `-i, --images`: Ruta al directorio de imágenes (opcional)
- `-v, --verbose`: Habilita logging detallado

**Ejemplo:**
```bash
python infolks_to_yolo.py -n mi_dataset -f "data/*.json" -i images/ -v
```

## Manejo de Errores

Todos los scripts incluyen manejo robusto de errores:

- **Archivos faltantes**: Verifica la existencia de archivos antes de procesarlos
- **JSON inválido**: Detecta y reporta errores de formato JSON
- **Datos faltantes**: Valida campos requeridos y omite entradas incompletas
- **Errores de red**: Maneja timeouts y errores de descarga con reintentos
- **Dimensiones inválidas**: Valida dimensiones de imágenes y bboxes
- **Coordenadas fuera de rango**: Clampea valores a rangos válidos [0, 1]

## Formato de Salida

Los scripts generan:

- **Archivos .txt**: Un archivo por imagen con anotaciones en formato YOLO
- **dataset.yaml** (labelbox2yolo): Configuración del dataset para YOLO
- **.names** (infolks_to_yolo): Archivo con nombres de clases
- **Logs detallados**: Información de procesamiento y errores

### Formato YOLO

Cada línea en los archivos .txt contiene:
```
<class_id> <x_center> <y_center> <width> <height>
```

Donde todas las coordenadas están normalizadas entre 0 y 1.

## Códigos de Salida

- `0`: Éxito
- `1`: Error (archivo no encontrado, JSON inválido, etc.)

Esto permite integración fácil en scripts y pipelines CI/CD.

## Logging

Los scripts utilizan el módulo `logging` de Python para proporcionar información detallada:

- **INFO**: Progreso general y estadísticas
- **WARNING**: Entradas omitidas o datos problemáticos
- **ERROR**: Errores críticos que impiden la conversión

Usa la opción `-v` o `--verbose` para obtener más detalles durante la ejecución.

## Basados en

- https://github.com/alexmihalyk23/COCO2YOLO/tree/master
- https://github.com/ultralytics/JSON2YOLO

## Mejoras Implementadas

Esta versión incluye mejoras significativas sobre las versiones originales:

1. **Control de errores exhaustivo** en todas las operaciones de I/O
2. **Validación de datos** completa para evitar crashes
3. **Sistema de logging** profesional para debugging
4. **Interfaz CLI mejorada** con ayuda detallada
5. **Manejo robusto de red** con timeouts y reintentos
6. **Validación de coordenadas** para garantizar datos válidos
7. **Códigos de salida apropiados** para automatización
8. **Documentación completa** y ejemplos de uso
