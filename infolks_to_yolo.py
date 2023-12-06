import json
import os
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import glob

# Función para crear directorios si no existen
def make_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

# Función para convertir INFOLKS JSON a formato YOLO
def convert_infolks_json(name, files, img_path):
    make_dirs('labels')

    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    names = []  # Lista para nombres de clases únicos
    name_file = name + '.txt'
    labels_dir = 'labels'

    with open(name_file, 'w') as nf:
        for img_data in tqdm(data, desc='Procesando Imágenes'):
            image_path = img_data['Labeled Data']
            image = Image.open(requests.get(image_path, stream=True).raw if image_path.startswith('http') else image_path)
            width, height = image.size

            label_file = Path(labels_dir) / (Path(img_data['External ID']).with_suffix('.txt').name)

            for label in img_data['Label']['objects']:
                top, left, h, w = label['bbox'].values()
                x_center = (left + w / 2) / width
                y_center = (top + h / 2) / height
                normalized_w = w / width
                normalized_h = h / height

                cls = label['value'].lower()
                if cls not in names:
                    names.append(cls)

                line = f"{names.index(cls)} {x_center:.6f} {y_center:.6f} {normalized_w:.6f} {normalized_h:.6f}\n"
                nf.write(line)

                # Guardar anotación en el archivo correspondiente
                with open(label_file, 'a') as lf:
                    lf.write(line)

    with open(name + '.names', 'w') as nnf:
        nnf.write('\n'.join(names))

    print(f'Conversión completada con éxito. Archivos de salida guardados en: {os.getcwd()}/{name}.txt y {os.getcwd()}/{labels_dir}/')

if __name__ == '__main__':
    convert_infolks_json('nombre_dataset', 'ruta_a_archivos_json/*.json', 'ruta_a_imagenes/')
