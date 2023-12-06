import json
import os
from pathlib import Path
import requests
import yaml
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile

def convert_labelbox_to_yolo(json_file, output_dir, zip_output=True):
    # Leer el archivo JSON de Labelbox
    with open(json_file) as f:
        labelbox_data = json.load(f)

    # Crear directorios de salida
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    class_names = []  # Nombres de clases

    for entry in tqdm(labelbox_data, desc=f'Converting {json_file}'):
        image_url = entry['Labeled Data']
        image = Image.open(requests.get(image_url, stream=True).raw if image_url.startswith('http') else image_url)
        width, height = image.size
        image_filename = os.path.join(output_image_dir, os.path.basename(entry['External ID']))
        image.save(image_filename, quality=95, subsampling=0)

        label_filename = os.path.join(output_label_dir, os.path.splitext(os.path.basename(entry['External ID']))[0] + '.txt')

        for label in entry['Label']['objects']:
            top, left, h, w = label['bbox'].values()
            x_center = (left + w / 2) / width
            y_center = (top + h / 2) / height
            width_norm = w / width
            height_norm = h / height

            class_name = label['value']
            if class_name not in class_names:
                class_names.append(class_name)

            label_line = f"{class_names.index(class_name)} {x_center} {y_center} {width_norm} {height_norm}\n"

            with open(label_filename, 'a') as label_file:
                label_file.write(label_line)

    # Guardar dataset.yaml
    dataset_info = {
        'path': output_dir,
        'train': "images/train",
        'val': "images/val",
        'test': "",
        'nc': len(class_names),
        'names': class_names
    }

    yaml_filename = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(dataset_info, yaml_file, sort_keys=False)

    # Comprimir si se solicita
    if zip_output:
        with ZipFile(f'{output_dir}.zip', 'w') as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    print('Conversión completada exitosamente!')

if __name__ == '__main__':
    convert_labelbox_to_yolo('export-2021-06-29T15_25_41.934Z.json', 'output_directory', zip_output=True)
