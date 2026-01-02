import json
import os
import sys
import argparse
import logging
from pathlib import Path
import requests
import yaml
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_labelbox_to_yolo(json_file, output_dir, zip_output=True):
    """
    Convert Labelbox JSON annotations to YOLO format.
    
    Args:
        json_file: Path to the Labelbox JSON file
        output_dir: Directory where output files will be saved
        zip_output: Whether to create a zip file of the output
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    # Validate input parameters
    if not json_file:
        logger.error("JSON file path is required")
        return False
    
    if not output_dir:
        logger.error("Output directory path is required")
        return False
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found: {json_file}")
        return False
    
    # Leer el archivo JSON de Labelbox
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            labelbox_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in file {json_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return False
    
    # Validate JSON structure
    if not isinstance(labelbox_data, list):
        logger.error("JSON file should contain a list of entries")
        return False
    
    if len(labelbox_data) == 0:
        logger.warning("JSON file contains no entries")
        return False

    # Crear directorios de salida
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    
    try:
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        logger.info(f"Created output directories: {output_image_dir}, {output_label_dir}")
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        return False

    class_names = []  # Nombres de clases
    processed_count = 0
    error_count = 0

    for idx, entry in enumerate(tqdm(labelbox_data, desc=f'Converting {json_file}'), 1):
        try:
            # Validate entry structure
            if 'Labeled Data' not in entry:
                logger.warning(f"Entry {idx} missing 'Labeled Data' field, skipping")
                error_count += 1
                continue
            
            if 'External ID' not in entry:
                logger.warning(f"Entry {idx} missing 'External ID' field, skipping")
                error_count += 1
                continue
            
            if 'Label' not in entry or 'objects' not in entry['Label']:
                logger.warning(f"Entry {idx} missing 'Label' or 'objects' field, skipping")
                error_count += 1
                continue
            
            image_url = entry['Labeled Data']
            
            # Load image with error handling
            try:
                if image_url.startswith('http'):
                    response = requests.get(image_url, stream=True, timeout=30)
                    response.raise_for_status()
                    image = Image.open(response.raw)
                else:
                    if not os.path.exists(image_url):
                        logger.warning(f"Image file not found: {image_url}, skipping entry {idx}")
                        error_count += 1
                        continue
                    image = Image.open(image_url)
            except requests.RequestException as e:
                logger.warning(f"Error downloading image from {image_url}: {e}, skipping entry {idx}")
                error_count += 1
                continue
            except Exception as e:
                logger.warning(f"Error opening image {image_url}: {e}, skipping entry {idx}")
                error_count += 1
                continue
            
            width, height = image.size
            
            if width <= 0 or height <= 0:
                logger.warning(f"Invalid image dimensions ({width}x{height}) for entry {idx}, skipping")
                error_count += 1
                continue
            
            image_filename = os.path.join(output_image_dir, os.path.basename(entry['External ID']))
            
            try:
                image.save(image_filename, quality=95, subsampling=0)
            except Exception as e:
                logger.warning(f"Error saving image {image_filename}: {e}, skipping entry {idx}")
                error_count += 1
                continue

            label_filename = os.path.join(output_label_dir, os.path.splitext(os.path.basename(entry['External ID']))[0] + '.txt')

            # Process annotations
            annotations_written = 0
            for label in entry['Label']['objects']:
                try:
                    if 'bbox' not in label or 'value' not in label:
                        logger.warning(f"Label missing 'bbox' or 'value' in entry {idx}, skipping annotation")
                        continue
                    
                    bbox = label['bbox']
                    required_keys = ['top', 'left', 'height', 'width']
                    if not all(key in bbox for key in required_keys):
                        logger.warning(f"Invalid bbox format in entry {idx}, skipping annotation")
                        continue
                    
                    top, left, h, w = bbox['top'], bbox['left'], bbox['height'], bbox['width']
                    
                    # Validate bbox values
                    if w <= 0 or h <= 0:
                        logger.warning(f"Invalid bbox dimensions in entry {idx}, skipping annotation")
                        continue
                    
                    x_center = (left + w / 2) / width
                    y_center = (top + h / 2) / height
                    width_norm = w / width
                    height_norm = h / height
                    
                    # Ensure normalized coordinates are valid
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width_norm <= 1 and 0 < height_norm <= 1):
                        logger.warning(f"Normalized coordinates out of range in entry {idx}, skipping annotation")
                        continue

                    class_name = label['value']
                    if class_name not in class_names:
                        class_names.append(class_name)

                    label_line = f"{class_names.index(class_name)} {x_center} {y_center} {width_norm} {height_norm}\n"

                    with open(label_filename, 'a', encoding='utf-8') as label_file:
                        label_file.write(label_line)
                    annotations_written += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing annotation in entry {idx}: {e}")
                    continue
            
            if annotations_written > 0:
                processed_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing entry {idx}: {e}")
            error_count += 1
            continue

    logger.info(f"Processed {processed_count} entries successfully, {error_count} entries had errors")
    
    if processed_count == 0:
        logger.error("No entries were successfully processed")
        return False
    
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
    try:
        with open(yaml_filename, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(dataset_info, yaml_file, sort_keys=False)
        logger.info(f"Dataset configuration saved to {yaml_filename}")
    except Exception as e:
        logger.error(f"Error saving dataset.yaml: {e}")
        return False

    # Comprimir si se solicita
    if zip_output:
        try:
            zip_filename = f'{output_dir}.zip'
            with ZipFile(zip_filename, 'w') as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
            logger.info(f"Output compressed to {zip_filename}")
        except Exception as e:
            logger.error(f"Error creating zip file: {e}")
            return False

    logger.info('Conversión completada exitosamente!')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Labelbox JSON annotations to YOLO format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python labelbox2yolo.py -j export.json -o output_directory --zip
        """
    )
    parser.add_argument(
        '-j', '--json',
        required=True,
        help='Path to the Labelbox JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--zip',
        action='store_true',
        help='Create a zip file of the output directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    success = convert_labelbox_to_yolo(args.json, args.output, args.zip)
    sys.exit(0 if success else 1)
