import json
import os
import sys
import argparse
import logging
import cv2
import requests
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Función para crear directorios si no existen
def make_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {d}: {e}")
            raise

# Función para convertir INFOLKS JSON a formato YOLO
def convert_infolks_json(name, files, img_path):
    """
    Convert INFOLKS JSON annotations to YOLO format.
    
    Args:
        name: Base name for output files
        files: Glob pattern for JSON files
        img_path: Path to images directory (not used but kept for compatibility)
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    # Validate inputs
    if not name:
        logger.error("Output name is required")
        return False
    
    if not files:
        logger.error("Files pattern is required")
        return False
    
    try:
        make_dirs('labels')
    except Exception as e:
        logger.error(f"Failed to create output directories: {e}")
        return False

    # Load JSON files
    data = []
    json_files = glob.glob(files)
    
    if not json_files:
        logger.error(f"No files found matching pattern: {files}")
        return False
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                jdata = json.load(f)
                jdata['json_file'] = file
                data.append(jdata)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON format in file {file}: {e}, skipping")
            continue
        except Exception as e:
            logger.warning(f"Error reading file {file}: {e}, skipping")
            continue
    
    if not data:
        logger.error("No valid JSON files were loaded")
        return False
    
    logger.info(f"Successfully loaded {len(data)} JSON files")

    names = []  # Lista para nombres de clases únicos
    name_file = name + '.txt'
    labels_dir = 'labels'
    
    processed_count = 0
    error_count = 0

    with open(name_file, 'w', encoding='utf-8') as nf:
        for idx, img_data in enumerate(tqdm(data, desc='Procesando Imágenes'), 1):
            try:
                # Validate entry structure
                if 'Labeled Data' not in img_data:
                    logger.warning(f"Entry {idx} missing 'Labeled Data' field, skipping")
                    error_count += 1
                    continue
                
                if 'External ID' not in img_data:
                    logger.warning(f"Entry {idx} missing 'External ID' field, skipping")
                    error_count += 1
                    continue
                
                if 'Label' not in img_data or 'objects' not in img_data['Label']:
                    logger.warning(f"Entry {idx} missing 'Label' or 'objects' field, skipping")
                    error_count += 1
                    continue
                
                image_path = img_data['Labeled Data']
                
                # Load image with error handling
                try:
                    if image_path.startswith('http'):
                        response = requests.get(image_path, stream=True, timeout=30)
                        response.raise_for_status()
                        image = Image.open(response.raw)
                    else:
                        if not os.path.exists(image_path):
                            logger.warning(f"Image file not found: {image_path}, skipping entry {idx}")
                            error_count += 1
                            continue
                        image = Image.open(image_path)
                except requests.RequestException as e:
                    logger.warning(f"Error downloading image from {image_path}: {e}, skipping entry {idx}")
                    error_count += 1
                    continue
                except Exception as e:
                    logger.warning(f"Error opening image {image_path}: {e}, skipping entry {idx}")
                    error_count += 1
                    continue
                
                width, height = image.size
                
                if width <= 0 or height <= 0:
                    logger.warning(f"Invalid image dimensions ({width}x{height}) for entry {idx}, skipping")
                    error_count += 1
                    continue

                label_file = Path(labels_dir) / (Path(img_data['External ID']).with_suffix('.txt').name)
                
                annotations_written = 0
                for label in img_data['Label']['objects']:
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
                        normalized_w = w / width
                        normalized_h = h / height
                        
                        # Ensure normalized coordinates are valid
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < normalized_w <= 1 and 0 < normalized_h <= 1):
                            logger.warning(f"Normalized coordinates out of range in entry {idx}, skipping annotation")
                            continue

                        cls = label['value'].lower()
                        if cls not in names:
                            names.append(cls)

                        line = f"{names.index(cls)} {x_center:.6f} {y_center:.6f} {normalized_w:.6f} {normalized_h:.6f}\n"
                        nf.write(line)

                        # Guardar anotación en el archivo correspondiente
                        with open(label_file, 'a', encoding='utf-8') as lf:
                            lf.write(line)
                        
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

    # Save class names
    try:
        with open(name + '.names', 'w', encoding='utf-8') as nnf:
            nnf.write('\n'.join(names))
        logger.info(f"Saved {len(names)} class names to {name}.names")
    except Exception as e:
        logger.error(f"Error saving class names: {e}")
        return False

    logger.info(f'Conversión completada con éxito. Archivos de salida guardados en: {os.getcwd()}/{name}.txt y {os.getcwd()}/{labels_dir}/')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert INFOLKS JSON annotations to YOLO format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python infolks_to_yolo.py -n dataset_name -f "data/*.json" -i images/
        """
    )
    parser.add_argument(
        '-n', '--name',
        required=True,
        help='Base name for output files'
    )
    parser.add_argument(
        '-f', '--files',
        required=True,
        help='Glob pattern for JSON files (e.g., "data/*.json")'
    )
    parser.add_argument(
        '-i', '--images',
        default='',
        help='Path to images directory (optional, kept for compatibility)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    success = convert_infolks_json(args.name, args.files, args.images)
    sys.exit(0 if success else 1)
