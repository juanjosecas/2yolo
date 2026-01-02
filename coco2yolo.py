import json
import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class COCO2YOLO:
    def __init__(self, json_file, output):
        self._check_file_and_dir(json_file, output)
        self.labels = self.load_json(json_file)
        
        # Validate JSON structure
        if not self._validate_coco_structure():
            raise ValueError("Invalid COCO JSON structure")
        
        self.coco_id_name_map = self.get_categories_mapping()
        self.coco_name_list = list(self.coco_id_name_map.values())
        logger.info(f"Total images: {len(self.labels.get('images', []))}")
        logger.info(f"Total categories: {len(self.labels.get('categories', []))}")
        logger.info(f"Total labels: {len(self.labels.get('annotations', []))}")
    
    @staticmethod
    def load_json(json_file):
        """Load and parse JSON file with error handling."""
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file {json_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading JSON file {json_file}: {e}")
            raise
    
    @staticmethod
    def _check_file_and_dir(file_path, dir_path):
        """Validate input file and create output directory."""
        if not file_path:
            raise ValueError("JSON file path is required")
        
        if not os.path.exists(file_path):
            raise ValueError(f"JSON file not found: {file_path}")
        
        if not dir_path:
            raise ValueError("Output directory path is required")
        
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created output directory: {dir_path}")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            raise
    
    def _validate_coco_structure(self):
        """Validate that the JSON has the required COCO structure."""
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in self.labels:
                logger.error(f"Missing required key in COCO JSON: {key}")
                return False
            if not isinstance(self.labels[key], list):
                logger.error(f"Key '{key}' should be a list in COCO JSON")
                return False
        
        if len(self.labels['categories']) == 0:
            logger.error("No categories found in COCO JSON")
            return False
        
        return True
    
    def get_categories_mapping(self):
        """Create a mapping from category IDs to names."""
        categories = {}
        try:
            for cls in self.labels['categories']:
                if 'id' not in cls or 'name' not in cls:
                    logger.warning(f"Category missing 'id' or 'name' field, skipping: {cls}")
                    continue
                categories[cls['id']] = cls['name']
        except Exception as e:
            logger.error(f"Error creating category mapping: {e}")
            raise
        
        if not categories:
            raise ValueError("No valid categories found")
        
        return categories
    
    def load_images_info(self):
        """Load image information with error handling."""
        images_info = {}
        skipped = 0
        
        for image in self.labels['images']:
            try:
                required_fields = ['id', 'file_name', 'width', 'height']
                if not all(field in image for field in required_fields):
                    logger.warning(f"Image missing required fields, skipping: {image}")
                    skipped += 1
                    continue
                
                img_id = image['id']
                file_name = os.path.basename(image['file_name'])
                w = image['width']
                h = image['height']
                
                if w <= 0 or h <= 0:
                    logger.warning(f"Invalid image dimensions ({w}x{h}) for image {img_id}, skipping")
                    skipped += 1
                    continue
                
                images_info[img_id] = (file_name, w, h)
            except Exception as e:
                logger.warning(f"Error processing image info: {e}, skipping")
                skipped += 1
                continue
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} images due to errors")
        
        if not images_info:
            raise ValueError("No valid images found in COCO JSON")
        
        return images_info
    
    def bbox_2_yolo(self, bbox, img_w, img_h):
        """Convert COCO bbox format to YOLO format with validation."""
        try:
            if len(bbox) < 4:
                raise ValueError(f"Invalid bbox format: {bbox}")
            
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Validate bbox values
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid bbox dimensions: w={w}, h={h}")
            
            if x < 0 or y < 0:
                logger.warning(f"Negative bbox coordinates: x={x}, y={y}, clamping to 0")
                x = max(0, x)
                y = max(0, y)
            
            centerx = x + w / 2
            centery = y + h / 2
            dw = 1 / img_w
            dh = 1 / img_h
            centerx *= dw
            w *= dw
            centery *= dh
            h *= dh
            
            # Clamp values to valid range [0, 1]
            centerx = max(0, min(1, centerx))
            centery = max(0, min(1, centery))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            return centerx, centery, w, h
        except Exception as e:
            logger.error(f"Error converting bbox {bbox}: {e}")
            raise
    
    def convert_annotations(self, images_info):
        """Convert annotations with error handling."""
        anno_dict = {}
        skipped = 0
        
        for anno in self.labels['annotations']:
            try:
                required_fields = ['bbox', 'image_id', 'category_id']
                if not all(field in anno for field in required_fields):
                    logger.warning(f"Annotation missing required fields, skipping")
                    skipped += 1
                    continue
                
                bbox = anno['bbox']
                image_id = anno['image_id']
                category_id = anno['category_id']

                # Check if image exists
                image_info = images_info.get(image_id)
                if not image_info:
                    logger.warning(f"Image {image_id} not found for annotation, skipping")
                    skipped += 1
                    continue
                
                # Check if category exists
                if category_id not in self.coco_id_name_map:
                    logger.warning(f"Category {category_id} not found in categories, skipping annotation")
                    skipped += 1
                    continue
                
                image_name = image_info[0]
                img_w = image_info[1]
                img_h = image_info[2]
                
                try:
                    yolo_box = self.bbox_2_yolo(bbox, img_w, img_h)
                except Exception as e:
                    logger.warning(f"Error converting bbox for image {image_id}: {e}, skipping")
                    skipped += 1
                    continue

                anno_info = (image_name, category_id, yolo_box)
                anno_infos = anno_dict.get(image_id)
                if not anno_infos:
                    anno_dict[image_id] = [anno_info]
                else:
                    anno_infos.append(anno_info)
                    anno_dict[image_id] = anno_infos
            except Exception as e:
                logger.warning(f"Error processing annotation: {e}, skipping")
                skipped += 1
                continue
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} annotations due to errors")
        
        if not anno_dict:
            raise ValueError("No valid annotations were converted")
        
        return anno_dict
    
    def save_classes(self, file_name='coco.names'):
        """Save class names to file."""
        try:
            sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
            logger.info(f'COCO names: {sorted_classes}')
            with open(file_name, 'w', encoding='utf-8') as f:
                for cls in sorted_classes:
                    f.write(cls + '\n')
            logger.info(f'Saved {file_name}')
        except Exception as e:
            logger.error(f"Error saving class names to {file_name}: {e}")
            raise
    
    def coco2yolo(self, output_folder):
        """Main conversion method with error handling."""
        try:
            logger.info("Loading image info...")
            images_info = self.load_images_info()
            logger.info(f"Loading done, total images: {len(images_info)}")

            logger.info("Start converting...")
            anno_dict = self.convert_annotations(images_info)
            logger.info(f"Converting done, total labels: {len(anno_dict)}")

            logger.info("Saving YOLO txt files...")
            self.save_yolo_txt(anno_dict, output_folder)
            logger.info("Saving done")
            return True
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False

    def save_yolo_txt(self, anno_dict, output_folder):
        """Save YOLO format annotations to text files."""
        saved = 0
        errors = 0
        
        for k, v in anno_dict.items():
            try:
                file_name = os.path.splitext(v[0][0])[0] + ".txt"
                file_path = os.path.join(output_folder, file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for obj in v:
                        try:
                            cat_name = self.coco_id_name_map.get(obj[1])
                            if cat_name is None:
                                logger.warning(f"Category {obj[1]} not found, skipping")
                                continue
                            
                            category_id = self.coco_name_list.index(cat_name)
                            box = ['{:.6f}'.format(x) for x in obj[2]]
                            box = ' '.join(box)
                            line = str(category_id) + ' ' + box
                            f.write(line + '\n')
                        except Exception as e:
                            logger.warning(f"Error writing annotation: {e}")
                            errors += 1
                            continue
                saved += 1
            except Exception as e:
                logger.warning(f"Error saving file {file_name}: {e}")
                errors += 1
                continue
        
        logger.info(f"Saved {saved} label files")
        if errors > 0:
            logger.warning(f"Encountered {errors} errors while saving")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert COCO annotations to YOLO format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python coco2yolo.py -j instances_train2017.json -o output_labels/
        """
    )
    parser.add_argument('-j', '--json', help='Path to COCO JSON file', dest='json', required=True)
    parser.add_argument('-o', '--output', help='Output folder for YOLO labels', dest='out', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        c2y = COCO2YOLO(args.json, args.out)
        success = c2y.coco2yolo(args.out)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
