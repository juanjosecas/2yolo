import json
import os
import argparse

class COCO2YOLO:
    def __init__(self, json_file, output):
        self._check_file_and_dir(json_file, output)
        self.labels = self.load_json(json_file)
        self.coco_id_name_map = self.get_categories_mapping()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("Total images:", len(self.labels['images']))
        print("Total categories:", len(self.labels['categories']))
        print("Total labels:", len(self.labels['annotations']))
    
    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    @staticmethod
    def _check_file_and_dir(file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("JSON file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def get_categories_mapping(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories
    
    def load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = os.path.basename(image['file_name'])
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)
        return images_info
    
    def bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = x + w / 2
        centery = y + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h
    
    def convert_annotations(self, images_info):
        anno_dict = {}
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self.bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict
    
    def save_classes(self, file_name='coco.names'):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('COCO names:', sorted_classes)
        with open(file_name, 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        print(f'Saved {file_name}')
    
    def coco2yolo(self, output_folder):
        print("Loading image info...")
        images_info = self.load_images_info()
        print("Loading done, total images", len(images_info))

        print("Start converting...")
        anno_dict = self.convert_annotations(images_info)
        print("Converting done, total labels", len(anno_dict))

        print("Saving YOLO txt files...")
        self.save_yolo_txt(anno_dict, output_folder)
        print("Saving done")

    @staticmethod
    def save_yolo_txt(anno_dict, output_folder):
        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            with open(os.path.join(output_folder, file_name), 'w', encoding='utf-8') as f:
                for obj in v:
                    cat_name = COCO2YOLO.coco_id_name_map.get(obj[1])
                    category_id = COCO2YOLO.coco_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format.')
    parser.add_argument('-j', help='JSON file', dest='json', required=True)
    parser.add_argument('-o', help='Output folder', dest='out', required=True)

    args = parser.parse_args()

    c2y = COCO2YOLO(args.json, args.out)
    c2y.coco2yolo(args.out)
