## INFOLKS JSON Annotations to YOLO Format Converter

This Python script allows you to convert INFOLKS JSON format annotations to YOLO format, making it easier to use in object detection tasks.

## Usage

1. Make sure you have Python installed on your system.
2. Clone this repository to your local machine or download the infolks_to_yolo.py file.
3. Run the script with the following arguments:

```bash
python infolks_to_yolo.py <dataset_name> <path_to_json_files/*.json> <path_to_images/>
```

- `<dataset_name>`: The name you want to give to your dataset in YOLO format.
- `<path_to_json_files/*.json>`: The path to the INFOLKS JSON files you want to convert.
- `<path_to_images/>`: The path to the folder containing the corresponding images.

The script will generate YOLO-format annotation files and a .names file containing the detected class names.
Requirements

Make sure you have the following Python libraries installed:

```
json
os
cv2
PIL
tqdm
pathlib
numpy
glob
```

You can install these libraries using pip:

```bash
pip install json os cv2 PILLOW tqdm pathlib numpy glob2
```

## Contributing

If you'd like to contribute to this project, feel free to fork it and submit a pull request with your improvements.

## Credits

This script was developed by Juan Casal, based on https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py 

## License

This project is licensed under the MIT License.
