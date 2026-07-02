
import os
from xml.etree import ElementTree

import cv2
import matplotlib.pyplot as plt

import config


def extract_data_from_xml(xml_file: str) -> dict:
    """Parse one XML annotation file into a dict with filename, image_size, bboxes."""
    root = ElementTree.parse(xml_file).getroot()
    data_dict = {"bboxes": []}

    for element in root:
        if element.tag == "filename":
            data_dict["filename"] = element.text
        elif element.tag == "size":
            data_dict["image_size"] = [int(e.text) for e in element]
        elif element.tag == "object":
            bbox = {}
            for obj_element in element:
                if obj_element.tag == "name":
                    bbox["class"] = obj_element.text
                elif obj_element.tag == "bndbox":
                    for bbox_element in obj_element:
                        bbox[bbox_element.tag] = int(bbox_element.text)
            data_dict["bboxes"].append(bbox)

    return data_dict


def convert_dict_to_yolo(data_dict: dict, save_dir: str = None):
    """Write one parsed annotation dict out as a YOLO-format .txt label file."""
    save_dir = save_dir or config.YOLO_ANNOTATIONS_DIR
    os.makedirs(save_dir, exist_ok=True)

    lines = []
    img_w, img_h, _ = data_dict["image_size"]

    for bbox in data_dict["bboxes"]:
        if bbox["class"] not in config.CLASS_TO_ID:
            print(f'[WARN] Unknown class "{bbox["class"]}" - skipping box.')
            continue
        class_id = config.CLASS_TO_ID[bbox["class"]]

        x_center = ((bbox["xmin"] + bbox["xmax"]) / 2) / img_w
        y_center = ((bbox["ymin"] + bbox["ymax"]) / 2) / img_h
        width = (bbox["xmax"] - bbox["xmin"]) / img_w
        height = (bbox["ymax"] - bbox["ymin"]) / img_h

        lines.append(f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}")

    save_path = os.path.join(save_dir, data_dict["filename"].replace("jpg", "txt"))
    with open(save_path, "w+") as f:
        f.write("\n".join(lines))


def convert_all_annotations():
    """Convert every XML annotation under config.ANNOTATIONS_DIR to YOLO format."""
    annot_files = sorted(
        os.path.join(config.ANNOTATIONS_DIR, f)
        for f in os.listdir(config.ANNOTATIONS_DIR) if f.endswith(".xml")
    )

    if os.path.exists(config.YOLO_ANNOTATIONS_DIR) and os.listdir(config.YOLO_ANNOTATIONS_DIR):
        print("[INFO] YOLO annotations already exist, skipping conversion.")
        return annot_files

    print("[INFO] Converting XML annotations to YOLO format...")
    for annot_file in annot_files:
        data_dict = extract_data_from_xml(annot_file)
        convert_dict_to_yolo(data_dict)
    print(f"[INFO] Converted {len(annot_files)} annotation files.")
    return annot_files


def plot_bboxes(img_file: str, annot_file: str, id_to_class: dict = None):
    """Draw YOLO-format boxes from annot_file onto img_file and show it."""
    id_to_class = id_to_class or config.ID_TO_CLASS

    image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape

    with open(annot_file, "r") as f:
        rows = [line.split(" ") for line in f.read().split("\n") if line.strip()]
        rows = [[float(v) for v in row] for row in rows]

    for class_idx, x_center, y_center, width, height in rows:
        xmin = max(0, int((x_center - width / 2) * img_w))
        ymin = max(0, int((y_center - height / 2) * img_h))
        xmax = min(img_w - 1, int((x_center + width / 2) * img_w))
        ymax = min(img_h - 1, int((y_center + height / 2) * img_h))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        label_y = 0 if ymin - 10 < 0 else ymin - 10
        cv2.putText(image, id_to_class[int(class_idx)], (xmin, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    plt.imshow(image)
    plt.axis(False)
