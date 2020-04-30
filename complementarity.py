import argparse
import json
import os

from box import Box

"""
Compute channels complementarity (see section 3.3 in paper).
TP stands for True Positives,
FP stands for False Positives
"""


def get_tp_fp(models_dir, confidence_threshold, iou_threshold):
    all_rgb_tp_fp = []
    all_additional_tp_fp = []
    for experiment_dir in os.scandir(models_dir):
        if not experiment_dir.is_dir() or \
                not (experiment_dir.name.startswith("additional") or experiment_dir.name.startswith("rgb")):
            continue
        print(experiment_dir.name)
        ground_truths, predictions = load_predictions_ground_truth(experiment_dir)
        predictions = filter_positives(predictions, confidence_threshold)
        false_positives, true_positives = get_tp_fp_for_one_experiment(ground_truths, predictions, iou_threshold)
        print("True positives : " + str(len(true_positives)))
        print("False positives : " + str(len(false_positives)))
        if experiment_dir.name.startswith("additional"):
            all_additional_tp_fp.append({"true_positives": true_positives, "false_positives": false_positives})
        elif experiment_dir.name.startswith("rgb"):
            all_rgb_tp_fp.append({"true_positives": true_positives, "false_positives": false_positives})
    return all_rgb_tp_fp, all_additional_tp_fp


def filter_positives(predictions, confidence_threshold):
    return [prediction for prediction in predictions if prediction["score"] > confidence_threshold]


def get_tp_fp_for_one_experiment(ground_truths, predictions, iou_threshold):
    false_positives = []
    true_positives = []
    for prediction in predictions:
        if is_true_positive(prediction, ground_truths, iou_threshold):
            true_positives.append(prediction)
        else:
            false_positives.append(prediction)
    return false_positives, true_positives


def is_true_positive(prediction, ground_truths, iou_threshold):
    is_true_positive = False
    for ground_truth in ground_truths:
        if ground_truth["image_id"] == prediction["image_id"] \
                and prediction["category_id"] == ground_truth["category_id"]:
            if prediction["box"].intersection_over_union(ground_truth["box"]) >= iou_threshold:
                is_true_positive = True
                break
    return is_true_positive


def load_predictions_ground_truth(experiment_dir):
    ground_truths = load_json(experiment_dir, "ground_truth.json")
    add_box_objects(ground_truths)
    predictions = load_json(experiment_dir, "predictions.json")
    add_box_objects(predictions)
    return ground_truths, predictions

def add_box_objects(json_loaded_boxes):
    for box_dict in json_loaded_boxes:
        x_min, y_min, width, height = box_dict["bbox"]
        box_dict["box"] = Box(x_min, y_min, width, height, is_absolute=True)

def load_json(experiment_dir, file_name):
    file_path = os.path.join(experiment_dir.path, file_name)
    return json.load(open(file_path, "rb"))


def is_included(tested_box, other_boxes, iou_threshold):
    return any(other_box["box"].intersection_over_union(tested_box) >= iou_threshold for other_box in other_boxes)


def compute_complementarity(results, all_rgb_tp_fp, all_additional_tp_fp, iou_threshold):
    for additional_tp_fp in all_additional_tp_fp:
        for rgb_tp_fp in all_rgb_tp_fp:
            additional_tp_not_included_in_rgb_tp_count = count_additional_tp_not_included_in_rgb_tp(additional_tp_fp,
                                                                                                    rgb_tp_fp,
                                                                                                    iou_threshold)
            rgb_fp_not_included_in_additional_fp_count = count_rgb_fp_not_included_in_additional_fp(additional_tp_fp,
                                                                                                    rgb_tp_fp,
                                                                                                    iou_threshold)

            results.append({
                "additional_tp_not_included_in_rgb_tp_count": additional_tp_not_included_in_rgb_tp_count,
                "rgb_tp_count": len(rgb_tp_fp["true_positives"]),
                "rgb_fp_not_included_in_additional_fp_count": rgb_fp_not_included_in_additional_fp_count,
                "rgb_fp_count": len(rgb_tp_fp["false_positives"])})
    return results


def count_additional_tp_not_included_in_rgb_tp(additional_tp_fp, rgb_tp_fp, iou_threshold):
    additional_tp_not_included_in_rgb_tp_count = 0
    for additional_tp in additional_tp_fp["true_positives"]:
        rgb_true_positives = [rgb_tp for rgb_tp in rgb_tp_fp["true_positives"]
                              if rgb_tp["image_id"] == additional_tp["image_id"]]
        if not is_included(additional_tp["box"], rgb_true_positives, iou_threshold):
            additional_tp_not_included_in_rgb_tp_count += 1
    return additional_tp_not_included_in_rgb_tp_count


def count_rgb_fp_not_included_in_additional_fp(additional_tp_fp, rgb_tp_fp, iou_threshold):
    rgb_fp_not_included_in_additional_fp_count = 0
    for rgb_fp in rgb_tp_fp["false_positives"]:
        additional_false_positives = [additional_fp for additional_fp in additional_tp_fp["false_positives"]
                                      if additional_fp["image_id"] == rgb_fp["image_id"]]
        if not is_included(rgb_fp["box"], additional_false_positives, iou_threshold):
            rgb_fp_not_included_in_additional_fp_count += 1
    return rgb_fp_not_included_in_additional_fp_count


def complementarity():
    results = []
    iou_threshold = 0.5
    print(f"iou : {iou_threshold}")
    for confidence_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"confidence : {confidence_threshold}")
        all_rgb_tp_fp, all_additional_tp_fp = get_tp_fp(models_dir, confidence_threshold, iou_threshold)
        results = compute_complementarity(results, all_rgb_tp_fp, all_additional_tp_fp, iou_threshold)

    # print(results)
    rgb_tp_count = sum(result["rgb_tp_count"] for result in results)
    additional_tp_not_included_in_rgb_tp_count = sum(
        result["additional_tp_not_included_in_rgb_tp_count"] for result in results)
    delta_tp = additional_tp_not_included_in_rgb_tp_count / rgb_tp_count
    print("Delta True Positives")
    print(delta_tp)

    rgb_fp_count = sum(result["rgb_fp_count"] for result in results)
    rgb_fp_not_included_in_additional_fp_count = sum(
        result["rgb_fp_not_included_in_additional_fp_count"] for result in results)
    delta_fp = rgb_fp_not_included_in_additional_fp_count / rgb_fp_count
    print("Delta False Positives")
    print(delta_fp)

    delta_additional = 2 / ((1 / delta_fp) + (1 / delta_tp))  # F1 score
    print("Delta additional")
    print(delta_additional)

    print("Number of couples")
    print(len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir",
                        help="Every subdirectory of this directory with name starting with 'additional' or 'rgb' will be taken into account.\n"
                             "Each should hold predictions.json and ground_truth.json")

    args = parser.parse_args()

    models_dir = args.models_dir

    complementarity()
