import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
	"""Calculate intersection over union of single predicted and ground truth box.

	Args:
	    prediction_box (np.array of floats): location of predicted object as
	        [xmin, ymin, xmax, ymax]
	    gt_box (np.array of floats): location of ground truth object as
	        [xmin, ymin, xmax, ymax]

	    returns:
	        float: value of the intersection of union for the two boxes.
	"""
	# YOUR CODE HERE
	# Compute intersection
	leftX = max(prediction_box[0], gt_box[0])
	rightX = min(prediction_box[2], gt_box[2])
	topY = max(prediction_box[1], gt_box[1])
	bottomY = min(prediction_box[3], gt_box[3])

	intersectionArea = max(0, rightX - leftX) * max(0, bottomY - topY)
	# Compute union
	prediction_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
	gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

	iou = intersectionArea / float(prediction_area + gt_area - intersectionArea)

	assert iou >= 0 and iou <= 1
	return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    return (num_tp/(num_tp + num_fp)) if (num_tp + num_fp) else 1
    


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    return (num_tp/(num_tp + num_fn)) if (num_tp + num_fp) else 0


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    box_matches = []

    best_box_matches = {}

    for prediction_idx in range(prediction_boxes.shape[0]):
    	for gt_idx in range(gt_boxes.shape[0]):
    		iou = calculate_iou(prediction_boxes[prediction_idx], gt_boxes[gt_idx])

    		if iou >= iou_threshold:
    			box_matches.append((prediction_idx, gt_idx, iou))

    if not box_matches:
    	return np.array([]), np.array([])

    # Sort all matches on IoU in descending order
    box_matches.sort(key=lambda x: x[2], reverse = True)

    # Find all matches with the highest IoU threshold using dict {prediction idx: (pred_idx, gt_idx, iou)}
    # -> Simply check what the best prediction box is for each ground truth box:
    for idx, tup in enumerate(box_matches):
    	if best_box_matches.get(tup[1]) and tup[2] > best_box_matches.get(tup[1])[2]:
    		best_box_matches[tup[1]] = tup

    	elif not best_box_matches.get(tup[1]):
    		best_box_matches[tup[1]] = tup

    # Return the best match boxes
    result_prediction_matches = np.array([prediction_boxes[i] for i in list(zip(*best_box_matches.values()))[0]])
    result_gt_boxes = np.array([gt_boxes[i] for i in best_box_matches.keys()])

    return result_prediction_matches, result_gt_boxes


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    prediction_matches, gt_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    TP = len(prediction_matches)
    FP = len(prediction_boxes) - TP
    FN = len(gt_boxes) - TP

    return {"true_pos": TP, "false_pos":FP, "false_neg": FN}



def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    
    tot_TP = 0
    tot_FP = 0
    tot_FN = 0

    for img in range (len(all_prediction_boxes)):
    	img_res = calculate_individual_image_result(all_prediction_boxes[img],
    	                                            all_gt_boxes[img], 
    	                                            iou_threshold)
    	tot_TP += img_res["true_pos"]
    	tot_FP += img_res["false_pos"]
    	tot_FN += img_res["false_neg"]

    precision = calculate_precision(tot_TP, tot_FP, tot_FN)
    recall = calculate_recall(tot_TP, tot_FP, tot_FN)

    return precision, recall

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
	"""Given a set of prediction boxes and ground truth boxes for all images,
	   calculates the recall-precision curve over all images.
	   for a single image.

	   NB: all_prediction_boxes and all_gt_boxes are not matched!

	Args:
	    all_prediction_boxes: (list of np.array of floats): each element in the list
	        is a np.array containing all predicted bounding boxes for the given image
	        with shape: [number of predicted boxes, 4].
	        Each row includes [xmin, xmax, ymin, ymax]
	    all_gt_boxes: (list of np.array of floats): each element in the list
	        is a np.array containing all ground truth bounding boxes for the given image
	        objects with shape: [number of ground truth boxes, 4].
	        Each row includes [xmin, xmax, ymin, ymax]
	    scores: (list of np.array of floats): each element in the list
	        is a np.array containting the confidence score for each of the
	        predicted bounding box. Shape: [number of predicted boxes]

	        E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
	Returns:
	    precisions, recalls: two np.ndarray with same shape.
	"""
	# Instead of going over every possible confidence score threshold to compute the PR
	# curve, we will use an approximation
	confidence_thresholds = np.linspace(0, 1, 500)
	# YOUR CODE HERE
	precisions = [] 
	recalls = []

	for thresh in range(len(confidence_thresholds)):
		threshold_predictions = []

		for img in range(len(confidence_scores)):
			img_predictions = []

			for score in range(confidence_scores[img].shape[0]):
				if confidence_scores[img][score] >= confidence_thresholds[thresh]:
					img_predictions.append(all_prediction_boxes[img][score])

			threshold_predictions.append(np.array(img_predictions))

		precision, recall = calculate_precision_recall_all_images(threshold_predictions,
																  all_gt_boxes, 
																  iou_threshold)
		precisions.append(precision)
		recalls.append(recall)

	return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    """			# For Task 1 plotting
    interpolated = np.zeros(11)
    j = 0
    indices = np.arange(0.0, 1.1, 0.1)
    print(indices)
    for i in indices:
    	while int(i*10) > int(recalls[j] * 10):
    		j = j + 1
    	interpolated[int(i*10)] = max(precisions[j:])

    plt.figure(figsize=(15, 10))
    plt.plot(recalls, precisions)
    plt.plot(indices, interpolated, "x")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.1])
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.savefig("precision_recall_curve.png")
    """

    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precisions_max_sum = 0

    for lvl in range(len(recall_levels)):
    	precision_max = 0

    	for n in range(recalls.shape[0]):
    		if (precisions[n] > precision_max) and (recalls[n] >= recall_levels[lvl]):
    			precision_max = precisions[n]

    	precisions_max_sum += precision_max

    average_precision = precisions_max_sum / float(len(recall_levels))

    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
	#plot_precision_recall_curve(precision, recall)