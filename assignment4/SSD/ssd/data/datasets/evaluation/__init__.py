import pathlib
from ssd.data.datasets import VOCDataset, COCODataset, MNISTDetection
from .coco import coco_evaluation
from .voc import voc_evaluation
from .mnist import mnist_evaluation


def evaluate(dataset, predictions, output_dir: pathlib.Path, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs,
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, MNISTDetection):
        return mnist_evaluation(**args)
    else:
        raise NotImplementedError
