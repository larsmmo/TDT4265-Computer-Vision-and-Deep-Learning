import numpy as np
import matplotlib.pyplot as plt
from train import get_parser
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from vizer.draw import draw_boxes
args = get_parser().parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

data_loader = make_data_loader(cfg, is_train=False)
if isinstance(data_loader, list):
    data_loader = data_loader[0]
dataset = data_loader.dataset
indices = list(range(len(dataset)))
np.random.shuffle(indices)
for idx in indices:
    image = dataset._read_image(idx)
    boxes, labels = dataset.get_annotation(idx)
    image = draw_boxes(
        image, boxes, labels, class_name_map=dataset.class_names
    )
    plt.imshow(image)
    plt.show()
