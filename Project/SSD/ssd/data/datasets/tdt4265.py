from .waymo import WaymoDataset


class TDT4265Dataset(WaymoDataset):

    class_names = (
        "__background__",
        "vehicle",
        "person",
        "sign",
        "cyclist",
    )
    include_automatic_annotation = False
    validation_percentage = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def filter_on_labels(self):
        if self.include_automatic_annotation:
            return self.image_ids
        keep = []
        for idx, image_id in enumerate(self.image_ids):
            annotation_completed = self.labels[image_id]["annotation_completed"]
            if annotation_completed:
                keep.append(idx)
        return [self.image_ids[i] for i in keep]

    def filter_images(self):
        self.image_ids = self.filter_on_labels()
        return super().filter_images()
