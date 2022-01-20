import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.data import MetadataCatalog

class Segmentor():
    def __init__(self):
        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg) # Add PointRend-specific config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get("coco_2017_val")

    def inference(self, im):
        outputs = self.predictor(im)

        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

        return point_rend_result[:, :, ::-1]

    def person_binary_inference(self, im):
        outputs = self.predictor(im)

        fields = outputs['instances'].get_fields()

        people = fields['pred_classes'] == 0
        confident_people = fields['scores'][people] > 0.5
        people_masks = fields['pred_masks'][people][confident_people]

        numpy_masks = people_masks.cpu().numpy()

        binary_mask = np.bitwise_or.reduce(numpy_masks, axis=0)

        return binary_mask


if __name__ == "__main__":
    im = cv2.imread("input.png")
    seg = Segmentor()
    point_rend_result = seg.inference(im)

    while True:
        cv2.imshow('res', point_rend_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break