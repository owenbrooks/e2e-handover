import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.data import MetadataCatalog

def inference(im):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg) # Add PointRend-specific config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    coco_metadata = MetadataCatalog.get("coco_2017_val")

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

    while True:
        cv2.imshow('res', point_rend_result[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    im = cv2.imread("2760_2021-12-14-23:19:09.png")
    inference(im)