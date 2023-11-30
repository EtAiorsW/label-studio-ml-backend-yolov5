import logging
import cv2
from typing import List, Dict
from module import loadmodel, loadimages, inference, nms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, DATA_UNDEFINED_NAME

from utils import yaml_load

logger = logging.getLogger(__name__)
model_name = '/home/wanggq/label-studio-ml-backend/label_studio_ml_backend_yolov5/yolov5l_1280_11_14_4060.engine'
class_name = '/home/wanggq/label-studio-ml-backend/label_studio_ml_backend_yolov5/NG.yaml'
model = loadmodel(model_name, class_name)


class Yolov5Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(Yolov5Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.model = model
        self.category_map = yaml_load(class_name).get('names')
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')

        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        self.labels_attrs = schema.get('labels_attrs')

        self.score_thresh = 0.4

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        return image_url

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """

        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)

        results = []
        all_scores = []
        image = cv2.imread(image_path)
        original_width, original_height = image.shape[1], image.shape[0]
        im, im0 = loadimages(image)
        pred = inference(self.model, im)
        rs_xyxy = nms(pred, im, im0)
        if len(rs_xyxy) != 0:
            for r in rs_xyxy:
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'score': r[4],
                    'value': {
                        'x': r[0] / original_width * 100,
                        'y': r[1] / original_height * 100,
                        'width': (r[2] - r[0]) / original_width * 100,
                        'height': (r[3] - r[1]) / original_height * 100,
                        'rotation': 0,
                        'rectanglelabels': [self.category_map[int(r[5])]]
                    }
                })
                all_scores.append(r[4])
            avg_score = sum(all_scores) / len(all_scores)
            return [{
                'result': results,
                'score': avg_score,
                'model_version': model_name,
            }]
        else:
            return [{
                'result': [],
                'score': 0,
                'model_version': model_name,
            }]
