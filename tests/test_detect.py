import numpy as np
import pytest

from src.detect import Detect


class TestDetect:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.detect = Detect("./models/yolov5n6-fp16.tflite", conf_thr=0.4)

    def test_detect(self):
        dummy_img = np.random.randn(300, 300, 3)
        boxes, scores, class_idx = self.detect.detect(dummy_img)
        assert boxes.shape[1] == 4
        assert isinstance(boxes, np.ndarray) and isinstance(scores, np.ndarray) and isinstance(class_idx, np.ndarray)

    def test_preprocess(self):
        dummy_img = np.random.randn(300, 300, 3)
        result = self.detect.preprocess(dummy_img)
        assert result.shape == (1, self.detect.height, self.detect.width, 3)
        assert result.dtype == np.float32

    def test_to_xyxy(self):
        xywh = np.array([[100, 100, 200, 200]])
        expect = np.array([[0, 0, 200, 200]])
        result = self.detect.to_xyxy(xywh)
        assert (result == expect).all()
