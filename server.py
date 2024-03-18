from flask import Flask, request, jsonify
from functools import cache
from io import BytesIO
from PIL import Image
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import requests


WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)
pred_counter = Counter("app_http_inference_count", "")
model = fasterrcnn_resnet50_fpn_v2(weights=WEIGHTS, box_score_thresh=0.9)

model.eval()


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url = data['url']
    response = requests.get(url)

    imgage_raw = Image.open(BytesIO(response.content))
    
    prediction = model(WEIGHTS.transforms()(imgage_raw)[None, :])[0]
    
    labels = [WEIGHTS.meta["categories"][i] for i in prediction["labels"]]

    pred_counter.inc()

    return jsonify({
        "objects": labels
    })


def main():
    
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
