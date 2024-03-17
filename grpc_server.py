from concurrent import futures
from io import BytesIO
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import grpc
import inference_pb2
import inference_pb2_grpc
import logging
import requests


class InstanceDetectorServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        self.transform = self.weights.transforms()

    def Predict(self, request, context):
        self.model.eval()

        raw_image = requests.get(request.url).content

        input_image = Image.open(BytesIO(raw_image))

        prediction = self.model(self.transform(input_image)[None, :])
        assert len(prediction) == 1
        prediction = prediction[0]

        return inference_pb2.InstanceDetectorOutput(
            objects=[self.weights.meta["categories"][label] for label in prediction["labels"]]
        )


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorServicer(), server)
    server.add_insecure_port('0.0.0.0:9090')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
