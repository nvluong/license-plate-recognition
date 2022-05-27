import cv2
from src.data_utils import get_output_layers
import numpy as np

# hàm nhận diện vùng chứa biển số
def detectLP(model_yolo, image):
    boxes = []
    classes_id = []
    confidences = []
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True,
                                 crop=False)
    height, width = image.shape[:2]

    # take image to model
    model_yolo.setInput(blob)

    # run forward
    outputs = model_yolo.forward(get_output_layers(model_yolo))

    for output in outputs:
        for i in range(len(output)):
            scores = output[i][5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])

            if confidence > 0.5:
                # coordinate of bounding boxes
                center_x = int(output[i][0] * width)
                center_y = int(output[i][1] * height)

                detected_width = int(output[i][2] * width)
                detected_height = int(output[i][3] * height)

                x_min = center_x - detected_width / 2
                y_min = center_y - detected_height / 2

                boxes.append([x_min, y_min, detected_width, detected_height])
                classes_id.append(class_id)
                confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    coordinates = []
    # -----------------
    crop_scale = 0.05
    # -----------------

    for i in indices:
        index = i[0]
        x_min, y_min, width, height = boxes[index]
        x_min = round(x_min)
        y_min = round(y_min)

        # --------------------------
        # x_min = abs(int(x_min - crop_scale * width))
        # y_min = abs(int(y_min - crop_scale * height))
        # width = abs(int((1 + 2 * crop_scale) * width))
        # height = abs(int((1 + 2 * crop_scale) * height))
        # ----------------------------

        coordinates.append((x_min, y_min, width, height))

    return coordinates


# hàm này chuyển đổi từ tọa độ tuple thành generator
def extractLP(model, image):
    coordinates = detectLP(model, image)
    print('codi', type(coordinates))
    if len(coordinates) == 0:
        ValueError('No images detected')

    for coordinate in coordinates:
        yield coordinate