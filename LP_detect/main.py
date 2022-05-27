import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

from src.data_utils import *
from yolo_detect import *

configPath = './src/weights/yolov4-custom.cfg'
weightsPath = './src/weights/yolov4-custom_1000.weights'
model_yolo = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread('src/image/19.jpg')
(H, W) = image.shape[:2]

ln = model_yolo.getLayerNames()
ln = [ln[i[0] - 1] for i in model_yolo.getUnconnectedOutLayers()]

from tensorflow.keras.models import load_model

model = load_model('./src/w/save_1.h5')

dict_temp = {7: 'A', 28: 'Y', 30: 'Z', 15: 'H', 6: '6', 19: 'N', 0: '0',
             1: '1', 11: 'E', 27: 'X', 21: 'P', 10: 'D', 29: '9', 2: '2',
             5: '5', 14: 'G', 8: 'B', 22: 'R', 26: 'V', 20: '8', 9: 'C',
             24: 'T', 13: 'F', 3: '3', 12: '7', 18: 'M', 25: 'U', 4: '4',
             17: 'L', 23: 'S', 16: 'K'}

dict_new = {1: '1', 2: '2', 10: '68', 21: '80', 23: '83', 11: '69', 27:
    '88', 15: '72', 20: '8', 9: '67', 18: '77', 5: '5', 25: '85',
            6: '6', 8: '66', 4: '4', 19: '78', 22: '82', 14: '71', 0: '0',
            26: '86', 30: '90', 12: '7', 29: '9', 3: '3', 17: '76',
            28: '89', 16: '75', 13: '70', 7: '65', 24: '84'}


def recognizeChar(candidates, candidates1):
    characters = []
    coordinates = []

    for char, coordinate in candidates:
        characters.append(char)
        coordinates.append(coordinate)

    characters = np.array(characters)
    result = model.predict(characters)
    print('rs', result)
    result_idx = np.argmax(result, axis=1)
    print('rs', result_idx[0])
    str = ""
    for i in range(len(result_idx)):
        candidates1.append((dict_temp[result_idx[i]], coordinates[i]))
        str += dict_temp[result_idx[i]]
    print('str ', str)


def segmentation1(LpRegion, candidates):
    V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
    print("V ", V.shape)

    blurred = imutils.resize(V, width=300)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 14)

    cv2.imshow('thew', thresh)
    cv2.waitKey(0)

    # connected components analysis
    # dòng này trả về các labels cho từng kết nối
    labels = measure.label(thresh, connectivity=2, background=0)

    # loop over the unique components
    # lặp lại qua từng thành phần kết nối
    dem = 0
    dem2 = 0
    for label in np.unique(labels):
        dem += 1
        # if this is background label, ignore it
        if label == 0:
            continue

        # nếu không, hãy xây dựng mặt nạ nhãn để chỉ hiển thị các thành phần được kết nối cho
        # nhãn hiện tại, sau đó tìm các đường viền trong mặt nạ nhãn
        # init mask to store the location of the character candidates
        # mask này là vùng chứa 1 ký tự trong rất nhiều ký tự của labels
        mask = np.zeros(thresh.shape, dtype="uint8")
        # print('hihihihihi',labels.shape)
        mask[labels == label] = 255

        # cv2.imshow('haha', mask)
        # cv2.waitKey(0)

        # find contours from mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)
            # rule to determine characters
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(LpRegion.shape[0])

            if 0.23 < aspectRatio < 1.5 and 0.15 < solidity < 1 and 0.15 < heightRatio < 3 and 26 < h < 175 and 12 < w < 70:
                dem2 += 1
                cv2.imshow('haha', mask)
                cv2.waitKey(0)

                # extract characters
                candidate = np.array(mask[y:y + h, x:x + w])

                square_candidate = convert2Square(candidate)
                # square_candidate = candidate

                square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                square_candidate = square_candidate.reshape((28, 28, 1))
                candidates.append((square_candidate, (y, x)))

            print('w ', w)
            print('h ', h)
            print('aspectRatio = ', aspectRatio)
            print('solidity = ', solidity)
            print('heightRatio = ', heightRatio)
        print('------------------   -----------')
    print('dem  = ', dem)
    print('dem2  = ', dem2)


def predict(image):
    image = image
    for coordinate in extractLP(model_yolo, image):  # detect license plate by yolov3
        candidates = []
        candidates1 = []
        # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
        pts = order_points(coordinate)

        pts = np.array((pts), dtype="float32")

        # crop number plate used by bird's eyes view transformation
        LpRegion = perspective.four_point_transform(image, pts)

        cv2.imshow('LPRegion', LpRegion)
        cv2.waitKey(0)
        segmentation1(LpRegion, candidates)

        # recognize characters
        recognizeChar(candidates, candidates1)

        # print('can di ', candidates)
        # format and display license plate
        license_plate = format(candidates1)

        # draw labels
        image = draw_labels_and_boxes(image, license_plate, coordinate)

    return image


predict(image)
cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)
