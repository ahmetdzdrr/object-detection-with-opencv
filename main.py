import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)

threshold = 0.5
class_names = []
class_file = 'coco.names'
with open(class_file, 'rt') as file:
    class_names = file.read().rstrip('\n').split('\n')

weight_file = 'frozen_inference_graph.pb'
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

model = cv2.dnn_DetectionModel(weight_file, config_file)
model.setInputSize(340, 340)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success:
        break

    classIds, configs, bbox = model.detect(img, confThreshold=threshold)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), configs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 255), thickness=3)
            cv2.putText(img, class_names[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
