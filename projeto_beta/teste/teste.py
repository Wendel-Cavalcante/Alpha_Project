import cv2
import time
import numpy as np

#CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#CARREGA AS CLASSES
class_names =[]
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#CAPTURA DO VIDEO
cap = cv2.VideoCapture(0)

#CARREGA OS PESOS DA REDE NEURAL
#net = cv2.dnn.readNet("weights;yolov4.weights", "cfg/yolov4.cfg")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#LENDO OS FRAMES DO VIDEO
while True:

    #CAPTURA DO FRAME
    _, frame = cap.read()

    #começo da contagem dos ms
    start = time.time()

    #detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    #fim da contagem dos ms
    end = time.time()

    #percorrer todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):

        #gerando uma cor para a classe
        color = COLORS[int(classid) % len (COLORS)]

        #PEGANDO O NOME DA CLASSE PELO ID E SEU SCORE DE ACURACIA
        label = f"{class_names[classid]} : {score}"

        #desenhando a box da detecção
        cv2.rectangle(frame, box, color, 2)

        #escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0],box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color,2)

    #Calculando o tempo que levou para fazer a detecção
    fps_label = f"fps: {round((1.0/(end - start)),2)}"

    #escrevendo o fps na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    #MOSTRANDO A IMAGEM
    cv2.imshow("detections",frame)

    #espera a resposta
    if cv2.waitKey(1) == 27:
        break

#liberação da camera e destroi todas as janelas
cap.release()
cv2.destroyAllWindows()