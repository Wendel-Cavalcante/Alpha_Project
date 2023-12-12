import cv2
import face_recognition as fr

imgWen = fr.load_image_file('wendel.png')
imgWen = cv2.cvtColor(imgWen, cv2.COLOR_BGR2RGB)
imgWenTest = fr.load_image_file('wendelTeste.png')
imgWenTest = cv2.cvtColor(imgWenTest, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgWen)[0]
cv2.rectangle(imgWen,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)


encodeWen = fr.face_encodings(imgWen)[0]
encodeWenTest = fr.face_encodings(imgWenTest)[0]

comparacao = fr.compare_faces([encodeWen],encodeWenTest)
distancia = fr.face_distance([encodeWen],encodeWenTest)

print(comparacao,distancia)

cv2.imshow('wendel',imgWen)
cv2.imshow('wendel Test',imgWenTest)
cv2.waitKey(0)