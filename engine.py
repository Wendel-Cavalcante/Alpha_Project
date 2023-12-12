import cv2
import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos

    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos  = []

    wendel1 = reconhece_face("./Pessoas/wendel.png")
    if(wendel1[0]):
        rostos_conhecidos.append(wendel1[1][0])
        nomes_dos_rostos.append("Wendel")

    Diogo1 = reconhece_face("./Pessoas/diogo.png")
    if(Diogo1[0]):
        rostos_conhecidos.append(Diogo1[1][0])
        nomes_dos_rostos.append("Diogo")

    return rostos_conhecidos, nomes_dos_rostos