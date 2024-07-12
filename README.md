# Reconhecimento Facial ao Vivo com Webcam (Magic Mirror)
Este projeto implementa um sistema de reconhecimento facial ao vivo usando Python. Ele captura vídeo de uma webcam, detecta e reconhece rostos em tempo real, e exibe os nomes reconhecidos no feed de vídeo.

## Funcionalidades
* Reconhecimento facial em tempo real usando a webcam.
* Utiliza OpenCV para captura de vídeo e processamento de imagens.
* Emprega face_recognition para codificação e comparação facial.
* Exibe caixas delimitadoras e nomes ao redor dos rostos reconhecidos.

## Requisitos
* Python 3.x
* OpenCV
* face_recognition
* numpy

## Instalação
### Clone o repositório:
```
git clone https://github.com/seuusuario/reconhecimento-facial-ao-vivo.git
cd reconhecimento-facial-ao-vivo
```
### Instale as bibliotecas necessárias:
```
pip install opencv-python face_recognition numpy
```
Certifique-se de que você tem um diretório chamado FaceRecImages na raiz do projeto, contendo imagens dos rostos que você deseja reconhecer. Nomeie os arquivos de imagem como Nome.jpg (ex.: JohnDoe.jpg).

## Uso
### Execute o script:
```
python main.py
```
A webcam será iniciada e o sistema começará a reconhecer rostos em tempo real. Pressione q para sair do aplicativo.

## Explicação do Código
```
import cv2
import numpy as np
import face_recognition
import os

path = 'FaceRecImages'
images = []
classNames = []

if os.path.exists(path):
    mylist = os.listdir(path)
    print(f'Imagens encontradas: {mylist}')
else:
    print(f'Pasta {path} não encontrada.')
    mylist = []

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f'Falha ao carregar imagem {cl}')

print(f'Nomes das classes: {classNames}')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("Nenhum rosto encontrado na imagem.")
    return encodeList

encodeListKnown = findEncodings(images)
print('Codificação completa')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Falha ao capturar a imagem da câmera.")
        break
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
```

Sinta-se à vontade para contribuir abrindo issues ou enviando pull requests. Qualquer melhoria ou sugestão é bem-vinda.

Este projeto é licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
