import cv2

# Carregue o vídeo da pasta
videoPath = 'YOLO/face.mp4x'
video = cv2.VideoCapture(videoPath)

# Carregue o classificador de faces
classificadorVideoFace = cv2.CascadeClassifier('YOLO/haarcascades/haarcascade_frontalface_default.xml')

while True:
    # Capture o frame atual do vídeo
    camera, frame = video.read()

    # Verifique se o frame foi capturado com sucesso
    if not camera:
        break

    # Converta o frame para tons de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte faces no frame em tons de cinza
    detecta = classificadorVideoFace.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))

    # Desenhe retângulos em torno das faces detectadas
    for (x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)

    # Contador de faces
    if len(detecta) == 0:
        print("No faces detected")
        continue
    else:
        contador = str(len(detecta))

    # Exiba a contagem de faces acima da caixa delimitadora
    cv2.putText(frame, contador, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exiba a contagem total de faces na parte inferior do quadro
    cv2.putText(frame, "Quantidade de Faces: " + contador, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exiba o frame com as faces detectadas e a contagem
    cv2.imshow("Deteção de Faces em Vídeo", frame)

    # Verifique se a tecla 'f' foi pressionada para interromper o loop
    if cv2.waitKey(1) == ord('q'):
        break

# Libere recursos
video.release()
cv2.destroyAllWindows()
