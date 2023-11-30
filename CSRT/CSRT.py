import cv2

def track_people(video_path):
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)

    # Cria o rastreador CSRT
    tracker = cv2.TrackerCSRT_create()

    # Inicia o rastreamento
    ret, frame = cap.read()
    bbox = cv2.selectROI("Selecione a pessoa a ser rastreada", frame)
    tracker.init(frame, bbox)

    # Loop sobre os frames do vídeo
    while cap.isOpened():
        # Lê o próximo frame
        ret, frame = cap.read()

        # Rastreia o objeto
        ret, bbox = tracker.update(frame)

        # Desenha o retângulo de rastreamento
        cv2.rectangle(frame, bbox, (255, 0, 0), 2)

        # Exibe o frame
        cv2.imshow("Rastreamento de pessoas", frame)

        # Pressione q para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Fecha o vídeo
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "CSRT/people_walking.mp4"
    track_people(video_path)