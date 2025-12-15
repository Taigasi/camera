import os
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import time
from collections import deque, Counter 
from flask import Flask, Response, jsonify
from flask_cors import CORS

# Configurações para deixar o log limpo
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)
CORS(app) 

# --- VARIÁVEIS GLOBAIS (O ESTADO DA ENTREVISTA) ---
estado_entrevista = {
    "fps": 0,
    "score_geral": 100, 	 
    "postura_status": "Analisando...", 
    "contato_visual": 0, 	 # Porcentagem (0 a 100)
    "emocao_dominante": "Neutro",
    "atencao_alerta": False, 
    "gestos_alerta": False 	
}

# --- CARREGAR IA DE EMOÇÃO ---
ONNX_PATH = "models/emotion-ferplus.onnx"
sess_emo = None
EMO_LABELS = ["Neutro", "Felicidade", "Surpresa", "Tristeza", "Raiva", "Nojo", "Medo", "Desprezo"]

try:
    if os.path.exists(ONNX_PATH):
        sess_emo = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        print("IA de Emoção: ATIVADA")
    else:
        print(f"IA de Emoção: DESATIVADA (Arquivo não encontrado em: {ONNX_PATH})")
except Exception as e:
    print(f"Erro ao carregar ONNX: {e}")
    sess_emo = None

# --- MEDIAPIPE (Corpo e Face) ---
mp_hol = mp.solutions.holistic
hol = mp_hol.Holistic(
    model_complexity=0, 
    smooth_landmarks=True,
    refine_face_landmarks=True 
)
mp_drawing = mp.solutions.drawing_utils

# --- FUNÇÕES AUXILIARES ---

def get_emotion(face_img):
    """Processa o ROI da face para prever a emoção dominante."""
    if sess_emo is None: return "Neutro"
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        input_name = sess_emo.get_inputs()[0].name
        input_tensor = resized[np.newaxis, np.newaxis, :, :]
        res = sess_emo.run(None, {input_name: input_tensor})[0][0]
        return EMO_LABELS[np.argmax(res)]
    except: 
        return "Neutro"

def calcular_olhar(face_landmarks, w, h):
    """Calcula a pontuação de contato visual (0 a 100) baseada no desvio da íris."""
    if not face_landmarks: return 0, True 
    
    # Índices da íris
    left_iris = face_landmarks.landmark[468] 
    right_iris = face_landmarks.landmark[473] 
    
    # Olho Esquerdo (do usuário, direito na tela)
    left_eye_inner = face_landmarks.landmark[362] # Canto interno
    left_eye_outer = face_landmarks.landmark[263] # Canto externo
    
    # Olho Direito (do usuário, esquerdo na tela)
    right_eye_inner = face_landmarks.landmark[133]
    right_eye_outer = face_landmarks.landmark[33]

    # --- Análise do Olho Esquerdo ---
    center_eye_l = (left_eye_inner.x + left_eye_outer.x) / 2
    dist_l = abs(left_iris.x - center_eye_l)
    
    # --- Análise do Olho Direito ---
    center_eye_r = (right_eye_inner.x + right_eye_outer.x) / 2
    dist_r = abs(right_iris.x - center_eye_r)

    # Threshold: Desvio máximo aceitável. (0.006 é um bom equilíbrio)
    MAX_DEVIATION = 0.006 
    
    # Calcula a distância média dos dois olhos
    dist_avg = (dist_l + dist_r) / 2
    
    # Score: Se dist_avg é 0, score é 100. Se dist_avg >= MAX_DEVIATION, score é 0.
    # O cálculo cria um score gradual entre 0 e 100
    score_norm = max(0, 1 - (dist_avg / MAX_DEVIATION))
    contato_visual = int(score_norm * 100)
    
    # O alerta é disparado se o score for muito baixo (ex: menos de 20%)
    atencao_alerta = contato_visual < 20 
        
    return contato_visual, atencao_alerta

# --- GERADOR DE VÍDEO ---
def generate_frames():
    global estado_entrevista
    
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    hist_postura = deque(maxlen=30)
    hist_emocao = deque(maxlen=15) 
    hist_contato = deque(maxlen=20) 
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processamento MediaPipe
        results = hol.process(rgb)
        
        score = 100
        postura_ruim = False
        contato_visual_pct = 0
        emocao_atual = "Neutro"
        atencao_alerta_status = False
        gestos_alerta_status = False

        # 1. ANÁLISE DE POSTURA E GESTOS
        if results.pose_landmarks:
            lmk = results.pose_landmarks.landmark
            ombro_e = lmk[mp_hol.PoseLandmark.LEFT_SHOULDER]
            ombro_d = lmk[mp_hol.PoseLandmark.RIGHT_SHOULDER]
            
            # Postura Desnivelada
            if abs(ombro_e.y - ombro_d.y) > 0.04: 
                postura_ruim = True
                score -= 10
            
            hist_postura.append(postura_ruim)

            # Desenha esqueleto simplificado
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_hol.POSE_CONNECTIONS)

        # 2. ANÁLISE FACIAL (Emoção e Contato Visual)
        if results.face_landmarks:
            
            # Recorte do Rosto para Emoção
            x_coords = [lmk.x for lmk in results.face_landmarks.landmark]
            y_coords = [lmk.y for lmk in results.face_landmarks.landmark]
            x_min = max(0, int(min(x_coords) * w) - 20)
            x_max = min(w, int(max(x_coords) * w) + 20)
            y_min = max(0, int(min(y_coords) * h) - 20)
            y_max = min(h, int(max(y_coords) * h) + 20)
            
            face_roi = frame[y_min:y_max, x_min:x_max]
            
            # Desenho do retângulo e cálculo da emoção
            if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                emocao_atual = get_emotion(face_roi)
                # Desenha o retângulo em volta do rosto (Verde: 0, 255, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
                
            hist_emocao.append(emocao_atual)

            # Contato Visual (Lógica corrigida e mais suave)
            contato_visual_pct, atencao_alerta_status = calcular_olhar(results.face_landmarks, w, h)
            hist_contato.append(contato_visual_pct)

            # (Linha de draw_landmarks removida para limpar o vídeo)

        
        # --- CÁLCULO DE MÉDIAS MÓVEIS E STATUS ---
        # Determina o status final da postura usando a maioria das detecções recentes
        postura_final = "Corrigir Postura" if sum(hist_postura) > (len(hist_postura) / 2) else "Excelente"
        
        # Emoção mais comum (dominante)
        emocao_dominante = Counter(hist_emocao).most_common(1)[0][0] if hist_emocao else "Neutro"
        
        # Média móvel do contato visual
        contato_visual_media = int(np.mean(hist_contato)) if hist_contato else 0

        # Alerta de atenção baseado na média móvel
        atencao_alerta_final = contato_visual_media < 50 

        # Deduções finais de score
        if postura_final == "Corrigir Postura": score -= 10
        if atencao_alerta_final: score -= 15
        if gestos_alerta_status: score -= 10 # Gesto de braços cruzados não foi implementado, mas a dedução permanece
        if emocao_dominante in ["Tristeza", "Raiva", "Medo", "Nojo"]: score -= 5
        
        # --- ATUALIZA ESTADO GLOBAL ---
        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        estado_entrevista["fps"] = int(current_fps)
        estado_entrevista["postura_status"] = postura_final
        estado_entrevista["contato_visual"] = contato_visual_media
        estado_entrevista["emocao_dominante"] = emocao_dominante
        estado_entrevista["atencao_alerta"] = atencao_alerta_final
        estado_entrevista["gestos_alerta"] = gestos_alerta_status
        estado_entrevista["score_geral"] = max(0, min(100, score))
        
        # Desenho para visualização
        cv2.putText(frame, f"SCORE: {estado_entrevista['score_geral']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"CONTATO: {contato_visual_media}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stream de vídeo
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROTAS DA API ---
@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dados')
def dados():
    return jsonify(estado_entrevista.copy())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)