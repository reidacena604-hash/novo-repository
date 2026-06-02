import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Configuração da página do Streamlit
st.set_page_config(
    page_title="Scanner com Yolo",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inicializa o modelo YOLO na sessão do Streamlit para evitar recarregamento a cada interação
@st.cache_resource
def logo_model():
    # Carrega o modelo YOLOv8 inicial (versão 'nano', ideal para deploys leves como no Render)
    return YOLO("yolov8n.pt")

model = logo_model()

# Interface do usuário
st.title("📸 Scanner com YOLOv8")
st.write("Abra a câmera abaixo para realizar a detecção de objetos em tempo real.")

# 4. Botão/Controle para abrir a câmera (O st.camera_input serve como o próprio ativador)
img_file_buffer = st.camera_input("Clique no botão abaixo para tirar uma foto")

# 3. Processamento da imagem capturada pela câmera e detecção de objetos
if img_file_buffer is not None:
    # Converte a imagem capturada para o formato PIL
    image = Image.open(img_file_buffer)
    
    # Executa a inferência do YOLO no frame capturado
    results = model(image)
    
    # Renderiza os resultados (caixas delimitadoras e labels) na imagem original
    annotated_img = results[0].plot()
    
    # Exibe a imagem processada de volta no dashboard do Streamlit
    st.image(annotated_img, caption="Objetos Detectados", use_column_width=True)