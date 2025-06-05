# YOLO DeepSort Tracker

Sistema de rastreamento de objetos em tempo real usando YOLO para detecção e DeepSort para tracking. Ideal para rastreamento de veículos, pessoas e outros objetos em vídeos.

## 🚀 Funcionalidades

- **Detecção com YOLO**: Suporte a modelos YOLOv8/YOLOv9/YOLOv10
- **Tracking com DeepSort**: Rastreamento robusto de múltiplos objetos
- **Filtros inteligentes**: Filtragem por classes específicas e ROI
- **Otimização de performance**: Processamento em GPU/CPU com otimizações
- **Configuração flexível**: Argumentos personalizáveis via linha de comando
- **Auto-detecção**: Configuração automática para vídeos de exemplo

## 📋 Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/yolo-deepsort-tracker.git
cd yolo-deepsort-tracker
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Baixe um modelo YOLO (exemplo):
```bash
# YOLOv8 nano (mais rápido)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# YOLOv8 small (mais preciso)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

## 🎯 Uso

### Versão Básica (TrackerV1)
```bash
python trackerV1.py --video caminho/para/video.mp4 --model yolov8n.pt
```

### Versão Avançada (TrackerV2)
```bash
python trackerV2.py --video caminho/para/video.mp4 --model yolov8n.pt
```

### 📺 Download de Vídeos do YouTube

O projeto inclui `yt-dlp` para baixar vídeos do YouTube diretamente:

#### Download Básico
```bash
# Download em qualidade padrão (somente vídeo, sem áudio)
yt-dlp -f "bv*" "https://www.youtube.com/watch?v=VIDEO_ID"

# Download com nome específico (somente vídeo)
yt-dlp -f "bv*" -o "meu_video.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### Download Otimizado para Tracking
```bash
# Download em MP4 com resolução específica (recomendado para tracking)
yt-dlp -f "bv*[height<=720][ext=mp4]" -o "sample_traffic.mp4" "URL_DO_VIDEO"

# Download apenas os primeiros 60 segundos (para testes)
yt-dlp --download-sections "*0-60" -f "bv*[ext=mp4]" "URL_DO_VIDEO"
```

#### Exemplo Completo: YouTube → Tracking
```bash
# 1. Baixar vídeo de tráfego do YouTube (somente vídeo, sem áudio)
yt-dlp -f "bv*[height<=720][ext=mp4]" -o "traffic_video.mp4" "https://youtube.com/watch?v=EXEMPLO"

# 2. Executar tracking no vídeo baixado
python trackerV2.py --video traffic_video.mp4 --model yolov8n.pt
```

#### Opções Úteis do yt-dlp
| Comando | Descrição |
|---------|-----------|
| `-f "bv*[height<=480]"` | Vídeo sem áudio, limita resolução (480p, 720p, 1080p) |
| `-f "bv*[ext=mp4]"` | Vídeo sem áudio, força formato MP4 |
| `--download-sections "*0-120"` | Baixa apenas os primeiros 2 minutos |
| `-o "%(title)s.%(ext)s"` | Usa título do vídeo como nome |
| `--list-formats` | Lista formatos disponíveis |

**💡 Dica:** Usar `bv*` baixa apenas o vídeo (sem áudio), reduzindo o tamanho do arquivo e evitando problemas de compatibilidade com codecs de áudio.

### Parâmetros Disponíveis (TrackerV2)

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|---------|-----------|
| `--video` | str | **obrigatório** | Caminho para o arquivo de vídeo |
| `--model` | str | **obrigatório** | Caminho para o modelo YOLO (.pt) |
| `--conf` | float | 0.6 | Limite de confiança para detecções |
| `--skip-frames` | int | 1 | Processa a cada N frames |
| `--resize-width` | int | None | Redimensiona largura para processamento |
| `--resize-height` | int | None | Redimensiona altura para processamento |
| `--filter-classes` | list | None | IDs das classes para filtrar |

### Exemplos de Uso

#### Rastreamento de Pessoas
```bash
python trackerV2.py --video people.mp4 --model yolov8n.pt --filter-classes 0
```

#### Rastreamento de Veículos
```bash
python trackerV2.py --video traffic.mp4 --model yolov8s.pt --filter-classes 2 3 5 7
```

#### Otimização de Performance
```bash
python trackerV2.py --video video.mp4 --model yolov8n.pt --skip-frames 2 --resize-width 640 --resize-height 480
```

## 🏷️ Classes COCO

| ID | Classe | ID | Classe | ID | Classe |
|----|--------|----|--------|----|--------|
| 0 | person | 1 | bicycle | 2 | car |
| 3 | motorcycle | 4 | airplane | 5 | bus |
| 6 | train | 7 | truck | 8 | boat |
| ... | ... | ... | ... | ... | ... |

[Ver lista completa das 80 classes COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## ⚡ Otimizações

### TrackerV2 inclui várias otimizações:

- **GPU/CPU automático**: Detecta automaticamente CUDA
- **Cache de variáveis**: Evita recálculos desnecessários
- **Processamento batch**: Operações vetorizadas
- **Skip frames**: Processa apenas frames selecionados
- **Redimensionamento**: Reduz resolução para maior velocidade
- **ROI filtering**: Filtra apenas regiões de interesse

## 🎯 ROI (Region of Interest) e Auto-Detecção

### Auto-Configuração para Vídeos de Exemplo

O TrackerV2 possui **detecção automática** para os vídeos de exemplo, configurando automaticamente:

#### `sample_video2.mp4` (Tráfego de Veículos)
```python
# Auto-detectado automaticamente:
--filter-classes 2 3 5 7  # car, motorcycle, bus, truck
# ROI da pista: x=[100,1250], y=[0,700]
```

#### `sample_video.mp4` (Pessoas)
```python
# Auto-detectado automaticamente:
--filter-classes 0  # person
# Sem ROI (rastreamento em toda a imagem)
```

### Como o ROI Melhora a Precisão

O **ROI (Region of Interest)** implementado no `sample_video2.mp4` oferece várias melhorias:

#### 🎯 **Redução de Falsos Positivos**
- Filtra detecções fora da área da pista
- Elimina veículos estacionados nas laterais
- Ignora objetos irrelevantes (pedestres, placas, etc.)

#### ⚡ **Melhoria de Performance**
- Processa apenas a região relevante
- Reduz carga computacional do tracker
- Diminui conflitos de ID entre objetos

#### 📐 **Implementação Técnica**
```python
# Coordenadas do ROI otimizadas para sample_video2.mp4
ROI_X_MIN, ROI_X_MAX = 100, 1250  # Largura da pista
ROI_Y_MIN, ROI_Y_MAX = 0, 700     # Altura útil da câmera

# Filtragem eficiente usando coordenadas em cache
if x1_scaled < roi_x_min or x2_scaled > roi_x_max or \
   y1_scaled < roi_y_min or y2_scaled > roi_y_max:
    continue  # Descarta detecção fora do ROI
```

### Visualização do ROI

O TrackerV2 desenha automaticamente o ROI em azul quando ativo:
- **Retângulo azul**: Delimita a área de interesse
- **Texto "ROI - pista"**: Indica que o filtro está ativo
- **Detecções verdes**: Apenas dentro da área ROI

## 🎮 Controles

- **`q`**: Sair do programa
- **`ESC`**: Fechar janela

## 📁 Estrutura do Projeto

```
yolo-deepsort-tracker/
├── trackerV1.py          # Versão básica
├── trackerV2.py          # Versão otimizada
├── requirements.txt      # Dependências
├── README.md            # Este arquivo
└── models/              # Pasta para modelos (criar se necessário)
    └── yolov8n.pt
```

## 🔧 Solução de Problemas

### Erro: "Could not open video file"
- Verifique se o caminho do arquivo está correto
- Confirme se o formato do vídeo é suportado (mp4, avi, mov, etc.)

### Performance baixa
- Use `--skip-frames 2` ou maior
- Redimensione o vídeo com `--resize-width` e `--resize-height`
- Use um modelo menor como `yolov8n.pt`

### CUDA não detectado
- **Verifique a compatibilidade da sua placa de vídeo**: [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- Para outras versões CUDA (11.7, 12.1, etc.), consulte a página oficial do PyTorch
- Instale PyTorch com suporte CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### ROI não aparece ou funciona incorretamente
- Verifique se está usando `trackerV2.py` (ROI não está no V1)
- ROI só é ativado automaticamente para `sample_video2.mp4`
- Para outros vídeos, modifique as coordenadas ROI no código conforme necessário

### Auto-detecção não funciona
- Certifique-se que o nome do arquivo contém exatamente `sample_video2.mp4` ou `sample_video.mp4`
- Use `--filter-classes` manualmente se necessário: `--filter-classes 0 2 3 5 7`

