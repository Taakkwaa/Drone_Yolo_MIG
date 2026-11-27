# Détection d'objets temps réel — YOLO26n sur Raspberry Pi 5

Détection YOLO26n sur webcam USB, en Python, avec ONNX Runtime (ou NCNN en option) :
- modèle COCO d'origine (80 classes), ou
- modèle **fine-tuné sur HERIDAL** (détection de personnes vues de drone).

Le pré-traitement (letterbox / stretch, normalisation) et le post-traitement
(décodage, NMS) sont écrits à la main en NumPy dans
[`yolo_pi/processing.py`](yolo_pi/processing.py) — résultats identiques à ceux
d'Ultralytics, au pixel près. YOLOv8n reste supporté (même format de sortie).

## Structure

```
detect.py              programme principal (Pi, console, sans écran)
mac_demo.py            interface fenêtre temps réel (Mac), avec émulation du Pi 5
benchmark.py           FPS moyens sur 100 frames + temps par étape
train.py               fine-tuning (ex. HERIDAL) + export ONNX dans models/
yolo_pi/
  processing.py        letterbox/stretch + normalisation, décodage + NMS
  backends.py          ONNX Runtime (défaut) ou NCNN ; lit noms de classes et famille du modèle
  detector.py          pipeline complet avec chrono par étape
  camera.py            capture dans un thread (toujours la dernière frame)
  draw.py              boîtes + labels
  pi_emulation.py      temps du Pi 5 et cadencement (mac_demo)
  memory.py            plafond de RAM (mac_demo)
  coco.py              noms des 80 classes COCO (repli)
models/                modèles exportés (.onnx, *_ncnn_model_*/)
```

## 1. Exporter le modèle COCO

À faire **sur ton Mac/PC** (PyTorch est lourd et inutile sur le Pi) :

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-export.txt

mkdir -p models && cd models
yolo export model=yolo26n.pt format=onnx              # -> yolo26n.onnx (640x640)
mv yolo26n.onnx yolo26n_640.onnx
yolo export model=yolo26n.pt format=onnx imgsz=320    # 320x320, ~4x moins de calcul
mv yolo26n.onnx yolo26n_320.onnx

# Optionnel : NCNN, le plus rapide sur CPU ARM
yolo export model=yolo26n.pt format=ncnn imgsz=320
mv yolo26n_ncnn_model yolo26n_ncnn_model_320
cd ..
```

Sur Windows:
python3 -m venv .venv && .\.venv\Scripts\activate.bat



(Pour YOLOv8n : mêmes commandes avec `yolov8n.pt`, ex. `yolo export model=yolov8n.pt format=onnx`.)

Les poids sont téléchargés automatiquement au premier export. La taille d'entrée et
les noms des classes sont enregistrés dans le modèle : le programme les lit tout seul.

## 2. Fine-tuning sur HERIDAL (personnes vues de drone)

Dataset : [Roboflow – HERIDAL Human Detection](https://universe.roboflow.com/drone-internship/heridal-human-detection)
(CC BY 4.0), téléchargé au format YOLOv8/YOLO26 dans
`Heridal Human Detection.v14i.yolo26/`.

```bash
python train.py --data "Heridal Human Detection.v14i.yolo26/data.yaml" --model models/yolo26n.pt --epochs 100 --name heridal
# -> runs/heridal/ (courbes, poids) et models/heridal_640.onnx
```

Sur un Mac Apple Silicon (GPU `mps`), une epoch prend environ 1 min ; l'arrêt
anticipé (`--patience 30`) coupe l'entraînement s'il ne progresse plus. Sur Colab
(GPU NVIDIA), c'est plus rapide.

Suivi pendant l'entraînement ([`train_monitor.py`](train_monitor.py)) :

```bash
tensorboard --logdir runs/tensorboard      # puis http://localhost:6006
```

- **Scalars** : pertes train/val, mAP50, mAP50-95, précision, rappel, learning rate,
  `best/map50` (record courant).
- **Images**, à chaque epoch, sur 6 images de validation fixes :
  `exemples/vue_complete` (vérité terrain en vert, prédictions en rouge avec score)
  et `exemples/zoom_personnes` (zoom x4 sur chaque personne ; cadre vert = trouvée,
  rouge = ratée). Seuil d'affichage 0.15 pour voir les détections émerger tôt.
- **Meilleur checkpoint** : `checkpoints/heridal_best.pt` est remplacé à chaque fois
  que le mAP50 bat son record (`checkpoints/heridal_best.json` : epoch et scores).
  C'est ce fichier qui est exporté en ONNX à la fin. Autre critère :
  `--best-metric map50-95` ou `--best-metric recall`.

Ce qu'il faut savoir sur ce dataset (analysé) :

- 2685 images (2212 train / 316 valid / 157 test), une classe `human`.
- Roboflow a **étiré** les photos d'origine (4000x3000) en **640x640**. Il faut donc
  utiliser **`--resize stretch`** à l'inférence, pour que les personnes aient la même
  forme qu'à l'entraînement.
- Les personnes y font **~11 px de médiane** : c'est très petit. On reste en **640**
  (en 320, 90 % d'entre elles feraient moins de 8 px). Pour faire mieux, il faudrait
  retélécharger le dataset **sans redimensionnement** et travailler par tuiles.

## 3. Installer sur le Raspberry Pi 5

Copie le projet sur le Pi (avec `models/`, sans `.venv` ni le dataset) :

```bash
rsync -av --exclude .venv --exclude runs --exclude "Heridal*" ./ pi@raspberrypi.local:~/yolo/
```

Raspberry Pi OS 64 bits (Bookworm). Tous les paquets existent en wheel précompilée
pour aarch64, rien à compiler.

**Caméra CSI (nappe plate, Pi Camera Module)** — Picamera2 s'installe par apt, le
venv doit donc voir les paquets système :

```bash
cd ~/yolo
rpicam-hello --list-cameras                  # la caméra doit être listée (ex. imx708)
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-pi.txt           # onnxruntime + psutil seulement
python check_camera.py                       # type, résolution, FPS de capture, photo test_camera.jpg
```

**Webcam USB** :

```bash
cd ~/yolo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install -y v4l-utils
v4l2-ctl --list-devices                     # trouve /dev/videoN
v4l2-ctl -d /dev/video0 --list-formats-ext  # vérifie que MJPG est proposé
python check_camera.py --camera 0
```

`--camera auto` (défaut partout) prend la caméra CSI si Picamera2 en trouve une,
sinon la caméra OpenCV n°0. On peut forcer `--camera csi` ou `--camera 1`.

Conseil : mets un dissipateur/ventilateur (Active Cooler). Sous charge continue,
le Pi 5 sans refroidissement baisse sa fréquence et les FPS chutent.

## 4. Lancer

```bash
source .venv/bin/activate

# Modèle COCO 640x640, caméra 0, seuil 0.25 (défauts)
python detect.py

# Modèle HERIDAL fine-tuné
python detect.py --model models/heridal_640.onnx --resize stretch --conf 0.3

# 320x320, plus rapide (objets proches seulement)
python detect.py --model models/yolo26n_320.onnx --conf 0.4

# Sauvegarder 1 image annotée sur 10, seulement s'il y a des détections
python detect.py --save-dir out --save-every 10 --save-only-detections

# Backend NCNN (taille lue dans metadata.yaml)
python detect.py --model models/yolo26n_ncnn_model_320
```

Sortie console (une ligne par frame) :

```
[    42]   7.3 FPS | pre   5.1 ms  inf  126.4 ms  post  4.9 ms | 2 obj | human 0.71 [120,40,140,70], human 0.52 [400,300,415,330]
```

Options principales (`python detect.py -h` pour tout voir) :

| Option | Rôle | Défaut |
|---|---|---|
| `--model` | `.onnx` ou dossier `*_ncnn_model*` | `models/yolo26n_640.onnx` |
| `--resize` | `letterbox` (poids COCO) ou `stretch` (dataset Roboflow étiré, HERIDAL) | `letterbox` |
| `--conf` / `--iou` | seuils de confiance / NMS | 0.25 / 0.45 |
| `--camera` | `auto`, `csi`, index OpenCV, `/dev/videoN` ou fichier vidéo | `auto` |
| `--cam-width/--cam-height` | résolution demandée à la caméra | 640x480 |
| `--save-dir`, `--save-every`, `--save-only-detections` | sauvegarde des images annotées | off |
| `--show` | fenêtre vidéo (nécessite un écran et `opencv-python` au lieu de `-headless`) | off |
| `--threads` | threads d'inférence | 4 |

`Ctrl+C` arrête proprement.

## 5. Tester sur Mac avec une fenêtre

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y opencv-python-headless && pip install opencv-python   # version avec fenêtres
python mac_demo.py                                                    # COCO 640 + 320
python mac_demo.py --models models/heridal_640.onnx --resize stretch  # modèle HERIDAL
```

- curseur **conf %** : seuil de confiance en direct
- **m** : change de modèle · **p** : émulation Pi on/off · **s** : capture dans `captures/` · **q** : quitter
- RAM plafonnée à 8 Go comme le Pi 5 (`--max-ram-gb`). macOS ignore les limites
  noyau, donc la RAM réelle est vérifiée à chaque frame et le programme s'arrête
  proprement au-delà ; sous Linux, une vraie limite noyau s'ajoute.
  Consommation mesurée : ~270 Mo avec deux modèles chargés.

Au premier lancement, macOS demande l'accès à la caméra pour Terminal / VS Code :
accepte, puis relance. Une autre caméra : `--camera 1`.

### Émulation des performances du Pi 5 (activée par défaut)

`mac_demo.py` bride le Mac comme un Pi 5 8 Go : 4 threads, 8 Go de RAM max, caméra
640x480, et surtout **le rythme d'images du Pi** : chaque frame est calculée
normalement puis la boucle attend jusqu'au temps qu'aurait mis le Pi. FPS, latence et
images sautées sont donc ceux du Pi. `--no-pi` coupe l'émulation au lancement.

| Modèle (ONNX) | Pi 5, 640 | Pi 5, 320 |
|---|---|---|
| YOLO26n (et HERIDAL fine-tuné) | ≈ 7.4 FPS | ≈ 26 FPS |
| YOLOv8n | ≈ 5.0 FPS | ≈ 17 FPS |

Ce sont des **estimations** (inférence YOLO26n 640 : chiffres officiels Ultralytics ;
le reste est extrapolé, voir [`yolo_pi/pi_emulation.py`](yolo_pi/pi_emulation.py)).
Pour avoir les temps réels de ton Pi :

```bash
# sur le Pi
python benchmark.py --model models/yolo26n_640.onnx --save-profile pi_profile.json
python benchmark.py --model models/yolo26n_320.onnx --save-profile pi_profile.json
# copie pi_profile.json sur le Mac, puis
python mac_demo.py --pi-profile pi_profile.json     # le panneau affiche "mesuré sur Pi"
```

## 6. Benchmark

```bash
# Calcul seul, reproductible (image fixe en boucle) : pour comparer modèles et backends
python benchmark.py --model models/yolo26n_640.onnx
python benchmark.py --model models/yolo26n_320.onnx
python benchmark.py --model models/yolo26n_ncnn_model_320

# Pipeline réel avec la webcam
python benchmark.py --model models/yolo26n_640.onnx --source camera
```

10 frames de chauffe puis 100 frames mesurées. Résultat : FPS moyens, latence
moyenne / p50 / p95, et le temps de chaque étape.

Mesures de référence sur un Mac Apple Silicon (image `bus.jpg`). **Le Pi 5 sera
nettement plus lent** : lance le benchmark sur le Pi pour avoir tes vrais chiffres.

| Modèle | FPS moyen | Inférence |
|---|---|---|
| YOLO26n ONNX 640 | 66.8 | 12.9 ms |
| YOLO26n ONNX 320 | 225 | 3.9 ms |
| YOLO26n NCNN 320 | 250 | 3.4 ms |
| YOLOv8n ONNX 640 | 44.6 | 19.7 ms |

L'inférence représente environ 90 % du temps : c'est là qu'il faut agir
(résolution, backend), pas sur le code Python autour.

## Pour aller plus vite sur le Pi

1. Backend NCNN (≈ 2x plus rapide qu'ONNX Runtime sur le Pi 5 d'après Ultralytics).
2. `320` au lieu de `640` — sauf pour HERIDAL, où les personnes sont trop petites.
3. Webcam en 640x480 MJPG (défaut) : inutile de capturer en 1080p.
4. Refroidissement actif, alimentation officielle 27 W.
