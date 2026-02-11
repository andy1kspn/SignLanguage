# TRADUCĂTOR LIMBAJ SEMNE ROMÂNESC - PROIECT COMPLET

## DESCRIERE GENERALĂ

Acest proiect implementează un sistem complet de traducere din limbajul semnelor românesc în text, cu interfață optimizată, control vocal, și suport pentru multiple tipuri de camere.

## STRUCTURA COMPLETĂ A PROIECTULUI

```
SignLanguage/
├── start_menu.py              # Interfața principală optimizată
├── realtime.py               # Aplicația de traducere în timp real
├── utils.py                  # Utilități MediaPipe optimizate
├── config.py                 # Configurări și constante
├── model.py                  # Arhitectura modelului neural
├── train_model.py            # Script pentru antrenarea modelului
├── collect_data.py           # Colectarea datelor pentru antrenament
├── augmentation.py           # Augmentarea datelor
├── generate_synthetic_data.py # Generarea datelor sintetice
├── check_camera.py           # Test simplu pentru camere
├── check_camera_dshow.py     # Test avansat cu DirectShow și IP
├── run_with_python311.bat    # Script pentru compatibilitate MediaPipe
├── install_dependencies.bat  # Instalare dependențe Windows
├── install_dependencies.sh   # Instalare dependențe Linux/Mac
├── check_dependencies.py     # Verificare dependențe instalate
├── requirements.txt          # Dependențe principale
├── requirements-dev.txt      # Dependențe pentru dezvoltare
├── README.md                 # Documentația proiectului
├── .gitignore               # Fișiere ignorate de Git
├── data/                    # Date pentru antrenament
│   ├── ajutor/             # 60 fișiere .npy
│   ├── apa/                # 60 fișiere .npy
│   ├── bine/               # Date pentru cuvântul "bine"
│   ├── buna/               # 60 fișiere .npy
│   ├── cum_te_cheama/      # 60 fișiere .npy
│   ├── da/                 # 60 fișiere .npy
│   └── ...                 # Alte cuvinte
└── models/                 # Modele antrenate
    └── lsr_model.keras     # Modelul principal
```

## FIȘIERE PRINCIPALE OPTIMIZATE

### 1. start_menu.py - INTERFAȚA PRINCIPALĂ OPTIMIZATĂ

**Caracteristici implementate:**
- ✅ Interfață Full HD (1920x1080) cu design modern și responsive
- ✅ Text centrat pe mijloc în carduri cu poziționare mai jos pentru echilibru vizual
- ✅ Culori calde și plăcute pentru ochi (albastru cald, gri cald, verde cald)
- ✅ Margini interioare generoase (4% padding) pentru carduri
- ✅ Protecție completă împotriva overflow-ului textului
- ✅ Control vocal continuu (hands-free) cu comenzi simple
- ✅ Suport complet mouse (click pe carduri)
- ✅ Navigare tastatură (1/2/3, Enter, ESC)
- ✅ Audio natural în română (Google TTS)
- ✅ Feedback audio și vizual instant
- ✅ Fără diacritice în tot textul
- ✅ Revenire automată la meniu după închidere aplicație

**Comenzi vocale:**
- "1"/"unu" → Traducere semne în text
- "2"/"doi" → Mod demo
- "3"/"trei" → Ieșire
- "stop" → Oprește audio complet
- "start" → Pornește audio
- "ajutor" → Repetă opțiunile

### 2. realtime.py - APLICAȚIA DE TRADUCERE OPTIMIZATĂ

**Optimizări implementate:**
- ✅ Suport pentru camere USB cu detectie automată (testează indexurile 0,1,2,3)
- ✅ Suport pentru camere IP (telefon cu DroidCam)
- ✅ Parametru `--camera` acceptă atât numere cât și URL-uri
- ✅ Mesaje de eroare utile cu instrucțiuni de rezolvare
- ✅ Detectie automată a camerei funcționale
- ✅ Demo mode optimizat cu statistici detaliate

**Utilizare:**
```bash
# Mod normal cu model antrenat
python realtime.py

# Mod demo pentru testare
python realtime.py --demo

# Cu camera IP (telefon)
python realtime.py --camera http://192.168.1.100:4747/video

# Cu prag de confidență personalizat
python realtime.py --threshold 0.7
```

### 3. utils.py - UTILITĂȚI MEDIAPIPE OPTIMIZATE

**Optimizări MediaPipe:**
- ✅ Parametri optimizați: `model_complexity=2`, `refine_face_landmarks=True`
- ✅ Praguri de detectie mai sensibile (0.3 în loc de 0.5)
- ✅ Funcție îmbunătățită `draw_landmarks_on_frame()` cu:
  - Puncte mari și vizibile pentru mâini (8px pentru vârfuri degete)
  - Puncte colorate distinct pentru ochi (galben), sprâncene (magenta), buze (roșu)
  - Analiza în timp real: ochi deschiși/închiși, gură deschisă/închisă
  - Informații de detectie clare chiar și fără față
  - Statistici detaliate în demo mode

### 4. Fișiere de suport create

**check_camera.py** - Test simplu pentru camere
**check_camera_dshow.py** - Test avansat cu DirectShow și IP camere
**run_with_python311.bat** - Script pentru compatibilitate MediaPipe

## DEPENDENȚE COMPLETE

### requirements.txt - ACTUALIZAT COMPLET
```
# Dependente principale
opencv-python>=4.8.0
mediapipe==0.10.14
tensorflow==2.15.0
keras==2.15.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
protobuf>=3.20.0,<5.0.0

# Dependente pentru interfata audio si vocala
gtts>=2.3.0
pygame>=2.5.0
SpeechRecognition>=3.10.0
pyaudio>=0.2.11

# Dependente pentru procesarea datelor
pandas>=2.0.0
matplotlib>=3.7.0

# Dependente optionale
imgaug>=0.4.0
```

### Scripturi de instalare create:
- **install_dependencies.bat** - Pentru Windows
- **install_dependencies.sh** - Pentru Linux/Mac
- **check_dependencies.py** - Verificare dependențe

## INSTALARE ȘI CONFIGURARE

### 1. Cerințe sistem
- **Python 3.10 sau 3.11** (OBLIGATORIU pentru MediaPipe 0.10.14)
- Windows 10/11, Linux, sau macOS
- Camera web sau telefon cu DroidCam
- Microfon pentru control vocal

### 2. Instalare dependențe

**Windows:**
```bash
# Rulează scriptul automat
install_dependencies.bat

# Sau manual
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Rulează scriptul automat
bash install_dependencies.sh

# Sau manual
pip3 install -r requirements.txt
```

### 3. Verificare instalare
```bash
python check_dependencies.py
```

### 4. Rulare aplicație
```bash
# Interfața principală
python start_menu.py

# Sau cu Python 3.11 specific
python3.11 start_menu.py
run_with_python311.bat
```

## PROBLEME COMUNE ȘI SOLUȚII

### 1. MediaPipe nu funcționează
**Problemă:** `AttributeError: module 'mediapipe' has no attribute 'solutions'`
**Soluție:** 
- Folosiți Python 3.10 sau 3.11
- Instalați MediaPipe 0.10.14: `pip install mediapipe==0.10.14`

### 2. Camera nu se deschide
**Problemă:** "EROARE: Nu se poate deschide camera!"
**Soluții:**
- Rulați `python check_camera_dshow.py` pentru diagnostic
- Pentru telefon: instalați DroidCam și folosiți IP camera
- Verificați că alte aplicații nu folosesc camera

### 3. PyAudio nu se instalează pe Windows
**Soluții:**
```bash
pip install pipwin
pipwin install pyaudio
```
Sau descărcați wheel de la: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### 4. TensorFlow/Keras import errors
**Problemă:** `AttributeError: module 'tensorflow' has no attribute 'keras'`
**Soluție:** Folosiți versiunile specificate:
```bash
pip install tensorflow==2.15.0 keras==2.15.0
```

## FUNCȚIONALITĂȚI IMPLEMENTATE

### Interfața principală (start_menu.py)
- [x] Design modern și responsive
- [x] Text centrat și optimizat
- [x] Culori calde și plăcute
- [x] Control vocal hands-free
- [x] Suport mouse și tastatură
- [x] Audio TTS în română
- [x] Fără diacritice
- [x] Protecție overflow text

### Aplicația de traducere (realtime.py)
- [x] Suport camere multiple (USB, IP)
- [x] Detectie automată camere
- [x] Demo mode optimizat
- [x] Mesaje de eroare utile
- [x] Statistici detaliate

### Detectie landmarks (utils.py)
- [x] Parametri MediaPipe optimizați
- [x] Puncte mari și vizibile
- [x] Analiza expresiilor faciale
- [x] Informații detectie clare
- [x] Funcționează fără față

### Suport și utilități
- [x] Scripturi instalare automată
- [x] Verificare dependențe
- [x] Test camere avansate
- [x] Compatibilitate Python 3.11
- [x] Documentație completă

## TESTARE

### 1. Test interfață
```bash
python start_menu.py
```
Verificați:
- Interfața se deschide corect
- Textul este centrat și lizibil
- Controlul vocal funcționează
- Mouse-ul funcționează pe carduri

### 2. Test camere
```bash
python check_camera_dshow.py
```
Verificați:
- Camera USB este detectată
- Imaginea este clară
- FPS-ul este stabil

### 3. Test demo mode
```bash
python realtime.py --demo
```
Verificați:
- Landmarks-urile sunt vizibile
- Punctele sunt mari și colorate
- Statisticile se actualizează
- Detectia funcționează fără față

## DEZVOLTARE ULTERIOARĂ

### Îmbunătățiri posibile:
1. **Model neural mai avansat** - Transformer cu atenție temporală
2. **Mai multe cuvinte** - Extinderea vocabularului
3. **Detectie gesturi complexe** - Secvențe de mișcări
4. **Interfață web** - Versiune browser
5. **Mobile app** - Aplicație pentru telefon
6. **Cloud deployment** - Serviciu online

### Structura pentru dezvoltare:
```bash
# Instalează dependențele de dezvoltare
pip install -r requirements-dev.txt

# Formatează codul
black *.py
isort *.py

# Rulează testele
pytest tests/

# Generează documentația
sphinx-build docs/ docs/_build/
```

## CONTACT ȘI SUPORT

Pentru probleme sau întrebări:
1. Verificați secțiunea "Probleme comune și soluții"
2. Rulați `python check_dependencies.py` pentru diagnostic
3. Verificați că folosiți Python 3.10/3.11
4. Testați camera cu `python check_camera_dshow.py`

## LICENȚĂ ȘI UTILIZARE

Acest proiect este dezvoltat pentru uz educațional și de cercetare în domeniul traducerii limbajului semnelor românesc.

---

**IMPORTANT:** Acest document conține toate informațiile necesare pentru recrearea completă a proiectului. Urmați pașii în ordine pentru o instalare reușită.