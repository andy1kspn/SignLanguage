# Traducător Limbaj Semne Românesc (LSR)

## 🚀 Start Rapid

### Meniu Vocal Modern (RECOMANDAT)
```bash
python meniu_vocal_vizual.py
```
- 🎨 UI modern cu design glassmorphism
- 🎤 Control vocal în română (Whisper)
- 🖱️ Control mouse + tastatură
- ✨ Animații și efecte vizuale

**Comenzi vocale:** "start", "unu", "doi", "trei", "stop"

### Demo Îmbunătățit
```bash
python start.py
```

---

## 🎤 Meniu Vocal - Caracteristici

### Control Multiplu
- **Vocal**: Whisper (recunoaștere offline în română)
  - "start" → Pornește dictorul
  - "unu/doi/trei" → Selectează opțiunea
  - "stop" → Oprește dictorul
- **Mouse**: Click pe opțiune
- **Tastatură**: 1/2/3, Enter, ESC

### UI Modern
- Design glassmorphism cu gradient
- Animații fluide (pulsație, glow)
- Status box animat cu culori per stare
- Cards stilizate pentru opțiuni
- Efecte de lumină de fundal

### Instalare Meniu Vocal
```bash
pip install -r requirements.txt
```
La prima rulare, Whisper descarcă automat modelul (~142 MB).

---

## 🎮 Demo Îmbunătățit - Funcționalități

### Taste disponibile:
- **Q / ESC** - Ieșire
- **R** - Pornește/Oprește înregistrare video
- **S** - Screenshot
- **+/-** - Zoom in/out
- **L** - Toggle landmarks (afișare/ascundere)
- **T** - Toggle trails (urmă mâini)
- **H** - Toggle heatmap (hartă activitate)
- **U** - Toggle UI (interfață)
- **I** - Toggle grafic FPS
- **D** - Toggle comparație cu dataset
- **F** - Fullscreen
- **X** - Curăță trails și heatmap

### Caracteristici:
- ✅ Analiză calitate în timp real
- ✅ Înregistrare video (salvat în `recordings/`)
- ✅ Screenshot-uri (salvate în `screenshots/`)
- ✅ Vizualizare trails pentru mâini
- ✅ Heatmap activitate
- ✅ Grafic FPS în timp real
- ✅ Comparație cu gesturi din dataset
- ✅ Statistici detaliate la final

---

## 📋 Alte Opțiuni

### Meniu complet (vechi, cu Google Speech):
```bash
python start_menu.py
```

### Traducere în timp real (cu model antrenat):
```bash
python realtime.py
```

### Colectare date noi:
```bash
python collect_data.py
```

### Antrenare model:
```bash
python train_model.py
```

---

## 📁 Structură Fișiere

### Aplicații Principale
- `meniu_vocal_vizual.py` - **Meniu vocal modern cu UI** ⭐
- `start.py` - Start rapid demo îmbunătățit
- `demo_enhanced.py` - Demo cu funcționalități avansate
- `start_menu.py` - Meniu vechi cu Google Speech
- `realtime.py` - Traducere în timp real

### Module Core
- `config.py` - Configurație centrală
- `model.py` - Arhitecturi model
- `train_model.py` - Antrenare model
- `collect_data.py` - Colectare date
- `augmentation.py` - Augmentare date
- `generate_synthetic_data.py` - Generare date sintetice
- `utils.py` - Funcții utilitare

---

## 🔧 Cerințe

```bash
pip install -r requirements.txt
```

### Dependențe Principale
- **TensorFlow** - Model de recunoaștere
- **MediaPipe** - Detectare landmarks
- **OpenCV** - Procesare video și UI
- **Whisper** - Recunoaștere vocală (română)
- **gTTS** - Text-to-Speech (română)
- **pygame** - Redare audio

---

## 💡 Recomandări

1. **Pentru utilizare normală**: `python meniu_vocal_vizual.py`
2. **Pentru testare rapidă**: `python start.py`
3. **Pentru antrenare model**: `python train_model.py`

---

**Mult succes! 🎉**
