# Traducător Limbaj Semne Românesc (LSR)

## Start Rapid

Pentru a porni demo-ul îmbunătățit direct:

```bash
python start.py
```

## Funcționalități Demo Îmbunătățit

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

## Alte Opțiuni

### Meniu complet (cu control vocal):
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

## Structură Fișiere

- `start.py` - Start rapid demo îmbunătățit
- `demo_enhanced.py` - Demo cu funcționalități avansate
- `start_menu.py` - Meniu principal cu control vocal
- `realtime.py` - Traducere în timp real
- `config.py` - Configurație centrală
- `model.py` - Arhitecturi model
- `train_model.py` - Antrenare model
- `collect_data.py` - Colectare date
- `augmentation.py` - Augmentare date
- `generate_synthetic_data.py` - Generare date sintetice

## Cerințe

```bash
pip install -r requirements.txt
```
