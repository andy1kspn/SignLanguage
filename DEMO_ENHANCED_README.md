# Mod Demo Îmbunătățit - LSR

## Prezentare Generală

Modul demo îmbunătățit oferă o experiență avansată de vizualizare și analiză pentru sistemul de recunoaștere a limbajului semnelor românesc (LSR). Include funcționalități interactive, vizualizări multiple și instrumente de analiză în timp real.

## Pornire Rapidă

### Din Meniul Principal
```bash
python start_menu.py
```
Selectează opțiunea **3** (Mod Demo Îmbunătățit)

### Direct
```bash
python demo_enhanced.py
```

### Cu IP Camera
```bash
python demo_enhanced.py --camera http://192.168.1.100:4747/video
```

## Funcționalități Principale

### 🎥 Înregistrare și Capturi
- **R** - Pornește/Oprește înregistrarea video (salvat în `recordings/`)
- **S** - Salvează screenshot (salvat în `screenshots/`)
- Format: MP4 pentru video, PNG pentru imagini
- Indicator roșu "REC" când înregistrarea este activă

### 🎨 Vizualizări
- **L** - Toggle landmarks (puncte de referință MediaPipe)
- **T** - Toggle trails (urmă colorată pentru mâini)
- **H** - Toggle heatmap (hartă termică pentru activitate)
- **U** - Toggle UI (ascunde/arată interfața)
- **F** - Toggle fullscreen
- **X** - Curăță trails și heatmap

### 🎯 Analiză Calitate
Sistem automat de evaluare a calității detecției:
- **Scor 90-100**: Excelent - Calitate perfectă
- **Scor 70-89**: Bine - Calitate bună
- **Scor 50-69**: Acceptabil - Poziționează-te mai bine
- **Scor 0-49**: Slab - Asigură-te că mâinile sunt vizibile

Afișează:
- Scor curent și mediu
- Detectare mâini (stânga/dreaptă)
- Detectare pose și față
- Features active din total
- Feedback în timp real

### 📊 Statistici și Grafice
- **I** - Toggle grafic FPS în timp real
- Statistici sesiune (afișate la ieșire):
  - Durată totală
  - Frame-uri procesate
  - Rata de detecție
  - Calitate medie
  - FPS mediu
  - Număr screenshots

### 🎨 Teme Vizuale
- **C** - Schimbă tema curentă
- **default**: Culori standard, bună vizibilitate
- **dark**: Fundal întunecat, culori atenuate
- **high_contrast**: Contrast maxim pentru vizibilitate
- **colorblind**: Optimizat pentru daltonism

### 📚 Comparație Dataset
- **D** - Toggle comparație cu dataset
- Compară landmarks-urile curente cu gesturi din dataset
- Afișează cel mai apropiat gest și similaritatea (%)
- Util pentru verificarea consistenței gesturilor

### 🎮 Mini-Joc
- **G** - Pornește/Oprește mini-jocul "Atinge Punctele"
- Durată: 30 secunde
- Folosește degetul arătător (mâna dreaptă) pentru a atinge țintele
- Scor final afișat la terminare

### 📋 Meniu Interactiv
- **M** - Toggle meniu cu toate comenzile
- Afișat în colțul dreapta-sus
- Lista completă de taste și funcții

### 🔍 Zoom și Control
- **+/=** - Zoom in (până la 200%)
- **-/_** - Zoom out (până la 50%)
- **Q/ESC** - Ieșire din demo

## Structura Fișierelor Generate

```
project/
├── recordings/          # Înregistrări video