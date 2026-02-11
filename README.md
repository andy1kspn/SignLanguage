Comenzi complete

  # 1. Instalare dependente
  pip install -r requirements.txt

  # 2. (Optional) Daca ai eroare protobuf, forteaza versiunea:
  pip install "protobuf>=5.28.0"

  # 3. Generare date sintetice (pentru test rapid)
  python generate_synthetic_data.py

  # 4. Antrenare model
  python train_model.py --arch lstm_attention --epochs 50 --augment 3

  # 5a. Mod DEMO (vizualizare landmarks, fara model)
  python realtime.py --demo

  # 5b. Mod COMPLET (inferenta cu modelul antrenat)
  python realtime.py --threshold 0.7

  # --- COLECTARE DATE REALE (inlocuieste datele sintetice) ---
  # 6. Colecteaza semne reale cu webcam
  python collect_data.py --sign buna --samples 50
  python collect_data.py --sign multumesc --samples 50
  python collect_data.py --all  # toate semnele

  # 7. Re-antreneaza pe date reale
  python train_model.py --arch lstm_attention --epochs 100 --augment 5


  <img width="814" height="653" alt="image" src="https://github.com/user-attachments/assets/43d4fa21-3566-408f-b598-0b5183f8842a" />
