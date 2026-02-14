"""
Meniu vocal cu interfață vizuală - folosind Whisper pentru recunoaștere în română.

Caracteristici:
- Interfață vizuală modernă (OpenCV)
- Recunoaștere vocală în română (Whisper)
- Text-to-Speech în română (gTTS)
- Control mouse (click pe opțiuni)
- Control tastatură (1/2/3, Enter, ESC)
- Control vocal (Whisper)
- Mașină de stări cu 2 stări (DICTOR / ASCULTARE)
"""

import cv2
import numpy as np
import subprocess
import sys
import threading
import time
import os
import tempfile

# Verificare dependențe
try:
    import whisper
    import sounddevice as sd
    import soundfile as sf
    from gtts import gTTS
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"ATENȚIE: Lipsesc dependențe pentru control vocal: {e}")
    print("Instalează: pip install openai-whisper gtts pygame sounddevice soundfile")
    WHISPER_AVAILABLE = False


class MeniuVocalVizual:
    """Meniu accesibil cu interfață vizuală și control vocal Whisper."""
    
    def __init__(self):
        """Inițializare meniu."""
        # Dimensiuni fereastră
        self.width = 1280
        self.height = 720
        
        self.selected_option = 0
        self.options = [
            "Traducere Semne -> Text (Camera Live)",
            "Mod Demo Imbunatatit - Testare Camera",
            "Iesire din Aplicatie"
        ]
        
        # Coordonate butoane pentru mouse
        self.button_coords = []
        
        # Mașina de stări
        self.is_speaking = False  # False = ASCULTARE, True = DICTOR
        self.system_active = False
        self.should_execute = False
        
        # Control audio
        self.audio_enabled = True
        self.temp_dir = tempfile.gettempdir()
        
        # Feedback vizual
        self.last_voice_command = ""
        self.voice_command_time = 0
        self.listening_indicator = False
        self.recording = False
        
        # Inițializare Whisper
        if WHISPER_AVAILABLE:
            print("Încărcare model Whisper (base pentru acuratețe mai bună)...")
            self.whisper_model = whisper.load_model("base")  # base e mai precis decât tiny
            self.sample_rate = 16000
            print("✓ Whisper încărcat")
        else:
            self.whisper_model = None
    
    def speak(self, text):
        """Redă mesaj TTS (non-blocking pentru UI)."""
        if not self.audio_enabled:
            return
        
        self.is_speaking = True
        
        def speak_thread():
            temp_file = os.path.join(self.temp_dir, f"tts_{int(time.time()*1000)}.mp3")
            
            try:
                tts = gTTS(text=text, lang='ro', slow=False, tld='ro')
                tts.save(temp_file)
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.set_volume(0.9)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    if not self.audio_enabled:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)
                
                pygame.mixer.music.unload()
                time.sleep(0.1)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            except Exception as e:
                print(f"EROARE TTS: {e}")
            
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
    
    def stop_audio(self):
        """Oprește audio instant."""
        self.audio_enabled = False
        try:
            pygame.mixer.music.stop()
        except:
            pass
        time.sleep(0.2)
        self.audio_enabled = True
    
    def listen_for_command(self):
        """Ascultă și recunoaște comandă vocală cu Whisper - OPTIMIZAT."""
        if not WHISPER_AVAILABLE or not self.whisper_model:
            return None
        
        self.listening_indicator = True
        self.recording = True
        
        try:
            print("🎙️  Înregistrare (2s)...", end=" ", flush=True)
            
            # Înregistrare audio MAI SCURTĂ (2 secunde în loc de 3)
            audio = sd.rec(
                int(2 * self.sample_rate),  # 2 secunde
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            self.recording = False
            print("Procesare...", end=" ", flush=True)
            
            # Salvare temporară
            temp_audio = os.path.join(self.temp_dir, f"audio_{int(time.time()*1000)}.wav")
            sf.write(temp_audio, audio, self.sample_rate)
            
            # Transcriere cu Whisper - OPTIMIZAT
            result = self.whisper_model.transcribe(
                temp_audio,
                language='ro',
                fp16=False,
                beam_size=1,  # Mai rapid (default e 5)
                best_of=1,    # Mai rapid (default e 5)
                temperature=0.0  # Mai deterministic
            )
            text = result['text'].strip().lower()
            
            # Curățare
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            self.listening_indicator = False
            
            if text:
                print(f"✓ '{text}'")
            else:
                print("✗ Nimic")
            
            return text if text else None
        
        except Exception as e:
            print(f"✗ Eroare: {e}")
            self.listening_indicator = False
            self.recording = False
            return None
    
    def handle_voice_command(self, command):
        """Procesare comandă vocală - OPTIMIZAT cu mai multe variante."""
        if not command:
            return
        
        self.last_voice_command = command
        self.voice_command_time = time.time()
        
        print(f"🎤 Comandă: '{command}'")
        
        # STOP - oprește DOAR dictorul dacă vorbește
        stop_keywords = ["stop", "oprește", "opreste", "stai", "gata", "taci"]
        if any(kw in command for kw in stop_keywords):
            if self.is_speaking:
                print("   → STOP (opresc dictorul)")
                self.stop_audio()
            else:
                print("   → STOP (dictorul nu vorbește)")
            return
        
        # START - pornește dictorul dacă tace (anunță opțiunile)
        start_keywords = ["start", "pornește", "porneste", "începe", "incepe", "hai"]
        if any(kw in command for kw in start_keywords):
            if not self.is_speaking:
                print("   → START (pornesc dictorul)")
                self.system_active = True
                welcome_msg = (
                    "Bună ziua! Aveți 3 opțiuni disponibile. "
                    "Spuneți UNU pentru prima opțiune. "
                    "Spuneți DOI pentru a doua opțiune. "
                    "Spuneți TREI pentru a treia opțiune."
                )
                self.speak(welcome_msg)
            else:
                print("   → START (dictorul deja vorbește)")
            return
        
        # Ignoră alte comenzi în timpul TTS
        if self.is_speaking:
            print("   → Ignorat (dictorul vorbește)")
            return
        
        # Pentru comenzi de opțiuni, activează automat sistemul dacă nu e activ
        if not self.system_active:
            print("   → Activare automată sistem")
            self.system_active = True
        
        # OPȚIUNEA 1 (variante multiple)
        opt1_keywords = ["unu", "1", "prima", "întâi", "intai", "traducere", "camera"]
        if any(kw in command for kw in opt1_keywords):
            print("   → Opțiunea 1")
            self.selected_option = 0
            self.speak("Opțiunea 1")
            time.sleep(1.0)
            self.should_execute = True
            return
        
        # OPȚIUNEA 2 (variante multiple)
        opt2_keywords = ["doi", "2", "doua", "a doua", "demo", "test"]
        if any(kw in command for kw in opt2_keywords):
            print("   → Opțiunea 2")
            self.selected_option = 1
            self.speak("Opțiunea 2")
            time.sleep(1.0)
            self.should_execute = True
            return
        
        # OPȚIUNEA 3 (variante multiple)
        opt3_keywords = ["trei", "3", "treia", "a treia", "ieșire", "iesire", "închide", "inchide", "exit"]
        if any(kw in command for kw in opt3_keywords):
            print("   → Opțiunea 3")
            self.selected_option = 2
            self.speak("La revedere!")
            time.sleep(1.0)
            self.should_execute = True
            return
        
        # Comandă necunoscută
        print(f"   → Necunoscut: '{command}'")
    
    def voice_control_loop(self):
        """Loop pentru control vocal (rulează pe thread separat) - OPTIMIZAT."""
        if not WHISPER_AVAILABLE:
            return
        
        print("🎙️  Control vocal activ (Whisper - model base)")
        print("💡 Sfat: Vorbește clar și aproape de microfon")
        print("💡 Comenzi: START, UNU, DOI, TREI, STOP\n")
        
        consecutive_empty = 0
        
        while not self.should_execute:
            try:
                # Nu asculta în timpul TTS
                if self.is_speaking:
                    time.sleep(0.5)
                    continue
                
                command = self.listen_for_command()
                
                if command:
                    consecutive_empty = 0
                    self.handle_voice_command(command)
                else:
                    consecutive_empty += 1
                    # Dacă nu detectează nimic de 3 ori, pauză mai lungă
                    if consecutive_empty >= 3:
                        time.sleep(1.0)
                        consecutive_empty = 0
                
            except Exception as e:
                print(f"EROARE voice loop: {e}")
                time.sleep(1.0)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback pentru mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (y_start, y_end) in enumerate(self.button_coords):
                if y_start <= y <= y_end and 100 <= x <= self.width - 100:
                    self.selected_option = i
                    self.stop_audio()
                    
                    messages = {
                        0: "Pornesc traducerea",
                        1: "Pornesc demo-ul",
                        2: "La revedere!"
                    }
                    if i in messages:
                        self.speak(messages[i])
                        time.sleep(1.2)
                    
                    self.should_execute = True
                    return
    
    def draw_menu(self):
        """Desenare meniu vizual - DESIGN MODERN."""
        # Fundal gradient modern (albastru închis -> negru)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.height):
            # Gradient de la albastru închis la negru
            ratio = i / self.height
            r = int(15 * (1 - ratio))
            g = int(25 * (1 - ratio))
            b = int(45 * (1 - ratio))
            frame[i, :] = (b, g, r)
        
        # Efecte de lumină de fundal (cercuri blur)
        overlay = frame.copy()
        cv2.circle(overlay, (200, 150), 300, (80, 40, 20), -1)
        cv2.circle(overlay, (self.width - 200, self.height - 150), 250, (40, 20, 80), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Header modern cu efect glassmorphism
        header_height = int(self.height * 0.25)
        
        # Box semi-transparent pentru header
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (0, 0), (self.width, header_height),
                     (40, 40, 40), -1)
        frame = cv2.addWeighted(frame, 0.6, header_overlay, 0.4, 0)
        
        # Linie accent gradient (albastru -> cyan)
        for i in range(6):
            color_ratio = i / 6
            b = int(255 * (1 - color_ratio) + 255 * color_ratio)
            g = int(100 * (1 - color_ratio) + 255 * color_ratio)
            r = int(50 * (1 - color_ratio) + 0 * color_ratio)
            cv2.rectangle(frame, (0, i), (self.width, i + 1), (b, g, r), -1)
        
        # Titlu principal cu efect glow
        title = "TRADUCATOR LIMBAJ SEMNE"
        title_scale = 1.8
        title_thickness = 3
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)[0]
        title_x = (self.width - title_size[0]) // 2
        title_y = int(header_height * 0.35)
        
        # Glow effect (multiple layers)
        for offset in range(8, 0, -1):
            alpha = 0.15
            glow_color = (int(255 * alpha), int(200 * alpha), int(100 * alpha))
            cv2.putText(frame, title, (title_x, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, title_scale, glow_color, title_thickness + offset, cv2.LINE_AA)
        
        # Text principal
        cv2.putText(frame, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
        
        # Subtitlu modern
        subtitle = "Sistem Inteligent de Recunoastere Vocala"
        subtitle_scale = 0.7
        subtitle_thickness = 2
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness)[0]
        subtitle_x = (self.width - subtitle_size[0]) // 2
        subtitle_y = int(header_height * 0.55)
        
        cv2.putText(frame, subtitle, (subtitle_x, subtitle_y),
                   cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (150, 200, 255), subtitle_thickness, cv2.LINE_AA)
        
        # Status vocal modern cu animație
        if WHISPER_AVAILABLE:
            status_y = int(header_height * 0.80)
            
            if self.recording:
                status = "INREGISTREZ"
                status_icon = "🔴"
                status_color = (0, 100, 255)
                pulse = int(20 * abs(np.sin(time.time() * 5)))  # Pulsație
            elif self.listening_indicator:
                status = "PROCESEZ"
                status_icon = "⚙️"
                status_color = (0, 200, 255)
                pulse = 0
            elif not self.system_active:
                status = "Spune START"
                status_icon = "🎙️"
                status_color = (100, 255, 150)
                pulse = int(15 * abs(np.sin(time.time() * 2)))
            elif self.is_speaking:
                status = "VORBESC"
                status_icon = "🔊"
                status_color = (200, 150, 255)
                pulse = 0
            else:
                status = "ASCULT"
                status_icon = "👂"
                status_color = (100, 255, 150)
                pulse = int(10 * abs(np.sin(time.time() * 3)))
            
            # Box status cu efect glassmorphism
            status_text = f"{status_icon} {status}"
            status_scale = 0.8
            status_thickness = 2
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_thickness)[0]
            status_x = (self.width - status_size[0]) // 2
            
            box_padding = 20 + pulse
            box_x1 = status_x - box_padding
            box_x2 = status_x + status_size[0] + box_padding
            box_y1 = status_y - 30
            box_y2 = status_y + 15
            
            # Box semi-transparent
            status_overlay = frame.copy()
            cv2.rectangle(status_overlay, (box_x1, box_y1), (box_x2, box_y2),
                         (30, 30, 30), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.7, status_overlay, 0.3, 0)
            
            # Border cu culoare status
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                         status_color, 2, cv2.LINE_AA)
            
            # Text status
            cv2.putText(frame, status_text, (status_x, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_color, status_thickness, cv2.LINE_AA)
        
        # Opțiuni meniu cu design modern (cards)
        self.button_coords = []
        start_y = header_height + 60
        button_height = 90
        button_spacing = 25
        button_margin = 120
        
        for i, option in enumerate(self.options):
            y_pos = start_y + i * (button_height + button_spacing)
            
            # Culori moderne
            if i == self.selected_option:
                # Gradient albastru pentru selecție
                bg_color1 = (255, 150, 50)   # Portocaliu
                bg_color2 = (255, 100, 30)   # Portocaliu închis
                text_color = (255, 255, 255)
                border_color = (255, 200, 100)
                shadow_offset = 8
            else:
                # Gri semi-transparent
                bg_color1 = (60, 60, 60)
                bg_color2 = (40, 40, 40)
                text_color = (200, 200, 200)
                border_color = (100, 100, 100)
                shadow_offset = 4
            
            # Umbră card
            shadow_overlay = frame.copy()
            cv2.rectangle(shadow_overlay,
                         (button_margin + shadow_offset, y_pos + shadow_offset),
                         (self.width - button_margin + shadow_offset, y_pos + button_height + shadow_offset),
                         (0, 0, 0), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.85, shadow_overlay, 0.15, 0)
            
            # Card cu gradient
            card_overlay = frame.copy()
            for j in range(button_height):
                ratio = j / button_height
                b = int(bg_color1[0] * (1 - ratio) + bg_color2[0] * ratio)
                g = int(bg_color1[1] * (1 - ratio) + bg_color2[1] * ratio)
                r = int(bg_color1[2] * (1 - ratio) + bg_color2[2] * ratio)
                cv2.rectangle(card_overlay,
                             (button_margin, y_pos + j),
                             (self.width - button_margin, y_pos + j + 1),
                             (b, g, r), -1)
            frame = cv2.addWeighted(frame, 0.3, card_overlay, 0.7, 0)
            
            # Border card
            cv2.rectangle(frame,
                         (button_margin, y_pos),
                         (self.width - button_margin, y_pos + button_height),
                         border_color, 2, cv2.LINE_AA)
            
            # Număr opțiune în cerc
            number = str(i + 1)
            circle_x = button_margin + 50
            circle_y = y_pos + button_height // 2
            circle_radius = 25
            
            # Cerc număr
            cv2.circle(frame, (circle_x, circle_y), circle_radius, border_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (circle_x, circle_y), circle_radius, text_color, 2, cv2.LINE_AA)
            
            # Număr
            number_scale = 1.2
            number_thickness = 2
            number_size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, number_scale, number_thickness)[0]
            number_x = circle_x - number_size[0] // 2
            number_y = circle_y + number_size[1] // 2
            cv2.putText(frame, number, (number_x, number_y),
                       cv2.FONT_HERSHEY_SIMPLEX, number_scale, text_color, number_thickness, cv2.LINE_AA)
            
            # Text opțiune
            text_scale = 0.8
            text_thickness = 2
            text_x = button_margin + 100
            text_y = y_pos + button_height // 2 + 8
            cv2.putText(frame, option, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness, cv2.LINE_AA)
            
            self.button_coords.append((y_pos, y_pos + button_height))
        
        # Footer cu instrucțiuni moderne
        footer_y = self.height - 80
        
        # Box semi-transparent pentru footer
        footer_overlay = frame.copy()
        cv2.rectangle(footer_overlay, (0, footer_y - 20), (self.width, self.height),
                     (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.7, footer_overlay, 0.3, 0)
        
        instructions = [
            ("🖱️", "Click pe optiune", (100, 255, 200)),
            ("⌨️", "Taste: 1/2/3, Enter, ESC", (255, 200, 100)),
            ("🎤", "Vocal: START, UNU, DOI, TREI", (200, 150, 255))
        ]
        
        total_width = sum([cv2.getTextSize(f"{icon} {text}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for icon, text, _ in instructions]) + 100
        start_x = (self.width - total_width) // 2
        
        for icon, text, color in instructions:
            instr_text = f"{icon} {text}"
            cv2.putText(frame, instr_text, (start_x, footer_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            text_width = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            start_x += text_width + 50
        
        # Ultima comandă vocală (dacă există)
        if self.last_voice_command and (time.time() - self.voice_command_time) < 3:
            cmd_text = f"Ultima comanda: {self.last_voice_command}"
            cmd_scale = 0.6
            cmd_thickness = 2
            cmd_size = cv2.getTextSize(cmd_text, cv2.FONT_HERSHEY_SIMPLEX, cmd_scale, cmd_thickness)[0]
            cmd_x = (self.width - cmd_size[0]) // 2
            cmd_y = footer_y - 10
            
            # Box pentru comandă
            cmd_overlay = frame.copy()
            cv2.rectangle(cmd_overlay,
                         (cmd_x - 15, cmd_y - 25),
                         (cmd_x + cmd_size[0] + 15, cmd_y + 5),
                         (50, 50, 50), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.7, cmd_overlay, 0.3, 0)
            
            cv2.putText(frame, cmd_text, (cmd_x, cmd_y),
                       cv2.FONT_HERSHEY_SIMPLEX, cmd_scale, (100, 255, 100), cmd_thickness, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Loop principal cu interfață vizuală."""
        cv2.namedWindow("Meniu Vocal", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Meniu Vocal", self.width, self.height)
        cv2.setMouseCallback("Meniu Vocal", self.mouse_callback)
        
        # Pornire thread control vocal
        if WHISPER_AVAILABLE:
            voice_thread = threading.Thread(target=self.voice_control_loop, daemon=True)
            voice_thread.start()
        
        print("\n" + "="*70)
        print("MENIU VOCAL VIZUAL ACTIV")
        print("="*70)
        print("\nControale:")
        print("  Mouse: Click pe opțiune")
        print("  Tastatură: 1/2/3, Enter, ESC")
        print("  Vocal: Spune START apoi UNU/DOI/TREI")
        print("="*70 + "\n")
        
        while not self.should_execute:
            # Desenare meniu
            frame = self.draw_menu()
            cv2.imshow("Meniu Vocal", frame)
            
            # Procesare taste
            key = cv2.waitKey(100) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.selected_option = 0
            elif key == ord('2'):
                self.selected_option = 1
            elif key == ord('3'):
                self.selected_option = 2
            elif key == 13:  # Enter
                self.should_execute = True
        
        cv2.destroyAllWindows()
        
        # Execută opțiunea selectată
        if self.should_execute:
            if self.selected_option == 0:
                print("\n✓ Lansare: Traducere semne")
                subprocess.run([sys.executable, "realtime.py"])
            elif self.selected_option == 1:
                print("\n✓ Lansare: Demo îmbunătățit")
                subprocess.run([sys.executable, "demo_enhanced.py"])
            elif self.selected_option == 2:
                print("\n✓ Ieșire")
        
        pygame.quit()


def main():
    """Punct de intrare principal."""
    menu = MeniuVocalVizual()
    menu.run()


if __name__ == "__main__":
    main()
