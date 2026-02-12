"""
Meniu de start accesibil OPTIMIZAT pentru aplicatia de traducere limbaj semne.

Caracteristici:
- Interfata Full HD (1920x1080) cu design modern
- Audio natural in romana (Google TTS)
- Control vocal continuu (hands-free)
- Suport complet mouse (click pe optiuni)
- Navigare tastatura (1/2/3, Enter, ESC)
- Feedback audio si vizual instant
- Sunet de "gandire" pentru nevazatori
- Comenzi simple: "1", "2", "3", "stop", "start"
- Revenire automata la meniu dupa inchidere aplicatie
"""

import cv2
import numpy as np
import subprocess
import sys
import threading
import time
import os
import tempfile
import random
from pathlib import Path

# Initializare TTS
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("ATENTIE gTTS sau pygame lipsesc. Instalati: pip install gtts pygame")

# Initializare Speech Recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("ATENTIE SpeechRecognition lipseste. Instalati: pip install SpeechRecognition pyaudio")


class AccessibleMenu:
    """Meniu accesibil optimizat cu control vocal, mouse si tastatura."""
    
    def __init__(self):
        """Initializare meniu cu design dinamic si adaptabil."""
        # Dimensiuni dinamice - se adapteaza la ecran
        import tkinter as tk
        root = tk.Tk()
        self.width = root.winfo_screenwidth()
        self.height = root.winfo_screenheight()
        root.destroy()
        
        # Minim pentru lizibilitate
        if self.width < 1200:
            self.width = 1200
        if self.height < 800:
            self.height = 800
            
        self.selected_option = 0
        self.options = [
            "Traducere Semne -> Text (Camera Live)",
            "Mod Demo Imbunatatit - Testare Camera",
            "Iesire din Aplicatie"
        ]
        
        # Coordonate butoane pentru mouse
        self.button_coords = []
        
        # Control audio
        self.audio_playing = False
        self.audio_enabled = True
        self.audio_paused = False  # Pentru stop/start complet
        self.audio_counter = 0
        self.temp_dir = tempfile.gettempdir()
        
        # Control vocal
        self.voice_control_active = True
        self.should_execute = False
        self.thinking_sound_active = False
        
        # Initializare Speech Recognition OPTIMIZAT
        self.speech_available = SPEECH_RECOGNITION_AVAILABLE
        if self.speech_available:
            try:
                self.recognizer = sr.Recognizer()
                # Configurare optima pentru romana - MAI SENSIBIL
                self.recognizer.energy_threshold = 300  # Mult mai sensibil
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                self.recognizer.non_speaking_duration = 0.5
                
                self.microphone = sr.Microphone()
                
                # Calibrare rapida
                print("Calibrare microfon...", end=" ", flush=True)
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print("OK")
                print(f"   Prag energie dupa calibrare: {self.recognizer.energy_threshold}")
            except Exception as e:
                print(f"EROARE Eroare microfon: {e}")
                self.recognizer = None
                self.microphone = None
                self.speech_available = False
        else:
            self.recognizer = None
    
    def speak(self, text, priority=False):
        """
        Vorbeste text folosind Google TTS.
        
        Args:
            text: Textul de rostit
            priority: Daca True, opreste audio-ul curent
        """
        if not TTS_AVAILABLE or not self.audio_enabled or self.audio_paused:
            return
        
        if priority:
            self.stop_audio(silent=True)
        
        self.audio_playing = True
        self.audio_counter += 1
        
        # Foloseste timestamp pentru nume unic
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        temp_file = os.path.join(self.temp_dir, f"lsr_{timestamp}_{random_id}.mp3")

        def speak_thread():
            try:
                if not self.audio_enabled or self.audio_paused:
                    self.audio_playing = False
                    return
                
                # Generare audio rapida
                tts = gTTS(text=text, lang='ro', slow=False, tld='com')
                tts.save(temp_file)

                if not self.audio_enabled or self.audio_paused:
                    self.audio_playing = False
                    self._cleanup_audio_file(temp_file)
                    return

                # Redare cu pygame
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Asteapta finalizare sau oprire
                while pygame.mixer.music.get_busy():
                    if not self.audio_enabled or self.audio_paused:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)

                self._cleanup_audio_file(temp_file)
                
            except Exception as e:
                print(f"EROARE TTS: {e}")
            finally:
                # Asteapta doar 0.2s pentru a permite comenzi imediate
                time.sleep(0.2)
                self.audio_playing = False

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
    
    def _cleanup_audio_file(self, filepath):
        """Sterge fisierul audio temporar."""
        try:
            time.sleep(0.1)
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
    
    def stop_audio(self, silent=False):
        """Opreste instant tot audio-ul."""
        self.audio_enabled = False
        self.audio_playing = False
        self.stop_thinking_sound()
        
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except:
            pass
        
        if not silent:
            print("OK Audio oprit")
        
        # Reactiveaza audio dupa 0.2s
        def reactivate():
            time.sleep(0.2)
            self.audio_enabled = True
        threading.Thread(target=reactivate, daemon=True).start()
    
    def play_thinking_sound(self):
        """Sunet de "gandire" pentru nevazatori (beep periodic)."""
        self.thinking_sound_active = True
        
        def thinking_thread():
            try:
                import winsound
                while self.thinking_sound_active and self.voice_control_active:
                    if not self.audio_playing:
                        winsound.Beep(600, 150)
                    time.sleep(3)
            except:
                pass
        
        threading.Thread(target=thinking_thread, daemon=True).start()
    
    def stop_thinking_sound(self):
        """Opreste sunetul de gandire."""
        self.thinking_sound_active = False
    
    def beep(self, frequency=1000, duration=0.1):
        """Feedback sonor instant."""
        try:
            import winsound
            threading.Thread(
                target=lambda: winsound.Beep(frequency, int(duration * 1000)),
                daemon=True
            ).start()
        except:
            pass
    
    def listen_continuously(self):
        """Ascultare vocala continua - POTI VORBI ORICAND."""
        if not self.recognizer or not self.microphone:
            return
        
        print("Microfon Ascultare vocala activa...")
        consecutive_errors = 0
        
        while self.voice_control_active:
            try:
                # SENSIBILITATE CONSTANTA - poti vorbi oricand
                # NU mai crestem pragul cand aplicatia vorbeste
                
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                
                print("Audio detectat, procesez...")
                
                try:
                    text = self.recognizer.recognize_google(audio, language='ro-RO')
                    text = text.lower().strip()
                    
                    # Filtreaza ecoul aplicatiei
                    app_message_words = ['bun', 'venit', 'optiuni', 'traducere', 'demo', 'iesire', 'imbunatatit']
                    has_valid_command = any(cmd in text for cmd in ['1', '2', '3', 'unu', 'doi', 'trei', 'stop', 'start', 'ajutor'])
                    
                    word_count = len(text.split())
                    is_likely_echo = (word_count > 4 and not has_valid_command and 
                                     any(phrase in text for phrase in app_message_words))
                    
                    if is_likely_echo:
                        print(f"Ignorat (ecou): '{text}'")
                        continue
                    
                    # Afiseaza si proceseaza comanda
                    print(f"Microfon '{text}'")
                    self.process_voice_command(text)
                    consecutive_errors = 0
                    
                except sr.UnknownValueError:
                    print("Nu am inteles")
                    pass
                except sr.RequestError as e:
                    consecutive_errors += 1
                    if consecutive_errors < 3:
                        print(f"Eroare Google Speech (reincerc...)")
                    else:
                        print(f"Eroare persistenta Google Speech")
                    time.sleep(1)
                
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"Eroare ascultare: {e}")
                time.sleep(0.5)
    
    def process_voice_command(self, command):
        """Procesare comenzi vocale optimizata."""
        
        # STOP - prioritate maxima - TACE COMPLET
        if 'stop' in command or 'taci' in command or 'opreste' in command:
            self.audio_paused = True
            self.stop_audio()
            print("Audio OPRIT complet - spune 'start' pentru a reactiva")
            return
        
        # START - REACTIVEAZA AUDIO
        if 'start' in command or 'porneste' in command:
            self.audio_paused = False
            self.audio_enabled = True
            self.play_thinking_sound()
            print("Audio PORNIT")
            self.speak("Audio activat", priority=True)
            return
        
        # Feedback sonor
        self.beep(800, 0.1)
        
        # Detectare optiuni
        if '1' in command or 'unu' in command or command.startswith('un'):
            self._select_and_execute(0, "Traducere")
            return
        
        if '2' in command or 'doi' in command or command.startswith('do'):
            self._select_and_execute(1, "Demo Imbunatatit")
            return
        
        if '3' in command or 'trei' in command or command.startswith('tr') or 'iesire' in command or 'exit' in command:
            self._select_and_execute(2, "Iesire")
            return
        
        # Ajutor
        if 'ajutor' in command or 'help' in command:
            self.stop_audio(silent=True)
            self.speak("Trei optiuni. Unu: traducere. Doi: demo imbunatatit. Trei: iesire.", priority=True)
            return
        
        print(f"Comanda necunoscuta: '{command}'")
    
    def _select_and_execute(self, option_index, option_name):
        """Selecteaza si executa o optiune."""
        self.selected_option = option_index
        print(f"OK Selectat: {option_name}")
        self.stop_audio(silent=True)
        self.beep(1200, 0.2)
        self.should_execute = True
        self.voice_control_active = False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback optimizat pentru mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (y_start, y_end) in enumerate(self.button_coords):
                if y_start <= y <= y_end and 100 <= x <= self.width - 100:
                    self.selected_option = i
                    self.beep(1200, 0.2)
                    print(f"Click -> Optiunea {i+1}")
                    self.stop_audio(silent=True)
                    self.should_execute = True
                    self.voice_control_active = False
                    return
        
        # Hover effect
        if event == cv2.EVENT_MOUSEMOVE:
            # Hover effect
            for i, (y_start, y_end) in enumerate(self.button_coords):
                if y_start <= y <= y_end and 100 <= x <= self.width - 100:
                    return
    
    def draw_menu(self):
        """Desenare meniu dinamic care se adapteaza la orice dimensiune de ecran."""
        # Fundal gradient elegant cu textura subtila
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.height):
            # Gradient vertical elegant cu variatie subtila
            gray_base = int(50 - (i / self.height) * 30)
            # Adauga textura subtila pentru profunzime
            noise = int((i % 3) * 2)
            gray_intensity = max(15, min(70, gray_base + noise))
            frame[i, :] = (gray_intensity, gray_intensity, gray_intensity)
        
        # Header dinamic cu design premium
        header_height = int(self.height * 0.22)
        # Fundal header cu gradient si efect glow
        for i in range(header_height):
            intensity = int(70 - (i / header_height) * 25)
            # Efect glow subtil in centru
            center_distance = abs(i - header_height//2) / (header_height//2)
            glow = max(0, int(8 * (1 - center_distance)))
            frame[i, :] = (intensity + glow//3, intensity + glow//3, intensity + glow//3)
        
        # Linie decorativa sus cu gradient elegant
        line_height = max(8, int(self.height * 0.008))
        for i in range(line_height):
            color_intensity = int(255 * (1 - i / line_height))
            cv2.rectangle(frame, (0, i), (self.width, i + 1), 
                         (color_intensity//3, int(color_intensity * 0.7), color_intensity), -1)
        
        # Titlu principal cu efecte premium - optimizat sa nu iasa de pe ecran
        title = "TRADUCATOR LIMBAJ SEMNE ROMANESC"
        max_title_width = self.width * 0.95  # Maxim 95% din latime
        
        title_scale = min(self.width / 1000, self.height / 700) * 1.9
        title_thickness = max(3, int(title_scale * 2))
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
        
        # Rescaleaza daca e prea mare
        if title_size[0] > max_title_width:
            title_scale = title_scale * (max_title_width / title_size[0])
            title_thickness = max(2, int(title_scale * 2))
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
        
        # Pozitionare garantata sa nu iasa
        title_x = max(10, (self.width - title_size[0]) // 2)
        if title_x + title_size[0] > self.width - 10:
            title_x = self.width - title_size[0] - 10
        title_y = int(header_height * 0.32)
        
        # Efect glow redus pentru ecrane mici
        glow_layers = min(6, int(self.width / 200))
        for offset in range(glow_layers, 0, -1):
            alpha = 0.4 - (offset * 0.05)
            glow_color = (int(80 * alpha), int(160 * alpha), int(255 * alpha))
            glow_x = max(0, title_x - offset//2)
            glow_y = min(self.height, title_y + offset//2)
            cv2.putText(frame, title, (glow_x, glow_y),
                       cv2.FONT_HERSHEY_TRIPLEX, title_scale, glow_color, title_thickness + offset, cv2.LINE_AA)
        
        # Umbra pentru profunzime
        shadow_offset = max(2, int(title_scale * 1.2))
        shadow_x = min(title_x + shadow_offset, self.width - title_size[0] - 5)
        shadow_y = min(title_y + shadow_offset, self.height - 10)
        cv2.putText(frame, title, (shadow_x, shadow_y),
                    cv2.FONT_HERSHEY_TRIPLEX, title_scale, (0, 0, 0), title_thickness + 1, cv2.LINE_AA)
        
        # Text principal garantat sa nu iasa
        final_x = max(5, min(title_x, self.width - title_size[0] - 5))
        final_y = max(title_size[1], min(title_y, self.height - 10))
        cv2.putText(frame, title, (final_x, final_y),
                    cv2.FONT_HERSHEY_TRIPLEX, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
        
        # Subtitlu elegant - fara diacritice, text mai mare
        subtitle = "Interfata Inteligenta - Control Vocal, Mouse si Tastatura"
        subtitle_scale = min(self.width / 1200, self.height / 800) * 1.0  # Mai mare
        subtitle_thickness = max(2, int(subtitle_scale * 2))
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness)[0]
        subtitle_x = max(10, (self.width - subtitle_size[0]) // 2)  # Nu iese de pe ecran
        subtitle_y = int(header_height * 0.6)
        
        cv2.putText(frame, subtitle, (subtitle_x + 1, subtitle_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (0, 0, 0), subtitle_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (subtitle_x, subtitle_y),
                    cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (190, 190, 200), subtitle_thickness, cv2.LINE_AA)  # Gri mai cald
        
        # Status microfon cu design premium - optimizat complet sa nu iasa de pe ecran
        if self.speech_available and self.voice_control_active:
            mic_status = "MICROFON ACTIV - Vorbeste oricand" if not self.audio_paused else "AUDIO OPRIT - Spune start"
            mic_color = (50, 255, 150) if not self.audio_paused else (150, 150, 150)
            
            # Margini de siguranta mai mari - 5% pe fiecare parte
            safety_margin = int(self.width * 0.05)
            max_text_width = self.width - (2 * safety_margin)
            
            # Text mai mare dar controlat
            status_scale = min(self.width / 1400, self.height / 900) * 0.8  # Mai mare
            status_thickness = max(2, int(status_scale * 2.5))
            
            # Verifica si ajusteaza scalarea pentru a incape complet
            status_size = cv2.getTextSize(mic_status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_thickness)[0]
            if status_size[0] > max_text_width:
                # Rescaleaza pentru a incape cu margini de siguranta
                status_scale = status_scale * (max_text_width / status_size[0]) * 0.95  # 5% buffer suplimentar
                status_thickness = max(2, int(status_scale * 2.5))
                status_size = cv2.getTextSize(mic_status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_thickness)[0]
            
            # Pozitionare centrata cu margini garantate
            status_x = max(safety_margin, (self.width - status_size[0]) // 2)
            # Dubla verificare ca nu iese pe dreapta
            if status_x + status_size[0] > self.width - safety_margin:
                status_x = self.width - status_size[0] - safety_margin
            
            status_y = int(header_height * 0.82)
            box_height = max(int(self.height * 0.045), status_size[1] + 20)  # Inaltime mai mare
            
            # Padding pentru box - mai generos
            box_padding = max(int(self.width * 0.025), 20)  # Minim 20px padding
            
            # Box principal cu margini garantate - fara glow pentru claritate
            box_left = max(safety_margin // 2, status_x - box_padding)
            box_right = min(self.width - safety_margin // 2, status_x + status_size[0] + box_padding)
            box_top = max(10, status_y - box_height//2)
            box_bottom = min(self.height - 10, status_y + box_height//2)
            
            # Fundal box cu gradient subtil
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (45, 45, 45), -1)
            
            # Bordura colorata cu grosime adaptiva
            border_thickness = max(2, int(self.width * 0.002))
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), 
                         mic_color, border_thickness, cv2.LINE_AA)
            
            # Efect glow subtil doar pe bordura
            glow_color = tuple(int(c * 0.3) for c in mic_color)
            cv2.rectangle(frame, (box_left - 1, box_top - 1), (box_right + 1, box_bottom + 1), 
                         glow_color, 1, cv2.LINE_AA)
            
            # Text centrat vertical in box cu pozitie garantata
            text_x = max(safety_margin, min(status_x, self.width - status_size[0] - safety_margin))
            text_y = status_y + status_size[1] // 2
            
            # Umbra pentru contrast
            cv2.putText(frame, mic_status, (text_x + 1, text_y + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, status_scale, (0, 0, 0), status_thickness, cv2.LINE_AA)
            
            # Text principal
            cv2.putText(frame, mic_status, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, status_scale, mic_color, status_thickness, cv2.LINE_AA)
        
        # Linie separatoare - dinamica
        separator_y = header_height + int(self.height * 0.02)
        separator_margin = int(self.width * 0.05)
        cv2.rectangle(frame, (separator_margin, separator_y), (self.width - separator_margin, separator_y + 2), (100, 100, 100), -1)
        
        # Sectiunea cardurilor - incepe DUPA linia separatoare cu mai mult spatiu
        cards_start_y = separator_y + int(self.height * 0.06)  # Mai mult spatiu dupa linie
        cards_available_height = int(self.height * 0.42)  # Inaltime disponibila pentru carduri
        card_spacing = cards_available_height // 3
        self.button_coords = []
        
        # Descrieri pentru fiecare optiune - fara diacritice
        option_descriptions = [
            "Traducere in timp real din limbajul semnelor in text",
            "Testare camera si vizualizare landmarks pentru debugging", 
            "Inchidere aplicatie si revenire la desktop"
        ]
        
        for i, (option, description) in enumerate(zip(self.options, option_descriptions)):
            y = cards_start_y + i * card_spacing
            card_height = int(card_spacing * 0.75)  # Carduri mai inalte
            self.button_coords.append((y - card_height//2, y + card_height//2))
            
            # Card design elegant - dinamic cu margini mai mari
            card_margin = int(self.width * 0.1)  # Margini mai mari - 10%
            card_width = self.width - 2 * card_margin
            
            if i == self.selected_option:
                # Selectat - card luminat cu culori mai calde
                border_thickness = max(4, int(self.width * 0.003))  # Bordura mai groasa
                cv2.rectangle(frame, (card_margin - border_thickness, y - card_height//2 - border_thickness), 
                            (card_margin + card_width + border_thickness, y + card_height//2 + border_thickness),
                            (120, 200, 255), border_thickness, cv2.LINE_AA)  # Albastru mai cald
                cv2.rectangle(frame, (card_margin, y - card_height//2), (card_margin + card_width, y + card_height//2),
                            (55, 55, 65), -1)  # Fundal mai cald
                
                text_color = (245, 245, 250)  # Alb crem mai cald
                desc_color = (220, 220, 230)  # Gri deschis mai cald
                
                # Indicator selectie - dinamic mai mare cu culoare mai calda
                circle_radius = int(min(self.width, self.height) * 0.015)  # Mai mare
                circle_x = card_margin + int(self.width * 0.04)
                cv2.circle(frame, (circle_x, y), circle_radius, (120, 200, 255), -1)  # Albastru mai cald
                cv2.circle(frame, (circle_x, y), circle_radius + 3, (100, 170, 230), 3)
                
            else:
                # Neselectat - card subtil cu culori mai calde
                cv2.rectangle(frame, (card_margin, y - card_height//2), (card_margin + card_width, y + card_height//2),
                            (50, 50, 55), -1)  # Fundal mai cald
                cv2.rectangle(frame, (card_margin, y - card_height//2), (card_margin + card_width, y + card_height//2),
                            (90, 90, 100), 2, cv2.LINE_AA)  # Bordura mai calda
                
                text_color = (210, 210, 220)  # Gri deschis mai cald
                desc_color = (160, 160, 170)  # Gri mediu mai cald
                
                # Cerc pentru numar mai mare cu culoare mai calda
                circle_radius = int(min(self.width, self.height) * 0.012)  # Mai mare
                circle_x = card_margin + int(self.width * 0.04)
                cv2.circle(frame, (circle_x, y), circle_radius, (65, 65, 70), -1)  # Fundal mai cald
                cv2.circle(frame, (circle_x, y), circle_radius, (120, 120, 130), 2)  # Bordura mai calda
            
            # Numar optiune - dimensiune mai mare
            num_text = f"{i + 1}"
            num_scale = min(self.width / 1400, self.height / 900) * 1.0  # Mai mare
            num_thickness = max(3, int(num_scale * 3))
            num_size = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_TRIPLEX, num_scale, num_thickness)[0]
            num_x = circle_x - num_size[0] // 2
            num_y = y + num_size[1] // 2
            cv2.putText(frame, num_text, (num_x, num_y),
                       cv2.FONT_HERSHEY_TRIPLEX, num_scale, text_color, num_thickness, cv2.LINE_AA)
            
            # Margini interioare pentru carduri - mult mai generoase
            card_inner_padding = int(self.width * 0.04)  # 4% padding interior
            card_center_x = card_margin + card_width // 2  # Centrul cardului
            max_text_width = card_width - (2 * card_inner_padding)  # Spatiu pentru margini interioare
            
            # Titlu optiune - text mai mare centrat pe mijloc
            title_scale = min(self.width / 1000, self.height / 700) * 1.3  # Mult mai mare
            title_thickness = max(3, int(title_scale * 2.5))
            
            # Verifica strict ca textul nu iese din cardul cu margini interioare
            title_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
            
            if title_size[0] > max_text_width:
                # Rescaleaza pentru a incape complet cu margini interioare
                title_scale = title_scale * (max_text_width / title_size[0]) * 0.9  # 10% buffer suplimentar
                title_thickness = max(2, int(title_scale * 2.5))
                title_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
            
            # Pozitionare centrata pe mijlocul cardului - MAI JOS
            title_x_centered = card_center_x - title_size[0] // 2
            # Verifica ca nu iese din marginile cardului cu padding
            title_x_final = max(card_margin + card_inner_padding, 
                               min(title_x_centered, card_margin + card_width - card_inner_padding - title_size[0]))
            
            cv2.putText(frame, option, (title_x_final, y - int(card_height * 0.05)),  # Mai jos: -0.05 in loc de -0.12
                       cv2.FONT_HERSHEY_TRIPLEX, title_scale, text_color, title_thickness, cv2.LINE_AA)
            
            # Descriere - text mai mare centrat pe mijloc - MAI JOS
            desc_scale = min(self.width / 1200, self.height / 800) * 0.9  # Mult mai mare
            desc_thickness = max(2, int(desc_scale * 2))
            
            # Verifica strict ca textul nu iese din cardul cu margini interioare
            desc_size = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, desc_scale, desc_thickness)[0]
            
            if desc_size[0] > max_text_width:
                # Rescaleaza pentru a incape complet cu margini interioare
                desc_scale = desc_scale * (max_text_width / desc_size[0]) * 0.9  # 10% buffer suplimentar
                desc_thickness = max(1, int(desc_scale * 2))
                desc_size = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, desc_scale, desc_thickness)[0]
            
            # Pozitionare centrata pe mijlocul cardului - MAI JOS
            desc_x_centered = card_center_x - desc_size[0] // 2
            # Verifica ca nu iese din marginile cardului cu padding
            desc_x_final = max(card_margin + card_inner_padding, 
                              min(desc_x_centered, card_margin + card_width - card_inner_padding - desc_size[0]))
            
            cv2.putText(frame, description, (desc_x_final, y + int(card_height * 0.25)),  # Mai jos: +0.25 in loc de +0.18
                       cv2.FONT_HERSHEY_SIMPLEX, desc_scale, desc_color, desc_thickness, cv2.LINE_AA)
        
        # Footer dinamic - incepe dupa carduri cu mai mult spatiu
        footer_y = cards_start_y + cards_available_height + int(self.height * 0.04)
        footer_height = self.height - footer_y
        
        # Fundal footer
        cv2.rectangle(frame, (0, footer_y), (self.width, self.height),
                     (35, 35, 35), -1)
        cv2.rectangle(frame, (0, footer_y), (self.width, footer_y + max(3, int(self.height * 0.003))),
                     (100, 180, 255), -1)
        
        # Sectiuni instructiuni - layout in 3 coloane egale cu margini mai mari
        col_width = self.width // 3
        section_y = footer_y + int(footer_height * 0.25)  # Mai mult spatiu sus
        
        # Dimensiuni dinamice pentru text - mult mai mari
        title_scale = min(self.width / 1200, self.height / 800) * 0.85  # Mult mai mare
        title_thickness = max(3, int(title_scale * 3.5))
        inst_scale = min(self.width / 1400, self.height / 900) * 0.7  # Mult mai mare
        inst_thickness = max(2, int(inst_scale * 2.5))
        
        # Sectiunea 1 - Control Vocal cu verificare overflow
        col1_center = col_width // 2
        vocal_title = "CONTROL VOCAL"
        vocal_title_size = cv2.getTextSize(vocal_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
        
        # Verifica ca titlul incape in coloana
        max_col_width = col_width * 0.9  # 90% din latime coloana
        if vocal_title_size[0] > max_col_width:
            title_scale_adj = title_scale * (max_col_width / vocal_title_size[0]) * 0.95
            title_thickness_adj = max(2, int(title_scale_adj * 3.5))
            vocal_title_size = cv2.getTextSize(vocal_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj, title_thickness_adj)[0]
        else:
            title_scale_adj = title_scale
            title_thickness_adj = title_thickness
        
        vocal_title_x = max(10, col1_center - vocal_title_size[0] // 2)
        if vocal_title_x + vocal_title_size[0] > col_width - 10:
            vocal_title_x = col_width - vocal_title_size[0] - 10
        
        cv2.putText(frame, vocal_title, (vocal_title_x, section_y),
                   cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj, (120, 220, 120), title_thickness_adj, cv2.LINE_AA)  # Verde mai cald
        
        vocal_instructions = [
            "Spune '1', '2' sau '3'",
            "'stop' - opreste audio",
            "'start' - porneste audio"
        ]
        
        line_spacing = max(int(footer_height * 0.12), 25)  # Minim 25px spatiere
        for j, instruction in enumerate(vocal_instructions):
            # Verifica ca instructiunea incape in coloana
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, inst_thickness)[0]
            if inst_size[0] > max_col_width:
                inst_scale_adj = inst_scale * (max_col_width / inst_size[0]) * 0.95
                inst_thickness_adj = max(1, int(inst_scale_adj * 2.5))
                inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, inst_thickness_adj)[0]
            else:
                inst_scale_adj = inst_scale
                inst_thickness_adj = inst_thickness
            
            inst_x = max(10, col1_center - inst_size[0] // 2)
            if inst_x + inst_size[0] > col_width - 10:
                inst_x = col_width - inst_size[0] - 10
                
            cv2.putText(frame, instruction, (inst_x, section_y + int(footer_height * 0.18) + j * line_spacing),
                       cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, (190, 190, 200), inst_thickness_adj, cv2.LINE_AA)  # Gri mai cald pentru vocal
        
        # Sectiunea 2 - Mouse cu verificare overflow
        col2_center = col_width + col_width // 2
        mouse_title = "MOUSE"
        mouse_title_size = cv2.getTextSize(mouse_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
        
        # Verifica ca titlul incape in coloana
        if mouse_title_size[0] > max_col_width:
            title_scale_adj2 = title_scale * (max_col_width / mouse_title_size[0]) * 0.95
            title_thickness_adj2 = max(2, int(title_scale_adj2 * 3.5))
            mouse_title_size = cv2.getTextSize(mouse_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj2, title_thickness_adj2)[0]
        else:
            title_scale_adj2 = title_scale
            title_thickness_adj2 = title_thickness
        
        mouse_title_x = max(col_width + 10, col2_center - mouse_title_size[0] // 2)
        if mouse_title_x + mouse_title_size[0] > 2 * col_width - 10:
            mouse_title_x = 2 * col_width - mouse_title_size[0] - 10
        
        cv2.putText(frame, mouse_title, (mouse_title_x, section_y),
                   cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj2, (255, 180, 120), title_thickness_adj2, cv2.LINE_AA)  # Portocaliu mai cald
        
        mouse_instructions = [
            "Click pe orice card",
            "pentru selectie automata"
        ]
        
        for j, instruction in enumerate(mouse_instructions):
            # Verifica ca instructiunea incape in coloana
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, inst_thickness)[0]
            if inst_size[0] > max_col_width:
                inst_scale_adj = inst_scale * (max_col_width / inst_size[0]) * 0.95
                inst_thickness_adj = max(1, int(inst_scale_adj * 2.5))
                inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, inst_thickness_adj)[0]
            else:
                inst_scale_adj = inst_scale
                inst_thickness_adj = inst_thickness
            
            inst_x = max(col_width + 10, col2_center - inst_size[0] // 2)
            if inst_x + inst_size[0] > 2 * col_width - 10:
                inst_x = 2 * col_width - inst_size[0] - 10
                
            cv2.putText(frame, instruction, (inst_x, section_y + int(footer_height * 0.18) + j * line_spacing),
                       cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, (190, 190, 200), inst_thickness_adj, cv2.LINE_AA)  # Gri mai cald pentru mouse
        
        # Sectiunea 3 - Tastatura cu verificare overflow
        col3_center = 2 * col_width + col_width // 2
        keyboard_title = "TASTATURA"
        keyboard_title_size = cv2.getTextSize(keyboard_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, title_thickness)[0]
        
        # Verifica ca titlul incape in coloana
        if keyboard_title_size[0] > max_col_width:
            title_scale_adj3 = title_scale * (max_col_width / keyboard_title_size[0]) * 0.95
            title_thickness_adj3 = max(2, int(title_scale_adj3 * 3.5))
            keyboard_title_size = cv2.getTextSize(keyboard_title, cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj3, title_thickness_adj3)[0]
        else:
            title_scale_adj3 = title_scale
            title_thickness_adj3 = title_thickness
        
        keyboard_title_x = max(2 * col_width + 10, col3_center - keyboard_title_size[0] // 2)
        if keyboard_title_x + keyboard_title_size[0] > self.width - 10:
            keyboard_title_x = self.width - keyboard_title_size[0] - 10
        
        cv2.putText(frame, keyboard_title, (keyboard_title_x, section_y),
                   cv2.FONT_HERSHEY_TRIPLEX, title_scale_adj3, (180, 150, 255), title_thickness_adj3, cv2.LINE_AA)  # Violet mai cald
        
        keyboard_instructions = [
            "1/2/3 - selectie directa",
            "Sus/Jos - navigare",
            "Enter - confirma | ESC - iesire"
        ]
        
        for j, instruction in enumerate(keyboard_instructions):
            # Verifica ca instructiunea incape in coloana
            inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, inst_thickness)[0]
            if inst_size[0] > max_col_width:
                inst_scale_adj = inst_scale * (max_col_width / inst_size[0]) * 0.95
                inst_thickness_adj = max(1, int(inst_scale_adj * 2.5))
                inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, inst_thickness_adj)[0]
            else:
                inst_scale_adj = inst_scale
                inst_thickness_adj = inst_thickness
            
            inst_x = max(2 * col_width + 10, col3_center - inst_size[0] // 2)
            if inst_x + inst_size[0] > self.width - 10:
                inst_x = self.width - inst_size[0] - 10
                
            cv2.putText(frame, instruction, (inst_x, section_y + int(footer_height * 0.18) + j * line_spacing),
                       cv2.FONT_HERSHEY_SIMPLEX, inst_scale_adj, (190, 190, 200), inst_thickness_adj, cv2.LINE_AA)  # Gri mai cald pentru tastatura
        
        return frame
    
    def run(self):
        """Rulare meniu cu toate controalele optimizate si fereastra normala."""
        # Creare fereastra normala (nu fullscreen)
        window_name = 'Meniu Start - LSR'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, self.width, self.height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Mesaj de bun venit optimizat fara diacritice
        welcome_msg = "Bun venit! Aveti trei optiuni. " \
                     "Unu: Traducere semne in text. " \
                     "Doi: Mod demo imbunatatit. " \
                     "Trei: Iesire. " \
                     "Spuneti numarul dorit sau faceti click."
        
        self.speak(welcome_msg)
        
        # Pornire sunet de gandire dupa 2s
        threading.Timer(2.0, lambda: self.play_thinking_sound() if self.voice_control_active else None).start()
        
        # Pornire ascultare vocala
        if self.speech_available:
            threading.Thread(target=self.listen_continuously, daemon=True).start()
            self._print_instructions(vocal=True)
        else:
            self._print_instructions(vocal=False)
        
        # Bucla principala
        while True:
            frame = self.draw_menu()
            cv2.imshow(window_name, frame)
            
            # Verificare executie - PRIORITATE MAXIMA
            if self.should_execute:
                time.sleep(0.1)
                cv2.destroyAllWindows()
                return self.execute_option()
            
            # Verificare daca trebuie sa iesim fara executie
            if not self.voice_control_active and not self.should_execute:
                cv2.destroyAllWindows()
                return False
            
            # Procesare taste
            key = cv2.waitKey(30) & 0xFF
            
            if key == 255:
                continue
            
            # Navigare
            if key in [82, ord('w'), ord('W')]:
                self.selected_option = (self.selected_option - 1) % len(self.options)
                self.beep(600, 0.05)
            
            elif key in [84, ord('s'), ord('S')]:
                self.selected_option = (self.selected_option + 1) % len(self.options)
                self.beep(600, 0.05)
            
            # Selectare directa
            elif key == ord('1'):
                self._select_and_execute(0, "Traducere")
            
            elif key == ord('2'):
                self._select_and_execute(1, "Demo Imbunatatit")
            
            elif key == ord('3'):
                self._select_and_execute(2, "Iesire")
            
            # Confirmare
            elif key in [13, 32]:
                self.beep(1200, 0.2)
                self.voice_control_active = False
                self.should_execute = True
                time.sleep(0.1)
            
            # Iesire
            elif key in [27, ord('q'), ord('Q')]:
                self.beep(400, 0.3)
                self.voice_control_active = False
                cv2.destroyAllWindows()
                return False
    
    def _print_instructions(self, vocal=True):
        """Afisare instructiuni in consola."""
        print("="*70)
        if vocal:
            print("CONTROL VOCAL ACTIV")
            print(f"Vorbeste clar in microfon. Prag: {self.recognizer.energy_threshold}")
        else:
            print("ATENTIE Control vocal indisponibil")
        print("="*70)
        
        if vocal:
            print("OK Comenzi vocale:")
            print("  '1' / 'unu' -> Traducere")
            print("  '2' / 'doi' -> Demo Imbunatatit")
            print("  '3' / 'trei' -> Iesire")
            print("  'stop' -> Opreste audio")
            print("  'start' -> Porneste audio")
            print("  'ajutor' -> Repeta optiunile")
        
        print("OK Mouse:")
        print("  Click pe orice optiune -> Selecteaza si porneste")
        
        print("OK Tastatura:")
        print("  1/2/3 -> Selecteaza direct")
        print("  Sageata sus/jos -> Navigheaza")
        print("  Enter -> Confirma")
        print("  ESC -> Iesire")
        print("="*70)
        
    def execute_option(self):
        """Executa optiunea selectata."""
        if self.selected_option == 0:
            # Traducere cu camera
            print("\n" + "="*70)
            print("PORNIRE: Traducere Semne -> Text (Camera Live)")
            print("="*70 + "\n")
            time.sleep(0.2)
            try:
                result = subprocess.run([sys.executable, "realtime.py"])
                if result.returncode != 0:
                    print(f"ATENTIE Aplicatia s-a inchis cu cod: {result.returncode}")
            except KeyboardInterrupt:
                print("\nATENTIE Intrerupt de utilizator")
            except Exception as e:
                print(f"EROARE Eroare: {e}")
            
            print("\n" + "="*70)
            print("OK Revenire la meniu...")
            print("="*70)
            return True
        
        elif self.selected_option == 1:
            # Demo mode imbunatatit
            print("\n" + "="*70)
            print("PORNIRE: Mod Demo Imbunatatit - Testare Camera")
            print("="*70 + "\n")
            time.sleep(0.2)
            try:
                result = subprocess.run([sys.executable, "demo_enhanced.py"])
                if result.returncode != 0:
                    print(f"ATENTIE Aplicatia s-a inchis cu cod: {result.returncode}")
            except KeyboardInterrupt:
                print("\nATENTIE Intrerupt de utilizator")
            except Exception as e:
                print(f"EROARE Eroare: {e}")
            
            print("\n" + "="*70)
            print("OK Revenire la meniu...")
            print("="*70)
            return True
        
        else:
            # Iesire
            print("\n" + "="*70)
            print("OK Iesire din aplicatie. La revedere!")
            print("="*70 + "\n")
            return False


def main():
    """Functie principala - bucla infinita cu revenire la meniu."""
    print("\n" + "="*70)
    print("  TRADUCATOR LIMBAJ SEMNE ROMANESC - Meniu Principal")
    print("="*70)
    
    while True:
        try:
            menu = AccessibleMenu()
            should_continue = menu.run()
            
            if not should_continue:
                break
            
            # Revenire la meniu dupa 2 secunde
            print("Revenire la meniu in 2 secunde...")
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\nATENTIE Intrerupt de utilizator. Iesire...")
            break
        except Exception as e:
            print(f"\nERORE Eroare critica: {e}")
            print("Reincercare in 3 secunde...")
            time.sleep(3)
    
    print("\nOK Aplicatie inchisa.\n")


if __name__ == "__main__":
    main()
