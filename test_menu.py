"""
Script de test pentru debugging meniu.
Rulează: python test_menu.py
"""

import sys

print("\n" + "="*70)
print("TEST MENIU - Debugging")
print("="*70)

# Test 1: Verifică dependențe
print("\n1. Verificare dependențe...")
try:
    import cv2
    print("   ✓ OpenCV instalat")
except:
    print("   ❌ OpenCV lipsește")

try:
    from gtts import gTTS
    import pygame
    print("   ✓ gTTS și pygame instalate")
except:
    print("   ❌ gTTS sau pygame lipsesc")

try:
    import speech_recognition as sr
    print("   ✓ SpeechRecognition instalat")
except:
    print("   ❌ SpeechRecognition lipsește")

# Test 2: Verifică microfon
print("\n2. Verificare microfon...")
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("   ✓ Microfon detectat")
    
    # Listează microfoane disponibile
    print("\n   Microfoane disponibile:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"      {index}: {name}")
except Exception as e:
    print(f"   ❌ Eroare microfon: {e}")

# Test 3: Verifică fișiere
print("\n3. Verificare fișiere...")
import os
if os.path.exists("start_menu.py"):
    print("   ✓ start_menu.py există")
else:
    print("   ❌ start_menu.py lipsește")

if os.path.exists("realtime.py"):
    print("   ✓ realtime.py există")
else:
    print("   ❌ realtime.py lipsește")

# Test 4: Pornește meniul
print("\n4. Pornire meniu...")
print("   Apasă Ctrl+C pentru a opri\n")
print("="*70)

try:
    from start_menu import main
    main()
except KeyboardInterrupt:
    print("\n\n✓ Test oprit de utilizator")
except Exception as e:
    print(f"\n\n❌ Eroare: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST FINALIZAT")
print("="*70 + "\n")
