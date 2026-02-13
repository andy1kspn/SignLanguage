"""
Start rapid - Pornește direct demo-ul îmbunătățit LSR.
"""

import sys
from demo_enhanced import run_enhanced_demo

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TRADUCATOR LIMBAJ SEMNE ROMANESC - Demo Îmbunătățit")
    print("="*70 + "\n")
    
    try:
        # Pornește direct demo-ul îmbunătățit
        run_enhanced_demo(camera_idx=0)
    except KeyboardInterrupt:
        print("\n\nATENTIE: Intrerupt de utilizator. Iesire...")
    except Exception as e:
        print(f"\nERORE: {e}")
        sys.exit(1)
    
    print("\nOK Aplicatie inchisa.\n")
