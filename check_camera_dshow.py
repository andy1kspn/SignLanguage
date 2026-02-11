import cv2

# Incearca cu DirectShow (backend Windows)
print("Testare cu DirectShow backend:\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

resolutions = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1280, 1024),
    (1600, 1200),
    (1920, 1080),
]

supported = []

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if actual_width == width and actual_height == height:
        print(f"✓ {width}x{height}")
        supported.append((width, height))
    else:
        print(f"✗ {width}x{height} (camera da: {int(actual_width)}x{int(actual_height)})")

cap.release()

if supported:
    best = supported[-1]
    print(f"\nCea mai buna rezolutie cu DirectShow: {best[0]}x{best[1]}")
else:
    print("\nDirectShow nu ofera rezolutii mai bune")
