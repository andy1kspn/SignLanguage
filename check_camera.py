import cv2

# Rezolutii comune de testat
resolutions = [
    (640, 480),    # VGA
    (800, 600),    # SVGA
    (1024, 768),   # XGA
    (1280, 720),   # HD
    (1280, 1024),  # SXGA
    (1600, 1200),  # UXGA
    (1920, 1080),  # Full HD
    (2560, 1440),  # 2K
    (3840, 2160),  # 4K
]

cap = cv2.VideoCapture(0)

print("Rezolutii suportate de camera ta:\n")
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
    print(f"\nCea mai buna rezolutie: {best[0]}x{best[1]}")
else:
    print("\nCamera suporta doar rezolutia implicita (probabil 640x480)")
