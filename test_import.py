import mediapipe as mp
print(f"Docs: {dir(mp)}")
try:
    import mediapipe.python.solutions as solutions
    print("Found solutions via mediapipe.python.solutions")
except ImportError as e:
    print(f"Could not import mediapipe.python.solutions: {e}")

try:
    from mediapipe import solutions
    print("Found solutions via from mediapipe import solutions")
    print(dir(solutions))
except ImportError as e:
    print(f"Could not import solutions from mediapipe: {e}")
