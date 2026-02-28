try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError:
    print("numpy not found")

try:
    import scipy
    print(f"scipy: {scipy.__version__}")
except ImportError:
    print("scipy not found")

try:
    import cv2
    print(f"opencv: {cv2.__version__}")
except ImportError:
    print("opencv not found")

try:
    import sklearn
    print(f"sklearn: {sklearn.__version__}")
except ImportError:
    print("sklearn not found")

try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError:
    print("matplotlib not found")
