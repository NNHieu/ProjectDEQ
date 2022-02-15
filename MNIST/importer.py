import os
import sys
SRC_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if SRC_LIB not in sys.path:
    sys.path.append(SRC_LIB)