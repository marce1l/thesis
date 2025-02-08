import os
import sys

# Needed for relative imports
modulepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if modulepath not in sys.path:
    sys.path.insert(0, modulepath)
