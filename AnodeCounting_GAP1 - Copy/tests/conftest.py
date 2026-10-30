import os
import sys

# Make the application modules importable from the tests folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
