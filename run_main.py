"""
Wrapper to run main.py with proper path setup.
"""
import sys
import os

# Ensure proper path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force reload modules
if 'modules' in sys.modules:
    del sys.modules['modules']

# Now import and run
import main
main.main()