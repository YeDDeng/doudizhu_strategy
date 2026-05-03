"""
LabelImg starter using pythonw
"""
import subprocess
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

# Use pythonw.exe to avoid console window issues
subprocess.Popen(
    [r"C:\Users\30330\AppData\Local\Programs\Python\Python314\pythonw.exe",
     "-c",
     "from labelImg.labelImg import main; main()"],
    cwd=os.getcwd()
)
