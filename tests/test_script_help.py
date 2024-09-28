import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")


def test_main_script_simple():
    files = os.listdir(SCRIPT_DIR)
    files.remove("count_freq.py")
    for file in files:
        script_path = os.path.join(SCRIPT_DIR, file)
        # ensure every can be imported
        subprocess.call(["python", script_path, "--help"])
