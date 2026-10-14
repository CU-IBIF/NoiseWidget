import os

# Qt must be told to run headless before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
