@echo off
set PYTHONPATH=%CD%;%PYTHONPATH%
python examples/track.py --source test_4.mp4 --tracking-method deepocsort --save --classes 0 --conf 0.2
