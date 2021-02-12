#!/bin/sh

python3 ./keras/PredictBoundaryBox.py
python3 ./keras/ShowBoundingBox.py
python3 ./kidney/SeparabilityMembrane.py