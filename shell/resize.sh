#!/bin/sh

for kn in k n
do
    for xyz in x y z
    do
        python3 ./keras/ResizeImage.py $kn $xyz
    done
done


