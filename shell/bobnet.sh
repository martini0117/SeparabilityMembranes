#!/bin/sh

for var in x y z  #範囲の書き方(Bash独自) => {0..4}
do
    python3 ./keras/BoBNet.py $var
done
