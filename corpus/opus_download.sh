#!/bin/bash
echo "Downloading OPUS... "
python3 -m pip install opustools-pkg
echo "  * If you see an error running opus_get, please run 'pip install -r requirements.txt"
mkdir -p raw/opus
cd raw/opus
opus_get -s ro -p raw -q
unzip -o -q \*.zip
gunzip *.gz
rm *.zip
rm *.gz
rm INFO
rm LICENSE
rm README
cd ..
cd ..
echo "Done."