#!/bin/bash
set -e
echo "Install requirements..."
pip3 install -r requirements.txt
chmod u+x ./wiki_download.sh
chmod u+x ./opus_download.sh
chmod u+x ./oscar_download.sh

apt-get install opus-tools
echo "Start Data Download..."
./wiki_download.sh
./opus_download.sh
./oscar_download.sh
echo "Start Data Cleaning..."
python3 wiki_clean.py
python3 opus_clean.py
python3 oscar_clean.py
echo "Merging Dataset.."
python3 merge_corpora.py

echo "Done. train.txt and validation.txt are created"