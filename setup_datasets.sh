#!/bin/bash

# Create dataset folder (and throw no exception if already exists)
mkdir -p dataset/sed2012/photos
mkdir -p logs

# Download MediaEval2012 data if not already there
cd dataset/sed2012
#Check if the fotos folder already has stuff in it
if [ ! -z "$( ls -A './photos' )" ]; then
    echo "MediaEval SED 2012 is already ready and unzipped"
else
    #Check if files have already been downloaded
    FILE=./sed2012_photos_part4.tar.gz
    if test -f "$FILE"; then
        echo "MediaEval 2012 has already been downloaded"
    else
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_evaluation_kit.zip
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_photos_license.zip
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_photos_part1.tar.gz
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_photos_part2.tar.gz
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_photos_part3.tar.gz
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_photos_part4.tar.gz
        wget https://skulddata.cs.umass.edu/traces/mmsys/2013/social2012/sed2012_test_kit.zip
    fi

    unzip -j sed2012_test_kit.zip -d .
    unzip -j sed2012_photos_license.zip -d .
    unzip -j sed2012_evaluation_kit.zip -d .
    tar -xvzf sed2012_photos_part1.tar.gz -C photos --strip-components=3
    tar -xvzf sed2012_photos_part2.tar.gz -C photos --strip-components=3
    tar -xvzf sed2012_photos_part3.tar.gz -C photos --strip-components=3
    tar -xvzf sed2012_photos_part4.tar.gz -C photos --strip-components=3
fi
cd ../..