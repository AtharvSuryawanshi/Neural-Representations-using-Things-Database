#!/bin/bash

mkdir -p preprocessed

cd preprocessed

# List of links to download
# # participant 4
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-1.fif?versionId=LGG8UJiqW83Z5Mgpiv5LQDgUgs0kYn9y"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-2.fif?versionId=tdx2wKa7QOpMjBzsN4py4Y.rYhW9vp26"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-3.fif?versionId=gajLhEHnXBlMs.EMnyXyFtJ_SVvw_gOT"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo.fif?versionId=LM9PMqdfCcx8h9qjfU_W69WU0Ol9ECdl"

# participant 1
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-1.fif?versionId=VEFpBYggusLPxNIKlN21eEtrjpo4DcBP"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-2.fif?versionId=hVx8E4e0xIymnQ3nHKFnNAhQHsr7jKlQ"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-3.fif?versionId=1jVDhw2KOJz7b_6ApO2a9J4vy_8jqqhn"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo.fif?versionId=_KO81vnVItjzcxrR91AFeHh8sA7YiuB9"

# participant 2
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-1.fif?versionId=yIZzH6fajlVFCen182u0nGLl5sbWKs0w"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-2.fif?versionId=qQ9bWzptOpO8CZlBhT4c5erDOA5Do67a"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-3.fif?versionId=uRES5t7k47rQTCDUNlFN5_rRmjNOemtW"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo.fif?versionId=yYjGg9sMwgC3FozM8Sc_HXmKoSd_6Qnv"

# participant 3
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-1.fif?versionId=JYdYRMFUzwsz8354Fz9sQQbnQ.Ozstbb"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-2.fif?versionId=mUABa6OLpBG1Vu0A9IsFpOkgWjMBTuG8"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-3.fif?versionId=LadUGeGvC.s2wAqYEMXEzD4_VjD1aRJg"
wget "https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo.fif?versionId=oaTbATaOOvX.UtsD5gA9ET7RJoZljTsB" 

cd ..

w3m images_meg.zip "https://osf.io/download/rdxy2/" 
unzip _image_database_things.zip 
# Password for things_images.zip: things4all

# rename folder
mv object_images images_meg
