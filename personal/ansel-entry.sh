#!/bin/bash -e
#
# I have two FlashAir cards, so check them sequentially.
#
~/ansel/personal/ansel-worker.sh ansel 10.0.0.3 '/DCIM' ~/Dropbox/ $ANSEL_PSID $ANSEL_TOKEN
~/ansel/personal/ansel-worker.sh ansel 10.0.0.4 '/DCIM' ~/Dropbox/ $ANSEL_PSID $ANSEL_TOKEN
