#!/bin/bash
if [ -f TakaoGothic.ttf ]; then
    exit
fi
wget https://launchpad.net/takao-fonts/trunk/15.03/+download/TakaoFonts_00303.01.zip
unzip TakaoFonts_00303.01.zip
mv TakaoFonts_00303.01/TakaoGothic.ttf .
rm -rf TakaoFonts_00303.01
rm -f TakaoFonts_00303.01.zip
