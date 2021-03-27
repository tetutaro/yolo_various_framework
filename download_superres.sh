#!/usr/bin/env bash

if [ ! -d "superres" ]; then
    mkdir superres
fi
# if [ ! -f "superres/LapSRN_x8.pb" ]; then
#     wget -P superres https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb
# fi
if [ ! -f "superres/ESPCN_x4.pb" ]; then
    wget -P superres https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb
fi
