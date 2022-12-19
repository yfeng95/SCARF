#!/bin/bash
mkdir -p ./data

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# scarf utilities
echo -e "\nDownloading SCARF data..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/n58Fzbzz7Ei9x2W/download -O ./data/scarf_utilities.zip
unzip ./data/scarf_utilities.zip -d ./data
rm ./data/scarf_utilities.zip

# download two examples 
echo -e "\nDownloading SCARF training data and trained avatars..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/geTtN4p5YTJaqPi/download -O scarf-exp-data-small.zip
unzip ./scarf-exp-data-small.zip -d .
rm ./scarf-exp-data-small.zip
mv ./scarf-exp-data-small ./exps
