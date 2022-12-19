# trained model for RobustVideoMatting
mkdir -p ./assets
mkdir -p ./assets/RobustVideoMatting
echo -e "Downloading RobustVideoMatting model..."
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth -P ./assets/RobustVideoMatting

# trained model for cloth-segmentation
# if failed, please download the model from
# https://drive.google.com/file/d/1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ/view
mkdir -p ./assets/cloth-segmentation
echo -e "Downloading cloth-segmentation model..."
FILEID=1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ
FILENAME=./assets/cloth-segmentation/cloth_segm_u2net_latest.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# trained model for PIXIE
# if failed, please check https://github.com/yfeng95/PIXIE/blob/master/fetch_model.sh
cd submodules/PIXIE
echo -e "Downloading PIXIE data..."
#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# PIXIE pretrained model and utilities
echo -e "\nYou need to register at https://pixie.is.tue.mpg.de/"
read -p "Username (PIXIE):" username
read -p "Password (PIXIE):" password
username=$(urle $username)
password=$(urle $password)
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/utilities.zip' --no-check-certificate --continue

cd ./data
unzip utilities.zip