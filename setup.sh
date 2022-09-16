python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
FILE=pretrain
if ! test -d "$FILE"; then
	echo star
	mkdir pretrain
fi
cd pretrain
FILE=RN101.pt
if ! test -f "$FILE"; then
    wget https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
else
	echo RN101.pt is already downloaded.
fi
FILE=RN50.pt
if ! test -f "$FILE"; then
	wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
else
	echo RN50.pt is already downloaded.
fi
FILE=ViT-L-14.pt
if ! test -f "$FILE"; then
	wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
else
	echo ViT-L-14.pt is already downloaded.
fi
FILE=ViT-L-14-336px.pt
if ! test -f "$FILE"; then
	wget https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
else
	echo ViT-L-14-336px.pt is already downloaded.
fi
cd ..
