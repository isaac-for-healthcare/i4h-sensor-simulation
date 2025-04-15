wget https://zenodo.org/api/records/7272660/files-archive
mv files-archive files.zip
mkdir ~/zenodo_dataset
unzip files.zip -d ~/zenodo_dataset
cd ~/zenodo_dataset
unzip Benign.zip
unzip Malignant.zip
unzip Normal.zip
sudo rm -r __MACOSX
sudo rm *.zip
