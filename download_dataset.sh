# Command to download the dataset
#   bash script_download_dataset.sh
# Timing
#   Dropbox downloading takes 1h20m
#   Unzip takes 20m

# Move to data directory
DIR=$1

if test -d "$DIR"; then
	echo -e "Dataset downloaded to directory $DIR."
	cd $DIR
else
	echo -e "No directory $DIR ! \nDataset downloaded to the current directory."
fi

# Download chunks of the zip file
FILE=genomic_dataset_chunk.z01
echo -e "\nDownloading $FILE (size is 10GB)..."
curl https://www.dropbox.com/s/fa14gza4cf9dsk3/genomic_dataset_chunk.z01?dl=1 -o genomic_dataset_chunk.z01 -J -L -k
FILE=genomic_dataset_chunk.z02
echo -e "\nDownloading $FILE (size is 10GB)..."
curl https://www.dropbox.com/s/i8pftsjmbpkj1a0/genomic_dataset_chunk.z02?dl=1 -o genomic_dataset_chunk.z02 -J -L -k
FILE=genomic_dataset_chunk.z03
echo -e "\nDownloading $FILE (size is 10GB)..."
curl https://www.dropbox.com/s/udlqbypizummctq/genomic_dataset_chunk.z03?dl=1 -o genomic_dataset_chunk.z03 -J -L -k
FILE=genomic_dataset_chunk.z04
echo -e "\nDownloading $FILE (size is 10GB)..."
curl https://www.dropbox.com/s/2qzbswupfg90tbq/genomic_dataset_chunk.z04?dl=1 -o genomic_dataset_chunk.z04 -J -L -k
FILE=genomic_dataset_chunk
echo -e "\nDownloading $FILE (size is 3GB)..."
curl https://www.dropbox.com/s/0suo9k6fhtdg4p3/genomic_dataset_chunk.zip?dl=1 -o genomic_dataset_chunk.zip -J -L -k

# Assemble the zip file
FILE=genomic_dataset.zip
echo -e "Assembling zip file $FILE (46GB)..."
zip --fix genomic_dataset_chunk --out genomic_dataset

# Unzip the file
echo -e "\nUnzipping $FILE (182GB)..."
unzip genomic_dataset.zip

mv genomic_dataset real

# Done
echo -e "Dataset downloaded, unzipped and ready to use.\n"

