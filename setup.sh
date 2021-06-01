# TODO: Create a proper bash script for setting up the whole environment - conda and everything

mkdir pretrained
mkdir figures

# Compile Raven
cd vendor/raven
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
cd ../../..
