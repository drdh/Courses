if [ ! -d "./venv/" ];then
mkdir venv
cd venv
python3 -m venv .
cd ..
fi
source venv/bin/activate

# build original lib and install python module
g++ src/eculid.cpp -fPIC -shared -o lib/libeculid.so

cd src
python3 setup.py clean
python3 setup.py build
python3 setup.py install

cd ../memory_profiler
python3 setup.py install

pip install Pillow

cd ..
# make libwrapper.so and tryload binary
cmake CMakeLists.txt
make

# remove
rm -rf build
rm -rf CMakeFiles
rm -rf src/build
rm -rf src/CMakeFiles

rm CMakeCache.txt cmake_install.cmake Makefile src/CMakeCache.txt src/cmake_install.cmake src/Makefile
