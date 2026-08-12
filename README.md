# compile

## linux
- g++ -o test main.cpp src/* -O3 -fopenmp -march=native

## win
- x86_64-w64-mingw32-g++ -o test.exe main.cpp src/*.cpp -O3 -fopenmp -static -lstdc++ -lwinpthread
