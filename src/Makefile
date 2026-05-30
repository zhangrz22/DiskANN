CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
TARGET = diskann_test

all: $(TARGET)

$(TARGET): main.cpp diskann.cpp diskann.h utils.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp diskann.cpp

clean:
	rm -f $(TARGET)

.PHONY: all clean
