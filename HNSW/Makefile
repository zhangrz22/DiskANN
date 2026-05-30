CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
TARGET = hnsw_test

all: $(TARGET)

$(TARGET): main.cpp hnsw.cpp hnsw.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp hnsw.cpp

clean:
	rm -f $(TARGET)

.PHONY: all clean
