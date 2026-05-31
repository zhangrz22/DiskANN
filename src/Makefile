CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
TARGETS = diskann_test epsilon_pairs

all: $(TARGETS)

diskann_test: main.cpp diskann.cpp diskann.h utils.h
	$(CXX) $(CXXFLAGS) -o diskann_test main.cpp diskann.cpp

epsilon_pairs: epsilon_pairs.cpp utils.h
	$(CXX) $(CXXFLAGS) -o epsilon_pairs epsilon_pairs.cpp

clean:
	rm -f $(TARGETS)

.PHONY: all clean
