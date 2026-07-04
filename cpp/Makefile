CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

cfna_run: cfna_engine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f cfna_run

.PHONY: clean
