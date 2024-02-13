CXX       = nvcc
CPPFLAGS  = -x cu --restrict -Isubmodules/harmonize
CXXFLAGS  = -std=c++11

SRC_DIR  := src

.PHONY: all
all: tdr-inverse

# Must run `make clean` first if source has not changed.
.PHONY: debug
debug: CXXFLAGS += -g
debug: all

tdr-inverse: $(SRC_DIR)/tdr-inverse.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) tdr-inverse
