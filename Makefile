CXX       = nvcc
CPPFLAGS  = -x cu --restrict -Isubmodules/harmonize -Iinclude
CXXFLAGS  = -std=c++11

SRC_DIR  := src

.PHONY: all
all: tdr-inverse

.PHONY: debug
debug: CXXFLAGS += -g -DDEBUG
debug: clean all

tdr-inverse: $(SRC_DIR)/tdr-inverse.cu $(SRC_DIR)/utils.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) tdr-inverse
