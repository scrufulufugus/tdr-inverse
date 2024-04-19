CXX       = nvcc
CPPFLAGS  = -x cu --restrict -Isubmodules/harmonize -Iinclude
CXXFLAGS  = -std=c++17 --ftz=true --use_fast_math --prec-div=true

SRC_DIR  := src

.PHONY: all
all: tdr-inverse inverse

.PHONY: debug
debug: CXXFLAGS += -g -DDEBUG
debug: clean all

tdr-inverse: $(SRC_DIR)/tdr-inverse.cu $(SRC_DIR)/utils.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

inverse: $(SRC_DIR)/inverse.cu $(SRC_DIR)/utils.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) tdr-inverse inverse
