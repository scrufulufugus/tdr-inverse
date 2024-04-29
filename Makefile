CXX       = nvcc
CPPFLAGS  = -x cu --restrict -Isubmodules/harmonize/harmonize/cpp -Iinclude
CXXFLAGS  = -std=c++17 --ftz=true --use_fast_math --prec-div=true

SRC_DIR  := src
BIN_DIR  := bin

.PHONY: all
all: tdr-inverse cpu-inverse inverse

.PHONY: debug
debug: CXXFLAGS += -g -DDEBUG
debug: clean all

tdr-inverse: $(BIN_DIR)/tdr-inverse

inverse: $(BIN_DIR)/inverse

cpu-inverse: $(BIN_DIR)/cpu-inverse

$(BIN_DIR)/%: $(SRC_DIR)/%.cu $(SRC_DIR)/utils.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(SRC_DIR)/utils.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) $(BIN_DIR)/*
