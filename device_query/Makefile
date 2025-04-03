NVCC = nvcc
NVCC_FLAGS = -O3
TARGET = devicequery

all: $(TARGET)

$(TARGET): devicequery.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
