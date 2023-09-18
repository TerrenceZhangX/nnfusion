# Please run pip install pycuda first to build the whl
import pycuda.driver as cuda

def print_cuda_device_attributes(device):
    attributes = {
        "MAX_THREADS_PER_BLOCK": cuda.device_attribute.MAX_THREADS_PER_BLOCK,
        "MAX_BLOCK_DIM_X": cuda.device_attribute.MAX_BLOCK_DIM_X,
        "MAX_BLOCK_DIM_Y": cuda.device_attribute.MAX_BLOCK_DIM_Y,
        "MAX_BLOCK_DIM_Z": cuda.device_attribute.MAX_BLOCK_DIM_Z,
        "MAX_GRID_DIM_X": cuda.device_attribute.MAX_GRID_DIM_X,
        "MAX_GRID_DIM_Y": cuda.device_attribute.MAX_GRID_DIM_Y,
        "MAX_GRID_DIM_Z": cuda.device_attribute.MAX_GRID_DIM_Z,
        "MAX_SHARED_MEMORY_PER_BLOCK": cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK,
        "TOTAL_CONSTANT_MEMORY": cuda.device_attribute.TOTAL_CONSTANT_MEMORY,
        "WARP_SIZE": cuda.device_attribute.WARP_SIZE,
        "MAX_PITCH": cuda.device_attribute.MAX_PITCH,
        "MAX_REGISTERS_PER_BLOCK": cuda.device_attribute.MAX_REGISTERS_PER_BLOCK,
        "CLOCK_RATE": cuda.device_attribute.CLOCK_RATE,
        "TEXTURE_ALIGNMENT": cuda.device_attribute.TEXTURE_ALIGNMENT,
        "MULTIPROCESSOR_COUNT": cuda.device_attribute.MULTIPROCESSOR_COUNT,
        "KERNEL_EXEC_TIMEOUT": cuda.device_attribute.KERNEL_EXEC_TIMEOUT,
        "MEMORY_CLOCK_RATE": cuda.device_attribute.MEMORY_CLOCK_RATE,
        "GLOBAL_MEMORY_BUS_WIDTH": cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH,
        "L2_CACHE_SIZE": cuda.device_attribute.L2_CACHE_SIZE,
    }

    for attr_name, attr_value in attributes.items():
        try:
            attr = device.get_attribute(attr_value)
            print(f"{attr_name}: {attr}")
        except cuda.CudaAPIError as e:
            print(f"{attr_name}: Error - {e}")

def main():
    cuda.init()

    device = cuda.Device(0)

    print(device.name())
    print("Total memory(GB):"+str(device.total_memory()/1000000000))
    print("CUDA Device Properties:")
    print_cuda_device_attributes(device)

if __name__ == "__main__":
    main()
