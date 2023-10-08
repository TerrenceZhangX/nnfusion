# Please run pip install pycuda first to build the whl
# Document: https://documen.tician.de/pycuda/driver.html#pycuda.driver.device_attribute
import pycuda.driver as cuda
import subprocess

def get_cuda_device_memory_attributes(device):
    attributes = {
        "L2_CACHE_SIZE": cuda.device_attribute.L2_CACHE_SIZE,
        "MAX_SHARED_MEMORY_PER_BLOCK": cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK,
        "MAX_REGISTERS_PER_BLOCK": cuda.device_attribute.MAX_REGISTERS_PER_BLOCK,
        # "MEMORY_CLOCK_RATE": cuda.device_attribute.MEMORY_CLOCK_RATE,
        # "GLOBAL_MEMORY_BUS_WIDTH": cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH,
    }

    for attr_name, attr_value in attributes.items():
        try:
            attr = device.get_attribute(attr_value)
            print(f"{attr_name}: {attr}")
        except cuda.CudaAPIError as e:
            print(f"{attr_name}: Error - {e}")
    
    return attributes

def get_cuda_device_compute_attributes(device):
    attributes = {
        "CLOCK_RATE": cuda.device_attribute.CLOCK_RATE,
        "MULTIPROCESSOR_COUNT": cuda.device_attribute.MULTIPROCESSOR_COUNT,
        "WARP_SIZE": cuda.device_attribute.WARP_SIZE,
        "MAX_PITCH": cuda.device_attribute.MAX_PITCH,
        "MAX_THREADS_PER_BLOCK": cuda.device_attribute.MAX_THREADS_PER_BLOCK,
        "MAX_BLOCK_DIM_X": cuda.device_attribute.MAX_BLOCK_DIM_X,
        "MAX_BLOCK_DIM_Y": cuda.device_attribute.MAX_BLOCK_DIM_Y,
        "MAX_BLOCK_DIM_Z": cuda.device_attribute.MAX_BLOCK_DIM_Z,
        "MAX_GRID_DIM_X": cuda.device_attribute.MAX_GRID_DIM_X,
        "MAX_GRID_DIM_Y": cuda.device_attribute.MAX_GRID_DIM_Y,
        "MAX_GRID_DIM_Z": cuda.device_attribute.MAX_GRID_DIM_Z,

    }

    for attr_name, attr_value in attributes.items():
        try:
            attr = device.get_attribute(attr_value)
            print(f"{attr_name}: {attr}")
        except cuda.CudaAPIError as e:
            print(f"{attr_name}: Error - {e}")
    
    return attributes

def main():
    cuda.init()
    device = cuda.Device(0)
    print(device.name())

    # Get HBM Size from nvidia-smi 
    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    hbm_size_megabytes = int(nvidia_smi_output.strip())
    print("\n----Memory----")
    print(f"HBM Size: {hbm_size_megabytes} MB")
    memory_attributes  = get_cuda_device_memory_attributes(device)
    print("\n----Compute----")
    compute_attributes = get_cuda_device_compute_attributes(device)

if __name__ == "__main__":
    main()
