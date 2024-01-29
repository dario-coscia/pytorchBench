"""
PyTorch benchmarking code for different hardware specifics.
Code adapted and modified from https://github.com/hibagus/PyTorch-Matmul-Benchmark.git
"""


import torch
import time
import argparse

TESTs = ['torch.mm', 'torch.linalg.svd', 'torch.linalg.qr', 'torch.linalg.inv', 'torch.linalg.det', 'torch.linalg.qr',
        'torch.linalg.pinv', 'torch.linalg.norm', 'torch.linalg.cond']
OPS = {'torch.mm' : torch.mm, 'torch.linalg.svd' : torch.linalg.svd, 'torch.linalg.qr' : torch.linalg.qr,
       'torch.linalg.inv' : torch.linalg.inv, 'torch.linalg.det' : torch.linalg.det,
       'torch.linalg.qr' : torch.linalg.qr, 'torch.linalg.pinv' : torch.linalg.pinv,
       'torch.linalg.norm' : torch.linalg.norm, 'torch.linalg.cond' : torch.linalg.cond}

def generate_matrices(matrix_size, data_format):
    #print("Generating Random Matrix", flush=True)
    matrix = torch.rand(matrix_size, matrix_size, dtype=torch.float64)

    # casting fp64 tensor as needed.
    if(data_format=='fp64'):
        matrix=matrix
    elif(data_format=='fp32'):
        matrix=matrix.float()
    elif(data_format=='fp16'):
        matrix=matrix.half()
    elif(data_format=='bfloat16'):
        matrix=matrix.bfloat16()
    elif(data_format=='int64'):
        matrix=matrix.long()
    elif(data_format=='int32'):
        matrix=matrix.int()
    elif(data_format=='int16'):
        matrix=matrix.short()
    elif(data_format=='int8'):
        matrix=matrix.char()
    else:
        print("Non-supported Data Type.")
        exit()

    #print("Matrix Generation is done", flush=True)
    return(matrix)


def run_mm_benchmark(matrix, matrix_size, device, operation):

    # get device
    useCUDA = True if device in ['cuda', 'mps'] else False

    # import synchronize 
    if device == 'mps':
        from torch.mps import synchronize, empty_cache
    elif device == 'cuda':
        from torch.cuda import synchronize, empty_cache

    # Calculating Number of Ops
    ops = 2*matrix_size**3

    # Copying matrix to GPU memory (if GPU is used)
    if(useCUDA==True):
        matrix = matrix.to(device=device)
    
    # Take a note for start time
    start = time.time()

    # Begin Multiplication
    operation(matrix, matrix)

    # Wait operation to finish
    if(useCUDA==True):
        synchronize()

    # Take a note for end time
    end = time.time()

    # Calculate Elapsed Time
    duration = end - start

    # Calculate TFLOPS
    tflops = ops / duration / 10**12
    print("{}x{} MM {} ops in {} sec = TFLOPS {}".format(matrix_size, matrix_size, ops, duration, tflops), flush=True)

    # Clean-up
    if(useCUDA==True):
        del matrix
        empty_cache()

def run_benchmark(matrix, matrix_size, device, operation):

    # get device
    useCUDA = True if device in ['cuda', 'mps'] else False

    # import synchronize 
    if device == 'mps':
        from torch.mps import synchronize, empty_cache
    elif device == 'cuda':
        from torch.cuda import synchronize, empty_cache

    # Calculating Number of Ops
    ops = 2*matrix_size**3

    # Copying matrix to GPU memory (if GPU is used)
    if(useCUDA==True):
        matrix = matrix.to(device=device)
    
    # Take a note for start time
    start = time.time()

    # Begin Multiplication
    operation(matrix)

    # Wait operation to finish
    if(useCUDA==True):
        synchronize()

    # Take a note for end time
    end = time.time()

    # Calculate Elapsed Time
    duration = end - start

    # Calculate TFLOPS
    tflops = ops / duration / 10**12
    print("{}x{} MM {} ops in {} sec = TFLOPS {}".format(matrix_size, matrix_size, ops, duration, tflops), flush=True)

    # Clean-up
    if(useCUDA==True):
        del matrix
        empty_cache()

def cmdline_args():
    # Make parser object
    parser = argparse.ArgumentParser(description='PyTorch operations benchmark.')
    parser.add_argument("--device", type=str, default='cpu', help='Device to use for benchmarking', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument("--precision", type=str, default='fp32', help='Data Precision for Benchmark', choices=['int8', 'int16', 'int32', 'int64', 'fp16', 'bfloat16', 'fp32', 'fp64'])
    parser.add_argument("--size", type=str, default='100, 500, 1000, 2000, 4000, 8000, 16000', help='List of matrices size separated by comma, e.g., 100, 500, 1000')
    parser.add_argument("--test", type=str, default='torch.mm',
                        help='Pytorch operation to benchmark. When "all" is passed all the functions are benchmarked', choices=TESTs+['all'])
    return(parser.parse_args())


if __name__ == '__main__':
    
    # Parse Arguments
    try:
        args = cmdline_args()
        #print(args)
    except:
        print("Launch argument error!")
        print("Example: $python <script_name> --device='cuda' --precision='fp32' --size='100, 500, 1000'")
        exit()

    # Parse Matrices Size
    try:
        size_list = [int(i) for i in args.size.split(",")]
    except:
        print("Invalid list of matrix size. Use only integer separated by comma to define the list of matrix size.")
        exit()


    if args.test == 'all':
        test_cases = TESTs
    else:
        test_cases = [args.test]

    for test in test_cases:
        print(f'PyTorch operation {test} testing')
        for matrix_size in size_list:
            matrix = generate_matrices(matrix_size,args.precision)
            if test == 'torch.mm':
                run_mm_benchmark(matrix, matrix_size, args.device, OPS[test])
            else:
                run_benchmark(matrix, matrix_size, args.device, OPS[test])
            del matrix
        print()
        
   
