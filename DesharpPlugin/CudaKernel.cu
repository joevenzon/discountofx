enum e_params
{
#define PARAMFLOAT(x, def, name, hint, minimum, maximum) _##x,
#define PARAMCOUNT(x) k_param_##x
#include "params.h"
#undef PARAMFLOAT
#undef PARAMCOUNT
};

__device__ __host__ float calcLuminance(float r, float g, float b)
{
    return r * 0.2126f + g * 0.7152f + b * 0.0722f;
}

__device__ __host__ float lerp(float x, float y, float s)
{
    return x + s*(y - x);
}

__device__ __host__ const float* sample(const float* p_Input, const int p_Width, const int p_Height, const int x, const int y)
{
    const int index = (max(0,min(p_Height-1,y) * p_Width) + max(0, min(p_Width - 1, x))) * 4;
    //const int index = ((y * p_Width) + x) * 4;
    return &p_Input[index];
}

__device__ __host__ void InternalGainAdjustKernel(const int p_Width, const int p_Height, int x, int y, const float* params, const float* p_Input, float* p_Output)
{
    const float* input = sample(p_Input, p_Width, p_Height, x, y);
    const float* left = sample(p_Input, p_Width, p_Height, x-1, y);
    const float* right = sample(p_Input, p_Width, p_Height, x+1, y);
    const float* above = sample(p_Input, p_Width, p_Height, x, y-1);
    const float* below = sample(p_Input, p_Width, p_Height, x, y+1);

    float sharp = params[_sharp] - params[_desharp];

#define OPERATION(i) p_Output[i] = input[i] + input[i]*4.0f*sharp + left[i]*(-1.0f)*sharp + right[i]*(-1.0f)*sharp + above[i]*(-1.0f)*sharp + below[i]*(-1.0f)*sharp;
//#define OPERATION(i) p_Output[i] = input[i]*params[_desharp];
    OPERATION(0);
    OPERATION(1);
    OPERATION(2);
#undef OPERATION
    
    p_Output[3] = input[3];
}

__global__ void GainAdjustKernel(int p_Width, int p_Height, const float* p_Params, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

       InternalGainAdjustKernel(p_Width, p_Height, x, y, p_Params, p_Input, &p_Output[index]);
   }
}

void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Params, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    const int paramBytes = sizeof(float) * k_param_count;
    float* d_Params;
    cudaMalloc(&d_Params, paramBytes);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_Params, p_Params, paramBytes, cudaMemcpyHostToDevice);

    GainAdjustKernel<<<blocks, threads, 0, stream>>>(p_Width, p_Height, d_Params, p_Input, p_Output);

    // Free device memory
    cudaFree(d_Params);
}
