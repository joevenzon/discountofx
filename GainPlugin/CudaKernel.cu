enum e_params
{
#define PARAMFLOAT(x, def, name, hint) _##x,
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

__device__ __host__ void InternalGainAdjustKernel(int x, int y, float* params, const float* p_Input, float* p_Output)
{
    p_Output[0] = p_Input[0] * params[_rscaler] + p_Input[1] * params[_rscaleg] + p_Input[2] * params[_rscaleb];
    p_Output[1] = p_Input[0] * params[_gscaler] + p_Input[1] * params[_gscaleg] + p_Input[2] * params[_gscaleb];
    p_Output[2] = p_Input[0] * params[_bscaler] + p_Input[1] * params[_bscaleg] + p_Input[2] * params[_bscaleb];

    float origLum = calcLuminance(p_Input[0], p_Input[1], p_Input[2]);
    float newLum = calcLuminance(p_Output[0], p_Output[1], p_Output[2]);
    float makeup = lerp(1.0f, origLum / max(newLum,0.01f), params[_preserveLum]);

    p_Output[0] *= makeup;
    p_Output[1] *= makeup;
    p_Output[2] *= makeup;
    
    p_Output[3] = p_Input[3];
}

__global__ void GainAdjustKernel(int p_Width, int p_Height, float* p_Params, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

       InternalGainAdjustKernel(x, y, p_Params, &p_Input[index], &p_Output[index]);
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
