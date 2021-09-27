enum e_params
{
#define PARAMFLOAT(x, def, name, hint) _##x,
#define PARAMCOUNT(x) k_param_##x
#include "params.h"
#undef PARAMFLOAT
#undef PARAMCOUNT
};

__device__ __host__ void InternalGainAdjustKernel(int x, int y, int p_Width, int p_Height, float* params, const float* p_Input, float* p_Output)
{
    float blackTBBorderWidth = params[_topBottomPercent] * params[_blackWidth] * 0.5f; // times 0.5 because each bar covers at most half the image
    float whiteTBBorderWidth = params[_topBottomPercent] * params[_whiteWidth] * 0.5f + blackTBBorderWidth; // +black because the white is the inner border

    float blackLRBorderWidth = params[_leftRightPercent] * params[_blackWidth] * 0.5f;
    float whiteLRBorderWidth = params[_leftRightPercent] * params[_whiteWidth] * 0.5f + blackLRBorderWidth;

    p_Output[0] = p_Input[0];
    p_Output[1] = p_Input[1];
    p_Output[2] = p_Input[2];
    p_Output[3] = p_Input[3];

    float w = (float) p_Width;
    float h = (float) p_Height;

    // check for white border
    // note that this will also cover the outer border
    if (x / w < whiteLRBorderWidth || x / w > 1.0f - whiteLRBorderWidth ||
        y / h < whiteTBBorderWidth || y / h > 1.0f - whiteTBBorderWidth)
    {
        p_Output[0] = 1.0f;
        p_Output[1] = 1.0f;
        p_Output[2] = 1.0f;
    }

    // check for black border
    // this will overwrite the white border on the outer edge
    if (x / w < blackLRBorderWidth || x / w > 1.0f - blackLRBorderWidth ||
        y / h < blackTBBorderWidth || y / h > 1.0f - blackTBBorderWidth)
    {
        p_Output[0] = 0.0f;
        p_Output[1] = 0.0f;
        p_Output[2] = 0.0f;
    }
}

__global__ void GainAdjustKernel(int p_Width, int p_Height, float* p_Params, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;

       InternalGainAdjustKernel(x, y, p_Width, p_Height, p_Params, &p_Input[index], &p_Output[index]);
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
