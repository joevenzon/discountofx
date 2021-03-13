enum e_params
{
#define PARAMFLOAT(x, def, name, hint, minimum, maximum) _##x,
#define PARAMCOUNT(x) k_param_##x
#include "params.h"
#undef PARAMFLOAT
#undef PARAMCOUNT
};

#define CLEARFLOAT3(x) x[0] = 0.f; x[1] = 0.f; x[2] = 0.f;

__device__ __host__ float calcLuminance(const float * rgb)
{
    //return r * 0.2126f + g * 0.7152f + b * 0.0722f;
    return rgb[0] * 0.299f + rgb[1] * 0.587f + rgb[2] * 0.114f;
}

__device__ __host__ float lerp(float x, float y, float s)
{
    return x + s*(y - x);
}

__device__ __host__ void rgb2yuv(const float * rgb, float * out_yuv) {
    out_yuv[0] = rgb[0] * (0.299f) + rgb[1] * (0.587f) + rgb[2] * (0.114f);
    out_yuv[1] = rgb[0] * (-0.147f) + rgb[1] * (-0.289f) + rgb[2] * (0.436f);
    out_yuv[2] = rgb[0] * (0.615f) + rgb[1] * (-0.515f) + rgb[2] * (-0.1f);
}

__device__ __host__ void yuv2rgb(const float * yuv, float * out_rgb) {
    out_rgb[0] = yuv[0] /* * (1.0f) + yuv[1] * (0.0f)*/ + yuv[2] * (1.14f);
    out_rgb[1] = yuv[0] /* * (1.0f)*/ + yuv[1] * (-0.395f) + yuv[2] * (-0.5806f);
    out_rgb[2] = yuv[0] /* * (1.0f)*/ + yuv[1] * (2.032f) /*+ yuv[2] * (0.0f)*/;
}

__device__ __host__ const float* sample(const float* p_Input, const int p_Width, const int p_Height, const int x, const int y)
{
    const int index = (max(0,min(p_Height-1,y) * p_Width) + max(0, min(p_Width - 1, x))) * 4;
    //const int index = ((y * p_Width) + x) * 4;
    return &p_Input[index];
}

__device__ __host__ void comparePatches(const int p_Width, const int p_Height, const float* params, const float* p_Input, int px, int py, int x, int y, float * result)
{
    float w[3];
    CLEARFLOAT3(w);

    float sigma = max(0.00001f, params[_amount]*params[_amount] * 0.2f);

    #define WINDOW 2

    for (int i = -WINDOW; i < WINDOW; i++)
    {
        for (int j = -WINDOW; j < WINDOW; j++)
        {
            float pcurrent[3];
            float ocurrent[3];

            rgb2yuv(sample(p_Input, p_Width, p_Height, px + i, py + j), pcurrent);
            rgb2yuv(sample(p_Input, p_Width, p_Height, x + i, y + j), ocurrent);

            w[0] = expf(-(pcurrent[0] - ocurrent[0]) * (pcurrent[0] - ocurrent[0]) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * 3.141593f) * sigma);
            w[1] = expf(-(pcurrent[1] - ocurrent[1]) * (pcurrent[1] - ocurrent[1]) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * 3.141593f) * sigma);
            w[2] = expf(-(pcurrent[2] - ocurrent[2]) * (pcurrent[2] - ocurrent[2]) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * 3.141593f) * sigma);
        }
    }

    const float k_denom = 1.0f / ((WINDOW + 1) * (2 * WINDOW + 1));

    result[0] = w[0] * k_denom;
    result[1] = w[1] * k_denom;
    result[2] = w[2] * k_denom;
}

__device__ __host__ void InternalGainAdjustKernel(const int p_Width, const int p_Height, int x, int y, const float* params, const float* p_Input, float* p_Output)
{
    const float* input = sample(p_Input, p_Width, p_Height, x, y);

    float result[3];
    CLEARFLOAT3(result);

    float yuv[3];
    rgb2yuv(input, yuv);

    float processed[3];
    float weights[3];
    CLEARFLOAT3(processed);
    CLEARFLOAT3(weights);

    #define KERNEL 8

    for (int i = -KERNEL; i < KERNEL; i++)
    {
        for (int j = -KERNEL; j < KERNEL; j++)
        {
            int px = x + i;
            int py = y + j;
            
            float w[3];
            CLEARFLOAT3(w);

            comparePatches(p_Width, p_Height, params, p_Input, px, py, x, y, w);

            const float* sampled = sample(p_Input, p_Width, p_Height, px, py);
            float sampledyuv[3];
            rgb2yuv(sampled, sampledyuv);

            processed[0] += w[0] * sampledyuv[0];
            processed[1] += w[1] * sampledyuv[1];
            processed[2] += w[2] * sampledyuv[2];

            weights[0] += w[0];
            weights[1] += w[1];
            weights[2] += w[2];
        }
    }

    result[0] = lerp(yuv[0], processed[0] / max(0.0001f,weights[0]), params[_luminance]);
    result[1] = lerp(yuv[1], processed[1] / max(0.0001f, weights[1]), params[_color]);
    result[2] = lerp(yuv[2], processed[2] / max(0.0001f, weights[2]), params[_color]);

    yuv2rgb(result, p_Output);

/*#define OPERATION(i) p_Output[i] = input[i] + input[i]*4.0f*sharp + left[i]*(-1.0f)*sharp + right[i]*(-1.0f)*sharp + above[i]*(-1.0f)*sharp + below[i]*(-1.0f)*sharp;
//#define OPERATION(i) p_Output[i] = input[i]*params[_desharp];
    OPERATION(0);
    OPERATION(1);
    OPERATION(2);
#undef OPERATION*/
    
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
