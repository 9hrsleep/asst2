#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

// for the prefix sum logic (optimization gave us 59/72)
#define BLOCKSIZE 256
#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"

// for the checking what circles are in each image section optimization
#define SQUARE_SIZE 16
#define THREADS_PER_SQUARE 256

class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numberOfCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData, 
        int circleIndex);
};


#endif
