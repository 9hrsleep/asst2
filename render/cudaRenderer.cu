#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// new global variables to have squares that know what circles belong in them
static unsigned int* circleInSquareBitMasks;
static float4* circleData;
static int numSquaresX = 0;
static int numSquaresY = 0;
static int numMaskWords = 0;
static size_t bitMaskArrayAllocSize = 0;

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// helper function to go through the set bits in this thread's circle bit mask and load those into shared memory
__device__ __inline__ void writeCirclesToSharedMem(unsigned int circleBitMask, int circleBitMaskIndex,
                                                   int myListStartIndex, int roundStart,
                                                   const float4* kernelCircleData,
                                                   float4* sharedCircleData, float3* sharedCircleColors,
                                                   bool isSnowflakes) {
    int currentPos = myListStartIndex;

    // loop through the set bits in this thread's circle bit mask
    while (circleBitMask != 0) {
        int sharedMemorySlot = currentPos - roundStart;
        // circle belongs to a later round, stop for now
        if (sharedMemorySlot >= BLOCKSIZE) {
            break;
        }

        // ffs: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
        // we searched for a cuda function to find the first set bit in a 32 bit number and found ffs
        int circleBitPosition = __ffs(circleBitMask) - 1;

        // clear the least significant bit set to 1
        circleBitMask &= circleBitMask - 1;

        if (sharedMemorySlot >= 0) {
            // get the index of the circle in the global circle data array
            int circleIndex = circleBitMaskIndex * 32 + circleBitPosition;

            // load the circle data into shared memory
            sharedCircleData[sharedMemorySlot] = kernelCircleData[circleIndex];
            if (!isSnowflakes) {
                sharedCircleColors[sharedMemorySlot] = *(float3*)(&cuConstRendererParams.color[3 * circleIndex]);
            }
        }
        currentPos++;
    }
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
// modified to stop fetching from GPU global memory - threads put all circle
// data into a __shared__ memory before shadePixel
__device__ __inline__ void 
shadePixel(float2 pixelCenter, float4 circle, float4* imagePtr, float3 circleColor, bool isSnowflakes) {
    float diffX = circle.x - pixelCenter.x;
    float diffY = circle.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;
    float rad = circle.w;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (isSnowflakes) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-circle.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        rgb = circleColor;
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    *imagePtr = newColor;
    // END SHOULD-BE-ATOMIC REGION
}

// // kernelRenderCircles -- (CUDA device code)
// //
// // Each thread renders a circle.  Since there is no protection to
// // ensure order of update or mutual exclusion on the output image, the
// // resulting image will be incorrect.
// __global__ void kernelRenderCircles() {

//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index >= cuConstRendererParams.numberOfCircles)
//         return;

//     int index3 = 3 * index;

//     // Read position and radius
//     float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
//     float  rad = cuConstRendererParams.radius[index];

//     // Compute the bounding box of the circle. The bound is in integer
//     // screen coordinates, so it's clamped to the edges of the screen.
//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;
//     short minX = static_cast<short>(imageWidth * (p.x - rad));
//     short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
//     short minY = static_cast<short>(imageHeight * (p.y - rad));
//     short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

//     // A bunch of clamps.  Is there a CUDA built-in for this?
//     short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
//     short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
//     short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
//     short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;

//     // For all pixels in the bounding box
//     for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
//         float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
//         for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
//             float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
//                                                  invHeight * (static_cast<float>(pixelY) + 0.5f));
//             shadePixel(pixelCenterNorm, circles, imgPtr, index);
//             imgPtr++;
//         }
//     }
// }

// Each thread handles one circle by figuring out which squares this circle is inside
__global__ void kernelPickSquaresThatCircleTouches(unsigned int* kernelCircleInSquareBitMasks, float4* kernelCircleData, int kernelNumSquaresX, 
                                                   int kernelNumSquaresY, int kernelNumMaskWords){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int index3 = 3 * index;
    
    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // update the circle data with the position and radius
    kernelCircleData[index] = make_float4(p.x, p.y, p.z, rad);

    // figure out the bounding box of the circle
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // get the square bounds from the pixel bounds
    int squareMinX = screenMinX / SQUARE_SIZE;
    int squareMaxX = screenMaxX / SQUARE_SIZE;
    int squareMinY = screenMinY / SQUARE_SIZE;
    int squareMaxY = screenMaxY / SQUARE_SIZE;

    // get the circle's bit position and the index of the bit mask array for this circle
    unsigned int circleBitPositon = 1u << (index & 31);
    int circleBitMaskIndex = index >> 5;

    // for each square that the circle is in, set the bit for this circle in the bit mask array
    for (int squareY = squareMinY; squareY <= squareMaxY; squareY++) {
        // get the bottom and top bounds of the square
        float squareBottomBound = squareY * SQUARE_SIZE * invHeight;
        float squareTopBound = (squareY + 1) * SQUARE_SIZE * invHeight;

        for (int squareX = squareMinX; squareX <= squareMaxX; squareX++) {
            // get the left and right bounds of the square
            float squareLeftBound = squareX * SQUARE_SIZE * invWidth;
            float squareRightBound = (squareX + 1) * SQUARE_SIZE * invWidth;

            // check if the circle is in the square
            if (circleInBox(p.x, p.y, rad, squareLeftBound, squareRightBound, squareTopBound, squareBottomBound)) {
                // get the index of the bit mask array for this square
                int squareBitMaskIndex = (squareY * kernelNumSquaresX + squareX) * kernelNumMaskWords + circleBitMaskIndex;
                // set the bit for this circle in the bit mask array for this square
                atomicOr(&kernelCircleInSquareBitMasks[squareBitMaskIndex], circleBitPositon);
            }
        }
    }                                           
}


/**  new function 
 after forming a bit array of all touching circles for each tile, use prefix sum to
 get indexes of circles and render the circles using information in circle
 data (already saved as shared var)
*/
__global__ void
kernelRenderSquares(unsigned int* kernelCircleInSquareBitMasks, const float4* kernelCircleData,
                    int kernelNumSquaresX, int kernelNumMaskWords, bool isSnowflakes) {
    // variables for image bounds for current thread
    int imageX = blockIdx.x * SQUARE_SIZE + threadIdx.x; 
    int imageY = blockIdx.y * SQUARE_SIZE + threadIdx.y; 
    int width  = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;
    float invWidth  = 1.f / width;
    float invHeight = 1.f / height;
    
    // flattened 2d coordinate for cuda block into 1d
    int linearThreadIndex = threadIdx.y * SQUARE_SIZE + threadIdx.x;
    // to check whether this thread is doing real work within the image boundary
    bool threadInImageBounds = (!(imageX >= width || imageY >= height));
    
    float2 pixelCenterNorm;
    float4* imgPtr;
    float4 pixelColor;
    
    if (threadInImageBounds) {
        pixelCenterNorm = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
                                            invHeight * (static_cast<float>(imageY) + 0.5f));
        imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (imageY * width + imageX)]);
        pixelColor = *imgPtr;
    }

    // offsets the huge global array of all squares and their circle bit masks to 
    // see our current square's bitmask chunk
    unsigned int* currSquareBitMasks =(kernelCircleInSquareBitMasks + 
                        ((size_t)blockIdx.y * kernelNumSquaresX + blockIdx.x) * kernelNumMaskWords);
    
    // Shared memory allocations
    __shared__ float4 sharedCircleData[BLOCKSIZE];
    __shared__ float3 sharedCircleColors[BLOCKSIZE];

    __shared__ uint prefixSumInput[THREADS_PER_SQUARE];
    __shared__ uint prefixSumOutput[THREADS_PER_SQUARE];
    __shared__ uint prefixSumScratch[2 * THREADS_PER_SQUARE];
    
    // 256 threads execute this in parallel
    for (int i = 0; i < kernelNumMaskWords; i+= THREADS_PER_SQUARE) {
        // count bits (number of circles in this square) from bitmask
        int currCircleBitIndex = i + linearThreadIndex;
        unsigned int currCircleBit = 0; 
        // array bound check for squareBitMasks
        if (currCircleBitIndex < kernelNumMaskWords) {
            currCircleBit = currSquareBitMasks[currCircleBitIndex];
        }
        if (currCircleBit != 0) {
            currSquareBitMasks[currCircleBitIndex] = 0u; 
            // clear the shared mask array for next frame
        }
        
        // popc: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
        // we searched up a cuda bit counter function to count the number of 1s in a 32 bit number
        int count = __popc(currCircleBit);
        prefixSumInput[linearThreadIndex] = count;
        __syncthreads();

        sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, THREADS_PER_SQUARE);
        __syncthreads();

        // CHECKKKKK
        int myListStartIndex = prefixSumOutput[linearThreadIndex];

        int currNumCirclesFound = prefixSumOutput[THREADS_PER_SQUARE - 1] + prefixSumInput[THREADS_PER_SQUARE - 1];

        if (currNumCirclesFound == 0) {
            // skip and move on to save computation
            continue;
        }

        // batch process circles
        for (int j = 0; j < currNumCirclesFound; j += BLOCKSIZE) {
            if (count >0) {
                // write circles
                writeCirclesToSharedMem(currCircleBit, currCircleBitIndex, myListStartIndex, j, kernelCircleData, sharedCircleData, sharedCircleColors, isSnowflakes);
            }
            __syncthreads(); // ensure all circle data is loaded

            if (threadInImageBounds) {
                // edge case for very last batch might be uneven number of circles
                int numCirclesRender = min(BLOCKSIZE, currNumCirclesFound - j);
                for (int k =0; k < numCirclesRender; k++) {
                    shadePixel(pixelCenterNorm, sharedCircleData[k], &pixelColor, sharedCircleColors[k], isSnowflakes);
                }
            }
            __syncthreads();
        }
    }
    
    if (threadInImageBounds) {
        *imgPtr = pixelColor;
    }
}


// // new function to render pixel by pixel
// // Each thread renders a pixel (for the ordering fix) 
// __global__ void kernelRenderPixels() {
//     int imageX = blockIdx.x * blockDim.x + threadIdx.x; 
//     int imageY = blockIdx.y * blockDim.y + threadIdx.y; 

//     int width = cuConstRendererParams.imageWidth;
//     int height = cuConstRendererParams.imageHeight;
//     float invWidth = 1.f / width;
//     float invHeight = 1.f / height;

//     // variables for box bounds for each block
//     float leftBound = (blockIdx.x * blockDim.x) * invWidth;
//     float rightBound = ((blockIdx.x + 1) * blockDim.x) * invWidth;
//     float bottomBound = (blockIdx.y * blockDim.y) * invHeight;
//     float topBound = ((blockIdx.y + 1) * blockDim.y) * invHeight;

//     // to check whether this thread is doing real work within the image boundary
//     bool threadInImageBounds = (!(imageX >= width || imageY >= height));

//     float2 pixelCenterNorm;
//     float4* imgPtr;
//     float4 pixelColor;

//     if (threadInImageBounds){
//         pixelCenterNorm = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
//                                             invHeight * (static_cast<float>(imageY) + 0.5f));
//         imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (imageY * width + imageX)]);
//         pixelColor = *imgPtr;
//     }
     
//     __shared__ int sharedCircleIndices[BLOCKSIZE];

//     // flattened 2d coordinate into 1d
//     int linearThreadIndex =  threadIdx.y * blockDim.y + threadIdx.x;

//     // arrays needed for prefix sum logic
//     __shared__ uint prefixSumInput[BLOCKSIZE];
//     __shared__ uint prefixSumOutput[BLOCKSIZE];
//     __shared__ uint prefixSumScratch[2 * BLOCKSIZE];

//     // interleaved assigment of circles for each thread
//     for (int i = 0; i < cuConstRendererParams.numberOfCircles; i+= BLOCKSIZE) {
        
//         int threadCircleIdx = i + linearThreadIndex;
//         int circleInBlock = 0;

//         // dividing the work for each thread to check if the circle is in the block
//         if (threadCircleIdx < cuConstRendererParams.numberOfCircles) {
//             float3 circleData = *(float3*)(&cuConstRendererParams.position[threadCircleIdx*3]);
//             float circleRadius = cuConstRendererParams.radius[threadCircleIdx];

//             circleInBlock = circleInBox(circleData.x, circleData.y, circleRadius, leftBound, rightBound, topBound, bottomBound);
//         }

//         // prefixSumInput is a shared memory array that stores whether the circle is in the block or not for each thread
//         prefixSumInput[linearThreadIndex] = circleInBlock;
//         __syncthreads();

//         // do exclusive scan on shared memory to get the indices of the circles that are in the block
//         sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE);
//         __syncthreads();

//         // if circle is in block store its index in sharedCircleIndices
//         if (circleInBlock) {
//             sharedCircleIndices[prefixSumOutput[linearThreadIndex]] = threadCircleIdx;
//         }
//         __syncthreads();

//         // find the total number of circles in the block
//         int totalCirclesInBlock = prefixSumOutput[BLOCKSIZE - 1] + prefixSumInput[BLOCKSIZE - 1];

//         if (threadInImageBounds) {
//             // loop through only the compacted circles
//             for (int j = 0; j < totalCirclesInBlock; j++) {
//                 int circleIdx = sharedCircleIndices[j];
//                 float3 circleData = *(float3*)(&cuConstRendererParams.position[circleIdx*3]);
//                 shadePixel(pixelCenterNorm, circleData, &pixelColor, circleIdx);
//             }
//         }
//         __syncthreads();
//     }
//     // only assign color in threads in bounds working on the image
//     if (threadInImageBounds) {
//         *imgPtr = pixelColor;
//     }
// }  

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (circleInSquareBitMasks) {
        cudaFree(circleInSquareBitMasks);
    }

    if (circleData) {
        cudaFree(circleData);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    // round up the number of squares we need to split up the width and height of the image
    numSquaresX = (image->width + SQUARE_SIZE - 1) / SQUARE_SIZE;
    numSquaresY = (image->height + SQUARE_SIZE - 1) / SQUARE_SIZE;
    numMaskWords = (numberOfCircles + 31) / 32;
    bitMaskArrayAllocSize = sizeof(unsigned int) * (size_t)numSquaresX * (size_t)numSquaresY * numMaskWords;

    cudaMalloc(&circleInSquareBitMasks, bitMaskArrayAllocSize);
    cudaMalloc(&circleData, sizeof(float4) * (size_t)numberOfCircles);
    cudaMemset(circleInSquareBitMasks, 0, bitMaskArrayAllocSize);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    // 256 threads per block is a healthy number
    dim3 circlesInSquaresBlockDim(256, 1);
    dim3 circlesInSquaresGridDim((numberOfCircles + circlesInSquaresBlockDim.x - 1) / circlesInSquaresBlockDim.x);

    // first figure out which squares each circle is in and store that information in a bit mask array
    kernelPickSquaresThatCircleTouches<<<circlesInSquaresGridDim, circlesInSquaresBlockDim>>>(
    circleInSquareBitMasks, circleData, numSquaresX, numSquaresY, numMaskWords);
    
    dim3 renderSquaresBlockDim(SQUARE_SIZE, SQUARE_SIZE, 1);
    dim3 renderSquaresGridDim(numSquaresX, numSquaresY, 1);
    
    bool isSnowflakes = (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME);

    // then actually render the squares using the bit mask array and the circle data
    kernelRenderSquares<<<renderSquaresGridDim, renderSquaresBlockDim>>>(circleInSquareBitMasks, circleData, numSquaresX, numMaskWords, isSnowflakes);
    
    cudaDeviceSynchronize();
}