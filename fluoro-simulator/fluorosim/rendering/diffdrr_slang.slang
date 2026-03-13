/**
 * Differentiable Digitally Reconstructed Radiograph (DRR) using Slang Autodiff
 *
 * This implementation uses Slang's automatic differentiation to compute exact
 * gradients of the rendered image with respect to pose parameters.
 *
 * Key Features:
 * - Software trilinear interpolation for differentiable volume sampling
 * - [Differentiable] marked functions for gradient computation
 * - Forward and backward pass entry points
 * - Compatible with PyTorch autograd via slangpy
 *
 * Physics:
 *   I(u,v) = I₀ · exp(-∫ μ(s) ds)
 */

// ============================================================================
// Configuration
// ============================================================================

static const int MAX_STEPS = 2048;
static const float EPS = 1e-12f;
static const float PI = 3.14159265359f;

// ============================================================================
// Data Structures
// ============================================================================

struct Pose
{
    float3 rotation;     // Euler angles (rx, ry, rz) in radians
    float3 translation;  // Translation (tx, ty, tz) in mm
};

struct CarmGeometry
{
    float sdd;              // Source-to-Detector Distance (mm)
    float sid;              // Source-to-Isocenter Distance (mm)
    float2 detectorSize;    // Detector physical size (mm)
    int2 detectorPixels;    // Detector resolution (pixels)
    float pixelSpacing;     // Pixel pitch (mm)
};

struct VolumeInfo
{
    float3 spacing;     // Voxel spacing (X, Y, Z) in mm
    int3 dimensions;    // Volume dimensions (X, Y, Z) in voxels
    float3 origin;      // Volume origin in world coordinates (mm)
};

// ============================================================================
// Differentiable Helper Functions
// ============================================================================

/**
 * Convert Euler angles to rotation matrix (ZXY convention).
 * Marked [Differentiable] for gradient propagation.
 */
[Differentiable]
float3x3 eulerToRotationMatrix(float3 euler)
{
    float cx = cos(euler.x);
    float sx = sin(euler.x);
    float cy = cos(euler.y);
    float sy = sin(euler.y);
    float cz = cos(euler.z);
    float sz = sin(euler.z);

    // ZXY convention: R = Rz * Rx * Ry
    float3x3 R;
    R[0][0] = cz * cy - sz * sx * sy;
    R[0][1] = -sz * cx;
    R[0][2] = cz * sy + sz * sx * cy;
    R[1][0] = sz * cy + cz * sx * sy;
    R[1][1] = cz * cx;
    R[1][2] = sz * sy - cz * sx * cy;
    R[2][0] = -cx * sy;
    R[2][1] = sx;
    R[2][2] = cx * cy;

    return R;
}

/**
 * Transform a point from local C-arm frame to world frame.
 */
[Differentiable]
float3 transformPoint(float3 localPoint, float3x3 R, float3 translation)
{
    return mul(R, localPoint) + translation;
}

/**
 * Ray-box intersection using slab method.
 * Returns (tNear, tFar). If tFar < tNear, ray misses box.
 */
[Differentiable]
float2 rayBoxIntersect(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax)
{
    // Safe inverse direction
    float3 invDir;
    invDir.x = 1.0f / (abs(rayDir.x) > EPS ? rayDir.x : sign(rayDir.x) * EPS);
    invDir.y = 1.0f / (abs(rayDir.y) > EPS ? rayDir.y : sign(rayDir.y) * EPS);
    invDir.z = 1.0f / (abs(rayDir.z) > EPS ? rayDir.z : sign(rayDir.z) * EPS);

    float3 t0 = (boxMin - rayOrigin) * invDir;
    float3 t1 = (boxMax - rayOrigin) * invDir;

    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);

    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);

    return float2(tNear, tFar);
}

/**
 * Software trilinear interpolation - fully differentiable.
 *
 * The gradient flows through the interpolation weights which depend on uvw.
 * The actual voxel values are sampled with no_diff() since we don't need
 * gradients w.r.t. the volume data itself (only w.r.t. pose).
 */
[Differentiable]
float trilinearSample(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 uvw,
    int3 dimensions
)
{
    // Clamp UVW to valid range [0, 1]
    float3 uvwClamped = clamp(uvw, float3(0.0f), float3(1.0f));

    // Convert to voxel coordinates
    float3 voxelCoord = uvwClamped * float3(dimensions) - float3(0.5f);
    voxelCoord = clamp(voxelCoord, float3(0.0f), float3(dimensions) - float3(1.001f));

    // Get integer indices of the 8 surrounding voxels
    int3 idx0 = int3(floor(voxelCoord));
    int3 idx1 = min(idx0 + int3(1), dimensions - int3(1));

    // Compute fractional position within the voxel cell [0, 1]
    float3 frac = voxelCoord - float3(idx0);

    // Compute trilinear weights - these ARE differentiable w.r.t. uvw!
    float w000 = (1.0f - frac.x) * (1.0f - frac.y) * (1.0f - frac.z);
    float w100 = frac.x * (1.0f - frac.y) * (1.0f - frac.z);
    float w010 = (1.0f - frac.x) * frac.y * (1.0f - frac.z);
    float w110 = frac.x * frac.y * (1.0f - frac.z);
    float w001 = (1.0f - frac.x) * (1.0f - frac.y) * frac.z;
    float w101 = frac.x * (1.0f - frac.y) * frac.z;
    float w011 = (1.0f - frac.x) * frac.y * frac.z;
    float w111 = frac.x * frac.y * frac.z;

    // Sample the 8 corners using hardware texture sampling (non-differentiable)
    float3 texelSize = float3(1.0f) / float3(dimensions);

    float3 uv000 = (float3(idx0) + float3(0.5f)) * texelSize;
    float3 uv100 = (float3(idx1.x, idx0.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv010 = (float3(idx0.x, idx1.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv110 = (float3(idx1.x, idx1.y, idx0.z) + float3(0.5f)) * texelSize;
    float3 uv001 = (float3(idx0.x, idx0.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv101 = (float3(idx1.x, idx0.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv011 = (float3(idx0.x, idx1.y, idx1.z) + float3(0.5f)) * texelSize;
    float3 uv111 = (float3(idx1) + float3(0.5f)) * texelSize;

    // Sample with no_diff() - we don't need gradients through the texture lookup
    float v000 = no_diff(volume.Sample(sampler, uv000));
    float v100 = no_diff(volume.Sample(sampler, uv100));
    float v010 = no_diff(volume.Sample(sampler, uv010));
    float v110 = no_diff(volume.Sample(sampler, uv110));
    float v001 = no_diff(volume.Sample(sampler, uv001));
    float v101 = no_diff(volume.Sample(sampler, uv101));
    float v011 = no_diff(volume.Sample(sampler, uv011));
    float v111 = no_diff(volume.Sample(sampler, uv111));

    // Weighted sum - gradients flow through the weights
    return w000 * v000 + w100 * v100 + w010 * v010 + w110 * v110 +
           w001 * v001 + w101 * v101 + w011 * v011 + w111 * v111;
}

/**
 * Differentiable ray-march through volume.
 * Uses [MaxIters] to satisfy Slang's autodiff loop requirements.
 */
[Differentiable]
float rayMarchIntegral(
    Texture3D<float> volume,
    SamplerState sampler,
    float3 rayOrigin,
    float3 rayDir,
    float tStart,
    float tEnd,
    VolumeInfo volInfo,
    float stepMM
)
{
    float3 volumeSize = float3(volInfo.dimensions) * volInfo.spacing;
    float integral = 0.0f;

    float rayLength = tEnd - tStart;
    int numSteps = int(rayLength / stepMM) + 1;
    numSteps = min(numSteps, MAX_STEPS);

    // [MaxIters] tells Slang the maximum iterations for autodiff
    [MaxIters(2048)]
    for (int i = 0; i < numSteps; i++)
    {
        float t = tStart + (float(i) + 0.5f) * stepMM;
        if (t >= tEnd)
            break;

        float3 worldPos = rayOrigin + rayDir * t;
        float3 uvw = (worldPos - volInfo.origin) / volumeSize;

        float mu = trilinearSample(volume, sampler, uvw, volInfo.dimensions);
        integral += mu * stepMM;
    }

    return integral;
}

// ============================================================================
// Core Differentiable Rendering Function
// ============================================================================

/**
 * Compute the intensity for a single pixel using differentiable ray-marching.
 *
 * This function is marked [Differentiable] so Slang will automatically
 * generate the backward pass for gradient computation.
 */
[Differentiable]
float computePixelIntensity(
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    VolumeInfo volInfo,
    CarmGeometry carm,
    float3 rotation,      // Euler angles in radians
    float3 translation,   // Translation in mm
    int2 pixelCoord,
    float stepMM,
    float i0
)
{
    // Build rotation matrix from Euler angles
    float3x3 R = eulerToRotationMatrix(rotation);

    // Volume center (isocenter)
    float3 volumeCenter = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing * 0.5f;

    // Source position in world coordinates
    float3 sourceLocal = float3(0.0f, 0.0f, -carm.sid);
    float3 sourceWorld = transformPoint(sourceLocal, R, translation + volumeCenter);

    // Detector pixel position in world coordinates
    float detZ = carm.sdd - carm.sid;
    float2 pixelOffset = float2(
        (float(pixelCoord.x) + 0.5f - float(carm.detectorPixels.x) * 0.5f) * carm.pixelSpacing,
        (float(pixelCoord.y) + 0.5f - float(carm.detectorPixels.y) * 0.5f) * carm.pixelSpacing
    );

    float3 detectorLocal = float3(pixelOffset.x, pixelOffset.y, detZ);
    float3 detectorWorld = transformPoint(detectorLocal, R, translation + volumeCenter);

    // Ray direction (normalized)
    float3 rayDir = normalize(detectorWorld - sourceWorld);

    // Volume bounding box
    float3 boxMin = volInfo.origin;
    float3 boxMax = volInfo.origin + float3(volInfo.dimensions) * volInfo.spacing;

    // Ray-box intersection
    float2 tHit = rayBoxIntersect(sourceWorld, rayDir, boxMin, boxMax);

    // Check for valid intersection
    if (tHit.x > tHit.y || tHit.y < 0.0f)
    {
        return i0;
    }

    // Clamp to positive t
    float tStart = max(tHit.x, 0.0f);
    float tEnd = tHit.y;

    // Compute line integral
    float integral = rayMarchIntegral(
        muVolume, volumeSampler,
        sourceWorld, rayDir,
        tStart, tEnd,
        volInfo, stepMM
    );

    // Beer-Lambert law: I = I₀ * exp(-∫μds)
    return i0 * exp(-integral);
}

// ============================================================================
// Entry Points for slangpy
// ============================================================================

/**
 * Forward pass: Render DRR image.
 * Called via slangpy's dispatch for each pixel.
 */
[CudaKernel]
void renderDRR_forward(
    uint3 dispatchThreadID,
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    RWTexture2D<float> outputImage,
    VolumeInfo volInfo,
    CarmGeometry carm,
    Pose pose,
    float stepMM,
    float i0
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= carm.detectorPixels.x || pixelCoord.y >= carm.detectorPixels.y)
        return;

    float intensity = computePixelIntensity(
        muVolume, volumeSampler,
        volInfo, carm,
        pose.rotation, pose.translation,
        pixelCoord,
        stepMM, i0
    );

    outputImage[pixelCoord] = intensity;
}

/**
 * Backward pass: Compute gradients of loss w.r.t. pose.
 * Uses Slang's automatic differentiation via bwd_diff().
 */
[CudaKernel]
void renderDRR_backward(
    uint3 dispatchThreadID,
    Texture3D<float> muVolume,
    SamplerState volumeSampler,
    Texture2D<float> gradOutput,        // ∂L/∂I (read-only)
    RWTexture2D<float4> gradRotation,   // ∂L/∂(rx,ry,rz)
    RWTexture2D<float4> gradTranslation,// ∂L/∂(tx,ty,tz)
    VolumeInfo volInfo,
    CarmGeometry carm,
    Pose pose,
    float stepMM,
    float i0
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= carm.detectorPixels.x || pixelCoord.y >= carm.detectorPixels.y)
        return;

    // Get upstream gradient for this pixel
    float dL_dI = gradOutput[pixelCoord];

    // Create differential pairs for inputs we want gradients for
    DifferentialPair<float3> dpRotation = diffPair(pose.rotation, float3(0.0f));
    DifferentialPair<float3> dpTranslation = diffPair(pose.translation, float3(0.0f));

    // Wrap constants in diffPair with zero derivatives
    DifferentialPair<float> dpStepMM = diffPair(stepMM, 0.0f);
    DifferentialPair<float> dpI0 = diffPair(i0, 0.0f);

    // Call backward differentiation
    bwd_diff(computePixelIntensity)(
        muVolume, volumeSampler,
        volInfo, carm,
        dpRotation,
        dpTranslation,
        pixelCoord,
        dpStepMM, dpI0,
        dL_dI
    );

    // Store per-pixel gradients
    gradRotation[pixelCoord] = float4(dpRotation.d, 0.0f);
    gradTranslation[pixelCoord] = float4(dpTranslation.d, 0.0f);
}

/**
 * Normalize image to [0, 1] range.
 */
[CudaKernel]
void normalizeImage(
    uint3 dispatchThreadID,
    RWTexture2D<float> image,
    float minVal,
    float maxVal,
    float eps,
    int2 imageSize
)
{
    int2 pixelCoord = int2(dispatchThreadID.xy);

    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y)
        return;

    float val = image[pixelCoord];
    image[pixelCoord] = (val - minVal) / (maxVal - minVal + eps);
}
