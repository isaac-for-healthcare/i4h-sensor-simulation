static __global__ void scan_convert_phased_kernel(cudaTextureObject_t input, uint2 input_size,
                                                  float* __restrict__ output, uint2 output_size,
                                                  float sector_angle, float far) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Black out pixels by default
  output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();

  // Convert sector_angle to radians and calculate half angle
  const float sector_angle_rad = (sector_angle / 180.0f) * M_PI;
  const float half_angle_rad = sector_angle_rad / 2.0f;

  // For phased array, the origin is at the center top of the image
  // Map pixel coordinates to normalized coordinates [0,1] x [0,1]
  const float u = float(index.x) / float(output_size.x - 1);  // [0,1] horizontal
  const float v = float(index.y) / float(output_size.y - 1);  // [0,1] vertical

  // Transform to physical coordinates (centered at origin, origin at top center)
  // For a sector scan, the top center (u=0.5, v=0) is the origin (probe position)
  const float x = (u - 0.5f);  // [-0.5, 0.5] centered horizontally
  const float y = v;           // [0, 1] depth from top

  // Skip pixels at or above the origin
  if (y <= 0.0f) { return; }

  // Convert to polar coordinates (r, theta) where:
  // - r is the normalized distance from the origin
  // - theta is the angle from the central ray (vertical down)
  const float theta = atan2f(x, y);

  // If outside the sector angle, skip this pixel
  if (fabsf(theta) > half_angle_rad) { return; }

  // Calculate the relative depth based on the distance from origin
  const float r = sqrtf(x * x + y * y);

  // Scale the radius to match the physical depth
  // Use a direct linear mapping for depth to preserve physical dimensions
  const float depth = r * far;

  // Normalized depth for texture lookup
  const float source_x = depth / far;

  // Check if we're beyond the max imaging depth
  if (source_x > 1.0f) { return; }

  // Calculate the texture column based on the angle
  // Map theta from [-half_angle, half_angle] to [0, 1] for texture lookup
  const float source_y = (theta + half_angle_rad) / sector_angle_rad;

  // Sample the texture at the calculated coordinates
  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}
