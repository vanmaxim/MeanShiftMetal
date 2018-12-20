
#include <metal_math>

using namespace metal;

// Maybe halfs ?

kernel void square_sum(device float4 *input    [[buffer(0)]],
                       device float *output    [[buffer(1)]],
                       constant uint &lenght   [[buffer(2)]],
                       uint gid                [[thread_position_in_grid]])
{
    if (gid >= lenght)
        return;
    
    float4 const x = input[gid];
    float4 const xx = pow(x, 2);
    
    output[gid] = xx[0] + xx[1] + xx[2] + xx[3];
}

kernel void calculate_force(device float4 *input                   [[buffer(0)]],
                            device float *squared_sum              [[buffer(1)]],
                            texture2d<float, access::write> output [[texture(0)]],
                            constant float &sigma                  [[buffer(2)]],
                            uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height())
        return;
    
    float const distance = squared_sum[gid.x] + squared_sum[gid.y] - 2.0 * dot(input[gid.x], input[gid.y]);
    
    output.write(exp(-0.5 * (distance / (sigma * sigma))), gid.xy);
}

kernel void normalize(texture2d<float, access::read_write> values [[texture(0)]],
                      texture1d<float, access::read> normCoefs    [[texture(1)]],
                      uint2 gid                                   [[thread_position_in_grid]])
{
    if (gid.x >= values.get_width() || gid.y >= values.get_height())
        return;
    
    float v = values.read(gid.xy)[0];
    float coef = normCoefs.read(gid.x)[0];
    values.write(v / coef, gid.xy);
}
