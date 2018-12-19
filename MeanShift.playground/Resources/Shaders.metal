
#include <metal_math>

using namespace metal;

// Maybe halfs ?

kernel void pow_sum(device float4 *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int &height [[buffer(2)]], // lenght
                    uint gid             [[thread_position_in_grid]])
{
    if (gid >= static_cast<uint>(height))
        return;
    
    float4 x = input[gid];
    float4 xx = pow(x, 2);
    output[gid] = xx[0] + xx[1] + xx[2] + xx[3];
}

kernel void calculate_force(device float4 *input                [[buffer(0)]],
                            device float *pow_sum               [[buffer(1)]],
                            constant int &width                 [[buffer(2)]], // use from out texture
                            constant int &height                [[buffer(3)]],
                            texture2d<float, access::write> out [[texture(0)]],
                            uint2 gid                           [[thread_position_in_grid]])
{
    if(gid.x >= static_cast<uint>(width) || gid.y >= static_cast<uint>(height))
        return;
    
    float const sigma = 0.7;
    float const distance = pow_sum[gid.x] + pow_sum[gid.y] - 2.0 * dot(input[gid.x], input[gid.y]);
    
    out.write(exp(-0.5 * (distance / (sigma * sigma))), gid.xy);
}

kernel void normalize(texture2d<float, access::read_write> input [[texture(0)]],
                      texture1d<float, access::read> norms       [[texture(1)]],
                      uint2 gid                                  [[thread_position_in_grid]])
{
    float v = input.read(gid.xy)[0];
    float norm = norms.read(gid.x)[0];
    input.write(v / norm, gid.xy);
}
