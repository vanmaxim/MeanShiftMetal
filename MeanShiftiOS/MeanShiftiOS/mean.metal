//
//  mean.metal
//  MeanShiftiOS
//
//  Created by Maksim on 12/20/18.
//  Copyright © 2018 Mapbox. All rights reserved.
//

#include <metal_stdlib>
#include <metal_math>

using namespace metal;

// Maybe halfs ?

kernel void square_sum(constant float4 *input    [[buffer(0)]],
                       device float *output    [[buffer(1)]],
                       constant uint &lenght   [[buffer(2)]],
                       uint gid                [[thread_position_in_grid]])
{
    if (gid >= lenght)
        return;
    
    int const i = static_cast<int>(gid);
    
    float4 const x = input[i];
    float4 const xx = pow(x, 2);
    
    output[gid] = xx[0] + xx[1] + xx[2] + xx[3];
}

kernel void calculate_force(constant float4 *input                      [[buffer(0)]],
                            constant float *squared_sum                 [[buffer(1)]],
                            texture2d<float, access::read_write> output [[texture(0)]],
                            constant float &sigma                       [[buffer(2)]],
                            uint2 gid                                   [[thread_position_in_grid]])
{
    uint const width = output.get_width();
    
    if (gid.x >= width || gid.y >= output.get_height())
        return;
    
    float const o = output.read(gid.xy)[0];
    if (o != 0.0)
        return;
    
    int const x = static_cast<int>(gid.x);
    int const y = static_cast<int>(gid.y);
    
    float const distance = squared_sum[x] + squared_sum[y] - 2 * dot(input[x], input[y]);
    float const v = exp(-0.5 * (distance / (sigma * sigma)));
    output.write(v, gid.xy);
    output.write(v, uint2(gid.y, gid.x));
}

kernel void normalize(texture2d<float, access::read> values         [[texture(0)]],
                      texture2d<float, access::read> normCoefs      [[texture(1)]],
                      texture2d<float, access::read_write> output   [[texture(2)]],
                      uint2 gid                                     [[thread_position_in_grid]])
{
    if (gid.x >= values.get_width() || gid.y >= values.get_height())
        return;
    
    float const v = values.read(gid.xy)[0];
    float const coef = normCoefs.read(uint2(0, gid.y))[0];
    
    float const nv = v / coef;
    output.write(nv, gid.xy);
}


