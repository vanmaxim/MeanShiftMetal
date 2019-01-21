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

kernel void square_sum(constant float4 *input   [[buffer(0)]],
                       device float *output     [[buffer(1)]],
                       constant uint &lenght    [[buffer(2)]],
                       ushort gid               [[thread_position_in_grid]])
{
    if (gid >= lenght)
        return;
    
    ushort const i = static_cast<ushort>(gid);
    
    float4 const x = input[i];
    float4 const xx = pow(x, 2);
    
    output[i] = xx[0] + xx[1] + xx[2] + xx[3];
}

kernel void calculate_distances(constant float4 *input                      [[buffer(0)]],
                                constant float *squared_sum                 [[buffer(1)]],
                                texture2d<float, access::read_write> output [[texture(0)]],
                                constant float &sigma                       [[buffer(2)]],
                                ushort2 gid                                 [[thread_position_in_grid]])
{
    float const o = output.read(gid)[0];
    if (o != 0.0)
        return;
    
    ushort const x = static_cast<ushort>(gid.x);
    ushort const y = static_cast<ushort>(gid.y);
    
    float distance = squared_sum[x] + squared_sum[y] - 2 * dot(input[x], input[y]);
    if (sigma != 0.0) {
        distance = exp(-0.5 * (distance / (sigma * sigma)));
    }
    
    output.write(distance, gid.xy);
    output.write(distance, uint2(gid.y, gid.x));
}

kernel void normalize(texture2d<float, access::read> values     [[texture(0)]],
                      texture2d<float, access::read> normCoefs  [[texture(1)]],
                      texture2d<float, access::write> output    [[texture(2)]],
                      ushort2 gid                               [[thread_position_in_grid]])
{
    float const v = values.read(gid)[0];
    float const coef = normCoefs.read(ushort2(0, gid.y))[0];
    
    float const nv = v / coef;
    output.write(nv, gid.xy);
}
