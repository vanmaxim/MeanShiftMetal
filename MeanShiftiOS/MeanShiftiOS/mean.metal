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
                       uint gid                 [[thread_position_in_grid]])
{
    if (gid >= lenght)
        return;
    
    int const i = static_cast<int>(gid);
    
    float4 const x = input[i];
    float4 const xx = pow(x, 2);
    
    output[i] = xx[0] + xx[1] + xx[2] + xx[3];
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
    
    float distance = squared_sum[x] + squared_sum[y] - 2 * dot(input[x], input[y]);
    if (sigma != 0.0) {
        distance = exp(-0.5 * (distance / (sigma * sigma)));
    }
    
    output.write(distance, gid.xy);
    output.write(distance, uint2(gid.y, gid.x));
}

kernel void normalize(texture2d<float, access::read> values     [[texture(0)]],
                      texture2d<float, access::read> normCoefs  [[texture(1)]],
                      texture2d<float, access::write> output     [[texture(2)]],
                      uint2 gid                                 [[thread_position_in_grid]])
{
    if (gid.x >= values.get_width() || gid.y >= values.get_height())
        return;
    
    float const v = values.read(gid.xy)[0];
    float const coef = normCoefs.read(uint2(0, gid.y))[0];
    
    float const nv = v / coef;
    output.write(nv, gid.xy);
}

/*
kernel void clasterize(texture2d<float, access::read> input [[texture(0)]],
                       device int *mask [[buffer(0)]],
                       volatile device atomic_int *cls [[ buffer(1) ]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= input.get_width())
        return;
    
    int const i = static_cast<int>(gid);
    int const w = static_cast<int>(input.get_width());
    
    if (mask[i] != 0.0)
        return;
    
    mask[i] = atomic_load_explicit(cls, memory_order_relaxed);
    
    for (int j = i + 1; j <= w; ++j) {
        if (mask[j] != 0.0)
            continue;
        float const val = input.read(uint2(i, j))[0];
        if (val < 0.1)
            mask[j] = atomic_load_explicit(cls, memory_order_relaxed);
    }
    
    cls++;
}
*/

kernel void clasterize(texture2d<float, access::read> input [[texture(0)]],
                       device int *mask                     [[buffer(0)]],
                       constant int &cls                    [[buffer(1)]],
                       constant int &i                      [[buffer(2)]],
                       uint gid                             [[thread_position_in_grid]])
{
    
    
    if (gid >= input.get_width())
        return;
    
    int const j = static_cast<int>(gid);
    
    if (mask[j] != 0)
        return;
    
    if (input.read(uint2(i, j))[0] < 0.1)
        mask[j] = cls;
}


