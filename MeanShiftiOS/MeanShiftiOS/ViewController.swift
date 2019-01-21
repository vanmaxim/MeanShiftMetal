//
//  ViewController.swift
//  MeanShiftiOS
//
//  Created by Maksim on 12/20/18.
//  Copyright © 2018 Mapbox. All rights reserved.
//

import UIKit

import Metal
import simd
import MetalPerformanceShaders

class ViewController: UIViewController {
    
    let device = MTLCreateSystemDefaultDevice()!
    let library = MTLCreateSystemDefaultDevice()!.makeDefaultLibrary()!
    let commandQueue = MTLCreateSystemDefaultDevice()!.makeCommandQueue()!
    
    
    @IBAction func action(_ sender: Any) {
        var sigma: Float = 0.7
        
        let data = getInput()
        
        let start = DispatchTime.now()
        
        let width = data.count
        var height = data.count
        
        let input = device.makeBuffer(bytes: data, length: MemoryLayout<float4>.size * height, options: .storageModeShared)!
        
        let inputMatrix = MPSMatrix(buffer: input, descriptor: MPSMatrixDescriptor(rows: height, columns: 4, rowBytes: MemoryLayout<Float>.size * 4, dataType: .float32))
        
        let squaredSum = device.makeBuffer(length: MemoryLayout<Float>.size * height, options: .storageModePrivate)!
        
        let forcesDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                  width: width,
                                                                  height: height,
                                                                  mipmapped: false)
        forcesDesc.resourceOptions = .storageModePrivate
        forcesDesc.usage = [.shaderWrite, .shaderRead]
        
        let reduce = MPSImageReduceRowSum(device: device)
        
        let reducedDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 1, height: height, mipmapped: false)
        reducedDesc.usage = [.shaderWrite, .shaderRead]
        reducedDesc.storageMode = .private
        let reduced = device.makeTexture(descriptor: reducedDesc)!
        
        let normalizedDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        normalizedDesc.usage = .shaderWrite
        normalizedDesc.storageMode = .private
        let normalized = device.makeTexture(descriptor: normalizedDesc)!
        
        let forcesBuff = device.makeBuffer(length: MemoryLayout<Float>.size * width * height, options: .storageModeShared)!
        
        let forcesMatrix = MPSMatrix(buffer: forcesBuff,
                                     descriptor: MPSMatrixDescriptor(rows: height, columns: width, rowBytes: MemoryLayout<Float>.size * width, dataType: .float32))
        let buffer = commandQueue.makeCommandBuffer()!
        for _ in 0..<6 {

            let encoder = buffer.makeComputeCommandEncoder()!
            
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(squaredSum, offset: 0, index: 1)
            encoder.setBytes(&height, length: MemoryLayout<Int>.size, index: 2)
            
            let forces = device.makeTexture(descriptor: forcesDesc)!
            
            dispathGridAsRow(device: device,
                             function: library.makeFunction(name: "square_sum")!,
                             encoder: encoder,
                             rowLenght: height)
            
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(squaredSum, offset: 0, index: 1)
            encoder.setTexture(forces, index: 0)
            encoder.setBytes(&sigma, length: MemoryLayout<Float>.size, index: 2)
            
            dispathGridAsMatrix(device: device,
                                function: library.makeFunction(name: "calculate_distances")!,
                                encoder: encoder,
                                width: width,
                                height: height)
            
            encoder.endEncoding()
            
            reduce.encode(commandBuffer: buffer, sourceTexture: forces, destinationTexture: reduced)
            
            let normEncoder = buffer.makeComputeCommandEncoder()!
            
            normEncoder.setTexture(forces, index: 0)
            normEncoder.setTexture(reduced, index: 1)
            normEncoder.setTexture(normalized, index: 2)
            
            dispathGridAsMatrix(device: device,
                                function: library.makeFunction(name: "normalize")!,
                                encoder: normEncoder,
                                width: width,
                                height: height)
            
            normEncoder.endEncoding()
            
            let blit = buffer.makeBlitCommandEncoder()!
            
            blit.copy(from: normalized, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: width, height: height, depth: 1), to: forcesBuff, destinationOffset: 0, destinationBytesPerRow: MemoryLayout<Float>.size * width, destinationBytesPerImage: forcesBuff.length)
            
            blit.endEncoding()
            
            let mult = MPSMatrixMultiplication(device: device, resultRows: inputMatrix.rows, resultColumns: inputMatrix.columns, interiorColumns: forcesMatrix.columns)

            mult.encode(commandBuffer: buffer,
                        leftMatrix: forcesMatrix,
                        rightMatrix: inputMatrix,
                        resultMatrix: inputMatrix)
        }

        
        let sedEnc = buffer.makeComputeCommandEncoder()!
        
        let distancesDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                     width: width,
                                                                     height: height,
                                                                     mipmapped: false)
        //distancesDesc.resourceOptions = .storageModePrivate
        distancesDesc.usage = [.shaderWrite, .shaderRead]
        let distances = device.makeTexture(descriptor: distancesDesc)!
        
        sedEnc.setBuffer(inputMatrix.data, offset: 0, index: 0)
        sedEnc.setBuffer(squaredSum, offset: 0, index: 1)
        sedEnc.setBytes(&height, length: MemoryLayout<Int>.size, index: 2)
        
        dispathGridAsRow(device: device,
                         function: library.makeFunction(name: "square_sum")!,
                         encoder: sedEnc,
                         rowLenght: height)
        
        sedEnc.setBuffer(input, offset: 0, index: 0)
        sedEnc.setBuffer(squaredSum, offset: 0, index: 1)
        var sedSigm = 0
        sedEnc.setBytes(&sedSigm, length: MemoryLayout<Float>.size, index: 2)
        sedEnc.setTexture(distances, index: 0)
        
        dispathGridAsMatrix(device: device,
                            function: library.makeFunction(name: "calculate_distances")!,
                            encoder: sedEnc,
                            width: width,
                            height: height)
        sedEnc.endEncoding()
        
        buffer.commit()
        buffer.waitUntilCompleted()
        
        var mask = [Int32](repeating: 0, count: width)
        let S = distances.toFloatArray(width: distances.width, height: distances.height, featureChannels: distances.depth)
        
        var cls: Int32 = 1
        for i in 0..<mask.count {
            if mask[i] != 0 {
                continue
            }
            mask[i] = cls
            
            for j in (i + 1)..<mask.count {
                if mask[j] != 0 {
                    continue
                }
                let idx = j * distances.width + i;
                if S[idx] < 0.1 {
                    mask[j] = cls;
                }
            }
            cls += 1
        }
        
        let end = DispatchTime.now()
        
        let diff = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        print(diff)
        
        
        let testRes: [Int32] =
            [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 2, 2, 2, 3, 3, 2, 2, 4, 2, 5, 2, 5, 5, 5,
            5, 2, 6, 5, 5, 2, 2, 6, 6, 6, 5, 5, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6]
        
        assert(mask == testRes)
        
        print("Hello")
    }
 
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    func getInput() -> [float4] {
        let path = Bundle.main.url(forResource: "input", withExtension: "txt")!
        let data: [float4] = try! String(contentsOf: path, encoding: .utf8)
            .components(separatedBy: .newlines)
            .dropLast()
            .map {
                let f = $0.components(separatedBy: " ").map { Float($0)! }
                return float4(f[0], f[1], f[2], f[3])
        }
        return data
    }
    
    func dispathGridAsRow(device: MTLDevice, function: MTLFunction, encoder: MTLComputeCommandEncoder, rowLenght: Int) {
        let pipeline = try! device.makeComputePipelineState(function: function)
        let w = pipeline.maxTotalThreadsPerThreadgroup
        let threadGroupSize = MTLSizeMake(w, 1, 1);
        let threadGroups = MTLSizeMake((rowLenght + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1);
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
    
    func dispathGridAsMatrix(device: MTLDevice, function: MTLFunction, encoder: MTLComputeCommandEncoder, width: Int, height: Int) {
        let pipeline = try! device.makeComputePipelineState(function: function)
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadGroupSize = MTLSizeMake(w, h, 1)
        let threadGroups = MTLSizeMake(
            (width  + threadGroupSize.width  - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        )
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
}
