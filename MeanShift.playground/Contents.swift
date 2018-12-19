import Cocoa
import Metal
import simd

public func createLibrary() -> MTLLibrary {
    var library: MTLLibrary?
    do {
        let path = Bundle.main.path(forResource: "Shaders", ofType: "metal")
        let source = try String(contentsOfFile: path!, encoding: .utf8)
        library = try device.makeLibrary(source: source, options: nil)
    } catch let error as NSError {
        fatalError("library error: " + error.description)
    }
    return library!
}

public func getInput() -> [float4] {
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

let device = MTLCreateSystemDefaultDevice()!
let library = createLibrary()

let commandQueue = device.makeCommandQueue()!
let buffer = commandQueue.makeCommandBuffer()!
let encoder = buffer.makeComputeCommandEncoder()!

let data = getInput()

var width = data.count
var height = data.count

let input = device.makeBuffer(bytes: data, length: MemoryLayout<float4>.size * height, options: .storageModeShared)

let powSums = device.makeBuffer(length: MemoryLayout<Float>.size * height, options: .storageModeShared)

let _ = {
    
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(powSums, offset: 0, index: 1)
    encoder.setBytes(&height, length: MemoryLayout<Int>.size, index: 2)
    
    let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "pow_sum")!)
    let w = pipeline.maxTotalThreadsPerThreadgroup
    let threadGroupSize = MTLSizeMake(w, 1, 1);
    let threadGroups = MTLSizeMake((height + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1);
    encoder.setComputePipelineState(pipeline)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
}()

let forces = device.makeBuffer(length: MemoryLayout<Float>.size * width * height, options: .storageModeShared)

let _ = {
    
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(powSums, offset: 0, index: 1)
    encoder.setBuffer(forces, offset: 0, index: 2)
    encoder.setBytes(&width, length: MemoryLayout<Int>.size, index: 3)
    encoder.setBytes(&height, length: MemoryLayout<Int>.size, index: 4)
    
    let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "calculate_force")!)
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
}()

encoder.endEncoding()

buffer.commit()
buffer.waitUntilCompleted()

let ptr = forces?.contents().bindMemory(to: Float.self, capacity: width * height)
let buf = UnsafeBufferPointer(start: ptr, count: width * height)
let a = Array(buf)
