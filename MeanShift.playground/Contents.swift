import Cocoa
import Metal
import simd
import MetalPerformanceShaders

func createLibrary() -> MTLLibrary {
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

var sigma: Float = 0.7
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

encoder.setBuffer(input, offset: 0, index: 0)
encoder.setBuffer(powSums, offset: 0, index: 1)
encoder.setBytes(&height, length: MemoryLayout<Int>.size, index: 2)
    
dispathGridAsRow(device: device,
                 function: library.makeFunction(name: "pow_sum")!,
                 encoder: encoder,
                 rowLenght: height)

let forces = device.makeTexture(descriptor: MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                                     width: width,
                                                                                     height: height,
                                                                                     mipmapped: false))!

encoder.setBuffer(input, offset: 0, index: 0)
encoder.setBuffer(powSums, offset: 0, index: 1)
encoder.setTexture(forces, index: 0)
encoder.setBytes(&sigma, length: MemoryLayout<Float>.size, index: 2)

dispathGridAsMatrix(device: device,
                    function: library.makeFunction(name: "calculate_force")!,
                    encoder: encoder,
                    width: width,
                    height: height)

encoder.endEncoding()

let reduce = MPSNNReduceColumnSum(device: device)
let reduced = MPSImage(device: device,
                       imageDescriptor: MPSImageDescriptor(channelFormat: .float32, width: width, height: 1, featureChannels: 1))

reduce.encode(commandBuffer: buffer,
              sourceImage: MPSImage(texture: forces, featureChannels: 1),
              destinationImage: reduced)

let normEncoder = buffer.makeComputeCommandEncoder()!

normEncoder.setTexture(forces, index: 0)
normEncoder.setTexture(reduced.texture, index: 1)

dispathGridAsMatrix(device: device,
                    function: library.makeFunction(name: "normalize")!,
                    encoder: normEncoder,
                    width: width,
                    height: height)

normEncoder.endEncoding()

buffer.commit()
buffer.waitUntilCompleted()

forces.toFloatArray(width: width, height: height, featureChannels: 1)
