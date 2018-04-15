#include <chrono>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>

#include <iostream>
#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvUtils.h>
using namespace nvinfer1;
using namespace nvuffparser;

#include <cuda_runtime_api.h>

#include "common.h"
static Logger gLogger;

// refer to this:
// https://github.com/dusty-nv/jetson-inference/blob/8ed492bfdc9e1b98f7711cd5ae224ce588635656/tensorNet.cpp#L192
// for creating a class

#define MAX_WORKSPACE (1 << 20)
#define MAX_BATCHSIZE 1
#define INPUT_H 300
#define INPUT_W 400
#define RETURN_AND_LOG(ret, severity, message)              \
    do {                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                              \
    } while(0)                                              \

/**
 * calculate the volume of a tensor with dimension of [d]
 * \param d
 * \return the volume of the tensor
 */
inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}
inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

/**
 * locate the file with the given filename and original directory
 * \param input
 * \return the file path
 */
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/"};
    return locateFile(input,dirs);
}

/**
 * allocate memory on GPU
 * \param memSize
 * \return memory pointer
 */
void* safeCudaMalloc(int64_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cout << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

void createInputBuffer(int64_t volume, DataType dataType, void *gpuMem, std::string fileName)
{
    assert(gpuMem);
    assert(volume == INPUT_H * INPUT_W);
    assert(elementSize(dataType) == sizeof(float));

    cv::Mat src = cv::imread(locateFile(fileName, std::vector<string>{"data/"}), 0);
    if (!src.data) {
        std::cout << "cannot read the image..." << std::endl;
    }
    std::cout << "input image size: " << src.size() << std::endl;
    float *srcInputData = new float[volume];
    for (int i = 0; i < volume; ++i) {
        srcInputData[i] = float(src.data[i]) / 255.0f;
    }

    int64_t memSize = volume * elementSize(dataType);
    assert(sizeof(float) == elementSize(dataType) && "dataType error!");
    CHECK(cudaMemcpy(gpuMem, srcInputData, memSize, cudaMemcpyHostToDevice));

    delete [] srcInputData;
}

float* getOutputBufferData(int64_t volume, DataType dataType, void* buffer)
{
    int64_t memSize = volume * elementSize(dataType);
    float *output = new float[volume];
    CHECK(cudaMemcpy(output, buffer, memSize, cudaMemcpyDeviceToHost));
    return output;
}

/**
 * calculate sizes of all the binding tensor for buffer allocation on the device
 * \param engine
 * \param nbBindings
 * \param batchSize
 * \return vector of binding tensor infos: [(size1, dataType1)-(size2, dataType2)-...]
 */
std::vector<std::pair<int64_t, DataType >>
calculateBindingBufferSizes(const ICudaEngine& engine,
                            int nbBindings,
                            int batchSize) {

    /* calculate sizes of each binding tensor
     * [tensor1, datatype1]-[tensor2, datatype2]-...*/
    std::vector<std::pair<int64_t , DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine.getBindingDimensions(i);
        DataType tensorType = engine.getBindingDataType(i);

        int64_t tensorVolume = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(tensorVolume, tensorType));
    }
    return sizes;
}



ICudaEngine* loadModelAndCreateEngine(const char *uffFile,
                                      IUffParser *parser)
{
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetwork();

    if (!parser->parse(uffFile, *network, DataType::kFLOAT)) {
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    }

    builder->setMaxBatchSize(MAX_BATCHSIZE);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (!engine) {
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    }

    network->destroy();
    builder->destroy();

    return engine;
}

void execute(ICudaEngine &engine)
{
    IExecutionContext *context = engine.createExecutionContext();

    int nbBindings = engine.getNbBindings();
    int nbLayers = engine.getNbLayers();
    std::cout << "Bindings: " << nbBindings
              << ", index:" << engine.getBindingIndex("input")
              << ", nbDims:" << engine.getBindingDimensions(0).nbDims << std::endl;
    std::cout << "Layers: " << nbLayers
              << ", index: " << engine.getBindingIndex("conv12/output")
              << ", nbDims:" << engine.getBindingDimensions(1).nbDims << std::endl;
    assert(nbBindings == 2);



    std::vector<void*> buffers(nbBindings);// allocated memory for input and output
    const int batchSize = 1;
    auto bufferSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIndexInput = 0;
    int bindingIndexOutput = 1;
    auto bufferSizeInput = bufferSizes[bindingIndexInput]; // [size, type]
    auto bufferSizeOutput = bufferSizes[bindingIndexOutput];

    // allocate gpu memory for input data
    buffers[bindingIndexInput] = safeCudaMalloc(bufferSizeInput.first *
                                                elementSize(bufferSizeInput.second));
    // allocate gpu memory for output data
    buffers[bindingIndexOutput] = safeCudaMalloc(bufferSizeOutput.first *
                                                 elementSize(bufferSizeOutput.second));

    for (int i = 0; i < 7; ++i) {
        // here!!!!!!!!!!!!!
        createInputBuffer(bufferSizeInput.first, bufferSizeInput.second, buffers[bindingIndexInput], )
        auto t_start = std::chrono::high_resolution_clock::now();
        context->execute(batchSize, &buffers[bindingIndexInput]);
        // copy the device data to the host
        float *output = getOutputBufferData(bufferSizeOutput.first,
                                            bufferSizeOutput.second,
                                            buffers[bindingIndexOutput]);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        cv::Mat dstImg(INPUT_H, INPUT_W, CV_32FC1, output);
        cv::imshow("dst", dstImg);
        cv::imwrite("dst.png", dstImg);
        std::cout << "done..." << std::endl;
        std::cout << "time: " << ms << "ms" << std::endl;
    }
}

int main(int argc, char** argv)
{

    auto filename = locateFile("model.uff");
    std::cout << filename << std::endl;

    auto parser = createUffParser();

    parser->registerInput("input", DimsCHW(1, 300, 400));
    parser->registerOutput("conv12/output");

    ICudaEngine *engine = loadModelAndCreateEngine(filename.c_str(), parser);

    if (!engine) {
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    }

    parser->destroy();
    auto t_start = std::chrono::high_resolution_clock::now();
    execute(*engine);
    engine->destroy();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "total time: " << ms << "ms" << std::endl;
    cv::waitKey(0);
    return EXIT_SUCCESS;
}