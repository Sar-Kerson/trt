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

#define INPUT_NAME "input"
#define OUTPUT_NAME "output"
#define MID_INPUT_NAME "conv6/bn"
#define MID_OUTPUT_NAME "conv6/conv"

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
    std::cout << input << std::endl;
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

void createInputBuffer(int64_t volume, DataType dataType, void *gpuMem, std::string fileName, cudaStream_t &stream)
{
    assert(gpuMem);
    assert(volume == INPUT_H * INPUT_W);
    assert(elementSize(dataType) == sizeof(float));

    cv::Mat src = cv::imread(fileName, 0);
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
    CHECK(cudaMemcpyAsync(gpuMem, srcInputData, memSize, cudaMemcpyHostToDevice, stream));

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

void execute(ICudaEngine &engine, ICudaEngine &engine_)
{
    IExecutionContext *context = engine.createExecutionContext();
    IExecutionContext *context_ = engine_.createExecutionContext();

    int nbBindings = engine.getNbBindings();
    int nbLayers = engine.getNbLayers();
    std::cout << "Bindings: " << nbBindings
              << ", index:" << engine.getBindingIndex(INPUT_NAME)
              << ", nbDims:" << engine.getBindingDimensions(0).nbDims << std::endl;
    std::cout << "Layers: " << nbLayers
              << ", index: " << engine.getBindingIndex(MID_OUTPUT_NAME)
              << ", nbDims:" << engine.getBindingDimensions(1).nbDims << std::endl;
    assert(nbBindings == 2);

    int nbBindings_ = engine_.getNbBindings();
    nbLayers = engine_.getNbLayers();
    std::cout << "Bindings: " << nbBindings_
              << ", index:" << engine_.getBindingIndex(MID_INPUT_NAME)
              << ", nbDims:" << engine_.getBindingDimensions(0).nbDims << std::endl;
    std::cout << "Layers: " << nbLayers
              << ", index: " << engine_.getBindingIndex(OUTPUT_NAME)
              << ", nbDims:" << engine_.getBindingDimensions(1).nbDims << std::endl;
    assert(nbBindings_ == 2);


    std::vector<void*> buffers(2), buffers_(2);// allocated memory for input and output
    const int batchSize = 1;
    auto bufferSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);
    auto bufferSizes_ = calculateBindingBufferSizes(engine_, nbBindings_, batchSize);

    int bindingIndexInput = 0;
    int bindingIndexOutput = 1;
    auto bufferSizeInput = bufferSizes[bindingIndexInput]; // [size, type]
    auto bufferSizeMidOutput = bufferSizes[bindingIndexOutput];
    auto bufferSizeMidInput = bufferSizes_[bindingIndexInput];
    auto bufferSizeOutput = bufferSizes_[bindingIndexOutput];

    // allocate gpu memory for input data
    buffers[bindingIndexInput] = safeCudaMalloc(bufferSizeInput.first *
                                                elementSize(bufferSizeInput.second));
    // allocate gpu memory for output data
    buffers[bindingIndexOutput] = safeCudaMalloc(bufferSizeMidOutput.first *
                                                 elementSize(bufferSizeMidOutput.second));
    // allocate gpu memory for input data
    buffers_[bindingIndexInput] = safeCudaMalloc(bufferSizeMidInput.first *
                                                elementSize(bufferSizeMidInput.second));
    // allocate gpu memory for output data
    buffers_[bindingIndexOutput] = safeCudaMalloc(bufferSizeOutput.first *
                                                 elementSize(bufferSizeOutput.second));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cv::Mat temp(INPUT_H, INPUT_W, CV_8UC1);
    for (int i = 0; i <= 7; ++i) {
        std::string str = "t_pre" + std::to_string(i) + ".png";
        std::cout << str << std::endl;
        std::string fileName = locateFile(str);
        // feed the imgae to gpu memory: (void*) buffers[bindingIndexInput]
        createInputBuffer(bufferSizeInput.first, bufferSizeInput.second, buffers[bindingIndexInput], fileName, stream);
        auto t_start = std::chrono::high_resolution_clock::now();
        // run the engine, and feed the gpu input to the context for inference
        context->enqueue(batchSize, &buffers[bindingIndexInput], stream, nullptr);
        //---------------bn-layer!!!!!!!!!!!!!!!-------------------
        // from buffer[output]--to-->buffer_[input]
        //---------------------------------------------------------
        // feed the mid-input from the device directly
        CHECK(cudaMemcpyAsync(buffers_[bindingIndexInput], buffers[bindingIndexOutput], 64 * INPUT_H * INPUT_W *
                sizeof(float), cudaMemcpyDeviceToDevice, stream));
        context_->enqueue(batchSize, &buffers_[bindingIndexInput], stream, nullptr);
        cudaStreamSynchronize(stream);
        // copy the output from the device to the host
        float *output = getOutputBufferData(bufferSizeOutput.first,
                                            bufferSizeOutput.second,
                                            buffers_[bindingIndexOutput]);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        cv::Mat dstImg(INPUT_H, INPUT_W, CV_32FC1, output);
        temp = dstImg * 255;
        cv::imwrite(std::to_string(i) + "dst.png", temp);
        std::cout << "done..." << std::endl;
        std::cout << "time: " << ms << "ms" << std::endl;
    }
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[bindingIndexInput]));
    CHECK(cudaFree(buffers[bindingIndexOutput]));
    CHECK(cudaFree(buffers_[bindingIndexInput]));
    CHECK(cudaFree(buffers_[bindingIndexOutput]));
}

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> size;
        wt.type = DataType::kFLOAT;

        float *val = reinterpret_cast<float *>(malloc(sizeof(float) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> val[x];
        }
        wt.values = val;
        printf("nums of values in %s: %u)\n", name.c_str(), size);
        wt.count = size;
        weightMap[name.c_str()] = wt;
    }
    return weightMap;
}

// Creat the Engine using only the API and not any parser.
std::pair<ICudaEngine *, ICudaEngine *>
createNetworkEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
    INetworkDefinition* network = builder->createNetwork();
    INetworkDefinition* network_ = builder->createNetwork();


    auto data = network->addInput(INPUT_NAME, dt, DimsCHW{ 1, INPUT_H, INPUT_W});
    assert(data != nullptr);
    std::map<std::string, Weights> weightMap = loadWeights(locateFile("weights.wt"));

    auto conv1 = network->addConvolution(*data, 64, DimsHW{3, 3}, weightMap["conv1/weights"], weightMap["conv1/biases"]);
    conv1->setStride(DimsHW{1, 1});
    conv1->setPadding(DimsHW{1, 1});
    auto relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    auto conv2 = network->addConvolution(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2/weights"], weightMap["conv2/biases"]);
    conv2->setStride(DimsHW{1, 1});
    conv2->setPadding(DimsHW{1, 1});
    auto relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

    auto conv3 = network->addConvolution(*relu2->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv3/weights"], weightMap["conv3/biases"]);
    conv3->setStride(DimsHW{1, 1});
    conv3->setPadding(DimsHW{1, 1});
    auto relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);

    auto conv4 = network->addConvolution(*relu3->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv4/weights"], weightMap["conv4/biases"]);
    conv4->setStride(DimsHW{1, 1});
    conv4->setPadding(DimsHW{1, 1});
    auto relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);

    auto conv5 = network->addConvolution(*relu4->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv5/weights"], weightMap["conv5/biases"]);
    conv5->setStride(DimsHW{1, 1});
    conv5->setPadding(DimsHW{1, 1});
    auto relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);

    auto conv6 = network->addConvolution(*relu5->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv6/weights"], weightMap["conv6/biases"]);
    conv6->setStride(DimsHW{1, 1});
    conv6->setPadding(DimsHW{1, 1});
    conv6->getOutput(0)->setName(MID_OUTPUT_NAME);
    network->markOutput(*conv6->getOutput(0));

    // --------------bn-------------------------
    auto data_ = network_->addInput(MID_INPUT_NAME, dt, DimsCHW{64, INPUT_H, INPUT_W});
    assert(data_ != nullptr);
    auto scale6 = network_->addScale(*data_, ScaleMode::kCHANNEL, weightMap["conv6/scale"], weightMap["conv6/offset"], Weights{DataType::kFLOAT,
                                                                                                                               nullptr, 0});
    auto relu6 = network_->addActivation(*scale6->getOutput(0), ActivationType::kRELU);
    auto conv7 = network_->addConvolution(*relu6->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv7/weights"], weightMap["conv7/biases"]);
    conv7->setStride(DimsHW{1, 1});
    conv7->setPadding(DimsHW{1, 1});
    auto relu7 = network_->addActivation(*conv7->getOutput(0), ActivationType::kRELU);

    auto conv8 = network_->addConvolution(*relu7->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv8/weights"], weightMap["conv8/biases"]);
    conv8->setStride(DimsHW{1, 1});
    conv8->setPadding(DimsHW{1, 1});
    auto relu8 = network_->addActivation(*conv8->getOutput(0), ActivationType::kRELU);

    auto conv9 = network_->addConvolution(*relu8->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv9/weights"], weightMap["conv9/biases"]);
    conv9->setStride(DimsHW{1, 1});
    conv9->setPadding(DimsHW{1, 1});
    auto relu9 = network_->addActivation(*conv9->getOutput(0), ActivationType::kRELU);

    auto conv10 = network_->addConvolution(*relu9->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv10/weights"], weightMap["conv10/biases"]);
    conv10->setStride(DimsHW{1, 1});
    conv10->setPadding(DimsHW{1, 1});
    auto relu10 = network_->addActivation(*conv10->getOutput(0), ActivationType::kRELU);

    auto conv11 = network_->addConvolution(*relu10->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv11/weights"], weightMap["conv11/biases"]);
    conv11->setStride(DimsHW{1, 1});
    conv11->setPadding(DimsHW{1, 1});
    auto relu11 = network_->addActivation(*conv11->getOutput(0), ActivationType::kRELU);

    auto conv12 = network_->addConvolution(*relu11->getOutput(0), 1, DimsHW{3, 3}, weightMap["conv12/weights"], weightMap["conv12/biases"]);
    conv12->setStride(DimsHW{1, 1});
    conv12->setPadding(DimsHW{1, 1});
    assert(conv12 != nullptr);
    conv12->getOutput(0)->setName(OUTPUT_NAME);
    network_->markOutput(*conv12->getOutput(0));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    ICudaEngine * engine1 = builder->buildCudaEngine(*network);
    ICudaEngine * engine2 = builder->buildCudaEngine(*network_);
    // we don't need the network any more
//    std::cout << network->getNbLayers() << std::endl;
//    float* conv = (float*)((IConvolutionLayer*)network->getLayer(0))->getKernelWeights().values;

//    std::cout << "layers" << std::endl;
//    for (int i = 0; i < network->getNbLayers(); ++i) {
//        if (((IConvolutionLayer*)network->getLayer(i))->getPadding().nbDims == 2) {
//            std::cout << ((IConvolutionLayer*)network->getLayer(0))->getPadding().d[0] << ", " << ((IConvolutionLayer*)network->getLayer(0))->getPadding().d[1] << std::endl;
//        }
//    }
//    std::cout << std::endl;

    network->destroy();
    network_->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return std::make_pair(engine1, engine2);
}

void APIToModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
                IHostMemory **modelStream,
                IHostMemory **modelStream_)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    auto engines = createNetworkEngine(maxBatchSize, builder, DataType::kFLOAT);
    auto engine = engines.first;
    auto engine_ = engines.second;
    assert(engine != nullptr);
    assert(engine_ != nullptr);
    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    (*modelStream_) = engine_->serialize();
    engine->destroy();
    engine_->destroy();
    builder->destroy();
}

int main(int argc, char** argv)
{

    IHostMemory *modelStream{nullptr};
    IHostMemory *modelStream_{nullptr};
    APIToModel(1, &modelStream, &modelStream_);

    IRuntime *runtime = createInferRuntime(gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    ICudaEngine *engine_ = runtime->deserializeCudaEngine(modelStream_->data(), modelStream_->size(), nullptr);
    if (modelStream) modelStream->destroy();
    if (modelStream_) modelStream_->destroy();

    //----------------------------------------------

    if (!engine) {
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    }
    if (!engine_) {
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    }

    int finalOutputIndex = engine->getBindingIndex("conv6/conv");

    nvinfer1::Dims outputDims = engine->getBindingDimensions(finalOutputIndex);

    printf("dims of relu: (c=%u h=%u w=%u)\n", outputDims.d[0], outputDims.d[1], outputDims.d[2]);
    std::cout << engine->getBindingDimensions(finalOutputIndex).nbDims << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    execute(*engine, *engine_);
    engine->destroy();
    engine_->destroy();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "total time: " << ms << "ms" << std::endl;
    cv::waitKey(0);
    return EXIT_SUCCESS;
}