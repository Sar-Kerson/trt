//
// Created by ubuntu on 14/04/18.
//
#include "common.h"
std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
    const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            std::cout << file << std::endl;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

float* convertToNCHW(cv::Mat &input)
{
    if (input.channels() == 1) return (float*)input.data;
    std::vector<cv::Mat> rgbs;
    cv::split(input, rgbs);
    float *NCHW = new float[input.channels() * input.rows * input.cols];
    if (input.type() == CV_8UC3) {
        std::cout << "type: CV_8UC3" << std::endl;
    } else {

    }
}
