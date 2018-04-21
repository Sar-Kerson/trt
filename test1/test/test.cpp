#include <cstring>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdint.h>

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex>
void loadWeights(const std::string file)
{
    std::ifstream input(file.c_str());
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        uint32_t size;
        std::string name;
        input >> name >> size;
        float *val = reinterpret_cast<float *>(malloc(sizeof(float) * size));
        for (uint32_t x = 0; x < size; ++x)
        {
            input >> val[x];
        }
        printf("%s:", name.c_str());
        for (uint32_t i = 0; i < size; ++i) {
            printf("%f ", val[i]);
        }
    }
}

int main()
{
    loadWeights("test.wt");
    return 0;
}
