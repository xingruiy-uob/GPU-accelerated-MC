#include "MarchingCube.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    mc::IsoVolume tsdfVolume({256, 256, 128}, {0.01, 0.01, 0.01});
    tsdfVolume.LoadFromFile("0.txt");
    auto mesh = mc::MarchingCube::ExtractIsoSurface(tsdfVolume, 3000000);
    mesh.WritePLY("out.ply");
}