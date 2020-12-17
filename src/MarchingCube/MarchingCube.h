#pragma once
#include <Eigen/Dense>

namespace mc
{
    struct IsoVolume
    {
        IsoVolume(Eigen::Vector3i dim, Eigen::Vector3f scale);
        void LoadFromFile(std::string file);

        float *voxels;
        Eigen::Vector3i dim;
        Eigen::Vector3f size;
        Eigen::Vector3f scale;
    };

    struct Mesh
    {
        Mesh(int size, bool device = true);
        void MoveToHost(Mesh &hostData);
        void WritePLY(std::string filename);
        uint *numVerts;
        Eigen::Vector3f *verts;
        Eigen::Vector3f *normal;
        int size;
        bool device;
    };

    class MarchingCube
    {
    public:
        static Mesh ExtractIsoSurface(IsoVolume &tsdfVolume, int size);
    };
} // namespace mc