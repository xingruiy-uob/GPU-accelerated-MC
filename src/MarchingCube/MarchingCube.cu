#include "MarchingCube.h"
#include "TriangleTable.h"
#include "CudaSafeCall.h"
#include <fstream>

namespace mc
{
    IsoVolume::IsoVolume(Eigen::Vector3i dim, Eigen::Vector3f scale) : dim(dim), scale(scale)
    {
        SafeCall(cudaMalloc((void **)&voxels, sizeof(float) * dim[0] * dim[1] * dim[2]));
    }

    void IsoVolume::LoadFromFile(std::string file)
    {
        std::ifstream inFile(file);
        float *hostData = new float[dim[0] * dim[1] * dim[2]];
        if (inFile.is_open())
        {
            int count = 0;
            float val = 1;
            while (inFile >> val)
            {
                hostData[count] = val;
                count++;
            }
            SafeCall(cudaMemcpy(voxels, hostData, sizeof(float) * dim[0] * dim[1] * dim[2], cudaMemcpyHostToDevice));
        }
    }

    Mesh::Mesh(int size, bool device) : size(size), device(device)
    {
        if (device)
        {
            SafeCall(cudaMalloc((void **)&numVerts, sizeof(uint)));
            SafeCall(cudaMalloc((void **)&verts, sizeof(Eigen::Vector3f) * size));
            SafeCall(cudaMemset(numVerts, 0, sizeof(uint)));
        }
        else
        {
            numVerts = new uint[1];
            numVerts[0] = 0;
            verts = new Eigen::Vector3f[size];
        }
    }

    void Mesh::MoveToHost(Mesh &hostData)
    {
        SafeCall(cudaMemcpy(hostData.verts, verts, sizeof(Eigen::Vector3f) * size, cudaMemcpyDeviceToHost));
        SafeCall(cudaMemcpy(hostData.numVerts, numVerts, sizeof(uint), cudaMemcpyDeviceToHost));
    }

    void WriteMeshToPLY(Mesh &out, std::string filename)
    {
        std::ofstream outPly(filename);
        if (outPly.is_open())
        {
            outPly << "ply\n"
                   << "format ascii 1.0\n"
                   << "element vertex " << out.numVerts[0] << "\n"
                   << "property float x\n"
                   << "property float y\n"
                   << "property float z\n"
                   << "end_header\n";
            for (int i = 0; i < out.numVerts[0]; ++i)
            {
                outPly << out.verts[i][0] << " " << out.verts[i][1] << " " << out.verts[i][2] << "\n";
            }
            outPly.close();
        }
    }

    void Mesh::WritePLY(std::string filename)
    {
        if (device)
        {
            Mesh out(size, false);
            MoveToHost(out);
            WriteMeshToPLY(out, filename);
        }
        else
        {
            WriteMeshToPLY(*this, filename);
        }
    }

    __device__ float InterpolateSDF(float &v1, float &v2)
    {
        if (fabs(0 - v1) < 1e-6)
            return 0;
        if (fabs(0 - v2) < 1e-6)
            return 1;
        if (fabs(v1 - v2) < 1e-6)
            return 0;
        return (0 - v1) / (v2 - v1);
    }

    __device__ float ReadSdf(IsoVolume tsdfVolume, Eigen::Vector3i pos)
    {
        auto dim = tsdfVolume.dim;
        return tsdfVolume.voxels[pos[2] * dim[0] * dim[1] + pos[1] * dim[0] + pos[0]];
    }

    __device__ void MakeSdf(IsoVolume tsdfVolume, float *sdf, Eigen::Vector3i pos)
    {
        sdf[0] = ReadSdf(tsdfVolume, pos);
        sdf[1] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(1, 0, 0));
        sdf[2] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(1, 1, 0));
        sdf[3] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(0, 1, 0));
        sdf[4] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(0, 0, 1));
        sdf[5] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(1, 0, 1));
        sdf[6] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(1, 1, 1));
        sdf[7] = ReadSdf(tsdfVolume, pos + Eigen::Vector3i(0, 1, 1));
    }

    __device__ int MakeVerts(IsoVolume tsdfVolume, Eigen::Vector3f *verts, Eigen::Vector3i pos)
    {
        float sdf[8];
        MakeSdf(tsdfVolume, sdf, pos);

        int CubeIdx = 0;
        if (sdf[0] < 0)
            CubeIdx |= 1;
        if (sdf[1] < 0)
            CubeIdx |= 2;
        if (sdf[2] < 0)
            CubeIdx |= 4;
        if (sdf[3] < 0)
            CubeIdx |= 8;
        if (sdf[4] < 0)
            CubeIdx |= 16;
        if (sdf[5] < 0)
            CubeIdx |= 32;
        if (sdf[6] < 0)
            CubeIdx |= 64;
        if (sdf[7] < 0)
            CubeIdx |= 128;

        if (edgeTable[CubeIdx] == 0)
            return -1;

        Eigen::Vector3f WorldPos = pos.cast<float>();
        if (edgeTable[CubeIdx] & 1)
        {
            float val = InterpolateSDF(sdf[0], sdf[1]);
            verts[0] = WorldPos + Eigen::Vector3f(val, 0, 0);
        }

        if (edgeTable[CubeIdx] & 2)
        {
            float val = InterpolateSDF(sdf[1], sdf[2]);
            verts[1] = WorldPos + Eigen::Vector3f(1, val, 0);
        }

        if (edgeTable[CubeIdx] & 4)
        {
            float val = InterpolateSDF(sdf[2], sdf[3]);
            verts[2] = WorldPos + Eigen::Vector3f(1 - val, 1, 0);
        }

        if (edgeTable[CubeIdx] & 8)
        {
            float val = InterpolateSDF(sdf[3], sdf[0]);
            verts[3] = WorldPos + Eigen::Vector3f(0, 1 - val, 0);
        }

        if (edgeTable[CubeIdx] & 16)
        {
            float val = InterpolateSDF(sdf[4], sdf[5]);
            verts[4] = WorldPos + Eigen::Vector3f(val, 0, 1);
        }

        if (edgeTable[CubeIdx] & 32)
        {
            float val = InterpolateSDF(sdf[5], sdf[6]);
            verts[5] = WorldPos + Eigen::Vector3f(1, val, 1);
        }

        if (edgeTable[CubeIdx] & 64)
        {
            float val = InterpolateSDF(sdf[6], sdf[7]);
            verts[6] = WorldPos + Eigen::Vector3f(1 - val, 1, 1);
        }

        if (edgeTable[CubeIdx] & 128)
        {
            float val = InterpolateSDF(sdf[7], sdf[4]);
            verts[7] = WorldPos + Eigen::Vector3f(0, 1 - val, 1);
        }

        if (edgeTable[CubeIdx] & 256)
        {
            float val = InterpolateSDF(sdf[0], sdf[4]);
            verts[8] = WorldPos + Eigen::Vector3f(0, 0, val);
        }

        if (edgeTable[CubeIdx] & 512)
        {
            float val = InterpolateSDF(sdf[1], sdf[5]);
            verts[9] = WorldPos + Eigen::Vector3f(1, 0, val);
        }

        if (edgeTable[CubeIdx] & 1024)
        {
            float val = InterpolateSDF(sdf[2], sdf[6]);
            verts[10] = WorldPos + Eigen::Vector3f(1, 1, val);
        }

        if (edgeTable[CubeIdx] & 2048)
        {
            float val = InterpolateSDF(sdf[3], sdf[7]);
            verts[11] = WorldPos + Eigen::Vector3f(0, 1, val);
        }

        return CubeIdx;
    }

    __global__ void ExtractIsoSurfaceKernel(IsoVolume tsdfVolume, Mesh surface)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x <= 1 || y <= 1 || x >= tsdfVolume.dim[0] - 1 || y >= tsdfVolume.dim[1] - 1)
            return;

        for (int z = 1; z < tsdfVolume.dim[2] - 1; ++z)
        {
            Eigen::Vector3f verts[12];
            auto VoxelPos = Eigen::Vector3i(x, y, z);
            int CubeIdx = MakeVerts(tsdfVolume, verts, VoxelPos);
            if (CubeIdx <= 0)
                continue;

            for (int i = 0; triTable[CubeIdx][i] != -1; i += 3)
            {
                uint TriangleIdx = atomicAdd(surface.numVerts, 1);
                if (TriangleIdx < surface.size)
                {
                    Eigen::Vector3f vert0 = verts[triTable[CubeIdx][i]] * tsdfVolume.scale[0];
                    Eigen::Vector3f vert1 = verts[triTable[CubeIdx][i + 1]] * tsdfVolume.scale[1];
                    Eigen::Vector3f vert2 = verts[triTable[CubeIdx][i + 2]] * tsdfVolume.scale[2];
                    surface.verts[TriangleIdx] = vert0;
                    surface.verts[TriangleIdx * 3 + 1] = vert1;
                    surface.verts[TriangleIdx * 3 + 2] = vert2;
                }
            }
        }
    }

    int DivUp(int a, int b)
    {
        return ((a % b) != 0) ? (a / b + 1) : (a / b);
    }

    Mesh MarchingCube::ExtractIsoSurface(IsoVolume &tsdfVolume, int size)
    {
        Mesh out(size);
        dim3 block(8, 8, 1);
        dim3 grid(DivUp(tsdfVolume.dim[0], block.x), DivUp(tsdfVolume.dim[1], block.y));
        ExtractIsoSurfaceKernel<<<block, grid>>>(tsdfVolume, out);
        SafeCall(cudaDeviceSynchronize());
        SafeCall(cudaGetLastError());
        return out;
    }
} // namespace mc