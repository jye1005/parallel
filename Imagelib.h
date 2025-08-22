// bmp_utils_portable.h
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>

using byte = unsigned char;

#define LOG_OUT(x)    (fprintf(stderr,"%s", (x)))
#define LOG_OUT_W(x)  // no-op
#define LOG_OUT_A(x)  (fprintf(stderr,"%s", (x)))

#pragma pack(push, 1)
struct BITMAPFILEHEADER_PORT {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BITMAPINFOHEADER_PORT {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

static inline uint32_t row_stride_24(int width) {
 
    return static_cast<uint32_t>((width * 3 + 3) & ~3u);
}

bool LoadBmp(const char* filename, ::byte** pImage, int& height, int& width);
bool SaveBmp(const char* filename, ::byte* pImage, int height, int width);
bool convert1Dto2D(::byte* src, double** dst_Y, double** dst_U, double** dst_V, int height, int width);
bool convert2Dto1D(double** src_Y, double** src_U, double** src_V, ::byte* dst, int height, int width);

void convert2Dto3D(double **src2D, double ***dst3D, int height, int width);
void convert3Dto2D(double ***src3D, double **dst2D, int height, int width);

double *dmatrix1D(int nH);
double **dmatrix2D(int nH, int nW);
double ***dmatrix3D(int nH, int nW, int nC);
double ****dmatrix4D(int nH, int nW, int nC, int nNum);

void free_dmatrix1D(double *Image, int nH);
void free_dmatrix2D(double **Image, int nH, int nW);
void free_dmatrix3D(double ***Image, int nH, int nW, int nC);
void free_dmatrix4D(double ****Image, int nH, int nW, int nC, int nNum);

double clip(double x, double minVal, double maxVal);
double** simpleUpsampling2x(double **Image, int nH, int nW);


bool LoadBmp(const char* filename, ::byte** pImage, int& height, int& width)
{
    *pImage = nullptr;
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) { LOG_OUT_A("fopen() error\n"); return false; }

    BITMAPFILEHEADER_PORT bmfh{};
    BITMAPINFOHEADER_PORT bmih{};

    if (fread(&bmfh, sizeof(bmfh), 1, fp) != 1) { fclose(fp); return false; }
    if (fread(&bmih, sizeof(bmih), 1, fp) != 1) { fclose(fp); return false; }

    if (bmfh.bfType != 0x4D42) { fclose(fp); LOG_OUT_A("not '.bmp' file !!!\n"); return false; }
    if (bmih.biBitCount != 24 || bmih.biCompression != 0) { fclose(fp); LOG_OUT_A("Only 24-bit BI_RGB supported.\n"); return false; }

    width  = bmih.biWidth;
    height = std::abs(bmih.biHeight);

    const uint32_t stride = row_stride_24(width);
    const uint32_t dataBytes = stride * height;

    if (std::fseek(fp, static_cast<long>(bmfh.bfOffBits), SEEK_SET) != 0) { fclose(fp); return false; }

    std::vector<::byte> fileBuf(dataBytes);
    if (fread(fileBuf.data(), 1, dataBytes, fp) != dataBytes) { fclose(fp); return false; }
    fclose(fp);

    const size_t outBytes = static_cast<size_t>(width) * height * 3;
    ::byte* out = (::byte*)std::malloc(outBytes);
    if (!out) return false;

    for (int y = 0; y < height; ++y) {
        int srcRow = (bmih.biHeight > 0) ? (height - 1 - y) : y;
        const ::byte* srcPtr = fileBuf.data() + srcRow * stride;
        ::byte* dstPtr = out + (size_t)y * width * 3;
        std::memcpy(dstPtr, srcPtr, (size_t)width * 3);
    }

    *pImage = out;
    return true;
}

bool SaveBmp(const char* filename, ::byte* pImage, int height, int width)
{
    if (!pImage || height <= 0 || width <= 0) return false;

    const uint32_t stride = row_stride_24(width);
    const uint32_t dataBytes = stride * height;

    BITMAPFILEHEADER_PORT bmfh{};
    BITMAPINFOHEADER_PORT bmih{};

    bmfh.bfType = 0x4D42; // 'BM'
    bmfh.bfOffBits = sizeof(BITMAPFILEHEADER_PORT) + sizeof(BITMAPINFOHEADER_PORT);
    bmfh.bfSize = bmfh.bfOffBits + dataBytes;
    bmfh.bfReserved1 = 0;
    bmfh.bfReserved2 = 0;

    bmih.biSize = sizeof(BITMAPINFOHEADER_PORT);
    bmih.biWidth = width;
    bmih.biHeight = height;
    bmih.biPlanes = 1;
    bmih.biBitCount = 24;
    bmih.biCompression = 0;
    bmih.biSizeImage = dataBytes;
    bmih.biXPelsPerMeter = 0;
    bmih.biYPelsPerMeter = 0;
    bmih.biClrUsed = 0;
    bmih.biClrImportant = 0;

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) { LOG_OUT_A("fopen() error\n"); return false; }

    if (fwrite(&bmfh, sizeof(bmfh), 1, fp) != 1) { fclose(fp); return false; }
    if (fwrite(&bmih, sizeof(bmih), 1, fp) != 1) { fclose(fp); return false; }

    std::vector<::byte> row(stride, 0);
    for (int y = 0; y < height; ++y) {
        int srcRow = height - 1 - y;
        const ::byte* srcPtr = pImage + (size_t)srcRow * width * 3;
        std::memcpy(row.data(), srcPtr, (size_t)width * 3);
        if (fwrite(row.data(), 1, stride, fp) != stride) { fclose(fp); return false; }
    }

    fclose(fp);
    return true;
}

bool convert1Dto2D(::byte* src, double** dst_Y, double** dst_U, double** dst_V, int height, int width) {
    int iR, iG, iB;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            iB = src[3 * width * y + 3 * x + 0];
            iG = src[3 * width * y + 3 * x + 1];
            iR = src[3 * width * y + 3 * x + 2];

            dst_Y[y][x] = iR * 0.299 + iG * 0.587 + iB * 0.114;
            dst_U[y][x] = (iB - dst_Y[y][x]) * 0.565;
            dst_V[y][x] = (iR - dst_Y[y][x]) * 0.713;

            dst_Y[y][x] = dst_Y[y][x] / 255.0; // [0,255] -> [0,1]
        }
    }
    return true;
}

bool convert2Dto1D(double** src_Y, double** src_U, double** src_V, ::byte* dst, int height, int width) {
    size_t iCount = 0;
    double R, G, B;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double Y255 = src_Y[y][x] * 255.0; // [0,1] -> [0,255]
            R = clip(Y255 + 1.403 * src_V[y][x], 0, 255);
            G = clip(Y255 - 0.344 * src_U[y][x] - 0.714 * src_V[y][x], 0, 255);
            B = clip(Y255 + 1.770 * src_U[y][x], 0, 255);

            dst[iCount + 0] = static_cast<::byte>(B);
            dst[iCount + 1] = static_cast<::byte>(G);
            dst[iCount + 2] = static_cast<::byte>(R);
            iCount += 3;
        }
    }
    return true;
}

double clip(double x, double minVal, double maxVal) {
    if (x < minVal) x = minVal;
    if (x > maxVal) x = maxVal;
    return x;
}

double** simpleUpsampling2x(double **Image, int nH, int nW) {
    double** outImg = dmatrix2D(nH * 2, nW * 2);
    for (int y = 0; y < nH; y++) {
        for (int x = 0; x < nW; x++) {
            outImg[2 * y + 0][2 * x + 0] = Image[y][x];
            outImg[2 * y + 0][2 * x + 1] = Image[y][x];
            outImg[2 * y + 1][2 * x + 0] = Image[y][x];
            outImg[2 * y + 1][2 * x + 1] = Image[y][x];
        }
    }
    return outImg;
}

double *dmatrix1D(int nH) { return new double[nH](); }

double **dmatrix2D(int nH, int nW) {
    double **Temp = new double*[nH];
    for (int y = 0; y < nH; y++) {
        Temp[y] = new double[nW]();
    }
    return Temp;
}

double ***dmatrix3D(int nH, int nW, int nC) {
    double ***Temp = new double**[nH];
    for (int y = 0; y < nH; y++) {
        Temp[y] = new double*[nW];
        for (int x = 0; x < nW; x++) {
            Temp[y][x] = new double[nC]();
        }
    }
    return Temp;
}

double ****dmatrix4D(int nH, int nW, int nC, int nNum) {
    double ****Temp = new double***[nH];
    for (int y = 0; y < nH; y++) {
        Temp[y] = new double**[nW];
        for (int x = 0; x < nW; x++) {
            Temp[y][x] = new double*[nC];
            for (int c = 0; c < nC; c++) {
                Temp[y][x][c] = new double[nNum]();
            }
        }
    }
    return Temp;
}

void free_dmatrix1D(double *Image, int /*nH*/) { delete[] Image; }

void free_dmatrix2D(double **Image, int nH, int /*nW*/) {
    for (int y = 0; y < nH; y++) delete[] Image[y];
    delete[] Image;
}

void free_dmatrix3D(double ***Image, int nH, int nW, int /*nC*/) {
    for (int y = 0; y < nH; y++) {
        for (int x = 0; x < nW; x++) delete[] Image[y][x];
        delete[] Image[y];
    }
    delete[] Image;
}

void free_dmatrix4D(double ****Image, int nH, int nW, int nC, int /*nNum*/) {
    for (int y = 0; y < nH; y++) {
        for (int x = 0; x < nW; x++) {
            for (int c = 0; c < nC; c++) delete[] Image[y][x][c];
            delete[] Image[y][x];
        }
        delete[] Image[y];
    }
    delete[] Image;
}

void convert2Dto3D(double **src2D, double ***dst3D, int height, int width) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            dst3D[y][x][0] = src2D[y][x];
}

void convert3Dto2D(double ***src3D, double **dst2D, int height, int width) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            dst2D[y][x] = src3D[y][x][0];
}
