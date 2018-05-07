#include "Watershed.h"

IMGP_Watershed::IMGP_Watershed(IMGP_WatershedType type)
{
	m_type = type;
}

IMGP_Watershed::~IMGP_Watershed()
{

}

template<typename T>
bool IMGP_Watershed::watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed)
{
	printf("watershed_tranform");
	return true;
}
template bool IMGP_Watershed::watershed_tranform(int *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(float *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(unsigned char *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(char *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(unsigned short *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(short *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(unsigned int *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
template bool IMGP_Watershed::watershed_tranform(double *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);