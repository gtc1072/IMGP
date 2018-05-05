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