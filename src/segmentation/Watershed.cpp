#include "Watershed.h"
class UnMarker_Watershed : public IMGP_Watershed
{
public:
	virtual ~UnMarker_Watershed(){}
	template<typename T>
	bool watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed)
	{
		return true;
	}
};

class Markered_Watershed : public IMGP_Watershed
{
public:
	virtual ~Markered_Watershed(){}
	template<typename T>
	bool watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed)
	{
		return true;
	}
};

std::shared_ptr<IMGP_Watershed> IMGP_Watershed::create(IMGP_WatershedType type)
{
	return nullptr;
}