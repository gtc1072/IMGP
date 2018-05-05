#ifndef IMGP_SRC_SEGMENTATION_WATERSHED___HHH
#define IMGP_SRC_SEGMENTATION_WATERSHED___HHH

#include "..\common\common.h"

class IMGP_Watershed
{
public:
	virtual ~IMGP_Watershed(){}
	template<typename T>
	bool watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed) = 0;
	static std::shared_ptr<IMGP_Watershed> create(IMGP_WatershedType type);
};

#endif