#ifndef IMGP_SRC_SEGMENTATION_WATERSHED___HHH
#define IMGP_SRC_SEGMENTATION_WATERSHED___HHH

#include "..\common\common.h"

class IMGP_Watershed
{
public:
	explicit IMGP_Watershed(IMGP_WatershedType type);
	virtual ~IMGP_Watershed();
	template<typename T>
	bool watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
private:
	IMGP_WatershedType m_type;
};

#endif