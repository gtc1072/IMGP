#ifndef IMGP_SRC_SEGMENTATION_WATERSHED___HHH
#define IMGP_SRC_SEGMENTATION_WATERSHED___HHH

#include "..\common\Imgp_Base.h"

class IMGP_Watershed : public IMGP_Base
{
public:
	typedef IMGP_Watershed			Self;
	typedef std::shared_ptr<Self>	Pointer;
	virtual ~IMGP_Watershed();
protected:
	IMGP_Watershed();
public:
	IMGP_NEW(Self);
	template<typename T>
	bool watershed_tranform(T *pData, int *pMarker, int width, int height, int bandcount, int *pBasin, unsigned char *pWatershed);
private:
	IMGP_WatershedType m_type;
};

#endif