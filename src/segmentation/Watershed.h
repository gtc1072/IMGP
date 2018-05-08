#ifndef IMGP_SRC_SEGMENTATION_WATERSHED___HHH
#define IMGP_SRC_SEGMENTATION_WATERSHED___HHH

#include "..\common\Imgp_Base.h"

class IMGP_Watershed
{
public:
	typedef IMGP_Watershed			Self;
	typedef std::shared_ptr<Self>	Pointer;
	virtual ~IMGP_Watershed();
protected:
	IMGP_Watershed();
public:
	IMGP_NEW(Self);
	bool watershed_tranform(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed);
private:
	bool watershed_unmarker(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed);
	bool watershed_markered(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed);
private:
	IMGP_WatershedType	m_watershedType;
	IMGP_ConnectType	m_connectType;
};

#endif