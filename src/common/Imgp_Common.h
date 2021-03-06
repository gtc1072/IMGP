#ifndef IMGP_SRC_COMMON___HHH
#define IMGP_SRC_COMMON___HHH

#include <memory>
#include <iostream>
#include <assert.h>

namespace IMGP{
	enum IMGP_WatershedType{ UNMARKER = 0, MARKERD };

	enum IMGP_ConnectType{ Four_Connection = 0, Eight_Connection };

#define IMGP_IMAGE_TYPE_BIT_SHIFT  3

#define IMGP_IMAGE_TYPE_DEPTH(i0) \
	((i0) & ((1<<IMGP_IMAGE_TYPE_BIT_SHIFT) - 1))

#define IMGP_IMAGE_TYPE_CHANNELS(i0) \
	((i0)>>3)

#define IMGP_IMAGE_TYPE_VALUE(i0, i1) \
	((i0) + (i1<<IMGP_IMAGE_TYPE_BIT_SHIFT))

	enum IMGP_ImageType{
		IMGP_U8C1 = IMGP_IMAGE_TYPE_VALUE(0, 1),
		IMGP_U8C3 = IMGP_IMAGE_TYPE_VALUE(0, 3),
		IMGP_S8C1 = IMGP_IMAGE_TYPE_VALUE(1, 1),
		IMGP_S8C3 = IMGP_IMAGE_TYPE_VALUE(1, 3),
		IMGP_U16C1 = IMGP_IMAGE_TYPE_VALUE(2, 1),
		IMGP_U16C3 = IMGP_IMAGE_TYPE_VALUE(2, 3),
		IMGP_S16C1 = IMGP_IMAGE_TYPE_VALUE(3, 1),
		IMGP_S16C3 = IMGP_IMAGE_TYPE_VALUE(3, 3),
		IMGP_U32C1 = IMGP_IMAGE_TYPE_VALUE(4, 1),
		IMGP_U32C3 = IMGP_IMAGE_TYPE_VALUE(4, 3),
		IMGP_S32C1 = IMGP_IMAGE_TYPE_VALUE(5, 1),
		IMGP_S32C3 = IMGP_IMAGE_TYPE_VALUE(5, 3),
		IMGP_F32C1 = IMGP_IMAGE_TYPE_VALUE(6, 1),
		IMGP_F32C3 = IMGP_IMAGE_TYPE_VALUE(6, 3),
		IMGP_F64C1 = IMGP_IMAGE_TYPE_VALUE(7, 1),
		IMGP_F64C3 = IMGP_IMAGE_TYPE_VALUE(7, 3)
	};

	enum IMGP_Color_Space{
		IMGP_GRAY = 0,
		IMGP_RGB,
		IMGP_XYZ,
		IMGP_HSV,
		IMGP_YUV,
		IMGP_LAB,
		IMGP_YCbCr
	};

#define IMGP_NEW(x) \
	static Pointer New() {\
		return 	Pointer(new (x)); \
			}\

#define IMGP_PI  3.1415926535897932384626433832795
#define IMGP_2PI 6.283185307179586476925286766559
}

#endif