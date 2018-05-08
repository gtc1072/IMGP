#ifndef IMGP_SRC_COMMON___HHH
#define IMGP_SRC_COMMON___HHH

#include <memory>
#include <iostream>
#include <assert.h>

enum IMGP_WatershedType{ UNMARKER = 0, MARKERD };

enum IMGP_ConnectType{ Four_Connection = 0, Eight_Connection };

enum IMGP_ImageType{
	IMGP_U8C1 = 0,
	IMGP_U8C3,
	IMGP_U16C1,
	IMGP_U16C3,
	IMGP_S16C1,
	IMGP_S16C3,
	IMGP_U32C1,
	IMGP_U32C3,
	IMGP_S32C1,
	IMGP_S32C3,
	IMGP_F32C1,
	IMGP_F32C3,
	IMGP_F64C1,
	IMGP_F64C3
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

#define IMGP_PI  3.1415926535

#endif