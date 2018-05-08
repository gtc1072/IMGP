#ifndef IMGP_SRC_COMMON___HHH
#define IMGP_SRC_COMMON___HHH

#include <memory>
#include <iostream>

enum IMGP_WatershedType{ UNMARKER = 0, MARKERD };

#define IMGP_NEW(x) \
	static Pointer New() {\
		return 	Pointer(new (x)); \
	}\

#endif