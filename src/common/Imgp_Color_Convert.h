#ifndef IMGP_SRC_COLOR_CONVERT___HHH
#define IMGP_SRC_COLOR_CONVERT___HHH

namespace IMGP{

	typedef void(*ColorSpaceConvertFunc)(const void *pSrc, unsigned int sp, void *pDst, unsigned int dp);

	static ColorSpaceConvertFunc get_color_space_convert_func(int id_s, int id_d)
	{
		static ColorSpaceConvertFunc func_table[][7] = {
			{},
			{},
			{},
			{},
			{},
			{},
			{}
		};
		return func_table[id_s][id_d];
	}
}

#endif