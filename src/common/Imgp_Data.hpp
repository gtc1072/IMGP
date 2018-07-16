#ifndef IMGP_SRC_DATA___HHH
#define IMGP_SRC_DATA___HHH

#include <iostream>
#include <limits.h>

namespace IMGP{

	template <typename TS, typename TD>
	inline void IMGP_data_convert(const TS *pSrc, unsigned int sp, TD *pDst, unsigned int dp, double ratio, double offset)
	{
		unsigned int width = sp < dp ? sp : dp;
		TD v_max = std::numeric_limits<TD>::max();
		TD v_min = std::numeric_limits<TD>::min();
		for (unsigned int i = 0; i < width; ++i)
		{
			double value = (double)pSrc[i] * ratio + offset;
			pDst[i] = value > v_max ? v_max : (value < v_min ? v_min : static_cast<TD>(value));
		}
	}

#define IMGP_DATA_FUNC(a,b,c,d) \
	inline void IMGP_data_##a##b##c##d##_convert(const void *pSrc, unsigned int sp, void *pDst, unsigned int dp, double ratio, double offset) \
		{\
		IMGP_data_convert((const a b*)pSrc, sp, (c d*)pDst, dp, ratio, offset);\
		}\

	IMGP_DATA_FUNC(unsigned, char, unsigned, char)
	IMGP_DATA_FUNC(unsigned, char, , char)
	IMGP_DATA_FUNC(unsigned, char, unsigned, short)
	IMGP_DATA_FUNC(unsigned, char, , short)
	IMGP_DATA_FUNC(unsigned, char, unsigned, int)
	IMGP_DATA_FUNC(unsigned, char, , int)
	IMGP_DATA_FUNC(unsigned, char, , float)
	IMGP_DATA_FUNC(unsigned, char, , double)

	IMGP_DATA_FUNC(, char, unsigned, char)
	IMGP_DATA_FUNC(, char, , char)
	IMGP_DATA_FUNC(, char, unsigned, short)
	IMGP_DATA_FUNC(, char, , short)
	IMGP_DATA_FUNC(, char, unsigned, int)
	IMGP_DATA_FUNC(, char, , int)
	IMGP_DATA_FUNC(, char, , float)
	IMGP_DATA_FUNC(, char, , double)

	IMGP_DATA_FUNC(unsigned, short, unsigned, char)
	IMGP_DATA_FUNC(unsigned, short, , char)
	IMGP_DATA_FUNC(unsigned, short, unsigned, short)
	IMGP_DATA_FUNC(unsigned, short, , short)
	IMGP_DATA_FUNC(unsigned, short, unsigned, int)
	IMGP_DATA_FUNC(unsigned, short, , int)
	IMGP_DATA_FUNC(unsigned, short, , float)
	IMGP_DATA_FUNC(unsigned, short, , double)

	IMGP_DATA_FUNC(, short, unsigned, char)
	IMGP_DATA_FUNC(, short, , char)
	IMGP_DATA_FUNC(, short, unsigned, short)
	IMGP_DATA_FUNC(, short, , short)
	IMGP_DATA_FUNC(, short, unsigned, int)
	IMGP_DATA_FUNC(, short, , int)
	IMGP_DATA_FUNC(, short, , float)
	IMGP_DATA_FUNC(, short, , double)

	IMGP_DATA_FUNC(unsigned, int, unsigned, char)
	IMGP_DATA_FUNC(unsigned, int, , char)
	IMGP_DATA_FUNC(unsigned, int, unsigned, short)
	IMGP_DATA_FUNC(unsigned, int, , short)
	IMGP_DATA_FUNC(unsigned, int, unsigned, int)
	IMGP_DATA_FUNC(unsigned, int, , int)
	IMGP_DATA_FUNC(unsigned, int, , float)
	IMGP_DATA_FUNC(unsigned, int, , double)

	IMGP_DATA_FUNC(, int, unsigned, char)
	IMGP_DATA_FUNC(, int, , char)
	IMGP_DATA_FUNC(, int, unsigned, short)
	IMGP_DATA_FUNC(, int, , short)
	IMGP_DATA_FUNC(, int, unsigned, int)
	IMGP_DATA_FUNC(, int, , int)
	IMGP_DATA_FUNC(, int, , float)
	IMGP_DATA_FUNC(, int, , double)

	IMGP_DATA_FUNC(, float, unsigned, char)
	IMGP_DATA_FUNC(, float, , char)
	IMGP_DATA_FUNC(, float, unsigned, short)
	IMGP_DATA_FUNC(, float, , short)
	IMGP_DATA_FUNC(, float, unsigned, int)
	IMGP_DATA_FUNC(, float, , int)
	IMGP_DATA_FUNC(, float, , float)
	IMGP_DATA_FUNC(, float, , double)

	IMGP_DATA_FUNC(, double, unsigned, char)
	IMGP_DATA_FUNC(, double, , char)
	IMGP_DATA_FUNC(, double, unsigned, short)
	IMGP_DATA_FUNC(, double, , short)
	IMGP_DATA_FUNC(, double, unsigned, int)
	IMGP_DATA_FUNC(, double, , int)
	IMGP_DATA_FUNC(, double, , float)
	IMGP_DATA_FUNC(, double, , double)



	typedef void(*DataConvertFunc)(const void *pSrc, unsigned int sp, void *pDst, unsigned int dp, double ratio, double offset);

	static DataConvertFunc get_data_convert_func(int sdepth, int ddepth)
	{
		static DataConvertFunc func_table[][8] = {
			{ IMGP_data_unsignedcharunsignedchar_convert, IMGP_data_unsignedcharchar_convert, IMGP_data_unsignedcharunsignedshort_convert, IMGP_data_unsignedcharshort_convert,
			IMGP_data_unsignedcharunsignedint_convert, IMGP_data_unsignedcharint_convert, IMGP_data_unsignedcharfloat_convert, IMGP_data_unsignedchardouble_convert },
			{ IMGP_data_charunsignedchar_convert, IMGP_data_charchar_convert, IMGP_data_charunsignedshort_convert, IMGP_data_charshort_convert,
			IMGP_data_charunsignedint_convert, IMGP_data_charint_convert, IMGP_data_charfloat_convert, IMGP_data_chardouble_convert },
			{ IMGP_data_unsignedshortunsignedchar_convert, IMGP_data_unsignedshortchar_convert, IMGP_data_unsignedshortunsignedshort_convert, IMGP_data_unsignedshortshort_convert,
			IMGP_data_unsignedshortunsignedint_convert, IMGP_data_unsignedshortint_convert, IMGP_data_unsignedshortfloat_convert, IMGP_data_unsignedshortdouble_convert },
			{ IMGP_data_shortunsignedchar_convert, IMGP_data_shortchar_convert, IMGP_data_shortunsignedshort_convert, IMGP_data_shortshort_convert,
			IMGP_data_shortunsignedint_convert, IMGP_data_shortint_convert, IMGP_data_shortfloat_convert, IMGP_data_shortdouble_convert },
			{ IMGP_data_unsignedintunsignedchar_convert, IMGP_data_unsignedintchar_convert, IMGP_data_unsignedintunsignedshort_convert, IMGP_data_unsignedintshort_convert,
			IMGP_data_unsignedintunsignedint_convert, IMGP_data_unsignedintint_convert, IMGP_data_unsignedintfloat_convert, IMGP_data_unsignedintdouble_convert },
			{ IMGP_data_intunsignedchar_convert, IMGP_data_intchar_convert, IMGP_data_intunsignedshort_convert, IMGP_data_intshort_convert,
			IMGP_data_intunsignedint_convert, IMGP_data_intint_convert, IMGP_data_intfloat_convert, IMGP_data_intdouble_convert },
			{ IMGP_data_floatunsignedchar_convert, IMGP_data_floatchar_convert, IMGP_data_floatunsignedshort_convert, IMGP_data_floatshort_convert,
			IMGP_data_floatunsignedint_convert, IMGP_data_floatint_convert, IMGP_data_floatfloat_convert, IMGP_data_floatdouble_convert },
			{ IMGP_data_doubleunsignedchar_convert, IMGP_data_doublechar_convert, IMGP_data_doubleunsignedshort_convert, IMGP_data_doubleshort_convert,
			IMGP_data_doubleunsignedint_convert, IMGP_data_doubleint_convert, IMGP_data_doublefloat_convert, IMGP_data_doubledouble_convert }
		};
		return func_table[sdepth][ddepth];
	}
}

#endif