// imgp.cpp : 定义控制台应用程序的入口点。
//

#include <tchar.h>
#include "..\src\segmentation\Watershed.h"
#include "..\src\common\Imgp_Data.hpp"

void test()
{
	IMGP_Watershed::Pointer p1 = IMGP_Watershed::New();
	std::cout << p1.use_count() << std::endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
	char p1, p2;
	DataConvertFunc fun = get_data_convert_func(IMGP_IMAGE_TYPE_DEPTH(IMGP_U16C1), IMGP_IMAGE_TYPE_DEPTH(IMGP_F64C1));
	fun((void*)&p1, 0, 0, (void*)&p2, 0, 0, 1, 0);
	return 0;
}
