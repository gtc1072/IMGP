// imgp.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <tchar.h>
#include "..\src\segmentation\Watershed.h"

void test()
{
	IMGP_Watershed::Pointer p1 = IMGP_Watershed::New();
	std::cout << p1.use_count() << std::endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
	test();
	return 0;
}

