// imgp.cpp : �������̨Ӧ�ó������ڵ㡣
//

//#include "stdafx.h"
#include <tchar.h>
#include "..\src\segmentation\Watershed.h"

int _tmain(int argc, _TCHAR* argv[])
{
	IMGP_Watershed w(UNMARKER);
	float *p = new float[100];
	w.watershed_tranform(p, nullptr, 10, 10, 1, nullptr, nullptr);
	return 0;
}

