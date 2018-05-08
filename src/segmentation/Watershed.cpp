#include "Watershed.h"

IMGP_Watershed::IMGP_Watershed()
{
	m_watershedType =	UNMARKER;
	m_connectType	=	Four_Connection;
}

IMGP_Watershed::~IMGP_Watershed()
{
	std::cout << "IMGP_Watershed Destrutor" << std::endl;
}

bool IMGP_Watershed::watershed_tranform(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed)
{
	bool ret = false;
	if (m_watershedType == UNMARKER)
	{
		ret = watershed_unmarker(Data, Marker, Basin, Watershed);
	}
	else if (m_watershedType == MARKERD)
	{
		ret = watershed_markered(Data, Marker, Basin, Watershed);
	}
	return ret;
}

bool IMGP_Watershed::watershed_unmarker(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed)
{
	return true;
}

bool IMGP_Watershed::watershed_markered(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed)
{
	return true;
}

