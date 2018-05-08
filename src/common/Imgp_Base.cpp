#include "Imgp_Base.h"

IMGP_Base::IMGP_Base()
{

}

IMGP_Base::~IMGP_Base()
{
	std::cout << "IMGP_Base Destrutor" << std::endl;
}

std::shared_ptr<IMGP_Base> IMGP_Base::New()
{
	return std::shared_ptr<IMGP_Base>(new IMGP_Base);
}