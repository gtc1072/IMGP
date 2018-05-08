#ifndef IMGP_BASE_HHHHH___
#define IMGP_BASE_HHHHH___

#include "common.h"

class IMGP_Base
{
public:
	typedef IMGP_Base				Self;
	typedef std::shared_ptr<Self>	Pointer;
	virtual ~IMGP_Base();
protected:
	IMGP_Base();
public:
	static Pointer New();
};

#endif