#ifndef IMGP_BASE_HHHHH___
#define IMGP_BASE_HHHHH___

#include "common.h"

class IMGP_Image
{
public:
	virtual ~IMGP_Image();
	IMGP_Image();
	IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type);
	IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, void *data, bool bShallowCopy = false);
	IMGP_Image(const IMGP_Image &img);
	IMGP_Image& operator = (const IMGP_Image &img);
	unsigned int	width() const;
	unsigned int	height() const;
	unsigned int	stride() const;
	unsigned int	step_length() const;
	unsigned int	channels() const;
	unsigned char*	data() const;
	IMGP_ImageType  type() const;
	IMGP_Color_Space color_space() const;
	IMGP_Image		convert(IMGP_ImageType type, double ratio, int offset);
	IMGP_Image		convert_to_color_space(IMGP_Color_Space type);
private:
	void allocate_buffer(IMGP_ImageType type, void *data, bool bShallow);
private:
	unsigned int			m_width;
	unsigned int			m_height;
	unsigned int			m_stride;
	unsigned int			m_step_length;
	unsigned int			m_channels;
	unsigned char			*m_pData;
	IMGP_ImageType			m_type;
	IMGP_Color_Space		m_color_space;
};

#endif