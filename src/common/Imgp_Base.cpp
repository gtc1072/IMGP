#include "Imgp_Base.h"

IMGP_Image::IMGP_Image()
{
	m_width = m_height = m_stride = m_channels = m_step_length = 0;
	m_pData = nullptr;
	m_type = IMGP_U8C1;
	m_color_space = IMGP_GRAY;
}

IMGP_Image::IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type)
{
	if (width > 0 && height > 0)
	{
		m_width = width;
		m_height = height;
		m_type = type;
		allocate_buffer(type, nullptr, false);
	}
}

IMGP_Image::IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, void *data, bool bShallowCopy)
{
	if (width > 0 && height > 0)
	{
		m_width = width;
		m_height = height;
		m_type = type;
		allocate_buffer(type, data, bShallowCopy);
	}
}

IMGP_Image::~IMGP_Image()
{
	if (m_pData)
	{
		delete m_pData;
	}
	std::cout << "IMGP_Image Destrutor" << std::endl;
}

IMGP_Image::IMGP_Image(const IMGP_Image &img)
{
	m_width = img.width();
	m_height = img.height();
	m_stride = img.stride();
	m_channels = img.channels();
	m_type = img.type();
	m_color_space = img.color_space();
	m_step_length = img.step_length();
	m_pData = new unsigned char[m_step_length * m_height];
	memcpy_s((void*)m_pData, m_step_length * m_height, (void*)(img.data()), m_step_length * m_height);
}

IMGP_Image& IMGP_Image::operator = (const IMGP_Image &img)
{
	if (this == &img)
	{
		return *this;
	}
	if (m_pData) 
	{ 
		delete m_pData; 
	}
	m_width = img.width();
	m_height = img.height();
	m_stride = img.stride();
	m_channels = img.channels();
	m_type = img.type();
	m_color_space = img.color_space();
	m_step_length = img.step_length();
	m_pData = new unsigned char[m_step_length * m_height];
	memcpy_s((void*)m_pData, m_step_length * m_height, (void*)(img.data()), m_step_length * m_height);
	return *this;
}

unsigned int	IMGP_Image::width() const
{
	return m_width;
}

unsigned int	IMGP_Image::height() const
{
	return m_height;
}

unsigned int	IMGP_Image::stride() const
{
	return m_stride;
}

unsigned int	IMGP_Image::step_length() const
{
	return m_step_length;
}

unsigned int	IMGP_Image::channels() const
{
	return m_channels;
}

IMGP_ImageType  IMGP_Image::type() const
{
	return m_type;
}

IMGP_Color_Space IMGP_Image::color_space() const
{
	return m_color_space;
}

unsigned char*	IMGP_Image::data() const
{
	return m_pData;
}

IMGP_Image	IMGP_Image::convert(IMGP_ImageType type, double ratio, int offset)
{
	IMGP_Image img(m_width, m_height, type);
	return img;
}

IMGP_Image	IMGP_Image::convert_to_color_space(IMGP_Color_Space type)
{
	IMGP_Image img(m_width, m_height, m_type);
	return img;
}

void IMGP_Image::allocate_buffer(IMGP_ImageType type, void *data, bool bShallow)
{
	m_step_length = 0;
	switch (type)
	{
	case IMGP_U8C1:
		m_channels = 1;
		m_step_length = (m_width + 3) / 4 * 4;
		m_stride = m_step_length;
		break;
	case IMGP_U8C3:
		m_channels = 3;
		m_step_length = (m_width * 3 + 3) / 4 * 4;
		m_stride = m_step_length;
		break;
	case IMGP_U16C1:
		m_channels = 1;
		m_step_length = (m_width * 2 + 3) / 4 * 4;
		m_stride = m_step_length / 2;
		break;
	case IMGP_U16C3:
		m_channels = 3;
		m_step_length = (m_width * 6 + 3) / 4 * 4;
		m_stride = m_step_length / 2;
		break;
	case IMGP_S16C1:
		m_channels = 1;
		m_step_length = (m_width * 2 + 3) / 4 * 4;
		m_stride = m_step_length / 2;
		break;
	case IMGP_S16C3:
		m_channels = 3;
		m_step_length = (m_width * 6 + 3) / 4 * 4;
		m_stride = m_step_length / 2;
		break;
	case IMGP_U32C1:
		m_channels = 1;
		m_step_length = m_width * 4;
		m_stride = m_width;
		break;
	case IMGP_U32C3:
		m_channels = 3;
		m_step_length = m_width * 12;
		m_stride = m_width * 3;
		break;
	case IMGP_S32C1:
		m_channels = 1;
		m_step_length = m_width * 4;
		m_stride = m_width;
		break;
	case IMGP_S32C3:
		m_channels = 3;
		m_step_length = m_width * 12;
		m_stride = m_width * 3;
		break;
	case IMGP_F32C1:
		m_channels = 1;
		m_step_length = m_width * 4;
		m_stride = m_width;
		break;
	case IMGP_F32C3:
		m_channels = 3;
		m_step_length = m_width * 12;
		m_stride = m_width * 3;
		break;
	case IMGP_F64C1:
		m_channels = 1;
		m_step_length = m_width * 8;
		m_stride = m_width;
		break;
	case IMGP_F64C3:
		m_channels = 3;
		m_step_length = m_width * 24;
		m_stride = m_width * 3;
		break;
	default:
		m_channels = 1;
		m_step_length = (m_width + 3) / 4 * 4;
		m_stride = m_step_length;
		break;
	}
	if (data && !bShallow) //deep copy
		memcpy_s((void*)m_pData, m_step_length * m_height, (void*)data, m_step_length * m_height);
	else if (data && bShallow) //shallow copy
		m_pData = (unsigned char*)data;
	else //allocate 
		m_pData = new unsigned char[m_step_length * m_height];
	if (m_channels == 1)
		m_color_space = IMGP_GRAY;
	else
		m_color_space = IMGP_RGB;
}