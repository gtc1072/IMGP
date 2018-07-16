#include "Imgp_Base.h"
#include "Imgp_Data.hpp"

namespace IMGP{

	IMGP_Image::IMGP_Image()
	{
		m_width = m_height = m_stride = m_channels = m_step_byte_row = m_step_byte_element = 0;
		m_pData = nullptr;
		m_type = IMGP_U8C1;
		m_color_space = IMGP_GRAY;
	}

	IMGP_Image::IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, IMGP_Color_Space space)
	{
		if (width > 0 && height > 0)
		{
			m_width = width;
			m_height = height;
			m_type = type;
			m_color_space = space;
			allocate_buffer(type, nullptr, false);
		}
	}

	IMGP_Image::IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, void *data, bool bShallowCopy, IMGP_Color_Space space)
	{
		if (width > 0 && height > 0)
		{
			m_width = width;
			m_height = height;
			m_type = type;
			m_color_space = space;
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
		m_step_byte_row = img.byte_stride_length();
		m_step_byte_element = img.byte_element_length();
		m_pData = new unsigned char[m_step_byte_row * m_height];
		memcpy_s((void*)m_pData, m_step_byte_row * m_height, (void*)(img.data()), m_step_byte_row * m_height);
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
		m_step_byte_row = img.byte_stride_length();
		m_step_byte_element = img.byte_element_length();
		m_pData = new unsigned char[m_step_byte_row * m_height];
		memcpy_s((void*)m_pData, m_step_byte_row * m_height, (void*)(img.data()), m_step_byte_row * m_height);
		return *this;
	}

	unsigned int IMGP_Image::width() const
	{
		return m_width;
	}

	unsigned int IMGP_Image::height() const
	{
		return m_height;
	}

	unsigned int IMGP_Image::stride() const
	{
		return m_stride;
	}

	unsigned int IMGP_Image::byte_stride_length() const
	{
		return m_step_byte_row;
	}

	unsigned int IMGP_Image::byte_element_length() const
	{
		return m_step_byte_row;
	}

	unsigned int IMGP_Image::channels() const
	{
		return m_channels;
	}

	IMGP_ImageType IMGP_Image::type() const
	{
		return m_type;
	}

	IMGP_Color_Space IMGP_Image::color_space() const
	{
		return m_color_space;
	}

	unsigned char* IMGP_Image::data(unsigned int i) const
	{
		if (i >= m_height) return nullptr;
		return m_pData + i * m_step_byte_row;
	}

	bool IMGP_Image::empty()
	{
		bool ret = false;
		if (m_width || m_height || !m_pData) ret = true;
		return ret;
	}

	IMGP_Image IMGP_Image::convert(IMGP_ImageType type, double ratio, double offset)
	{
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) != IMGP_IMAGE_TYPE_CHANNELS(type))
		{
			return IMGP_Image();
		}
		IMGP_Image img(m_width, m_height, type);
		DataConvertFunc func = get_data_convert_func(IMGP_IMAGE_TYPE_DEPTH(m_type), IMGP_IMAGE_TYPE_DEPTH(type));
		for (unsigned int j = 0; j < m_height; ++j)
		{
			func((void*)(m_pData + j * m_step_byte_row), m_width * m_channels, (void*)(img.data(j)), m_width * m_channels, ratio, offset);
		}
		return img;
	}

	IMGP_Image IMGP_Image::convert_to_color_space(IMGP_Color_Space type)
	{
		IMGP_Image img(m_width, m_height, m_type);
		return img;
	}

	void IMGP_Image::allocate_buffer(IMGP_ImageType type, void *data, bool bShallow)
	{
		m_channels = IMGP_IMAGE_TYPE_CHANNELS(type);
		int idx = IMGP_IMAGE_TYPE_DEPTH(type);
		if (idx == 0 || idx == 1) //8
		{
			m_step_byte_row = (m_width * m_channels + 3) / 4 * 4;
			m_step_byte_element = m_channels;
			m_stride = m_step_byte_row;
		}
		else if (idx == 2 || idx == 3) //16
		{
			m_step_byte_row = (m_width * m_channels * 2 + 3) / 4 * 4;
			m_step_byte_element = m_channels * 2;
			m_stride = m_step_byte_row / 2;
		}
		else if (idx == 4 || idx == 5 || idx == 6) //32
		{
			m_step_byte_row = m_width * m_channels * 4;
			m_step_byte_element = m_channels * 4;
			m_stride = m_step_byte_row / 4;
		}
		else //64
		{
			m_step_byte_row = m_width * m_channels * 8;
			m_step_byte_element = m_channels * 8;
			m_stride = m_step_byte_row / 8;
		}
		if (data && !bShallow) //deep copy
			memcpy_s((void*)m_pData, m_step_byte_row * m_height, (void*)data, m_step_byte_row * m_height);
		else if (data && bShallow) //shallow copy
			m_pData = (unsigned char*)data;
		else //allocate 
			m_pData = new unsigned char[m_step_byte_row * m_height];
		if (m_channels == 1)
			m_color_space = IMGP_GRAY;
		else
			m_color_space = IMGP_RGB;
	}

	IMGP_Image& IMGP_Image::create(unsigned int width, unsigned int height, IMGP_ImageType type, IMGP_Color_Space space)
	{
		if (width > 0 && height > 0)
		{
			if (!empty())
			{
				delete m_pData;
				m_pData = nullptr;
			}
			m_width = width;
			m_height = height;
			m_type = type;
			m_color_space = space;
			allocate_buffer(type, nullptr, false);
		}
		return *this;
	}

	IMGP_Image IMGP_Image::operator + (const IMGP_Image &img)
	{
		IMGP_Image out;
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return out;
	}

	IMGP_Image IMGP_Image::operator - (const IMGP_Image &img)
	{
		IMGP_Image out;
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return out;
	}

	IMGP_Image IMGP_Image::operator * (const IMGP_Image &img)
	{
		IMGP_Image out;
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return out;
	}

	IMGP_Image IMGP_Image::operator / (const IMGP_Image &img)
	{
		IMGP_Image out;
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return out;
	}

	IMGP_Image&	IMGP_Image::operator += (const IMGP_Image &img)
	{
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return *this;
	}

	IMGP_Image&	IMGP_Image::operator -= (const IMGP_Image &img)
	{
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return *this;
	}
	
	IMGP_Image&	IMGP_Image::operator *= (const IMGP_Image &img)
	{
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return *this;
	}

	IMGP_Image&	IMGP_Image::operator /= (const IMGP_Image &img)
	{
		if (IMGP_IMAGE_TYPE_CHANNELS(m_type) == IMGP_IMAGE_TYPE_CHANNELS(img.type())
			&& m_color_space == img.color_space())
		{

		}
		return *this;
	}

	IMGP_Image IMGP_Image::log()
	{
		IMGP_Image out;
		return out;
	}

	IMGP_Image IMGP_Image::exp()
	{
		IMGP_Image out;
		return out;
	}

	IMGP_Image IMGP_Image::power(double value)
	{
		IMGP_Image out;
		if (value > 0.0)
		{

		}
		return out;
	}

	IMGP_Image IMGP_Image::transpose()
	{
		IMGP_Image out;
		return out;
	}

	IMGP_Image IMGP_Image::flip(bool enable_direction_x, bool enable_direction_y)
	{
		IMGP_Image out;
		return out;
	}

	IMGP_Image IMGP_Image::inverse()
	{
		IMGP_Image out;
		return out;
	}
}