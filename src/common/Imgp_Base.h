#ifndef IMGP_BASE_HHHHH___
#define IMGP_BASE_HHHHH___

#include "Imgp_Common.h"

namespace IMGP{

	class IMGP_Image
	{
	public:
		virtual ~IMGP_Image();
		IMGP_Image();
		IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, IMGP_Color_Space space = IMGP_GRAY);
		IMGP_Image(unsigned int width, unsigned int height, IMGP_ImageType type, void *data, bool bShallowCopy = false, IMGP_Color_Space space = IMGP_GRAY);
		IMGP_Image(const IMGP_Image &img);
		IMGP_Image&			create(unsigned int width, unsigned int height, IMGP_ImageType type, IMGP_Color_Space space = IMGP_GRAY);
		//arithmetic operation
		IMGP_Image&			operator = (const IMGP_Image &img);
		IMGP_Image			operator + (const IMGP_Image &img);
		IMGP_Image			operator - (const IMGP_Image &img);
		IMGP_Image			operator * (const IMGP_Image &img);
		IMGP_Image			operator / (const IMGP_Image &img);
		IMGP_Image&			operator += (const IMGP_Image &img);
		IMGP_Image&			operator -= (const IMGP_Image &img);
		IMGP_Image&			operator *= (const IMGP_Image &img);
		IMGP_Image&			operator /= (const IMGP_Image &img);
		IMGP_Image			log();
		IMGP_Image			exp();
		IMGP_Image			power(double value);
		IMGP_Image			transpose();
		IMGP_Image			flip(bool enable_direction_x, bool enable_direction_y);
		IMGP_Image			inverse();
		template <typename T>
		IMGP_Image			operator + (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image			operator - (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image			operator * (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image			operator / (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image&			operator += (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image&			operator -= (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image&			operator *= (const T value)
		{
			IMGP_Image out;
			return out;
		}
		template <typename T>
		IMGP_Image&			operator /= (const T value)
		{
			IMGP_Image out;
			return out;
		}
		unsigned int		width() const;
		unsigned int		height() const;
		unsigned int		stride() const;
		unsigned int		byte_stride_length() const;
		unsigned int		byte_element_length() const;
		unsigned int		channels() const;
		unsigned char*		data(unsigned int i = 0) const;
		IMGP_ImageType		type() const;
		bool				empty();
		IMGP_Color_Space	color_space() const;
		IMGP_Image			convert(IMGP_ImageType type, double ratio, double offset);
		IMGP_Image			convert_to_color_space(IMGP_Color_Space type);
	private:
		void				allocate_buffer(IMGP_ImageType type, void *data, bool bShallow);
	private:
		unsigned int		m_width;
		unsigned int		m_height;
		unsigned int		m_stride;
		unsigned int		m_step_byte_row, m_step_byte_element;
		unsigned int		m_channels;
		unsigned char		*m_pData;
		IMGP_ImageType		m_type;
		IMGP_Color_Space	m_color_space;
	};
}

#endif