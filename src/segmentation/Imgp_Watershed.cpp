#include "Imgp_Watershed.h"
#include <vector>
#include <queue>

namespace IMGP{

	class pt
	{
	public:
		pt() :m_x(0), m_y(0){}
		pt(int x, int y){
			m_x = x;
			m_y = y;
		}
		pt(const pt &p){
			m_x = p.m_x;
			m_y = p.m_y;
		}
		pt& operator = (const pt &p)
		{
			this->m_x = p.m_x;
			this->m_y = p.m_y;
			return *this;
		}
		void getvalue(int &x, int &y)
		{
			x = m_x;
			y = m_y;
		}
	private:
		int m_x, m_y;
	};

	IMGP_Watershed::IMGP_Watershed()
	{
		
	}

	IMGP_Watershed::~IMGP_Watershed()
	{
		std::cout << "IMGP_Watershed Destrutor" << std::endl;
	}

	bool IMGP_Watershed::watershed_tranform(IMGP_Image Data, IMGP_Image &Basin, IMGP_Image &Watershed, IMGP_WatershedType type, IMGP_Image *Marker)
	{
		bool ret = false;
		if (type == UNMARKER)
		{
			ret = watershed_unmarker(Data, Basin, Watershed);
		}
		else if (type == MARKERD && Marker)
		{
			ret = watershed_markered(Data, *Marker, Basin, Watershed);
		}
		return ret;
	}

	bool IMGP_Watershed::watershed_unmarker(IMGP_Image Data, IMGP_Image &Basin, IMGP_Image &Watershed)
	{
#define WSHD -1
#define INITVALUE -2
#define INQUEUE 0
		unsigned char *image = Data.data();
		
		int width = Data.width();
		int height = Data.height();
		if (width < 10 || height < 10 || Data.type() != IMGP_U8C1 || !(Basin.empty()) || !(Watershed.empty()))
		{
			return false;
		}
		IMGP_Image Marker(width, height, IMGP_S32C1);
		int *label = (int*)(Marker.data());
		Basin = IMGP_Image(width, height, IMGP_S32C1);
		Watershed = IMGP_Image(width, height, IMGP_U8C1);

		int size = width * height;
		std::vector<int> t_stat(256, 0);
		std::vector<int> t_space(256, 0);
		std::vector<pt> t_sort(size);
		int step = Data.stride();
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				unsigned char value = image[j*step + i];
				t_stat[value]++;
			}
		}
		for (int j = 1; j < 256; ++j)
		{
			t_stat[j] += t_stat[j - 1];
		}
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				unsigned char value = image[j*step + i];
				int index = t_stat[value] - ++t_space[value];
				t_sort[index] = pt(i, j);
			}
		}
		for (int j = 255; j > 0; j--)
		{
			t_stat[j] -= t_stat[j - 1];
		}
		for (int j = 1; j < height - 1; ++j)
		{
			for (int i = 1; i < width - 1; ++i)
			{
				label[j * width + i] = INITVALUE;
			}
		}
		for (int j = 0; j < width; ++j)
		{
			label[j] = label[(height - 1)*width + j] = WSHD;
		}
		for (int j = 0; j < height; ++j)
		{
			label[j*width] = label[j*width + width - 1] = WSHD;
		}
		int prepos = 0;
		int postpos = 0;
		int label_count = 0;
		std::queue<pt> seed;
		int startIdx = 0;
		for (int j = 0; j < 256; ++j)
		{
			if (t_stat[j] > 0)
			{
				startIdx = j;
				break;
			}
		}
		for (int j = startIdx; j < 256; ++j)
		{
			prepos = postpos;
			postpos += t_stat[j];
			for (int jj = prepos; jj < postpos; ++jj)
			{
				int x, y;
				t_sort[jj].getvalue(x, y);
				if (x == 0 || x == width - 1 || y == 0 || y == height - 1) continue;
				int index = y*width + x;
				int index_l = index - 1;
				int index_r = index + 1;
				int index_t = index - width;
				int index_b = index + width;
				if (label[index] == INITVALUE)
				{
					if (label[index_l] > INQUEUE || label[index_r] > INQUEUE || label[index_t] > INQUEUE || label[index_b] > INQUEUE)
					{
						seed.push(pt(x, y));
						label[index] = INQUEUE;
					}
				}
			}
			while (!seed.empty())
			{
				pt p = seed.front();
				seed.pop();
				int x, y;
				p.getvalue(x, y);
				int index = y*width + x;
				int index_l = index - 1;
				int index_r = index + 1;
				int index_t = index - width;
				int index_b = index + width;
				if (label[index] == INQUEUE)
				{
					int lab = 0, t;
					t = label[index_l];
					if (t > 0) lab = t;
					t = label[index_t];
					if (t > 0)
					{
						if (lab == 0) lab = t;
						else
						{
							if (lab != t) lab = -1;
						}
					}
					t = label[index_r];
					if (t > 0)
					{
						if (lab == 0) lab = t;
						else
						{
							if (lab != t) lab = -1;
						}
					}
					t = label[index_b];
					if (t > 0)
					{
						if (lab == 0) lab = t;
						else
						{
							if (lab != t) lab = -1;
						}
					}
					label[index] = lab;
					if (label[index_l] == INITVALUE && image[y*step + x - 1] == j)
					{
						seed.push(pt(x - 1, y));
						label[index_l] = INQUEUE;
					}
					if (label[index_r] == INITVALUE && image[y*step + x + 1] == j)
					{
						seed.push(pt(x + 1, y));
						label[index_r] = INQUEUE;
					}
					if (label[index_t] == INITVALUE && image[(y - 1)*step + x] == j)
					{
						seed.push(pt(x, y - 1));
						label[index_t] = INQUEUE;
					}
					if (label[index_b] == INITVALUE && image[(y + 1)*step + x] == j)
					{
						seed.push(pt(x, y + 1));
						label[index_b] = INQUEUE;
					}
				}
			}
			for (int jj = prepos; jj < postpos; ++jj)
			{
				int x, y;
				t_sort[jj].getvalue(x, y);
				if (x == 0 || x == width - 1 || y == 0 || y == height - 1) continue;
				int index = y*width + x;
				int index_l = index - 1;
				int index_r = index + 1;
				int index_t = index - width;
				int index_b = index + width;
				if (label[index] == INITVALUE)
				{
					if (label[index_l] <= WSHD && label[index_r] <= WSHD && label[index_t] <= WSHD && label[index_b] <= WSHD)
					{
						seed.push(pt(x, y));
						label[index] = INQUEUE;
					}
				}
				while (!seed.empty())
				{
					pt p = seed.front();
					seed.pop();
					p.getvalue(x, y);
					index = y*width + x;
					index_l = index - 1;
					index_r = index + 1;
					index_t = index - width;
					index_b = index + width;
					if (label[index] == INQUEUE)
					{
						if (label[index_l] <= WSHD && label[index_r] <= WSHD && label[index_t] <= WSHD && label[index_b] <= WSHD)
						{
							label_count++;
							label[index] = label_count;
						}
						else
						{
							int lab = 0, t;
							t = label[index_l];
							if (t > 0) lab = t;
							t = label[index_t];
							if (t > 0)
							{
								if (lab == 0) lab = t;
								else
								{
									if (lab != t) lab = WSHD;
								}
							}
							t = label[index_r];
							if (t > 0)
							{
								if (lab == 0) lab = t;
								else
								{
									if (lab != t) lab = WSHD;
								}
							}
							t = label[index_b];
							if (t > 0)
							{
								if (lab == 0) lab = t;
								else
								{
									if (lab != t) lab = WSHD;
								}
							}
							label[index] = lab;
						}
						if (label[index_l] == INITVALUE && image[y*step + x - 1] == j)
						{
							seed.push(pt(x - 1, y));
							label[index_l] = INQUEUE;
						}
						if (label[index_r] == INITVALUE && image[y*step + x + 1] == j)
						{
							seed.push(pt(x + 1, y));
							label[index_r] = INQUEUE;
						}
						if (label[index_t] == INITVALUE && image[(y - 1)*step + x] == j)
						{
							seed.push(pt(x, y - 1));
							label[index_t] = INQUEUE;
						}
						if (label[index_b] == INITVALUE && image[(y + 1)*step + x] == j)
						{
							seed.push(pt(x, y + 1));
							label[index_b] = INQUEUE;
						}
					}
				}
			}
		}
		int *pBasin = (int*)(Basin.data());
		unsigned char *pShed = Watershed.data();
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				if (label[j * width + i] == WSHD)
				{
					pShed[j*step + i] = 255;
				}
				else
				{
					pShed[j*step + i] = 0;
					pBasin[j*width + i] = label[j * width + i];
				}
			}
		}
#undef WSHD
#undef INITVALUE
#undef VISITED 
		return true;
	}

	bool IMGP_Watershed::watershed_markered(IMGP_Image Data, IMGP_Image &Marker, IMGP_Image &Basin, IMGP_Image &Watershed)
	{
#define WSHD -1
#define INITVALUE -2
#define INQUEUE 0
		unsigned char *image = Data.data();
		int *label = (int*)(Marker.data());
		int width = Data.width();
		int height = Data.height();
		if (width != Marker.width() || height != Marker.height() || Data.type() != IMGP_U8C1 || Marker.type() != IMGP_S32C1
			|| !(Basin.empty()) || !(Watershed.empty()))
		{
			return false;
		}
		Basin = IMGP_Image(width, height, IMGP_S32C1);
		Watershed = IMGP_Image(width, height, IMGP_U8C1);

		int step = (width + 3) / 4 * 4;
		for (int j = 1; j < height - 1; ++j)
		{
			for (int i = 1; i < width - 1; ++i)
			{
				if (label[j * width + i] < 1)
					label[j * width + i] = INITVALUE;
			}
		}
		for (int j = 0; j < width; ++j)
		{
			label[j] = label[(height - 1)*width + j] = WSHD;
		}
		for (int j = 0; j < height; ++j)
		{
			label[j*width] = label[j*width + width - 1] = WSHD;
		}
		std::vector<std::queue<pt>> seeds(256);
		int activeseed;
		for (int j = 1; j < height - 1; ++j)
		{
			for (int i = 1; i < width - 1; ++i)
			{
				int index = j*width + i;
				if (label[index] == INITVALUE)
				{
					int index_l = index - 1;
					int index_r = index + 1;
					int index_t = index - width;
					int index_b = index + width;
					int diff = 256, t;
					if (label[index_l] > INQUEUE)
					{
						t = abs(image[j*step + i] - image[j*step + i - 1]);
						if (t < diff) diff = t;
					}
					if (label[index_r] > INQUEUE)
					{
						t = abs(image[j*step + i] - image[j*step + i + 1]);
						if (t < diff) diff = t;
					}
					if (label[index_t] > INQUEUE)
					{
						t = abs(image[j*step + i] - image[(j - 1)*step + i]);
						if (t < diff) diff = t;
					}
					if (label[index_b] > INQUEUE)
					{
						t = abs(image[j*step + i] - image[(j + 1)*step + i]);
						if (t < diff) diff = t;
					}
					if (diff < 256)
					{
						label[index] = INQUEUE;
						seeds[diff].push(pt(i, j));
					}
				}
			}
		}
		for (;;)
		{
			activeseed = 256;
			for (int j = 0; j < 256; ++j)
				if (!seeds[j].empty())
				{
					activeseed = j;
					break;
				}
			if (activeseed == 256) break;
			while (!seeds[activeseed].empty())
			{
				pt p = seeds[activeseed].front();
				seeds[activeseed].pop();
				int x, y;
				p.getvalue(x, y);
				int index = y * width + x;
				int index_l = index - 1;
				int index_r = index + 1;
				int index_t = index - width;
				int index_b = index + width;
				int lab = 0, t;
				t = label[index_l];
				if (t > 0) lab = t;
				t = label[index_t];
				if (t > 0)
				{
					if (lab == 0) lab = t;
					else
					{
						if (lab != t) lab = WSHD;
					}
				}
				t = label[index_r];
				if (t > 0)
				{
					if (lab == 0) lab = t;
					else
					{
						if (lab != t) lab = WSHD;
					}
				}
				t = label[index_b];
				if (t > 0)
				{
					if (lab == 0) lab = t;
					else
					{
						if (lab != t) lab = WSHD;
					}
				}
				label[index] = lab;
				if (lab == 0) continue;
				if (label[index_l] == INITVALUE)
				{
					int diff = abs(image[y*step + x] - image[y*step + x - 1]);
					seeds[diff].push(pt(x - 1, y));
					label[index_l] = INQUEUE;
				}
				if (label[index_r] == INITVALUE)
				{
					int diff = abs(image[y*step + x] - image[y*step + x + 1]);
					seeds[diff].push(pt(x + 1, y));
					label[index_r] = INQUEUE;
				}
				if (label[index_t] == INITVALUE)
				{
					int diff = abs(image[y*step + x] - image[(y - 1)*step + x]);
					seeds[diff].push(pt(x, y - 1));
					label[index_t] = INQUEUE;
				}
				if (label[index_b] == INITVALUE)
				{
					int diff = abs(image[y*step + x] - image[(y + 1)*step + x]);
					seeds[diff].push(pt(x, y + 1));
					label[index_b] = INQUEUE;
				}
			}
		}
		int *pBasin = (int*)(Basin.data());
		unsigned char *pShed = Watershed.data();
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				if (label[j * width + i] == WSHD)
				{
					pShed[j*step + i] = 255;
				}
				else
				{
					pShed[j*step + i] = 0;
					pBasin[j*width + i] = label[j * width + i];
				}
			}
		}
#undef WSHD
#undef INITVALUE
#undef VISITED 
		return true;
	}

}