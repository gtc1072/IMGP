// imgp.cpp : 定义控制台应用程序的入口点。
//

#include <tchar.h>
#include <conio.h>
#include "..\src\segmentation\Imgp_Watershed.h"
#include "..\src\common\Imgp_Base.h"
#include "..\src\classify\Imgp_MLP.h"
#include <Windows.h>
#include <io.h>

using namespace IMGP;

std::vector<double> readBmp(const char *pPath)
{
	FILE *pFile;
	errno_t err = fopen_s(&pFile, pPath, "rb");
	std::vector<double> ret;
	if (!err)
	{
		BITMAPFILEHEADER bh;
		BITMAPINFOHEADER bi;
		fread(&bh, sizeof(BITMAPFILEHEADER), 1, pFile);
		if (bh.bfType == 0x4D42)
		{
			fread(&bi, sizeof(BITMAPINFOHEADER), 1, pFile);
			int width = bi.biWidth;
			int height = bi.biHeight;
			if (bi.biBitCount == 8)
			{
				int step = (width + 3) / 4 * 4;
				unsigned char *pImg = new unsigned char[step * height];
				double *pTemp = new double[width * height];
				fseek(pFile, 1024, SEEK_CUR);
				fread(pImg, sizeof(unsigned char), step * height, pFile);
				for (int j = 0; j < height; j++)
				{
					pTemp[j*width] = pImg[j*step];
					pTemp[j*width + width - 1] = pImg[j*step + width - 1];
				}
				for (int j = 0; j < width; j++)
				{
					pTemp[j] = pImg[j];
					pTemp[(height - 1)*width + j] = pImg[(height - 1)*step + j];
				}
				for (int j = 1; j < height - 1; j++)
				{
					for (int i = 1; i < width - 1; i++)
					{
						pTemp[j*width + i] = ((double)(pImg[j*step + i]) + (double)(pImg[j*step + i - 1]) + (double)(pImg[j*step + i + 1])
							+ (double)(pImg[(j - 1)*step + i]) + (double)(pImg[(j - 1)*step + i - 1]) + (double)(pImg[(j - 1)*step + i + 1])
							+ (double)(pImg[(j + 1)*step + i]) + (double)(pImg[(j + 1)*step + i - 1]) + (double)(pImg[(j + 1)*step + i + 1])) / 9.0;
					}
				}
				for (int j = 0; j < height; j++)
				{
					for (int i = 0; i < width; i++)
					{
						ret.push_back(pTemp[j*width + i] / 256.0);
					}
				}
				delete pImg;
				delete pTemp;
			}
		}
		fclose(pFile);
	}
	else
	{
		printf("open file fail\n");
	}
	return ret;
}

std::vector<std::string> findFileNames(std::string path, std::string exd)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string pathName, exdName;
	std::vector<std::string> files;
	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
			{
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return files;
}

void test_MNIST()
{
	std::vector<std::vector<double>> data, data_v;
	std::vector<int> out, out_v;

	std::vector<std::string> files = findFileNames("E:\\mldata\\mnist\\train_images", "bmp");
	std::vector<std::string> files_v = findFileNames("E:\\mldata\\mnist\\test_images", "bmp");

	std::vector<double> ret;
	for (int i = 0; i < files.size(); ++i)
	{
		ret = readBmp(("E:\\mldata\\mnist\\train_images\\" + files[i]).c_str());
		if (ret.size() > 0)
		{
			data.push_back(ret);
			switch (files[i].front())
			{
			case '0':
				out.push_back(0);
				break;
			case '1':
				out.push_back(1);
				break;
			case '2':
				out.push_back(2);
				break;
			case '3':
				out.push_back(3);
				break;
			case '4':
				out.push_back(4);
				break;
			case '5':
				out.push_back(5);
				break;
			case '6':
				out.push_back(6);
				break;
			case '7':
				out.push_back(7);
				break;
			case '8':
				out.push_back(8);
				break;
			case '9':
				out.push_back(9);
				break;
			}
		}
	}

	for (int i = 0; i < files_v.size(); ++i)
	{
		ret = readBmp(("E:\\mldata\\mnist\\test_images\\" + files_v[i]).c_str());
		if (ret.size() > 0)
		{
			data_v.push_back(ret);
			switch (files_v[i].front())
			{
			case '0':
				out_v.push_back(0);
				break;
			case '1':
				out_v.push_back(1);
				break;
			case '2':
				out_v.push_back(2);
				break;
			case '3':
				out_v.push_back(3);
				break;
			case '4':
				out_v.push_back(4);
				break;
			case '5':
				out_v.push_back(5);
				break;
			case '6':
				out_v.push_back(6);
				break;
			case '7':
				out_v.push_back(7);
				break;
			case '8':
				out_v.push_back(8);
				break;
			case '9':
				out_v.push_back(9);
				break;
			}
		}
	}

	std::vector<unsigned int> sparse;
	sparse.push_back(50);
	sparse.push_back(20);
	sparse.push_back(5);
	std::vector<int> layer;
	layer.push_back(1024);
	layer.push_back(512);
	layer.push_back(256);
	layer.push_back(64);
	ML::IMGP_Mlp::Pointer mlp = ML::IMGP_Mlp::create();
	mlp->set_activation_function(ML::LEAKY_RELU);
	mlp->set_cost_function(ML::CROSS_ENTROPY);
	mlp->set_dropout_enable(false);
	mlp->set_dropout_regulation_para(0.1);
	mlp->set_hidden_layers(layer);
	mlp->set_input_preprocess(ML::ZERO_MEAN_GAUSSIAN_DENOISE);
	mlp->set_l1_l2_regulation_para(0.5);
	mlp->set_leaky_relu_para(0.05);
	mlp->set_learning_rate(0.0001);
	mlp->set_maxnormconstraint_regulation_para(1.0);
	mlp->set_mini_batch_size(512);
	mlp->set_regulation_type(ML::MAX_NORM_CONSTRAINT);
	mlp->set_solver_type(ML::ADAM, 0.9, 0.999);
	mlp->set_termination_para(3000, 5.0e-002);
	mlp->set_weight_constant_value(0.01);
	mlp->set_weight_initialization(ML::XAVIERFILLER);
	mlp->set_weight_xavierfiller_variance_norm(ML::AVERAGE);
	mlp->set_weight_msrafiller_variance_norm(ML::AVERAGE);
	mlp->set_weight_uniform_min_max_value(0.0, 1.0);
	mlp->set_weight_positive_unitball_min_max_value(-1.0, 1.0);
	mlp->set_weight_gaussian_mean_stddev_sparse(0.0, 0.1, sparse);
	mlp->set_validation_sample(data_v, out_v);
	if(mlp->train(data, out))
		mlp->save_model("model5.txt");
}

void test_MLP()
{
	std::vector<unsigned int> sparse;
	sparse.push_back(50);
	sparse.push_back(20);
	sparse.push_back(5);
	sparse.push_back(1);
	std::vector<int> layer;
	layer.push_back(500);
	layer.push_back(200);
	layer.push_back(50);
	layer.push_back(10);
	ML::IMGP_Mlp::Pointer mlp = ML::IMGP_Mlp::create();
	mlp->set_activation_function(ML::LEAKY_RELU);
	mlp->set_cost_function(ML::CROSS_ENTROPY);
	mlp->set_dropout_enable(false);
	mlp->set_dropout_regulation_para(0.5);
	mlp->set_hidden_layers(layer);
	mlp->set_input_preprocess(ML::ZERO_MEAN_GAUSSIAN_DENOISE);
	mlp->set_l1_l2_regulation_para(0.5);
	mlp->set_leaky_relu_para(0.05);
	mlp->set_learning_rate(0.0002);
	mlp->set_maxnormconstraint_regulation_para(1.0);
	mlp->set_mini_batch_size(51);
	mlp->set_regulation_type(ML::MAX_NORM_CONSTRAINT);
	mlp->set_solver_type(ML::ADAM, 0.9, 0.999);
	mlp->set_termination_para(10000, 1.0e-004);
	mlp->set_weight_constant_value(0.01);
	mlp->set_weight_initialization(ML::XAVIERFILLER);
	mlp->set_weight_xavierfiller_variance_norm(ML::AVERAGE);
	mlp->set_weight_msrafiller_variance_norm(ML::AVERAGE);
	mlp->set_weight_uniform_min_max_value(0.0, 1.0);
	mlp->set_weight_positive_unitball_min_max_value(-1.0, 1.0);
	mlp->set_weight_gaussian_mean_stddev_sparse(0.0, 0.1, sparse);
	mlp->train("data.txt");
	mlp->save_model("model.txt");
}

void test_watershed()
{
	IMGP_Watershed::Pointer pWatershed = IMGP_Watershed::New();
	IMGP_Image image(100, 100, IMGP_U8C1);
	IMGP_Image basin, watershed;
	pWatershed->watershed_tranform(image, basin, watershed);
}

int _tmain(int argc, _TCHAR* argv[])
{
	test_MNIST();
	_getch();
	return 0;
}
