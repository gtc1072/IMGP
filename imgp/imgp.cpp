// imgp.cpp : 定义控制台应用程序的入口点。
//

#include <tchar.h>
#include <conio.h>
#include "..\src\segmentation\Imgp_Watershed.h"
#include "..\src\common\Imgp_Base.h"
#include "..\src\classify\Imgp_MLP.h"

using namespace IMGP;

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
	mlp->set_learning_rate(0.0001);
	mlp->set_maxnormconstraint_regulation_para(1.0);
	mlp->set_mini_batch_size(51);
	mlp->set_regulation_type(ML::MAX_NORM_CONSTRAINT);
	mlp->set_solver_type(ML::ADAM, 0.9, 0.9);
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
	test_MLP();
	_getch();
	return 0;
}
