#ifndef IMGP_MLP_HHHHH___
#define IMGP_MLP_HHHHH___

#include <memory>
#include <vector>

namespace IMGP
{
	namespace ML
	{
		enum ACTIVATION_FUNCTION_TYPE{ SIGMOD = 0, SIGMOD_SYM = 1, TANH, RELU, LEAKY_RELU, SOFT_PLUS };
		enum COST_FUNCTION_TYPE{ MEAN_SQUARE_ERROR = 0, CROSS_ENTROPY };
		enum WEIGHT_INIT_TYPE{ CONSTANT = 0, UNIFORM, GAUSSIAN, POSITIVE_UNITBALL, XAVIERFILLER, MSRAFILLER, BILINEARFILLER };
		enum INPUT_PREPROCESS_TYPE{ NOPROCESS = 0, NORMALISE_0, NORMALISE_1, ZERO_MEAN, ZERO_MEAN_GAUSSIAN, ZERO_MEAN_GAUSSIAN_DENOISE };
		enum VARIANCENORM{ FAN_IN = 0, FAN_OUT, AVERAGE };
		enum REGULATION_TYPE{ NONE_NORM = 0, L1_NORM, L2_NORM, MAX_NORM_CONSTRAINT };
		enum SOLVER_TYPE{ SGD_NORMAL, SGD_MOMENT, NAG, RMSPROP, ADAM };
		class IMGP_Mlp
		{
		public:
			virtual ~IMGP_Mlp(){}
			virtual bool train(std::string data_file_path) = 0;
			virtual bool train(std::vector<std::vector<double>> data) = 0;
			virtual bool load_model(std::string model_path) = 0;
			virtual bool save_model(std::string model_path) = 0;
			virtual void set_activation_function(ACTIVATION_FUNCTION_TYPE type) = 0;
			virtual void set_cost_function(COST_FUNCTION_TYPE type) = 0;
			virtual void set_learning_rate(double learn_rate) = 0;
			virtual void set_leaky_relu_para(double alpha) = 0;
			virtual void set_termination_para(int max_iters, double eps) = 0;
			virtual void set_hidden_layers(std::vector<int> layer) = 0;
			virtual void set_weight_initialization(WEIGHT_INIT_TYPE type) = 0;
			virtual void set_input_preprocess(INPUT_PREPROCESS_TYPE type) = 0;
			virtual void set_regulation_type(REGULATION_TYPE type) = 0;
			virtual void set_dropout_enable(bool bEnable) = 0;
			virtual void set_solver_type(SOLVER_TYPE type, double moment = 0.5, double decay = 0.9) = 0;

			virtual void set_l1_l2_regulation_para(double lamda) = 0;
			virtual void set_maxnormconstraint_regulation_para(double c) = 0;
			virtual void set_dropout_regulation_para(double ratio) = 0;


			virtual void set_weight_constant_value(double constant_value) = 0;
			virtual void set_weight_uniform_min_max_value(double min_value, double max_value) = 0;
			virtual void set_weight_gaussian_mean_stddev_sparse(double mean, double stddev, std::vector<unsigned int> sparse) = 0;
			virtual void set_weight_positive_unitball_min_max_value(double min_value, double max_value) = 0;
			virtual void set_weight_xavierfiller_variance_norm(VARIANCENORM norm) = 0;
			virtual void set_weight_msrafiller_variance_norm(VARIANCENORM norm) = 0;

			virtual void set_mini_batch_size(int size) = 0;

			static std::shared_ptr<IMGP_Mlp> create();
		};
	}
}

#endif