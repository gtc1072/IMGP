#ifndef IMGP_MLP_IMPL_HHHHH___
#define IMGP_MLP_IMPL_HHHHH___

#define ALG_MLP_UPPER_LIMIT 500

#define CHECK_ALG_MLP_UPPER_LIMIT(x) \
	if((x) > ALG_MLP_UPPER_LIMIT) (x) = ALG_MLP_UPPER_LIMIT; \
	if((x) < -ALG_MLP_UPPER_LIMIT) (x) = -ALG_MLP_UPPER_LIMIT;

#define ALG_MLP_SOFTMAX_LOW_LIMIT 1.0e-004
#define ALG_MLP_SOFTMAX_HIGH_LIMIT 0.9999

#define CHECK_ALG_MLP_SOFTMAX_LIMIT(x)\
	if((x) < ALG_MLP_SOFTMAX_LOW_LIMIT) (x) = ALG_MLP_SOFTMAX_LOW_LIMIT; \
	if((x) > ALG_MLP_SOFTMAX_HIGH_LIMIT) (x) = ALG_MLP_SOFTMAX_HIGH_LIMIT;

#include <math.h>
#include "IMGP_MLP.h"

namespace IMGP
{
	namespace ML
	{
		inline double activation_sigmod(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			return 1.0 / (1.0 + std::exp(-val));
		}

		inline double activation_sigmod_sym(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			double q = std::exp(-val);
			return (1.0 - q) / (1.0 + q);
		}

		inline double activation_tanh(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			double p = std::exp(val);
			double q = std::exp(-val);
			return (p - q) / (p + q);
		}

		inline double activation_relu(double val, double alpha)
		{
			return val > 0 ? val : 0.0;
		}

		inline double activation_leaky_relu(double val, double alpha)
		{
			return val > 0 ? val : alpha * val;
		}

		inline double activation_softplus(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			return std::log(1.0 + std::exp(val));
		}

		typedef double(*activation_fun)(double val, double alpha);

		static activation_fun get_activation_fun(unsigned int idx)
		{
			static activation_fun fun_table[] = { activation_sigmod, activation_sigmod_sym, activation_tanh,
				activation_relu, activation_leaky_relu, activation_softplus
			};
			return fun_table[idx];
		}

		inline double deactivation_sigmod(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			double q = 1.0 / (1.0 + std::exp(-val));
			return q * (1.0 - q);
		}

		inline double deactivation_sigmod_sym(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			double q = std::exp(-val);
			return 2.0 * q / ((1.0 + q) * (1.0 + q));
		}

		inline double deactivation_tanh(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			double p = std::exp(val);
			double q = std::exp(-val);
			return 1.0 - std::pow((p - q) / (p + q), 2.0);
		}

		inline double deactivation_relu(double val, double alpha)
		{
			return val > 0 ? 1.0 : 0.0;
		}

		inline double deactivation_leaky_relu(double val, double alpha)
		{
			return val > 0 ? 1.0 : alpha;
		}

		inline double deactivation_softplus(double val, double alpha)
		{
			CHECK_ALG_MLP_UPPER_LIMIT(val);
			return 1.0 / (1.0 + std::exp(-val));
		}

		typedef double(*deactivation_fun)(double val, double alpha);

		static deactivation_fun get_deactivation_fun(unsigned int idx)
		{
			static deactivation_fun fun_table_de[] = { deactivation_sigmod, deactivation_sigmod_sym, deactivation_tanh,
				deactivation_relu, deactivation_leaky_relu, deactivation_softplus
			};
			return fun_table_de[idx];
		}

		class IMGP_Mlp_impl :public IMGP_Mlp
		{
		public:
			IMGP_Mlp_impl();
			virtual ~IMGP_Mlp_impl();
			virtual bool train(std::string data_file_path);
			virtual bool train(std::vector<std::vector<double>> &data, std::vector<int> &out);
			virtual void set_activation_function(ACTIVATION_FUNCTION_TYPE type);
			virtual bool load_model(std::string model_path);
			virtual bool save_model(std::string model_path);
			virtual bool set_validation_sample(std::string data_file_path);
			virtual bool set_validation_sample(std::vector<std::vector<double>> &data, std::vector<int> &out);
			virtual void set_cost_function(COST_FUNCTION_TYPE type);
			virtual void set_learning_rate(double learn_rate);
			virtual void set_leaky_relu_para(double alpha);
			virtual void set_termination_para(int max_iters, double eps);
			virtual void set_hidden_layers(std::vector<int> layer);
			virtual void set_weight_initialization(WEIGHT_INIT_TYPE type);
			virtual void set_input_preprocess(INPUT_PREPROCESS_TYPE type);
			virtual void set_regulation_type(REGULATION_TYPE type);
			virtual void set_dropout_enable(bool bEnable);
			virtual void set_solver_type(SOLVER_TYPE type, double moment = 0.5, double decay = 0.9);

			virtual void set_l1_l2_regulation_para(double lamda);
			virtual void set_maxnormconstraint_regulation_para(double c);
			virtual void set_dropout_regulation_para(double ratio);

			virtual void set_weight_constant_value(double constant_value);
			virtual void set_weight_uniform_min_max_value(double min_value, double max_value);
			virtual void set_weight_gaussian_mean_stddev_sparse(double mean, double stddev, std::vector<unsigned int> sparse);
			virtual void set_weight_positive_unitball_min_max_value(double min_value, double max_value);
			virtual void set_weight_xavierfiller_variance_norm(VARIANCENORM norm);
			virtual void set_weight_msrafiller_variance_norm(VARIANCENORM norm);

			virtual void set_mini_batch_size(int size);
		private:
			int getPredict(std::vector<double> &hist, int idx, double &err);
			bool load_sample(std::string  path);
			bool init_classifier();
			void front_propagation_batch(int idx);
			void back_propagation_bacth(std::vector<double> &sample, std::vector<double> &target_out);
			void back_propagation_bacth_solve();
			void do_solve_sgd_normal();
			void do_solve_sgd_moment();
			void do_solve_nag();
			void do_solve_rmsprop();
			void do_solve_adam();
			bool validate_sample_and_output();
			void get_output_count();
			void init_weight_mat();
			void set_weight_mat_init_value();
			void preprocess_input();
			bool train_batch();
			std::vector<double> get_error_value(std::vector<double> cur_out, std::vector<double> target_out);

			void weight_initialization_constant();
			void weight_initialization_uniform();
			void weight_initialization_gaussian();
			void weight_initialization_positiveunitball();
			void weight_initialization_xavierfiller();
			void weight_initialization_marafiller();
			void weight_initialization_bilinearfiller();

			void input_process_normalise0();
			void input_process_normalise1();
			void input_process_zero_mean();
			void input_process_zero_gaussian_mean();
			void input_process_zero_gaussian_mean_denoise();
		private:
			typedef std::vector<std::vector<double>> MLP_Mat;
			std::vector<int>					m_hidden_layer_sizes;
			int									m_max_iter;
			std::vector<std::vector<double>>	m_input;
			std::vector<int>					m_output;
			std::vector<std::vector<double>>	m_validation_input;
			std::vector<int>					m_validation_output;
			ACTIVATION_FUNCTION_TYPE			m_activation_fun_id;
			COST_FUNCTION_TYPE					m_err_type;
			WEIGHT_INIT_TYPE					m_weight_initialization_type;
			INPUT_PREPROCESS_TYPE				m_input_preprocess_type;
			REGULATION_TYPE						m_regulation_type;
			SOLVER_TYPE							m_solver_type;

			unsigned int						m_sample_count;
			unsigned int						m_sample_demension;
			unsigned int						m_output_demension;
			std::vector<MLP_Mat>				m_weights;
			std::vector<MLP_Mat>				m_weight_graditude, m_pre_weight_graditude, m_weight_graditude_batch, m_weight_decay, m_pre_weight_decay;
			MLP_Mat								m_neuron_value;
			MLP_Mat								m_neuron_graditude;
			MLP_Mat								m_neuron_derivate;
			activation_fun						m_act_function;
			deactivation_fun					m_deact_function;
			std::vector<std::pair<int, std::vector<double>>>  m_target_output;
			std::vector<double>					m_err_value;
			double								m_learn_rate;
			double								m_learn_moment;
			double								m_learn_decay;
			double								m_leaky_relu_alpha;
			double								m_error_eps;
			double								m_pre_error_sum;
			double								m_weight_constant_value;
			double								m_weight_uniform_min_value, m_weight_uniform_max_value;
			double								m_weight_gaussian_mean_value, m_weight_gaussian_stddev_value;
			std::vector<unsigned int>			m_weight_gaussian_sparse;
			double								m_weight_positive_unitball_min_value, m_weight_positive_unitball_max_value;
			VARIANCENORM						m_weight_xavierfiller_variance_norm, m_weight_msrafiller_variance_norm;

			std::vector<double>					m_input_mean, m_input_min, m_input_max, m_input_stddev;

			unsigned int						m_batch_size;

			double								m_regulation_norm_lamda;
			double								m_regulation_max_norm;
			double								m_regulation_dropout_ratio;
			bool								m_regulation_enable_dropout;
			std::vector<std::vector<bool>>		m_dropout_map;
			int									m_weight_count;

			std::vector<int>					m_loop_index;
			int									m_iter_batch;
		};
	}
}

#endif

