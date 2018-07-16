//#include "stdafx.h"
#include "IMGP_Mlp_impl.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

namespace IMGP
{
	namespace ML
	{
		std::shared_ptr<IMGP_Mlp> IMGP_Mlp::create()
		{
			return std::make_shared<IMGP_Mlp_impl>();
		}

		IMGP_Mlp_impl::IMGP_Mlp_impl()
		{
			m_max_iter = 10000;
			m_error_eps = 1.0e-004;
			m_pre_error_sum = 1.0e+004;
			m_hidden_layer_sizes.push_back(500);
			m_hidden_layer_sizes.push_back(200);
			m_hidden_layer_sizes.push_back(50);
			m_hidden_layer_sizes.push_back(10);
			m_activation_fun_id = LEAKY_RELU;
			m_act_function = get_activation_fun(m_activation_fun_id);
			m_deact_function = get_deactivation_fun(m_activation_fun_id);
			m_sample_count = m_sample_demension = m_output_demension = 0;
			m_err_type = CROSS_ENTROPY;
			m_weight_initialization_type = XAVIERFILLER;
			m_input_preprocess_type = ZERO_MEAN_GAUSSIAN_DENOISE;
			m_regulation_type = MAX_NORM_CONSTRAINT;
			m_regulation_norm_lamda = 0.5;
			m_regulation_max_norm = 1.0;
			m_regulation_dropout_ratio = 0.5;
			m_regulation_enable_dropout = true;
			m_weight_count = 0;
			m_solver_type = ADAM;
			m_batch_size = 51;
			m_learn_rate = 0.0001;
			m_learn_moment = 0.9;
			m_learn_decay = 0.9;
			m_leaky_relu_alpha = 0.05;
			m_weight_constant_value = 0.01;
			m_weight_uniform_min_value = 0.0;
			m_weight_uniform_max_value = 1.0;
			m_weight_gaussian_mean_value = 0.0;
			m_weight_gaussian_stddev_value = 1.0;
			m_weight_positive_unitball_min_value = -1.0;
			m_weight_positive_unitball_max_value = 1.0;
			m_weight_xavierfiller_variance_norm = m_weight_msrafiller_variance_norm = AVERAGE;
		}

		IMGP_Mlp_impl::~IMGP_Mlp_impl()
		{

		}

		void IMGP_Mlp_impl::set_activation_function(ACTIVATION_FUNCTION_TYPE type)
		{
			m_activation_fun_id = type;
			m_act_function = get_activation_fun(m_activation_fun_id);
			m_deact_function = get_deactivation_fun(m_activation_fun_id);
		}

		void IMGP_Mlp_impl::set_cost_function(COST_FUNCTION_TYPE type)
		{
			m_err_type = type;
		}

		void IMGP_Mlp_impl::set_learning_rate(double learn_rate)
		{
			m_learn_rate = learn_rate;
			m_learn_rate = m_learn_rate < 0 ? 0 : (m_learn_rate > 1.0 ? 1.0 : m_learn_rate);
		}

		void IMGP_Mlp_impl::set_leaky_relu_para(double alpha)
		{
			m_leaky_relu_alpha = alpha;
			m_leaky_relu_alpha = m_leaky_relu_alpha < 0 ? 0 : (m_leaky_relu_alpha > 1.0 ? 1.0 : m_leaky_relu_alpha);
		}

		void IMGP_Mlp_impl::set_termination_para(int max_iters, double eps)
		{
			m_max_iter = max_iters < 10 ? 10 : m_max_iter;
			m_error_eps = eps < 1.0e-009 ? 1.0e-009 : eps;
		}

		void IMGP_Mlp_impl::set_hidden_layers(std::vector<int> layer)
		{
			if (!layer.empty())
			{
				m_hidden_layer_sizes = layer;
			}
		}

		void IMGP_Mlp_impl::set_weight_initialization(WEIGHT_INIT_TYPE type)
		{
			m_weight_initialization_type = type;
		}

		void IMGP_Mlp_impl::set_input_preprocess(INPUT_PREPROCESS_TYPE type)
		{
			m_input_preprocess_type = type;
		}

		void IMGP_Mlp_impl::set_regulation_type(REGULATION_TYPE type)
		{
			m_regulation_type = type;
		}

		void IMGP_Mlp_impl::set_solver_type(SOLVER_TYPE type, double moment, double decay)
		{
			m_solver_type = type;

			m_learn_moment = moment;
			m_learn_moment = m_learn_moment < 0 ? 0 : (m_learn_moment > 0.9 ? 0.9 : m_learn_moment);

			m_learn_decay = decay;
			m_learn_decay = m_learn_decay < 0 ? 0 : (m_learn_decay > 0.9 ? 0.9 : m_learn_decay);
		}

		void IMGP_Mlp_impl::set_l1_l2_regulation_para(double lamda)
		{
			m_regulation_norm_lamda = lamda < 0 ? 0 : lamda;
		}

		void IMGP_Mlp_impl::set_maxnormconstraint_regulation_para(double c)
		{
			m_regulation_max_norm = c < 0.1 ? 0.1 : c;
		}

		void IMGP_Mlp_impl::set_dropout_regulation_para(double ratio)
		{
			m_regulation_dropout_ratio = ratio < 0 ? 0 : (ratio > 0.9 ? 0.9 : ratio);
		}

		void IMGP_Mlp_impl::set_dropout_enable(bool bEnable)
		{
			m_regulation_enable_dropout = bEnable;
		}

		void IMGP_Mlp_impl::input_process_normalise0()
		{
			int input_size = m_input.size();
			if (0 == input_size) return;
			int input_demension = m_input[0].size();
			if (0 == input_demension) return;
			m_input_min = std::vector<double>(input_demension, 1.0e+020);
			m_input_max = std::vector<double>(input_demension, -1.0e+020);
			m_input_mean = std::vector<double>(input_demension, 0);
			double value = 0.0;
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					value = m_input[j][i];
					if (value < m_input_min[i]) m_input_min[i] = value;
					if (value > m_input_max[i]) m_input_max[i] = value;
				}
			}
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input[j][i] = (m_input[j][i] - m_input_min[i]) / (m_input_max[i] - m_input_min[i]);
				}
			}
		}

		void IMGP_Mlp_impl::input_process_normalise1()
		{
			int input_size = m_input.size();
			if (0 == input_size) return;
			int input_demension = m_input[0].size();
			if (0 == input_demension) return;
			m_input_min = std::vector<double>(input_demension, 1.0e+020);
			m_input_max = std::vector<double>(input_demension, -1.0e+020);
			m_input_mean = std::vector<double>(input_demension, 0);
			double value = 0.0;
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					value = m_input[j][i];
					if (value < m_input_min[i]) m_input_min[i] = value;
					if (value > m_input_max[i]) m_input_max[i] = value;
				}
			}
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input[j][i] = (m_input[j][i] - m_input_min[i]) / (m_input_max[i] - m_input_min[i]);
					m_input[j][i] = m_input[j][i] * 2.0 - 1.0;
				}
			}
		}

		void IMGP_Mlp_impl::input_process_zero_mean()
		{
			int input_size = m_input.size();
			if (0 == input_size) return;
			int input_demension = m_input[0].size();
			if (0 == input_demension) return;
			m_input_mean = std::vector<double>(input_demension, 0);
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input_mean[i] += m_input[j][i];
				}
			}

			for (int i = 0; i < input_demension; ++i)
			{
				m_input_mean[i] /= input_size;
			}
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input[j][i] -= m_input_mean[i];
				}
			}
		}

		void IMGP_Mlp_impl::input_process_zero_gaussian_mean()
		{
			int input_size = m_input.size();
			if (0 == input_size) return;
			int input_demension = m_input[0].size();
			if (0 == input_demension) return;
			m_input_mean = std::vector<double>(input_demension, 0);
			m_input_stddev = std::vector<double>(input_demension, 0);
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input_mean[i] += m_input[j][i];
				}
			}

			for (int i = 0; i < input_demension; ++i)
			{
				m_input_mean[i] /= input_size;
			}
			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input_stddev[i] += (m_input[j][i] - m_input_mean[i]) * (m_input[j][i] - m_input_mean[i]);
				}
			}
			for (int i = 0; i < input_demension; ++i)
			{
				m_input_stddev[i] = std::sqrt(m_input_stddev[i] / input_size);
			}

			double value = 0.0;

			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					value = m_input[j][i] - m_input_mean[i];
					m_input[j][i] = 1.0 - std::exp(-0.5 * value * value / (m_input_stddev[i] * m_input_stddev[i]));
					m_input[j][i] *= value < 0 ? -1.0 : 1.0;
				}
			}
		}

		void IMGP_Mlp_impl::input_process_zero_gaussian_mean_denoise()
		{
			int input_size = m_input.size();
			if (0 == input_size) return;
			int input_demension = m_input[0].size();
			if (0 == input_demension) return;
			m_input_mean = std::vector<double>(input_demension, 0);
			m_input_stddev = std::vector<double>(input_demension, 0);
			std::vector<std::vector<double>> prob;
			std::vector<double> prob_i(input_demension, 1.0);
			for (int j = 0; j < input_size; ++j)
			{
				prob.push_back(prob_i);
			}

			std::vector<double> pre_mean(input_demension, 0.0);

			double eps_diff = 1.0e+002;
			double r = std::sqrt(2.0 * 3.1415926535);

			auto f = [&](){
				std::vector<double> sum_prob(input_demension, 0.0);
				for (int j = 0; j < input_size; ++j)
				{
					for (int i = 0; i < input_demension; ++i)
					{
						sum_prob[i] += prob[j][i];
						m_input_mean[i] += m_input[j][i] * prob[j][i];
					}
				}

				for (int i = 0; i < input_demension; ++i)
				{
					m_input_mean[i] /= sum_prob[i];
				}
				for (int j = 0; j < input_size; ++j)
				{
					for (int i = 0; i < input_demension; ++i)
					{
						m_input_stddev[i] += (m_input[j][i] - m_input_mean[i]) * (m_input[j][i] - m_input_mean[i]);
					}
				}
				for (int i = 0; i < input_demension; ++i)
				{
					m_input_stddev[i] = std::sqrt(m_input_stddev[i] / input_size);
				}
			};

			while (eps_diff > 1.0e-003)
			{
				f();

				for (int j = 0; j < input_size; ++j)
				{
					for (int i = 0; i < input_demension; ++i)
					{
						prob[j][i] = std::exp(-0.5 * (m_input[j][i] - m_input_mean[i]) * (m_input[j][i] - m_input_mean[i]) / (m_input_stddev[i] * m_input_stddev[i]));
					}
				}
				eps_diff = -1.0;
				for (int i = 0; i < input_demension; ++i)
				{
					if (fabs(m_input_mean[i] - pre_mean[i]) > eps_diff)
					{
						eps_diff = fabs(m_input_mean[i] - pre_mean[i]);
					}
				}
				pre_mean = m_input_mean;
			}

			f();

			for (int j = 0; j < input_size; ++j)
			{
				for (int i = 0; i < input_demension; ++i)
				{
					m_input[j][i] = ((m_input[j][i] - m_input_mean[i]) < 0 ? -1.0 : 1.0) * std::exp(-0.5 * (m_input[j][i] - m_input_mean[i]) * (m_input[j][i] - m_input_mean[i]) / (m_input_stddev[i] * m_input_stddev[i]));
				}
			}
		}

		void IMGP_Mlp_impl::preprocess_input()
		{
			switch (m_input_preprocess_type)
			{
			case NOPROCESS:
				break;
			case NORMALISE_0:
				input_process_normalise0();
				break;
			case NORMALISE_1:
				input_process_normalise1();
				break;
			case ZERO_MEAN:
				input_process_zero_mean();
				break;
			case ZERO_MEAN_GAUSSIAN:
				input_process_zero_gaussian_mean();
				break;
			case ZERO_MEAN_GAUSSIAN_DENOISE:
				input_process_zero_gaussian_mean_denoise();
				break;
			default:
				break;
			}
		}

		bool IMGP_Mlp_impl::train(std::string data_file_path)
		{
			bool ret = load_sample(data_file_path);
			if (ret)
			{
				preprocess_input();
				ret = init_classifier();
			}
			return ret;
		}

		bool IMGP_Mlp_impl::train(std::vector<std::vector<double>> data)
		{
			bool ret = false;
			preprocess_input();
			ret = init_classifier();
			return ret;
		}


		bool IMGP_Mlp_impl::load_sample(std::string  path)
		{
			bool ret = true;
			std::vector<std::string> vec;
			std::filebuf fb;
			if (fb.open(path.c_str(), std::ios::in))
			{
				std::istream is(&fb);
				while (!(is.eof()))
				{
					std::string str;
					std::getline(is, str, '\n');
					vec.push_back(str);
				}
				fb.close();
			}
			else
			{
				ret = false;
			}
			for (std::vector<std::string>::iterator it = vec.begin(); it != vec.end(); ++it)
			{
				if ((*it)[0] == 'p')
				{
					it->erase(0, 4);
					it->pop_back();
					it->pop_back();
					size_t pre_pos = 0;
					size_t cur_pos = it->find(' ', pre_pos);
					if (cur_pos != std::string::npos)
					{
						std::string s = it->substr(pre_pos, cur_pos - pre_pos);
						m_activation_fun_id = (ACTIVATION_FUNCTION_TYPE)atoi(s.c_str());
						pre_pos = cur_pos + 1;
						cur_pos = it->find(' ', pre_pos);
					}
					if (cur_pos != std::string::npos)
					{
						std::string s = it->substr(pre_pos, cur_pos - pre_pos);
						m_max_iter = atoi(s.c_str());
						pre_pos = cur_pos + 1;
						cur_pos = it->find(' ', pre_pos);
					}
					if (cur_pos != std::string::npos)
					{
						m_hidden_layer_sizes.clear();
					}
					while (cur_pos != std::string::npos)
					{
						std::string s = it->substr(pre_pos, cur_pos - pre_pos);
						m_hidden_layer_sizes.push_back(atoi(s.c_str()));
						pre_pos = cur_pos + 1;
						cur_pos = it->find(' ', pre_pos);
					}
					std::string s = it->substr(pre_pos, it->size() - pre_pos);
					if (!(s.empty()))
						m_hidden_layer_sizes.push_back(atoi(s.c_str()));
				}
				else
				{
					m_output.push_back(atoi((&(*it)[0])));
					it->erase(0, 4);
					it->pop_back();
					it->pop_back();
					size_t pre_pos = 0;
					size_t cur_pos = it->find(' ', pre_pos);
					std::vector<double> fv;
					while (cur_pos != std::string::npos)
					{
						std::string s = it->substr(pre_pos, cur_pos - pre_pos);
						fv.push_back(atof(s.c_str()));
						pre_pos = cur_pos + 1;
						cur_pos = it->find(' ', pre_pos);
					}
					std::string s = it->substr(pre_pos, it->size() - pre_pos);
					if (!(s.empty()))
						fv.push_back(atof(s.c_str()));
					m_input.push_back(fv);
				}
			}
			if (m_input.empty() || m_input.size() != m_output.size())
				ret = false;
			return ret;
		}

		bool IMGP_Mlp_impl::init_classifier()
		{
			bool ret = validate_sample_and_output();
			if (ret)
			{
				init_weight_mat();
				set_weight_mat_init_value();
				ret = train_batch();
			}
			printf("final result:\n");
			if (ret)
			{
				for (size_t i = 0; i < m_input.size(); ++i)
				{
					double err_i = 0;
					int cls = getPredict(m_input[i], m_output[i], err_i);
					printf("%d---[%d,%d,%.4f]\n", i, cls, m_output[i], err_i);
				}
			}
			return ret;
		}

		bool IMGP_Mlp_impl::load_model(std::string model_path)
		{
			return true;
		}

		bool IMGP_Mlp_impl::save_model(std::string model_path)
		{
			FILE *fp;
			errno_t err = fopen_s(&fp, model_path.c_str(), "wb");
			if (!err)
			{
				std::string text;
				char buf[4096];

				text = "MLP Model:\r\n\r\n";
				fwrite(text.c_str(), 1, text.length(), fp);

				sprintf_s(buf, 4096, "Max Iterations: %d\r\nStop Criteration: %.8f\r\n\r\n", m_max_iter, m_error_eps);
				fwrite(buf, 1, strlen(buf), fp);

				switch (m_activation_fun_id)
				{
				case SIGMOD:
					sprintf_s(buf, 4096, "Activation Function: sigmod\r\n\r\n");
					break;
				case SIGMOD_SYM:
					sprintf_s(buf, 4096, "Activation Function: sigmod symetric\r\n\r\n");
					break;
				case TANH:
					sprintf_s(buf, 4096, "Activation Function: tanh\r\n\r\n");
					break;
				case RELU:
					sprintf_s(buf, 4096, "Activation Function: relu\r\n\r\n");
					break;
				case LEAKY_RELU:
					sprintf_s(buf, 4096, "Activation Function: leaky relu %.5f\r\n\r\n", m_leaky_relu_alpha);
					break;
				case SOFT_PLUS:
					sprintf_s(buf, 4096, "Activation Function: soft plus\r\n\r\n");
					break;
				default:
					sprintf_s(buf, 4096, "Activation Function: none\r\n\r\n");
					break;
				}
				fwrite(buf, 1, strlen(buf), fp);

				if (m_err_type == CROSS_ENTROPY)
					text = "Cost Function: cross entropy\r\n\r\n";
				else
					text = "Cost Function: mean square\r\n\r\n";
				fwrite(text.c_str(), 1, text.length(), fp);

				switch (m_weight_initialization_type)
				{
				case CONSTANT:
					sprintf_s(buf, 4096, "Weight Initialization Type: constant %.4f\r\n\r\n", m_weight_constant_value);
					break;
				case UNIFORM:
					sprintf_s(buf, 4096, "Weight Initialization Type: uniform %.4f %.4f\r\n\r\n", m_weight_uniform_min_value, m_weight_uniform_max_value);
					break;
				case GAUSSIAN:
					sprintf_s(buf, 4096, "Weight Initialization Type: gaussian %.4f %.4f\r\n\r\n", m_weight_gaussian_mean_value, m_weight_gaussian_stddev_value);
					break;
				case POSITIVE_UNITBALL:
					sprintf_s(buf, 4096, "Weight Initialization Type: positive unitball %.4f %.4f\r\n\r\n", m_weight_positive_unitball_min_value, m_weight_positive_unitball_max_value);
					break;
				case XAVIERFILLER:
					sprintf_s(buf, 4096, "Weight Initialization Type: xavierfiller %s\r\n\r\n", m_weight_xavierfiller_variance_norm == AVERAGE ? "average" : (m_weight_xavierfiller_variance_norm == FAN_IN ? "fan_in" : "fan_out"));
					break;
				case MSRAFILLER:
					sprintf_s(buf, 4096, "Weight Initialization Type: marafiller %s\r\n\r\n", m_weight_msrafiller_variance_norm == AVERAGE ? "average" : (m_weight_msrafiller_variance_norm == FAN_IN ? "fan_in" : "fan_out"));
					break;
				default:
					sprintf_s(buf, 4096, "Weight Initialization Type: none\r\n\r\n");
					break;
				}
				fwrite(buf, 1, strlen(buf), fp);

				switch (m_input_preprocess_type)
				{
				case NOPROCESS:
					text = "Input Sample Preprocession Type: noprocess\r\n\r\n";
					break;
				case NORMALISE_0:
					text = "Input Sample Preprocession Type: normalise 0\r\n\r\n";
					break;
				case NORMALISE_1:
					text = "Input Sample Preprocession Type: normalise 1\r\n\r\n";
					break;
				case ZERO_MEAN:
					text = "Input Sample Preprocession Type: zero mean\r\n\r\n";
					break;
				case ZERO_MEAN_GAUSSIAN:
					text = "Input Sample Preprocession Type: zero mean gaussian\r\n\r\n";
					break;
				case ZERO_MEAN_GAUSSIAN_DENOISE:
					text = "Input Sample Preprocession Type: zero mean gaussian denoise\r\n\r\n";
					break;
				default:
					text = "Input Sample Preprocession Type: none\r\n\r\n";
					break;
				}
				fwrite(text.c_str(), 1, text.length(), fp);

				switch (m_regulation_type)
				{
				case NONE_NORM:
					sprintf_s(buf, 4096, "Regulation Type: none norm\r\n\r\n");
					break;
				case L1_NORM:
					sprintf_s(buf, 4096, "Regulation Type: l1 norm %.4f\r\n\r\n", m_regulation_norm_lamda);
					break;
				case L2_NORM:
					sprintf_s(buf, 4096, "Regulation Type: l2 norm %.4f\r\n\r\n", m_regulation_norm_lamda);
					break;
				case MAX_NORM_CONSTRAINT:
					sprintf_s(buf, 4096, "Regulation Type: max norm constraint %.4f\r\n\r\n", m_regulation_max_norm);
					break;
				default:
					sprintf_s(buf, 4096, "Regulation Type: none\r\n\r\n");
					break;
				}
				fwrite(buf, 1, strlen(buf), fp);

				if (m_regulation_enable_dropout)
				{
					sprintf_s(buf, 4096, "Enable Dropout: true %.4f\r\n\r\n", m_regulation_dropout_ratio);
				}
				else
				{
					sprintf_s(buf, 4096, "Enable Dropout: false\r\n\r\n");
				}
				fwrite(buf, 1, strlen(buf), fp);

				switch (m_solver_type)
				{
				case SGD_NORMAL:
					sprintf_s(buf, 4096, "Solver Type: sgd normal\r\nBatch Size:%d\r\nLearning Step:%.8f\r\n\r\n", m_batch_size, m_learn_rate);
					break;
				case SGD_MOMENT:
					sprintf_s(buf, 4096, "Solver Type: sgd moment\r\nBatch Size:%d\r\nLearning Step:%.8f\r\nLearning Moment:%.8f\r\n\r\n", m_batch_size, m_learn_rate, m_learn_moment);
					break;
				case NAG:
					sprintf_s(buf, 4096, "Solver Type: nag\r\nBatch Size:%d\r\nLearning Step:%.8f\r\nLearning Moment:%.8f\r\n\r\n", m_batch_size, m_learn_rate, m_learn_moment);
					break;
				case RMSPROP:
					sprintf_s(buf, 4096, "Solver Type: rmsprop\r\nBatch Size:%d\r\nLearning Step:%.8f\r\nLearning Decay:%.8f\r\n\r\n", m_batch_size, m_learn_rate, m_learn_decay);
					break;
				case ADAM:
					sprintf_s(buf, 4096, "Solver Type: adam\r\nBatch Size:%d\r\nLearning Step:%.8f\r\nLearning Moment:%.8f\r\nLearning Decay:%.8f\r\n\r\n", m_batch_size, m_learn_rate, m_learn_moment, m_learn_decay);
					break;
				default:
					sprintf_s(buf, 4096, "Solver Type: none\r\n\r\n");
					break;
				}
				fwrite(buf, 1, strlen(buf), fp);

				text = "MLP Weights:\r\n\r\n";
				fwrite(text.c_str(), 1, text.length(), fp);

				for (size_t k = 0; k < m_weights.size(); ++k)
				{
					sprintf_s(buf, 4096, "Layer %d:\r\n", k);
					fwrite(buf, 1, strlen(buf), fp);
					double *pData = new double[m_weights[k][0].size()];
					for (size_t j = 0; j < m_weights[k][0].size(); ++j)
					{
						pData[j] = 0.0;
					}
					for (size_t j = 0; j < m_weights[k].size(); ++j)
					{
						text = "";
						for (size_t i = 0; i < m_weights[k][j].size(); ++i)
						{
							sprintf_s(buf, 4096, "%s%.4f ", m_weights[k][j][i] < 0 ? "" : "+", m_weights[k][j][i]);
							text += buf;
							pData[i] += fabs(m_weights[k][j][i]);
						}
						text += "\r\n";
						fwrite(text.c_str(), 1, text.length(), fp);
					}
					text = "\r\nWeight sum:\r\n";
					for (size_t j = 0; j < m_weights[k][0].size(); ++j)
					{
						sprintf_s(buf, 4096, "%.4f ", pData[j]);
						text += buf;
					}
					fwrite(text.c_str(), 1, text.length(), fp);
					fwrite("\r\n\r\n", 1, strlen("\r\n\r\n"), fp);
					delete pData;
				}
			}
			fclose(fp);
			return 0 == err ? true : false;
		}

		void IMGP_Mlp_impl::set_mini_batch_size(int size)
		{
			if (m_sample_count > 1 && size <= m_sample_count)
				m_batch_size = size;
		}

		bool IMGP_Mlp_impl::train_batch()
		{
			bool ret = true;

			int iter = 0;
			m_iter_batch = 0;
			double cur_error_sum = 1.0e+005;

			if (m_weight_count > 0) m_regulation_norm_lamda /= (m_weight_count * m_regulation_dropout_ratio);

			m_loop_index.clear();
			m_loop_index.resize(m_sample_count);
			for (std::vector<int>::iterator it = m_loop_index.begin(); it != m_loop_index.end(); ++it)
			{
				*it = iter;
				iter++;
			}

			std::random_shuffle(m_loop_index.begin(), m_loop_index.end());
			m_iter_batch = m_batch_size;
			iter = 0;

			while (iter++ < m_max_iter && m_error_eps < cur_error_sum)
			{
				if (m_regulation_enable_dropout)
				{
					if (m_dropout_map.empty())
					{
						for (int j = 0; j < m_weights.size(); ++j)
						{
							std::vector<bool> o_v(m_weights[j].size(), true);
							m_dropout_map.push_back(o_v);
						}
					}
					for (int j = 0; j < m_dropout_map.size() - 1; ++j)
					{
						int dropout_count = m_dropout_map[j].size() * m_regulation_dropout_ratio;
						std::vector<bool> temp_v(m_dropout_map[j].size(), true);
						for (int i = 0; i < dropout_count; ++i)
						{
							temp_v[i] = false;
						}
						std::random_device rd;
						std::mt19937 g(rd());
						std::shuffle(temp_v.begin(), temp_v.end(), g);
						m_dropout_map[j] = temp_v;
					}
				}
				int idx = m_iter_batch - m_batch_size;
				for (int i = idx; i < m_iter_batch; ++i)
				{
					front_propagation_batch(m_loop_index[i]);
				}
				back_propagation_bacth_solve();
				cur_error_sum = 0;
				if (ret)
				{
					for (size_t i = 0; i < m_input.size(); ++i)
					{
						double err_i = 0;
						int cls = getPredict(m_input[i], m_output[i], err_i);
						cur_error_sum += err_i;
						printf("%02d---[%d,%d,%.4f]\n", i, cls, m_output[i], err_i);
					}
					cur_error_sum /= m_input.size();
				}
				m_iter_batch += m_batch_size;
				if (m_iter_batch > m_sample_count)
				{
					std::random_shuffle(m_loop_index.begin(), m_loop_index.end());
					m_iter_batch = m_batch_size;
				}
			}
			printf("train times: %d\n", iter);
			m_dropout_map.clear();


			return ret;
		}

		void IMGP_Mlp_impl::front_propagation_batch(int idx)
		{
			std::vector<double> sample = m_input[idx];
			std::vector<double> target_out;
			double cur_err_sum = 0;
			for (std::vector<std::pair<int, std::vector<double>>>::iterator it = m_target_output.begin(); it != m_target_output.end(); ++it)
			{
				if (m_output[idx] == it->first)
				{
					target_out = it->second;
					break;
				}
			}
			if (target_out.empty()) return;
			size_t input_demension = m_input[0].size();
			if (m_regulation_enable_dropout)
			{
				for (size_t i = 0; i < m_neuron_value[0].size(); ++i)
				{
					if (!m_dropout_map[0][i]) continue;
					double s = 0;
					for (size_t j = 0; j < input_demension; ++j)
					{
						s += sample[j] * m_weights[0][i][j];
					}
					s += m_weights[0][i][input_demension];
					s /= (1.0 - m_regulation_dropout_ratio);
					m_neuron_value[0][i] = m_act_function(s, m_leaky_relu_alpha);
					m_neuron_derivate[0][i] = m_deact_function(s, m_leaky_relu_alpha);
				}
				for (size_t i = 1; i < m_weights.size() - 1; ++i)
				{
					for (size_t j = 0; j < m_neuron_value[i].size(); ++j)
					{
						if (!m_dropout_map[i][j]) continue;
						double s = 0;
						for (size_t k = 0; k < m_neuron_value[i - 1].size(); ++k)
						{
							if (!m_dropout_map[i - 1][k]) continue;
							s += m_neuron_value[i - 1][k] * m_weights[i][j][k];
						}
						s += m_weights[i][j][m_neuron_value[i - 1].size()];
						s /= (1.0 - m_regulation_dropout_ratio);
						m_neuron_value[i][j] = m_act_function(s, m_leaky_relu_alpha);
						m_neuron_derivate[i][j] = m_deact_function(s, m_leaky_relu_alpha);
					}
				}
			}
			else
			{
				for (size_t i = 0; i < m_neuron_value[0].size(); ++i)
				{
					double s = 0;
					for (size_t j = 0; j < input_demension; ++j)
					{
						s += sample[j] * m_weights[0][i][j];
					}
					s += m_weights[0][i][input_demension];
					m_neuron_value[0][i] = m_act_function(s, m_leaky_relu_alpha);
					m_neuron_derivate[0][i] = m_deact_function(s, m_leaky_relu_alpha);
				}
				for (size_t i = 1; i < m_weights.size() - 1; ++i)
				{
					for (size_t j = 0; j < m_neuron_value[i].size(); ++j)
					{
						double s = 0;
						for (size_t k = 0; k < m_neuron_value[i - 1].size(); ++k)
						{
							s += m_neuron_value[i - 1][k] * m_weights[i][j][k];
						}
						s += m_weights[i][j][m_neuron_value[i - 1].size()];
						m_neuron_value[i][j] = m_act_function(s, m_leaky_relu_alpha);
						m_neuron_derivate[i][j] = m_deact_function(s, m_leaky_relu_alpha);
					}
				}
			}

			double soft_max_denominator = 0;
			for (size_t j = 0; j < m_neuron_value[m_weights.size() - 1].size(); ++j)
			{
				double s = 0;
				for (size_t k = 0; k < m_neuron_value[m_weights.size() - 2].size(); ++k)
				{
					s += m_neuron_value[m_weights.size() - 2][k] * m_weights[m_weights.size() - 1][j][k];
				}
				s += m_weights[m_weights.size() - 1][j][m_neuron_value[m_weights.size() - 2].size()];
				CHECK_ALG_MLP_UPPER_LIMIT(s);
				m_neuron_value[m_weights.size() - 1][j] = std::exp(s);
				soft_max_denominator += m_neuron_value[m_weights.size() - 1][j];
			}
			soft_max_denominator += ALG_MLP_SOFTMAX_LOW_LIMIT;
			for (size_t j = 0; j < m_neuron_value[m_weights.size() - 1].size(); ++j)
			{
				m_neuron_value[m_weights.size() - 1][j] /= soft_max_denominator;
			}

			back_propagation_bacth(sample, target_out);
		}

		void IMGP_Mlp_impl::back_propagation_bacth(std::vector<double> &sample, std::vector<double> &target_out)
		{
			double es = 0.0;
			size_t count = m_neuron_graditude.size();
			size_t c1 = m_neuron_graditude[count - 1].size();
			if (m_err_type == MEAN_SQUARE_ERROR)
			{
				for (size_t i = 0; i < c1; ++i)
				{
					double g = 0;
					double os_cur = m_neuron_value[count - 1][i];
					for (size_t j = 0; j < c1; ++j)
					{
						double os = m_neuron_value[count - 1][j];
						if (i == j)
							g += ((os - target_out[i]) * os * (1.0 - os));
						else
							g += (-1.0 * (os - target_out[i]) * os * os_cur);
					}
					m_neuron_graditude[count - 1][i] = g;
				}
			}
			else if (m_err_type == CROSS_ENTROPY)
			{
				for (size_t i = 0; i < c1; ++i)
				{
					double g = 0;
					double os_cur = m_neuron_value[count - 1][i];
					CHECK_ALG_MLP_SOFTMAX_LIMIT(os_cur);
					for (size_t j = 0; j < c1; ++j)
					{
						double os = m_neuron_value[count - 1][j];
						CHECK_ALG_MLP_SOFTMAX_LIMIT(os);
						if (i == j)
							g += (-1.0 * (target_out[j] / os - (1.0 - target_out[j]) / (1.0 - os)) * os * (1.0 - os));
						else
							g += ((target_out[j] / os - (1.0 - target_out[j]) / (1.0 - os)) * os * os_cur);
					}
					m_neuron_graditude[count - 1][i] = g;
				}
			}

			if (m_regulation_enable_dropout)
			{
				for (int i = count - 2; i >= 0; i--)
				{
					c1 = m_neuron_graditude[i].size();
					size_t c2 = m_neuron_graditude[i + 1].size();
					for (size_t j = 0; j < c1; ++j)
					{
						if (!m_dropout_map[i][j]) continue;
						double v = 0.0;
						for (size_t k = 0; k < c2; ++k)
						{
							if (!m_dropout_map[i][k]) continue;
							v += m_neuron_graditude[i + 1][k] * m_weights[i + 1][k][j];
						}
						m_neuron_graditude[i][j] = v * m_neuron_derivate[i][j];
					}
				}
				for (int i = m_weights.size() - 1; i >= 0; i--)
				{
					for (size_t j = 0; j < m_neuron_value[i].size(); ++j)
					{
						if (!m_dropout_map[i][j]) continue;
						if (i > 0)
						{
							for (size_t k = 0; k < m_neuron_value[i - 1].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_neuron_graditude[i][j] * m_neuron_value[i - 1][k];
								m_weight_graditude_batch[i][j][k] += m_weight_graditude[i][j][k];
							}
							m_weight_graditude[i][j][m_neuron_value[i - 1].size()] = m_neuron_graditude[i][j];
							m_weight_graditude_batch[i][j][m_neuron_value[i - 1].size()] += m_weight_graditude[i][j][m_neuron_value[i - 1].size()];
						}
						else
						{
							for (size_t k = 0; k < sample.size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_neuron_graditude[i][j] * sample[k];
								m_weight_graditude_batch[i][j][k] += m_weight_graditude[i][j][k];
							}
							m_weight_graditude[i][j][sample.size()] = m_neuron_graditude[i][j];
							m_weight_graditude_batch[i][j][sample.size()] += m_weight_graditude[i][j][sample.size()];
						}
					}
				}
			}
			else
			{
				for (int i = count - 2; i >= 0; i--)
				{
					c1 = m_neuron_graditude[i].size();
					size_t c2 = m_neuron_graditude[i + 1].size();
					for (size_t j = 0; j < c1; ++j)
					{
						double v = 0.0;
						for (size_t k = 0; k < c2; ++k)
						{
							v += m_neuron_graditude[i + 1][k] * m_weights[i + 1][k][j];
						}
						m_neuron_graditude[i][j] = v * m_neuron_derivate[i][j];
					}
				}
				for (int i = m_weights.size() - 1; i >= 0; i--)
				{
					for (size_t j = 0; j < m_neuron_value[i].size(); ++j)
					{
						if (i > 0)
						{
							for (size_t k = 0; k < m_neuron_value[i - 1].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_neuron_graditude[i][j] * m_neuron_value[i - 1][k];
								m_weight_graditude_batch[i][j][k] += m_weight_graditude[i][j][k];
							}
							m_weight_graditude[i][j][m_neuron_value[i - 1].size()] = m_neuron_graditude[i][j];
							m_weight_graditude_batch[i][j][m_neuron_value[i - 1].size()] += m_weight_graditude[i][j][m_neuron_value[i - 1].size()];
						}
						else
						{
							for (size_t k = 0; k < sample.size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_neuron_graditude[i][j] * sample[k];
								m_weight_graditude_batch[i][j][k] += m_weight_graditude[i][j][k];
							}
							m_weight_graditude[i][j][sample.size()] = m_neuron_graditude[i][j];
							m_weight_graditude_batch[i][j][sample.size()] += m_weight_graditude[i][j][sample.size()];
						}
					}
				}
			}
		}

		void IMGP_Mlp_impl::back_propagation_bacth_solve()
		{
			for (int k = 0; k < m_weight_graditude_batch.size(); ++k)
			{
				for (int j = 0; j < m_weight_graditude_batch[k].size(); ++j)
				{
					for (int i = 0; i < m_weight_graditude_batch[k][j].size(); ++i)
					{
						m_weight_graditude[k][j][i] = m_weight_graditude_batch[k][j][i] / m_batch_size;
						m_weight_graditude_batch[k][j][i] = 0.0;
					}
				}
			}

			switch (m_solver_type)
			{
			case SGD_NORMAL:
				do_solve_sgd_normal();
				break;
			case SGD_MOMENT:
				do_solve_sgd_moment();
				break;
			case NAG:
				do_solve_nag();
				break;
			case RMSPROP:
				do_solve_rmsprop();
				break;
			case ADAM:
				do_solve_adam();
				break;
			default:
				do_solve_sgd_moment();
				break;
			}
		}

		void IMGP_Mlp_impl::do_solve_sgd_normal()
		{
			if (m_regulation_enable_dropout)
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
			else
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
		}

		void IMGP_Mlp_impl::do_solve_sgd_moment()
		{
			if (m_regulation_enable_dropout)
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
			else
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weight_graditude[i][j][k] += m_learn_rate * m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * m_learn_rate;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
				m_pre_weight_graditude = m_weight_graditude;
			}
		}

		void IMGP_Mlp_impl::do_solve_nag()
		{
			if (m_regulation_enable_dropout)
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
			else
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment;
								m_weights[i][j][k] -= m_weight_graditude[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}

			int idx = m_iter_batch - m_batch_size;
			for (int i = idx; i < m_iter_batch; ++i)
			{
				front_propagation_batch(m_loop_index[i]);
			}

			for (int k = 0; k < m_weight_graditude_batch.size(); ++k)
			{
				for (int j = 0; j < m_weight_graditude_batch[k].size(); ++j)
				{
					for (int i = 0; i < m_weight_graditude_batch[k][j].size(); ++i)
					{
						m_weight_graditude[k][j][i] = m_weight_graditude_batch[k][j][i] / m_batch_size;
						m_weight_graditude_batch[k][j][i] = 0.0;
					}
				}
			}
			do_solve_sgd_moment();
		}

		void IMGP_Mlp_impl::do_solve_rmsprop()
		{
			double epsilon = 1.0e-10;
			if (m_regulation_enable_dropout)
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
			else
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
				m_pre_weight_decay = m_weight_decay;
			}
		}

		void IMGP_Mlp_impl::do_solve_adam()
		{
			double epsilon = 1.0e-10;
			if (m_regulation_enable_dropout)
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							if (!m_dropout_map[i][j]) continue;
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
			else
			{
				if (m_regulation_type == NONE_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L1_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * (fabs(m_weights[i][j][k]) < 1.0e-4 ? 0 : (m_weights[i][j][k] > 0 ? 1.0 : -1.0));
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == L2_NORM)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_graditude[i][j][k] += m_regulation_norm_lamda * m_weights[i][j][k];
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
							}
						}
					}
				}
				else if (m_regulation_type == MAX_NORM_CONSTRAINT)
				{
					for (size_t i = 0; i < m_weights.size(); ++i)
					{
						for (size_t j = 0; j < m_weights[i].size(); ++j)
						{
							for (size_t k = 0; k < m_weights[i][j].size(); ++k)
							{
								m_weight_graditude[i][j][k] = m_pre_weight_graditude[i][j][k] * m_learn_moment + m_weight_graditude[i][j][k] * (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] = m_learn_decay * m_pre_weight_decay[i][j][k] + (1.0 - m_learn_decay) * m_weight_graditude[i][j][k] * m_weight_graditude[i][j][k];
								m_pre_weight_graditude[i][j][k] = m_weight_graditude[i][j][k];
								m_pre_weight_decay[i][j][k] = m_weight_decay[i][j][k];
								m_weight_graditude[i][j][k] /= (1.0 - m_learn_moment);
								m_weight_decay[i][j][k] /= (1.0 - m_learn_decay);
								m_weights[i][j][k] -= (m_learn_rate / sqrt(m_weight_decay[i][j][k] + epsilon) * m_weight_graditude[i][j][k]);
								if (fabs(m_weights[i][j][k]) > m_regulation_max_norm)
								{
									m_weights[i][j][k] = m_weights[i][j][k] > 0 ? m_regulation_max_norm : -m_regulation_max_norm;
								}
							}
						}
					}
				}
			}
		}

		std::vector<double> IMGP_Mlp_impl::get_error_value(std::vector<double> cur_out, std::vector<double> target_out)
		{
			std::vector<double> ret;
			if (cur_out.size() != target_out.size()) return ret;
			ret.resize(cur_out.size());
			if (m_err_type == MEAN_SQUARE_ERROR)
			{
				for (size_t i = 0; i < cur_out.size(); ++i)
				{
					ret[i] = 0.5 * (cur_out[i] - target_out[i]) * (cur_out[i] - target_out[i]);
				}
			}
			else if (m_err_type == CROSS_ENTROPY)
			{
				for (size_t i = 0; i < cur_out.size(); ++i)
				{
					ret[i] = -(std::log(cur_out[i]) * target_out[i] + std::log(1.0 - cur_out[i]) * (1.0 - target_out[i]));
				}
			}
			return ret;
		}

		void IMGP_Mlp_impl::set_weight_constant_value(double constant_value)
		{
			m_weight_constant_value = constant_value;
		}

		void IMGP_Mlp_impl::set_weight_uniform_min_max_value(double min_value, double max_value)
		{
			if (min_value >= max_value)
			{
				min_value = 0.0;
				max_value = 1.0;
			}
			m_weight_uniform_max_value = max_value;
			m_weight_uniform_min_value = min_value;
		}

		void IMGP_Mlp_impl::set_weight_gaussian_mean_stddev_sparse(double mean, double stddev, std::vector<unsigned int> sparse)
		{
			m_weight_gaussian_mean_value = mean;
			m_weight_gaussian_stddev_value = stddev;
			m_weight_gaussian_sparse = sparse;
		}

		void IMGP_Mlp_impl::set_weight_positive_unitball_min_max_value(double min_value, double max_value)
		{
			if (min_value >= max_value)
			{
				min_value = 0.0;
				max_value = 1.0;
			}
			m_weight_positive_unitball_min_value = min_value;
			m_weight_positive_unitball_max_value = max_value;
		}

		void IMGP_Mlp_impl::set_weight_xavierfiller_variance_norm(VARIANCENORM norm)
		{
			m_weight_xavierfiller_variance_norm = norm;
		}

		void IMGP_Mlp_impl::set_weight_msrafiller_variance_norm(VARIANCENORM norm)
		{
			m_weight_msrafiller_variance_norm = norm;
		}

		void IMGP_Mlp_impl::weight_initialization_constant()
		{
			int layer_size = m_weights.size();
			int n0, n1;
			for (int k = 0; k < layer_size; ++k)
			{
				n0 = m_weights[k].size();
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = m_weight_constant_value;
					}
					m_weights[k][j][n1 - 1] = 0;
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_uniform()
		{
			int layer_size = m_weights.size();
			int n0, n1;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution(m_weight_uniform_min_value, m_weight_uniform_max_value);
			for (int k = 0; k < layer_size; ++k)
			{
				n0 = m_weights[k].size();
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = distribution(generator);
					}
					m_weights[k][j][n1 - 1] = 0;
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_gaussian()
		{
			int layer_size = m_weights.size();
			int n0 = m_input[0].size(), n1 = 0;
			int sparse_size = m_weight_gaussian_sparse.size();
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::normal_distribution<double> distribution(m_weight_gaussian_mean_value, m_weight_gaussian_stddev_value);
			for (int k = 0; k < layer_size; ++k)
			{
				n0 = m_weights[k].size();
				int sp_size = k >= sparse_size ? 0 : m_weight_gaussian_sparse[k];
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = distribution(generator);
					}
					m_weights[k][j][n1 - 1] = 0;
					unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
					std::default_random_engine generator1(seed1);
					std::uniform_int_distribution<int> distribution1(0, n1 - 1);
					std::vector<int> sparse_index;
					while (sparse_index.size() < sp_size)
					{
						int idx = distribution1(generator1);
						std::vector<int>::iterator it = std::find(sparse_index.begin(), sparse_index.end(), idx);
						if (it == sparse_index.end())
						{
							sparse_index.push_back(idx);
						}
					}
					for (int i = 0; i < sp_size; ++i)
					{
						m_weights[k][j][sparse_index[i]] = 0.0;
					}
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_positiveunitball()
		{
			int layer_size = m_weights.size();
			int n0, n1;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution(m_weight_positive_unitball_min_value, m_weight_positive_unitball_max_value);
			for (int k = 0; k < layer_size; ++k)
			{
				n0 = m_weights[k].size();
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					double s = 0;
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = distribution(generator);
						s += fabs(m_weights[k][j][i]);
					}
					s = 1.0 / s;
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] *= s;
					}
					m_weights[k][j][n1 - 1] = 0;
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_xavierfiller()
		{
			int layer_size = m_weights.size();
			int n0, n1;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution;
			for (int k = 0; k < layer_size; ++k)
			{
				int n_in, n_out;

				if (0 == k)
				{
					n_in = m_input[0].size();
					n_out = m_weights[k].size();
				}
				else
				{
					n_in = m_weights[k - 1].size();
					n_out = m_weights[k].size();
				}

				if (m_weight_xavierfiller_variance_norm == FAN_IN)
				{
					distribution = std::uniform_real_distribution<double>(-std::sqrt(3.0 / n_in), std::sqrt(3.0 / n_in));
				}
				else if (m_weight_xavierfiller_variance_norm == FAN_OUT)
				{
					distribution = std::uniform_real_distribution<double>(-std::sqrt(3.0 / n_out), std::sqrt(3.0 / n_out));
				}
				else
				{
					distribution = std::uniform_real_distribution<double>(-std::sqrt(6.0 / (n_in + n_out)), std::sqrt(6.0 / (n_in + n_out)));
				}

				n0 = m_weights[k].size();
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = distribution(generator);
					}
					m_weights[k][j][n1 - 1] = 0;
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_marafiller()
		{
			int layer_size = m_weights.size();
			int n0, n1;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::normal_distribution<double> distribution;
			for (int k = 0; k < layer_size; ++k)
			{
				int n_in, n_out;

				if (0 == k)
				{
					n_in = m_input[0].size();
					n_out = m_weights[k].size();
				}
				else
				{
					n_in = m_weights[k - 1].size();
					n_out = m_weights[k].size();
				}

				if (m_weight_xavierfiller_variance_norm == FAN_IN)
				{
					distribution = std::normal_distribution<double>(0, std::sqrt(2.0 / n_in));
				}
				else if (m_weight_xavierfiller_variance_norm == FAN_OUT)
				{
					distribution = std::normal_distribution<double>(0, std::sqrt(2.0 / n_out));
				}
				else
				{
					distribution = std::normal_distribution<double>(0, std::sqrt(4.0 / (n_in + n_out)));
				}

				n0 = m_weights[k].size();
				for (int j = 0; j < n0; ++j)
				{
					n1 = m_weights[k][j].size();
					for (int i = 0; i < n1 - 1; ++i)
					{
						m_weights[k][j][i] = distribution(generator);
					}
					m_weights[k][j][n1 - 1] = 0;
				}
			}
		}

		void IMGP_Mlp_impl::weight_initialization_bilinearfiller()//Î´ÊµÏÖ
		{
			int layer_size = m_weights.size() - 1;
			int n0 = m_input[0].size(), n1 = 0;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution(m_weight_positive_unitball_min_value, m_weight_positive_unitball_max_value);
			double val = 0, G = 0;
			for (int k = 0; k < layer_size; ++k)
			{
				n1 = m_hidden_layer_sizes[k];
				G = n1 > 2 ? 0.7 * std::pow((double)n0, 1.0 / (n1 - 1.0)) : 1.0;
				for (int j = 0; j < n1; ++j)
				{
					double s = 0;
					for (int i = 0; i < n0; ++i)
					{
						val = distribution(generator);
						m_weights[k][j][i] = val;
						s += val;
					}
					s = s / n0;
					for (int i = 0; i < n0; ++i)
					{
						m_weights[k][j][i] -= s;
					}
					if (m_activation_fun_id != 0)
					{
						s = 0;
						for (int i = 0; i < n0; ++i)
						{
							s += fabs(m_weights[k][j][i]);
						}
						s = 1.0 / s;
						for (int i = 0; i < n0; ++i)
						{
							m_weights[k][j][i] *= s;
						}
					}
					m_weights[k][j][n0] = 0;
				}
				n0 = n1;
			}
			n1 = m_output_demension;
			G = n1 > 2 ? 0.7 * std::pow((double)n0, 1.0 / (n1 - 1.0)) : 1.0;
			for (int j = 0; j < n1; ++j)
			{
				double s = 0;
				for (int i = 0; i < n0; ++i)
				{
					val = distribution(generator);
					m_weights[m_weights.size() - 1][j][i] = val;
					s += val;
				}
				s = s / n0;
				for (int i = 0; i < n0; ++i)
				{
					m_weights[m_weights.size() - 1][j][i] -= s;
				}
				if (m_activation_fun_id != 0)
				{
					s = 0;
					for (int i = 0; i < n0; ++i)
					{
						s += fabs(m_weights[m_weights.size() - 1][j][i]);
					}
					s = 1.0 / s;
					for (int i = 0; i < n0; ++i)
					{
						m_weights[m_weights.size() - 1][j][i] *= s;
					}
				}
				m_weights[m_weights.size() - 1][j][n0] = 0;
			}
		}

		void IMGP_Mlp_impl::set_weight_mat_init_value()
		{
			switch (m_weight_initialization_type)
			{
			case CONSTANT:
				weight_initialization_constant();
				break;
			case UNIFORM:
				weight_initialization_uniform();
				break;
			case GAUSSIAN:
				weight_initialization_gaussian();
				break;
			case POSITIVE_UNITBALL:
				weight_initialization_positiveunitball();
				break;
			case XAVIERFILLER:
				weight_initialization_xavierfiller();
				break;
			case MSRAFILLER:
				weight_initialization_marafiller();
				break;
			case BILINEARFILLER:
				weight_initialization_bilinearfiller();
				break;
			default:
				weight_initialization_uniform();
				break;
			}
		}

		void IMGP_Mlp_impl::get_output_count()
		{
			m_output_demension = 1;
			m_target_output.clear();
			std::vector<int> s_vec = m_output;
			std::sort(s_vec.begin(), s_vec.end());
			for (size_t i = 1; i < s_vec.size(); ++i)
			{
				if (s_vec[i] != s_vec[i - 1])
					m_output_demension++;
			}
			s_vec.erase(std::unique(s_vec.begin(), s_vec.end()), s_vec.end());

			for (size_t i = 0; i < m_output_demension; ++i)
			{
				std::vector<double> v(m_output_demension, 0.0);
				v[i] = 1.0;
				m_target_output.push_back(std::make_pair(s_vec[i], v));
			}
		}

		bool IMGP_Mlp_impl::validate_sample_and_output()
		{
			bool ret = true;
			if (m_input.empty() || m_output.empty() || m_hidden_layer_sizes.empty())
			{
				ret = false;
			}
			else
			{
				m_sample_count = m_input.size();
				get_output_count();
				if (m_output_demension < 2)
				{
					ret = false;
				}
				else
				{
					m_sample_demension = m_input[0].size();
					if (m_sample_demension > 0)
					{
						for (int i = 1; i < m_input.size(); ++i)
						{
							if (m_input[i].size() != m_sample_demension)
							{
								ret = false;
								break;
							}
						}
					}
					else
					{
						ret = false;
					}
				}
			}
			return ret;
		}

		void IMGP_Mlp_impl::init_weight_mat()
		{
			MLP_Mat mat;
			m_weight_count = 0;
			for (int i = 0; i < m_hidden_layer_sizes[0]; ++i)
			{
				mat.push_back(std::vector<double>(m_sample_demension + 1, 0.0));
				m_weight_count += m_sample_demension + 1;
			}
			m_weights.push_back(mat);
			m_weight_graditude.push_back(mat);
			m_weight_graditude_batch.push_back(mat);
			m_weight_decay.push_back(mat);
			m_pre_weight_decay.push_back(mat);
			m_pre_weight_graditude.push_back(mat);
			for (size_t i = 1; i < m_hidden_layer_sizes.size(); ++i)
			{
				mat.clear();
				for (int j = 0; j < m_hidden_layer_sizes[i]; ++j)
				{
					mat.push_back(std::vector<double>(m_hidden_layer_sizes[i - 1] + 1, 0.0));
					m_weight_count += m_hidden_layer_sizes[i - 1] + 1;
				}
				m_weights.push_back(mat);
				m_weight_graditude.push_back(mat);
				m_weight_graditude_batch.push_back(mat);
				m_weight_decay.push_back(mat);
				m_pre_weight_decay.push_back(mat);
				m_pre_weight_graditude.push_back(mat);
			}
			mat.clear();
			for (int i = 0; i < m_output_demension; ++i)
			{
				mat.push_back(std::vector<double>(m_hidden_layer_sizes[m_hidden_layer_sizes.size() - 1] + 1, 0.0));
				m_weight_count += m_hidden_layer_sizes[m_hidden_layer_sizes.size() - 1] + 1;
			}
			m_weights.push_back(mat);
			m_weight_graditude.push_back(mat);
			m_weight_graditude_batch.push_back(mat);
			m_weight_decay.push_back(mat);
			m_pre_weight_decay.push_back(mat);
			m_pre_weight_graditude.push_back(mat);
			for (int i = 0; i < m_weights.size(); ++i)
			{
				std::vector<double> v(m_weights[i].size(), 0.0);
				m_neuron_value.push_back(v);
				m_neuron_graditude.push_back(v);
				m_neuron_derivate.push_back(v);
			}
		}

		int IMGP_Mlp_impl::getPredict(std::vector<double> &hist, int idx, double &err)
		{
			int cls = -1;
			std::vector<double> sample = hist;
			size_t input_demension = m_input[0].size();
			for (size_t i = 0; i < m_neuron_value[0].size(); ++i)
			{
				double s = 0;
				for (size_t j = 0; j < input_demension; ++j)
				{
					s += sample[j] * m_weights[0][i][j];
				}
				s += m_weights[0][i][input_demension];
				m_neuron_value[0][i] = m_act_function(s, m_leaky_relu_alpha);
			}
			for (size_t i = 1; i < m_weights.size() - 1; ++i)
			{
				for (size_t j = 0; j < m_neuron_value[i].size(); ++j)
				{
					double s = 0;
					for (size_t k = 0; k < m_neuron_value[i - 1].size(); ++k)
					{
						s += m_neuron_value[i - 1][k] * m_weights[i][j][k];
					}
					s += m_weights[i][j][m_neuron_value[i - 1].size()];
					m_neuron_value[i][j] = m_act_function(s, m_leaky_relu_alpha);
				}
			}
			double soft_max_denominator = 0;
			for (size_t j = 0; j < m_neuron_value[m_weights.size() - 1].size(); ++j)
			{
				double s = 0;
				for (size_t k = 0; k < m_neuron_value[m_weights.size() - 2].size(); ++k)
				{
					s += m_neuron_value[m_weights.size() - 2][k] * m_weights[m_weights.size() - 1][j][k];
				}
				s += m_weights[m_weights.size() - 1][j][m_neuron_value[m_weights.size() - 2].size()];
				CHECK_ALG_MLP_UPPER_LIMIT(s);
				m_neuron_value[m_weights.size() - 1][j] = std::exp(s);
				soft_max_denominator += m_neuron_value[m_weights.size() - 1][j];
			}
			soft_max_denominator += ALG_MLP_SOFTMAX_LOW_LIMIT;
			for (size_t j = 0; j < m_neuron_value[m_weights.size() - 1].size(); ++j)
			{
				m_neuron_value[m_weights.size() - 1][j] /= soft_max_denominator;
			}
			int idx_max = 0;
			double v_max = m_neuron_value[m_neuron_value.size() - 1][0];
			printf("%s%.5f\t", v_max >= 0.0 ? "+" : "", v_max);
			for (size_t i = 1; i < m_neuron_value[m_neuron_value.size() - 1].size(); ++i)
			{
				printf("%s%.5f\t", m_neuron_value[m_neuron_value.size() - 1][i] >= 0.0 ? "+" : "", m_neuron_value[m_neuron_value.size() - 1][i]);
				if (m_neuron_value[m_neuron_value.size() - 1][i] > v_max)
				{
					v_max = m_neuron_value[m_neuron_value.size() - 1][i];
					idx_max = i;
				}
			}
			for (size_t i = 0; i < m_target_output.size(); ++i)
			{
				if (fabs(m_target_output[i].second[idx_max] - 1.0) < 0.001)
				{
					cls = m_target_output[i].first;
					break;
				}
			}
			std::vector<double> target_os;
			for (size_t i = 0; i < m_target_output.size(); ++i)
			{
				if (m_target_output[i].first == idx)
				{
					target_os = m_target_output[i].second;
					break;
				}
			}
			err = 0.0;
			for (size_t i = 0; i < m_neuron_value[m_neuron_value.size() - 1].size(); ++i)
			{
				err += (m_neuron_value[m_neuron_value.size() - 1][i] - target_os[i]) * (m_neuron_value[m_neuron_value.size() - 1][i] - target_os[i]);
			}
			return cls;
		}
	}
}
