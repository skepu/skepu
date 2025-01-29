#pragma once

//#ifdef SKEPU_HYBRID

#include <vector> 
#include <utility> 
#include <cmath> 


namespace skepu
{
	/*!
	 * \class ExecutionTimeModel
	 * 
	 * \brief A class that describes an execution time model based on measured data points
	 * 
	 * This class is used for the tuning of the \em hybrid backend. The class is used to model
	 * the execution time of the OpenMP and CUDA/OpenCL backends to find the optimal partition
	 * ratio between them.
	 * 
	 * Add data points to the model by using the addDataPoint() method. Then calculate the 
	 * model from the data using fitModel(). New data point can be added after the model is fitted, 
	 * but the fitModel() method must be executed again, to refit the model.
	 */
	class ExecutionTimeModel
	{
		std::vector<std::pair<size_t, double> > m_entries;
		double m_sumX = 0.0;
		double m_sumY = 0.0;
		
		// The model y = a*x + b
		double m_a = 0.0;
		double m_b = 0.0;
		
		bool m_fitted;
		
	public:
		ExecutionTimeModel() : m_fitted{false} 
		{
		};
		
		void addDataPoint(size_t problemSize, double executionTime) {
			m_fitted = false;
			
			m_entries.push_back(std::make_pair(problemSize, executionTime));
			m_sumX += (double) problemSize;
			m_sumY += executionTime;
		}
		
		double getPredictedTime(size_t problemSize) const {
			if(not m_fitted)
				SKEPU_ERROR("getPredictedTime(): ExecutionTimeModel is not fitted!");
			
			return m_a*problemSize + m_b;
		}
		
		double getR2Error() const {
			if(not m_fitted)
				SKEPU_ERROR("getR2Error(): ExecutionTimeModel is not fitted!");
			
			double yMean = m_sumY / (double) m_entries.size();
			double res = 0.0;
			double tot = 0.0;
			
			for(size_t i = 0; i < m_entries.size(); ++i) {
				double error = (m_entries[i].second - getPredictedTime(m_entries[i].first));
				res += error*error;
				tot += (m_entries[i].second - yMean)*(m_entries[i].second - yMean);
			}
			
			return 1 - (res / tot);
		}
		
		
		double getA() const {
			return m_a;
		}
		
		double getB() const {
			return m_b;
		}
		
		
		/*!
		 * Simple linear regression implemented based on: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
		 */
		void fitModel() {
			int numPoints = m_entries.size();
			if(numPoints < 2)
				SKEPU_ERROR("Must have at least two points for a polynomial fit!");
			
			double nominator = 0.0;
			double denominator = 0.0;
			double xMean = m_sumX / (double)numPoints;
			double yMean = m_sumY / (double)numPoints;
			
			for(size_t i = 0; i < numPoints; ++i) {
				double xi = (double) m_entries[i].first;
				double yi = m_entries[i].second;
				nominator += (xi - xMean)*(yi - yMean);
				denominator += (xi - xMean)*(xi - xMean);
			}
			
			if(std::abs(denominator) < 1e-9)
				SKEPU_ERROR("Error in linear fitting, seems to be a vertical line");
			
			m_a = nominator / denominator;
			m_b = yMean - m_a*xMean;
			if(m_a < 0.0)
				m_a = 0.0; // Don't allow execution time to reduce with the input size
			
			m_fitted = true;
		}
		
		
		bool isFitted() const {
			return m_fitted;
		}
		
		
		/*!
		 * Predict the optimal CPU partition ratio for a given problem size based on two ExecutionTimeModels. Both models must be fitted
		 * before this funciton is called.
		 */
		static float predictCPUPartitionRatio(const ExecutionTimeModel& cpuModel, const ExecutionTimeModel& gpuModel, size_t problemSize) {
			if(not cpuModel.isFitted())
				SKEPU_ERROR("CPU model is not fitted!");
			if(not gpuModel.isFitted())
				SKEPU_ERROR("GPU model is not fitted!");
			
			// Based on cpu.a*x*pS + cpu.b == gpu.a*(1-x)*pS + gpu.b, where x is CPU partition ratio and pS is problemSize.
			double nominator = gpuModel.m_a*problemSize - cpuModel.m_b + gpuModel.m_b;
			double denominator = (cpuModel.m_a + gpuModel.m_a)*problemSize;

			if(std::abs(denominator) < 1e-9) {
				// Lines does not cross, choose to map whole problemSize to CPU or GPU
				return cpuModel.getPredictedTime(problemSize) < gpuModel.getPredictedTime(problemSize) ? 1.0 : 0.0;
			}

			double ratio = nominator/denominator;
			if(ratio < 0.0)
				ratio = 0.0;
			else if(ratio > 1.0)
				ratio = 1.0;
			
			ratio = roundf(ratio*100.0)/100.0;
			return ratio;
		}
		
	};
}

//#endif // SKEPU_HYBRID
