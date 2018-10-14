#ifndef __DBSCAN_H__
#define __DBSCAN_H__
#include <vector>
#include <algorithm>

#define NOISE_ID -1

template<class Type>
class DBSCAN
{
private:
	double (*disFunc)(const Type&, const Type&);
public:
	typedef decltype(disFunc) DisFunc;

	/**
	 * \brief Density-Based Spatial Clustering of Applications with Noise
	 * \param e Minimum neighbor region percentage, (0,1)
	 * \param minElems Minimum number of points required to form a dense region
	 * \param func Function to calculate distance
	 */
	DBSCAN(double e, size_t minElems, DisFunc func = nullptr);
	DBSCAN();
	~DBSCAN();

	void setParameters(double e, size_t minElems, DisFunc func = nullptr);
	/**
	 * \brief Set the function to calculate distance
	 * \param func Pointer to function
	 */
	void setDistanceFunction(DisFunc func);
	/**
	 * \brief Set whether to solve the distance matrix in advance, 
	 * it is recommended to turn on to small data and close to big data.
	 * \param enable Default on
	 */
	void setPreCalDisMatrix(bool enable = true);

	/**
	 * \brief Do clustering
	 * \param data Cluster data in vector
	 * \return Labels
	 */
	std::vector<int> fit(const std::vector<Type>& data);

	/**
	 * \brief Release memory
	 */
	void reset();
private:
	double epsPercent;								//Minimum neighbor region percentage
	double epsDis;									//Minimum neighbor region distance
	size_t minElements;								//Minimum number of neighbor points
	bool preCalDisMatrix;							//whether to solve the distance matrix in advance

	std::vector<int> labels;
	std::vector<std::vector<double>> disMatrix;
	std::vector<Type> const* dataPtr;

	void prepareLabels(size_t n);
	void prepareDisMatrix();
	std::vector<size_t> findNeighbors(size_t index);

	void checkDataPtr();
	void checkFuncPtr();
};

template <class Type>
DBSCAN<Type>::DBSCAN(double e, size_t minElems, DisFunc func) :
	epsDis(0.0),
	preCalDisMatrix(true),
	dataPtr(nullptr)
{
	this->setParameters(e, minElems, func);
}

template <class Type>
DBSCAN<Type>::DBSCAN() :
	disFunc(nullptr),
	epsPercent(0.1),
	epsDis(0.0),
	minElements(1),
	preCalDisMatrix(true),
	dataPtr(nullptr)
{
}

template <class Type>
DBSCAN<Type>::~DBSCAN()
= default;

template <class Type>
void DBSCAN<Type>::setParameters(double e, size_t minElems, DisFunc func)
{
	this->epsPercent = std::max(e, 0.0);
	this->minElements = std::max(minElems, 0ULL);
	if (func != nullptr)
	{
		disFunc = func;
	}
}

template <class Type>
void DBSCAN<Type>::setDistanceFunction(DisFunc func)
{
	if (func != nullptr)
	{
		disFunc = func;
	}
}

template <class Type>
void DBSCAN<Type>::setPreCalDisMatrix(bool enable)
{
	this->preCalDisMatrix = enable;
}

template <class Type>
std::vector<int> DBSCAN<Type>::fit(const std::vector<Type>& data)
{
	std::vector<bool> visited(data.size());

	// for private functions use
	this->dataPtr = &data;
	this->prepareLabels(data.size());
	this->prepareDisMatrix();

	int clusterID = 1;
	for (auto pId = 0u; pId < data.size(); ++pId)
	{
		if (!visited[pId])
		{
			visited[pId] = true;
			auto neighbors = findNeighbors(pId);
			if (neighbors.size() >= minElements)
			{
				labels[pId] = clusterID;
				for (auto i = 0u;i < neighbors.size();++i)
				{
					const auto nId = neighbors[i];
					if (!visited[nId])
					{
						visited[nId] = true;
						auto newNeighbors = findNeighbors(nId);
						if (newNeighbors.size() >= minElements)
						{
							neighbors.insert(neighbors.end(), newNeighbors.begin(), newNeighbors.end());
						}
						if (labels[nId] == NOISE_ID)
						{
							labels[nId] = clusterID;
						}
					}
				}
				++clusterID;
			}
		}
	}

	return this->labels;
}

template <class Type>
void DBSCAN<Type>::reset()
{
	std::vector<int>().swap(this->labels);
	std::vector<std::vector<double>>().swap(this->disMatrix);
}

template <class Type>
void DBSCAN<Type>::prepareLabels(size_t n)
{
	this->labels.resize(n, NOISE_ID);
}

template <class Type>
void DBSCAN<Type>::prepareDisMatrix()
{
	checkDataPtr();
	checkFuncPtr();
	const auto& data = *(this->dataPtr);
	double dMax = 0.0, dMin = std::numeric_limits<double>::max();
	if (this->preCalDisMatrix)
	{
		// Calculate and save distance
		disMatrix.resize(data.size());
		for(auto i = 0u; i < data.size(); ++i)
		{
			disMatrix[i].resize(data.size(), 0.0);
		}
		for (auto i = 0u; i < data.size(); ++i)
		{
			for (auto j = i + 1; j < data.size(); ++j)
			{
				disMatrix[j][i] = disMatrix[i][j] = disFunc(data[i], data[j]);
				dMin = std::min(dMin, disMatrix[j][i]);
			}
			const auto mm = std::max_element(disMatrix[i].begin(), disMatrix[i].end());
			dMax = std::max(dMax, *mm);
		}
	}
	else
	{
		// Only calculate distance for epsDis
		for (auto i = 0u; i < data.size(); ++i)
		{
			for (auto j = i + 1; j < data.size(); ++j)
			{
				const auto dis = disFunc(data[i], data[j]);
				dMax = std::max(dMax, dis);
				dMin = std::min(dMin, dis);
			}
		}
	}
	epsDis = (dMax - dMin) * this->epsPercent + dMin;
}

template <class Type>
std::vector<size_t> DBSCAN<Type>::findNeighbors(size_t index)
{
	std::vector<size_t> neighbors;
	checkDataPtr();
	if (index >= dataPtr->size())
	{
		return neighbors;
	}
	if (this->preCalDisMatrix)
	{
		// Use pre-calculated distance matrix
		for (auto i = 0u; i < dataPtr->size(); ++i)
		{
			/*
			 * Do not need to avoid index == i,
			 * because index is 'visited'
			 */
			if (disMatrix[index][i] < epsDis)
			{
				neighbors.push_back(i);
			}
		}
	}
	else
	{
		checkFuncPtr();
		//calculate distance directly
		const auto& data = *(this->dataPtr);
		for (auto i = 0u; i < data.size(); ++i)
		{
			/*
			 * Do not need to avoid index == i,
			 * because index is 'visited'
			 */
			if(disFunc(data[i], data[index]) < epsDis)
			{
				neighbors.push_back(i);
			}
		}
	}
	return neighbors;
}

template <class Type>
void DBSCAN<Type>::checkDataPtr()
{
	if (this->dataPtr == nullptr)
	{
		throw std::runtime_error("Data not set!");
	}
}

template <class Type>
void DBSCAN<Type>::checkFuncPtr()
{
	if (this->disFunc == nullptr)
	{
		throw std::runtime_error("Distance function not set!");
	}
}

#endif
