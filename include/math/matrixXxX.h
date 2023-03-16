#pragma once

#include <vector>
#include <cassert>
#include <Eigen/Dense>
#include <cuda-utils.h>

namespace rmagine
{
template<typename T>
struct MatrixXxX
{
    T* m_data;
    int m_numRows;
    int m_numCols;

    MatrixXxX() : m_numRows(0), m_numCols(0) {}

    ~MatrixXxX()
    {
        if (m_data != nullptr)
        {
            cudaFree(m_data);
            m_data = nullptr;
        }
    }
    
    MatrixXxX(int numRows, int numCols) : m_numRows(numRows), m_numCols(numCols)
    {
        cudaMallocManaged(&m_data, sizeof(T) * m_numRows * m_numCols);
    }
    
    RMAGINE_INLINE_FUNCTION
    int numRows() const { return m_numRows; }
    
    RMAGINE_INLINE_FUNCTION
    int numCols() const { return m_numCols; }
    
    RMAGINE_INLINE_FUNCTION
    T& operator()(int row, int col)
    {
        assert(row >= 0 && row < m_numRows);
        assert(col >= 0 && col < m_numCols);
        return m_data[row + col * m_numRows];
    }
    
    RMAGINE_INLINE_FUNCTION
    const T& operator()(int row, int col) const
    {
        assert(row >= 0 && row < m_numRows);
        assert(col >= 0 && col < m_numCols);
        return m_data[row + col * m_numRows];
    }

    /**
     * Copies(!) to a Eigen dynamic matrix.
     * TODO change to reinterpret castable structure
    */
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigen() const
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigenMat(m_numRows, m_numCols);
        for (int col = 0; col < m_numCols; ++col)
        {
            for (int row = 0; row < m_numRows; ++row)
            {
                eigenMat(row, col) = operator()(row, col);
            }
        }
        return eigenMat;
    }

    void toEigenInpl(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &eigenMat) const
    {
        for (int col = 0; col < m_numCols; ++col)
        {
            for (int row = 0; row < m_numRows; ++row)
            {
                eigenMat(row, col) = operator()(row, col);
            }
        }
    }

    void toEigenInpl(Eigen::Matrix<T, Eigen::Dynamic, 1> &eigenMat) const
    {
        assert(m_numCols == 1);
        for (int col = 0; col < m_numCols; ++col)
        {
            for (int row = 0; row < m_numRows; ++row)
            {
                eigenMat(row, col) = operator()(row, col);
            }
        }
    }
};

using MatrixXd = MatrixXxX<double>;
using MatrixXf = MatrixXxX<float>;

}
