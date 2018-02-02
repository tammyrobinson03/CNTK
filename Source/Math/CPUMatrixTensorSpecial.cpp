#include "stdafx.h"

#ifdef USE_MKL

#include "CPUMatrixTensorImpl.h"
#include "mkl_cblas.h"
#include "mkl_vml.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& /*regularOpDims*/, const array<SmallVector<ptrdiff_t>, 2>& /*regularStrides*/,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& /*reducingStrides*/)
{
    if (alpha == 1.0f && beta == 0.0f &&
        offsets[0] == 0 && offsets[1] == 0 &&
        a.GetNumElements() == o.GetNumElements() &&
        reducingOpDims.size() == 0)
    {
        int N = (int)a.GetNumElements();
        switch(op)
        {
        case ElementWiseOperator::opLinearRectifier:
            vsAbs(N, a.Data(), o.Data());
            cblas_saxpby(N, 0.5f, a.Data(), 1, 0.5f, o.Data(), 1); // o = (a + abs(a))/2
            return true;
        }
    }
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, const CPUMatrix<float>& b, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& /*regularOpDims*/, const array<SmallVector<ptrdiff_t>, 3>& /*regularStrides*/,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& /*reducingStrides*/)
{
    if (alpha == 1.0f && beta == 0.0f &&
        offsets[0] == 0 && offsets[1] == 0 && offsets[2] == 0 &&
        reducingOpDims.size() == 0)
    {
        if (a.GetNumRows() == b.GetNumRows() &&
            a.GetNumRows() == o.GetNumRows() &&
            a.GetNumRows() > 1 &&
            ((a.GetNumCols() == 1 && o.GetNumCols() == b.GetNumCols()) ||
            (b.GetNumCols() == 1 && o.GetNumCols() == a.GetNumCols())))
        {
            // plus/multiply parameter (no dynamic axes, or GetNumCols() == 1)
            float* dataWithDynamicAxes = (a.GetNumCols() == 1 ? b.Data() : a.Data());
            float* dataParameter = (a.GetNumCols() == 1 ? a.Data() : b.Data());
            int N = (int)a.GetNumRows();
            switch (op)
            {
            case ElementWiseOperator::opSum:
                for (int col = 0; col < o.GetNumCols(); ++col)
                {
                    vsAdd(N, dataWithDynamicAxes + col * N, dataParameter, o.Data() + col * N);
                }
                return true;
            case ElementWiseOperator::opElementwiseProduct:
                for (int col = 0; col < o.GetNumCols(); ++col)
                {
                    vsMul(N, dataWithDynamicAxes + col * N, dataParameter, o.Data() + col * N);
                }
                return true;
            }
        }
        else if (a.GetNumElements() == b.GetNumElements() && a.GetNumElements() == o.GetNumElements())
        {
            // elementwise operation with no broadcast/reduction
            int N = (int)a.GetNumElements();
            switch (op)
            {
            case ElementWiseOperator::opSum:
                vsAdd(N, a.Data(), b.Data(), o.Data());
                return true;
            case ElementWiseOperator::opElementwiseProduct:
                vsMul(N, a.Data(), b.Data(), o.Data());
                return true;
            case ElementWiseOperator::opDifference:
                vsSub(N, a.Data(), b.Data(), o.Data());
                return true;
            }
        }
        else if ((a.GetNumElements() == 1 && o.GetNumElements() == b.GetNumElements()) ||
                 (b.GetNumElements() == 1 && o.GetNumElements() == a.GetNumElements()))
        {
            int N = (int)o.GetNumElements();
            float scalar = (a.GetNumElements() == 1 ? a.Data()[0] : b.Data()[0]);
            float* input = (a.GetNumElements() == 1 ? b.Data() : a.Data());
            switch (op)
            {
            case ElementWiseOperator::opElementwiseProduct:
                cblas_saxpby(N, scalar, input, 1, 0.0f, o.Data(), 1);
                return true;
            case ElementWiseOperator::opSum:
                memcpy(o.Data(), input, N * sizeof(float));
                cblas_saxpby(N, 1.0f, &scalar, 0, 1.0f, o.Data(), 1);
                return true;
            }
        }
    }
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<float>(float /*beta*/, const CPUMatrix<float>& /*a*/, const CPUMatrix<float>& /*b*/, const CPUMatrix<float>& /*c*/, CPUMatrix<float>& /*o*/, float /*alpha*/, ElementWiseOperator /*op*/, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 4>& /*offsets*/,
    const SmallVector<size_t>& /*regularOpDims*/, const array<SmallVector<ptrdiff_t>, 4>& /*regularStrides*/,
    const SmallVector<size_t>& /*reducingOpDims*/, const array<SmallVector<ptrdiff_t>, 4>& /*reducingStrides*/)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<double>(double, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<double>(double, const CPUMatrix<double>&, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<double>(double, const CPUMatrix<double>&, const CPUMatrix<double>&, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<half>(half, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<half>(half, const CPUMatrix<half>&, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<half>(half, const CPUMatrix<half>&, const CPUMatrix<half>&, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&)
{
    return false;
}

}}}

#endif