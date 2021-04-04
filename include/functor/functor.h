#ifndef HPE_FUNCTOR_H
#define HPE_FUNCTOR_H

#include "reg_term.h"
#include "ext_params_reproj_err.h"
#include "shape_coef_reproj_err.h"
#include "expr_coef_reproj_err.h"
#include "point_reproj_err.h"
#include "db_params.h"


namespace mhpe
{
    namespace Utils
    {
        void InitCeresProblem(ceres::Solver::Options& opts)
        {
            opts.max_num_iterations = N_CERES_ITERATIONS; 
            opts.num_threads = N_CERES_THREADS; 
            opts.minimizer_progress_to_stdout = B_CERES_STDCOUT;             
        }
    }
}


#endif // HPE_FUNCTOR_H