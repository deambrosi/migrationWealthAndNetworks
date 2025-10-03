function Ue = computeUtilityGivenProductivity(A_vec, matrices, A_indexer)
% COMPUTEUTILITYGIVENPRODUCTIVITY  Per-period utility given productivity.
%
%   Ue = COMPUTEUTILITYGIVENPRODUCTIVITY(A_VEC, MATRICES) returns the matrix of
%   per-period utilities evaluated at the productivity vector A_VEC (length N).
%   The function relies on precomputed components stored in MATRICES during
%   constructMatrix.m. In particular, MATRICES must contain the fields:
%       .cons_base              Baseline consumption component (without wages)
%       .employed_income_base   Contribution of productivity to income
%       .amenity_weight         Amenity scales used in utility
%   A_indexer : array of size matching MATRICES.cons_base that maps into the
%               location dimension. Typically INDEXES.I_Np from setGridsAndIndices.
%
%   The returned matrix has the same shape as MATRICES.cons_base. Infeasible
%   consumption (≤0) is penalized with −realmax to align with the convention in
%   constructMatrix.m.
%
%   AUTHOR: OpenAI ChatGPT
%   DATE  : October 2023
% ======================================================================
    if nargin < 3 || isempty(A_indexer)
        error('computeUtilityGivenProductivity:MissingIndexer', ...
            'A_indexer (e.g., indexes.I_Np) must be supplied.');
    end

    A_full = A_vec(A_indexer);

    cons = matrices.cons_base + matrices.employed_income_base .* A_full;

    Ue = zeros(size(cons), 'like', cons);

    feasible = cons > 0;
    Ue(~feasible) = -realmax;
    if any(feasible, 'all')
        Ue(feasible) = matrices.amenity_weight(feasible) .* log(cons(feasible));
    end

end

