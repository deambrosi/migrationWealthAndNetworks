function funcs = setFunctionalForms(params)
% SETFUNCTIONALFORMS Defines key functional forms for the model.
%
%   funcs = setFunctionalForms(params) initializes and returns a structure
%   containing function handles for the model's core functional forms.
%
%   INPUT:
%       params - Structure containing model parameters, including:
%                * ssigma - Elasticity of substitution for utility.
%                * A      - Vector of location productivities.
%                * alpha  - Parameter for the wage function.
%
%   OUTPUT:
%       funcs - Structure with the following fields:
%               * Uu - Utility function handle:
%                      Uu(x) = (x^(1-ssigma))/(1-ssigma)
%
%               * w  - Wage function handle:
%                      w(x) = A .* ((0.5 + x).^(-alpha))
%
%   EXAMPLE:
%       params = SetParameters(dims);
%       funcs = setFunctionalForms(params);
%       utilityValue = funcs.Uu(10);
%       wageValue    = funcs.w(2);
%
%   AUTHOR: Agustin Deambrosi
%   DATE: February 2025
%   VERSION: 1.1
% =========================================================================

    %% Define Functional Forms
    
    % Period utility function:
    funcs.Uu = @(x) (x.^(1 - params.ssigma)) ./ (1 - params.ssigma);
    
end
