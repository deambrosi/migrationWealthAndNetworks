function dynamics = TransitionDynamics(m0, T, ss, dims, params, grids, indexes, funcs, settings)
% TRANSITIONDYNAMICS Solves the transition dynamics of the model.
%
%   This function computes the transition path from an initial distribution
%   M0 to the steady-state distribution M_ss by iteratively updating a
%   parameterized guess for the migration path.
%
%   PROCEDURE:
%   1. Load steady-state results.
%   2. Initialize agent states for the transition.
%   3. Define the simulation time horizon and weights.
%   4. Initialize the guessed migration path.
%   5. Outer loop: Update transition path until convergence.
%   6. Finalize outputs and save results.
%
%   INPUTS:
%       m0      - Initial migration distribution.
%       T       - Time horizon for the transition.
%       ss      - Steady-state equilibrium structure.
%       dims    - Structure containing model dimensions.
%       params  - Model parameters.
%       grids   - Model grids.
%       indexes - Indexing structures.
%       funcs   - Functional forms of the model.
%       settings- Iteration settings.
%
%   OUTPUT:
%       dynamics - Structure containing transition path results:
%            * M - Migration path (N x T matrix).
%            * ahat - Asset policy function.
%            * mhat - Migration policy function.
%
%   AUTHOR: Agustin Deambrosi
%   DATE: February 2025
%   VERSION: 1.1
% =========================================================================

    %% Initialize Time Grid and Weights
    t = (1:T)';
    M0 = accumarray([m0.location]', 1, [dims.N, 1]);
    M_hat = GetMhat(M0, ss.M_ss, t);
    
    % Define weights: earlier periods have higher weight.
    we = (T:-1:1)';
    we = we / sum(we);
    
    %% Outer Iteration: Update Transition Path Until Convergence
    MaxIterOut = 100;
    tolTheta = 1e-4;
    
    for iter = 1:MaxIterOut
        fprintf('Outer iteration %d\n', iter);
        
        % Compute policy functions using the current guessed migration path.
        din = PolicyDynamics(T, M_hat, ss, dims, params, grids, indexes, funcs);
        
        % Simulate agents using the computed policies.
        M_new = simulateAgents(m0, din, T, dims, params, grids, settings);
        
        % Compute error and update guessed migration path.
        E_current = sum((M_new - M_hat).^2 .* we', 'all');
        fprintf('Current error E = %e\n', E_current);
        
        if E_current > 1e-3
            M_hat = 0.6 * M_hat + 0.4 * M_new;
        else
            M_hat = 0.8 * M_hat + 0.2 * M_new;
        end
        
        if E_current < tolTheta
            fprintf('Convergence reached at iteration %d\n', iter);
            break;
        end
    end
    
    %% Finalize Outputs and Save Results
    dynamics.M = M_hat;
    dynamics.ahat = din.a;
    dynamics.mhat = din.m;
    
    % Save results
    save(fullfile('output_data', 'TransitionDynamics.mat'), 'dynamics');
end
