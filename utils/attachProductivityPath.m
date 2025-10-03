function params = attachProductivityPath(params, settings, dims)
% ATTACHPRODUCTIVITYPATH  Embed an exogenous productivity path into PARAMS.
%
%   PARAMS = ATTACHPRODUCTIVITYPATH(PARAMS, SETTINGS, DIMS) augments the
%   parameter struct with fields describing the time-varying productivity
%   profile for each location. In particular, location 1 (Venezuela) is
%   assumed to experience an exogenous decline in productivity from 1.0 to
%   0.1 over a configurable horizon (default 10 periods, via
%   SETTINGS.T_decline_A_VEN). After the decline window, productivity remains
%   at 0.1 through the rest of SETTINGS.T. The resulting
%   path is stored in PARAMS.A_path (N×T). Convenience fields with the
%   initial and terminal productivity vectors are also provided and the
%   baseline PARAMS.A is updated to the terminal values so that stationary
%   objects (e.g., the terminal value function) use the long-run productivities.
%
%   INPUTS
%   ------
%   params   : struct
%       Parameter struct returned by SetParameters.
%   settings : struct
%       Simulation and iteration settings. Only SETTINGS.T is required here.
%   dims     : struct (optional)
%       Provides the number of locations (dims.N). If omitted, the size of
%       PARAMS.A is used to infer N.
%
%   OUTPUT
%   ------
%   params : struct
%       Same as input PARAMS with additional fields
%         .A_path     [N×T] time path of productivity by location
%         .A_initial  [N×1] productivity vector in period t = 1
%         .A_terminal [N×1] productivity vector in period t = T
%         .A          [N×1] overwritten with .A_terminal for convenience
%
%   NOTES
%   -----
%   • The decline in productivity for Venezuela is implemented via a linear
%     interpolation between the endpoints 1.0 and 0.1. Other locations retain
%     their baseline productivity levels throughout the horizon.
%   • The path is recomputed each time the function is called so changes to the
%     decline horizon in SETTINGS are immediately reflected.
%
%   AUTHOR: OpenAI ChatGPT
%   DATE  : October 2023
% ======================================================================

    if nargin < 3 || isempty(dims)
        N = numel(params.A);
    else
        N = dims.N;
    end

    T = settings.T;

    if isfield(settings, 'T_decline_A_VEN') && ~isempty(settings.T_decline_A_VEN)
        decline_horizon = settings.T_decline_A_VEN;
    else
        decline_horizon = T;
    end

    % Ensure the decline horizon is an integer number of periods within [1, T].
    decline_horizon = max(1, min(T, round(decline_horizon)));

    A_path = repmat(params.A(:), 1, T);

    % Venezuela (location 1) experiences a linear decline from 1 to 0.1 over
    % DECLINE_HORIZON periods and remains at the terminal value afterwards.
    start_val = 1.0;
    end_val   = 0.1;
    A_path(1, 1:decline_horizon) = linspace(start_val, end_val, decline_horizon);
    if decline_horizon < T
        A_path(1, decline_horizon+1:end) = end_val;
    end

    params.A_path     = A_path;
    params.A_initial  = A_path(:, 1);
    params.A_terminal = A_path(:, T);

    % Update the baseline productivity vector to the terminal values so that
    % stationary calculations (e.g., the no-help equilibrium) use the long-run
    % productivities.
    params.A = params.A_terminal;

end

