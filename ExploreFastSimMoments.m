%% ExploreFastSimMoments
% Script to run `fastSimulatedMoments` with custom transition options and
% visualize the resulting time-series of migrant outcomes.
%
% The script performs the following steps:
%   1. Adds the `core` and `utils` folders to the MATLAB path.
%   2. Defines runtime options for the fast simulation (transition paths for
%      $A_1$ and $B_1$, horizon length, etc.).
%   3. Shows how to override structural parameters without editing
%      `SetParameters.m`.
%   4. Calls `fastSimulatedMoments` and produces diagnostic plots describing
%      the simulated economy.
%
% Figures produced:
%   • Migrant shares by location over time.
%   • Networked migrant shares by location over time.
%   • Average wealth of migrants remaining in Venezuela, alongside migration
%     costs to alternative destinations.
%   • Average labor income of employed migrants in each location.
%
% NOTE: All time series are quarterly. The horizontal axis is reported in
% years for easier interpretation.

%% Housekeeping ---------------------------------------------------------------
clearvars; clc;
close all;

thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, 'core')));
addpath(genpath(fullfile(thisDir, 'utils')));

%% Simulation options --------------------------------------------------------
opt = struct();
opt.A1_start          = 2;
opt.A1_end            = 1;
opt.B1_start          = 1.2;
opt.B1_end            = 0.01;
opt.TransitionPeriods = 10;   % quarters over which A(1) and B(1) move
opt.years             = 20;   % total simulation horizon in years

%% Example parameter overrides -----------------------------------------------
% Specify structural parameters that should differ from the defaults in
% `SetParameters`. The values below match the baseline configuration, so they
% serve as a template: change them here instead of editing the function.
paramOverrides = struct();
paramOverrides.ggamma = 0.4;  % elasticity of help probability w.r.t. networks
paramOverrides.cchi   = 0.15; % probability that network ties decay outside VEN

%% Run fast simulation -------------------------------------------------------
out = fastSimulatedMoments(paramOverrides, opt);

%% Prepare auxiliary objects -------------------------------------------------
T                = out.settings.T;
quartersPerYear  = 4;
timeInYears      = (0:T-1) ./ quartersPerYear;
locationLabels   = defaultLocationLabels(out.dims.N);

[M_total, M_network] = deal(out.M_total, out.M_network);

% Retrieve grids needed for wealth/income calculations.
[grids, ~] = setGridsAndIndices(out.dims); %#ok<ASGLU>
agrid    = grids.agrid;
psiGrid  = grids.psi;

locationTraj = double(out.agentData.location);
stateTraj    = double(out.agentData.state);
wealthIdx    = double(out.agentData.wealth);
skillVec     = double(out.agentData.skill(:));

Nagents = size(locationTraj, 1);

%% Plot 1: Migrant mass shares by location -----------------------------------
figure('Name', 'Migrant Shares by Location');
plot(timeInYears, M_total', 'LineWidth', 1.25);
legend(locationLabels, 'Location', 'eastoutside');
xlabel('Years since start');
ylabel('Share of simulated agents');
title('Total migrant mass by location');
ylim([0, 1]);
grid on;

%% Plot 2: Networked migrant shares by location ------------------------------
figure('Name', 'Networked Migrant Shares');
plot(timeInYears, M_network', 'LineWidth', 1.25);
legend(locationLabels, 'Location', 'eastoutside');
xlabel('Years since start');
ylabel('Share of simulated agents');
title('Networked migrant mass by location');
ylim([0, 1]);
grid on;

%% Plot 3: Average wealth in Venezuela + migration costs ---------------------
wealthLevels = agrid(wealthIdx);
maskVzla     = (locationTraj == 1);
countVzla    = sum(maskVzla, 1);
wealthSum    = sum(wealthLevels .* maskVzla, 1);
avgWealthVzla = wealthSum ./ max(countVzla, 1);
avgWealthVzla(countVzla == 0) = NaN;  % no residents -> undefined mean

figure('Name', 'Wealth in Venezuela');
plot(timeInYears, avgWealthVzla, 'LineWidth', 1.5);
hold on;
migCostsFromVzla = out.params.ttau(1, 2:end);
for j = 1:numel(migCostsFromVzla)
    yline(migCostsFromVzla(j), '--', sprintf('Cost to %s', locationLabels{j+1}), ...
        'LabelHorizontalAlignment', 'left');
end
hold off;
legendLabels = ["Average wealth in Venezuela"; compose("Cost to %s", string(locationLabels(2:end)))];
legend(legendLabels, 'Location', 'eastoutside');
xlabel('Years since start');
ylabel('Assets / migration cost units');
title('Average assets of migrants in Venezuela vs. migration costs');
grid on;

%% Plot 4: Average labor income by location ----------------------------------
% Compute quarterly wages for employed agents, following the same logic as in
% `computeRequestedMoments`.
Tused = size(locationTraj, 2);
skillMat   = repmat(skillVec, 1, Tused);
if isfield(out.params, 'A_timePath') && ~isempty(out.params.A_timePath) && ...
        size(out.params.A_timePath, 1) == out.dims.N
    colIdx = repmat(1:Tused, Nagents, 1);
    linIdx = sub2ind(size(out.params.A_timePath), locationTraj, colIdx);
    A_vals = reshape(out.params.A_timePath(linIdx), [Nagents, Tused]);
else
    A_vals = out.params.A(locationTraj);
end

thetaIdx  = sub2ind([out.dims.S, out.dims.N], skillMat, locationTraj);
thetaVals = out.params.theta_s(thetaIdx);
psiIdx    = mod(stateTraj - 1, out.dims.k) + 1;
psiVals   = psiGrid(psiIdx);

wageVals = A_vals .* thetaVals .* (1 + psiVals) .^ out.params.theta_k;
isEmployed = stateTraj > out.dims.k;
wageVals(~isEmployed) = NaN;

avgIncomeByLoc = nan(out.dims.N, Tused);
for loc = 1:out.dims.N
    employedMask = (locationTraj == loc) & isEmployed;
    denom = sum(employedMask, 1);
    numer = sum(wageVals .* double(employedMask), 1, 'omitnan');
    hasWorkers = denom > 0;
    avgIncomeLoc = nan(1, Tused);
    avgIncomeLoc(hasWorkers) = numer(hasWorkers) ./ denom(hasWorkers);
    avgIncomeByLoc(loc, :) = avgIncomeLoc;
end

figure('Name', 'Average Labor Income by Location');
plot(timeInYears, avgIncomeByLoc', 'LineWidth', 1.25);
legend(locationLabels, 'Location', 'eastoutside');
xlabel('Years since start');
ylabel('Average labor income (employed only)');
title('Average labor income among employed migrants');
grid on;

%% Helper functions ----------------------------------------------------------
function labels = defaultLocationLabels(numLocations)
%DEFAULTLOCATIONLABELS  Provide fallback legend names for locations.
%   The first location is Venezuela; remaining destinations receive generic
%   labels unless the user provides custom names.
    labels = cell(numLocations, 1);
    if numLocations >= 1
        labels{1} = 'Venezuela';
    end
    for ii = 2:numLocations
        labels{ii} = sprintf('Location %d', ii);
    end
end