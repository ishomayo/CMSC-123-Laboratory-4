function [Alpha_pos, Alpha_score, Convergence_curve] = greyWolfOptimizer(X, y, input_layer_size, hidden_layer_size, num_labels, Max_iter, SearchAgents_no, lb, ub, dim)
% Grey Wolf Optimizer implementation for neural network training
%
% Inputs:
%   X - Training data features
%   y - Training data labels
%   input_layer_size - Number of input features
%   hidden_layer_size - Number of hidden neurons
%   num_labels - Number of output classes
%   Max_iter - Maximum number of iterations
%   SearchAgents_no - Number of search agents (wolves)
%   lb - Lower bound of search space
%   ub - Upper bound of search space
%   dim - Dimensionality of the problem (total number of weights)
%
% Outputs:
%   Alpha_pos - Best position (weights) found
%   Alpha_score - Best cost value
%   Convergence_curve - History of best cost values

% Initialize alpha, beta, and delta positions and scores
Alpha_pos = zeros(1, dim);
Alpha_score = inf;

Beta_pos = zeros(1, dim);
Beta_score = inf;

Delta_pos = zeros(1, dim);
Delta_score = inf;

% Initialize positions of search agents randomly within bounds
Positions = initializePositions(SearchAgents_no, dim, ub, lb);

% Create lambda value for regularization
lambda = 0.1;  % Regularization parameter (can be tuned)

% Initialize convergence curve
Convergence_curve = zeros(1, Max_iter);

% Main loop
for l = 1:Max_iter
    % For each search agent
    for i = 1:SearchAgents_no
        % Evaluate fitness of search agent (neural network weights)
        nn_params = Positions(i, :);

        % Calculate cost using nnCostFunction
        fitness = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

        % Update Alpha, Beta, Delta
        if fitness < Alpha_score
            Alpha_score = fitness;
            Alpha_pos = Positions(i, :);
        end

        if fitness > Alpha_score && fitness < Beta_score
            Beta_score = fitness;
            Beta_pos = Positions(i, :);
        end

        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
            Delta_score = fitness;
            Delta_pos = Positions(i, :);
        end
    end

    % Update a (linearly decreased from 2 to 0)
    a = 2 - l * (2 / Max_iter);

    % Update position of each search agent
    for i = 1:SearchAgents_no
        for j = 1:dim
            % Update coefficients
            r1 = rand(); % r1 is a random number in [0,1]
            r2 = rand(); % r2 is a random number in [0,1]

            A1 = 2 * a * r1 - a; % Eq. (3.3)
            C1 = 2 * r2; % Eq. (3.4)

            % Distance from alpha
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j)); % Eq. (3.5)
            X1 = Alpha_pos(j) - A1 * D_alpha; % Eq. (3.6)

            r1 = rand();
            r2 = rand();

            A2 = 2 * a * r1 - a;
            C2 = 2 * r2;

            % Distance from beta
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
            X2 = Beta_pos(j) - A2 * D_beta;

            r1 = rand();
            r2 = rand();

            A3 = 2 * a * r1 - a;
            C3 = 2 * r2;

            % Distance from delta
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
            X3 = Delta_pos(j) - A3 * D_delta;

            % Update position based on alpha, beta, and delta positions
            Positions(i, j) = (X1 + X2 + X3) / 3; % Eq. (3.7)

            % Check bounds
            if Positions(i, j) > ub
                Positions(i, j) = ub;
            end
            if Positions(i, j) < lb
                Positions(i, j) = lb;
            end
        end
    end

    % Save best result so far
    Convergence_curve(l) = Alpha_score;

    % Display iteration information
    if mod(l, 5) == 0
        fprintf('Iteration %d: Best Cost = %f\n', l, Alpha_score);
    end
end

end

function Positions = initializePositions(SearchAgents_no, dim, ub, lb)
% Initialize the positions of search agents randomly
% This function is equivalent to randInitializeWeights but for the entire
% population of weights vectors

Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;

end
