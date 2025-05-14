function [Alpha_pos, Alpha_score, Convergence_curve, fitness_history] = greyWolfOptimizer(X, y, input_layer_size, hidden_layer_size, num_labels, Max_iter, SearchAgents_no, lb, ub, dim)

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
lambda = 0.1;  % Regularization parameter

% Initialize convergence curve and fitness history
Convergence_curve = zeros(1, Max_iter);
fitness_history = zeros(SearchAgents_no, Max_iter);

% Main loop
for l = 1:Max_iter
    % Display iteration header
    fprintf('\n======= Iteration %d =======\n', l);

    % For each search agent
    for i = 1:SearchAgents_no
        % Evaluate fitness of search agent (neural network weights)
        nn_params = Positions(i, :);

        % Calculate cost using nnCostFunction
        fitness = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

        % Store fitness in history
        fitness_history(i, l) = fitness;

        % Display current wolf's fitness
        fprintf('Wolf %d Fitness: %f\n', i, fitness);

        % Update Alpha, Beta, Delta
        if fitness < Alpha_score
            Alpha_score = fitness;
            Alpha_pos = Positions(i, :);

            % Display when a new best solution is found
            fprintf('>>> New Best Solution Found! Wolf %d with fitness %f <<<\n', i, fitness);
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

    % Display summary statistics for this iteration
    fprintf('\n--- Iteration %d Summary ---\n', l);
    fprintf('Best Fitness (Alpha): %f\n', Alpha_score);
    fprintf('Second Best (Beta): %f\n', Beta_score);
    fprintf('Third Best (Delta): %f\n', Delta_score);
    fprintf('Average Fitness: %f\n', mean(fitness_history(:, l)));
    fprintf('Worst Fitness: %f\n', max(fitness_history(:, l)));
    fprintf('Standard Deviation: %f\n', std(fitness_history(:, l)));

    % Save best result so far
    Convergence_curve(l) = Alpha_score;

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
end

% Plot the convergence curve at the end
figure;
plot(1:Max_iter, Convergence_curve, 'LineWidth', 2);
title('Convergence Curve (Best Fitness per Generation)');
xlabel('Iteration');
ylabel('Best Cost');
grid on;

fprintf('\n======= Optimization Complete =======\n');
fprintf('Final Best Fitness: %f\n', Alpha_score);

end

function Positions = initializePositions(SearchAgents_no, dim, ub, lb)
% Initialize the positions of search agents randomly
Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;

end
