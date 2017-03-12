function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    hypothesis = X*theta;
    error = hypothesis-y;
    X_col1 = X(:,1);
    X_col2 = X(:,2);

    theta = theta-(alpha*(1/m)*(error' *X)');   %Note that error matrix has to be inverted.


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
