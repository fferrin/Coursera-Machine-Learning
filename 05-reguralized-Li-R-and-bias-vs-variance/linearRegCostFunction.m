function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

aux = theta;
aux(1) = 0;

diff = (X * theta) - y;
J = sum(diff.**2);
J = (J + lambda * aux' * aux) ./ (2 * m);

for i = 1:size(X, 2)
	aux2 = (X(:, i) .* diff);
	grad(i) = (sum(aux2) + lambda * aux(i)) / m;
end

% =========================================================================

grad = grad(:);

end
