function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0; % cost function
grad = zeros(size(theta)); % gradient

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% aux theta
aux = theta;
aux(1) = 0;

% J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} {(h_{\theta}(x^{(i)}) - y^{(i)})^2} +
%             \frac{\lambda}{2m} \sum_{j=1}^{n} {\theta_{j}^{2}}
diff = (X * theta) - y;
J = sum(diff.**2);
J = (J + lambda * aux' * aux) ./ (2 * m);

% \frac{\partial J(\theta)}{\partial \theta_{j}} =
%   \frac{1}{m} \sum_{i=1}^{m} {(h_{\theta}(x^{(i)}) - y^{(i)}) x_{j}^{(i)}} +
%   \frac{\lambda}{m} \theta_{j}
for i = 1:size(X, 2)
	aux2 = (X(:, i) .* diff);
	grad(i) = (sum(aux2) + lambda * aux(i)) / m;
end

% =========================================================================

grad = grad(:);

end
