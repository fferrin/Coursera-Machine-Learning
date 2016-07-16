function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0; % cost function
grad = zeros(size(theta)); % gradient

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% logistic regression hypothesis
h_theta = sigmoid(X * theta);
one = ones(m, 1);

% find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% compute cost function and gradient in logistic regression with regularization
% J(\theta) = \frac{1}{m} \sum_{i=1}^{m} {-y^{(i)} log(h_{\theta}(x^{(i)})) -
% (1 - y^{(i)}) log(1 - h_{\theta}(x^{(i)}))} + \frac{\lambda}{2m} \sum_{i=1}^{m}
% {\theta_{j}^{2}}
J = -(y' * log(h_theta) + (one - y)' * log(one - h_theta)) / m + lambda / (2 * m) * (theta' * theta - theta(1)^2);

% \frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=1}^{m} {
% (h_{\theta}(x^{(i)}) - y^{(i)}) x_{j}^{(i)} } + \frac{\lambda}{m} \theta_{j}
grad = X' * (h_theta - y) / m + (lambda / m) .* theta - lambda * [theta(1); zeros(size(theta) - 1, 1)] / m;

% =============================================================

end
