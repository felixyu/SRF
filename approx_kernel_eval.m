function [f, fx, pdf, cutoff] = approx_kernel_eval(theta, params, kernelfunc)
% Calculates the cost for a given theta.
%
% Output: f is total cost, fx predict kernel
theta = theta'; % minFunc likes theta transposed
D = length(theta)/2; % Should be equal to num_gaussians
dim = params.dim;
cutoff = max(find_cutoff(dim,eps./abs(theta(D+1:2*D)),abs(theta(1:D))));
x = 0:2/(params.num_points_x-1):2; % grid in x-space
w = 0:cutoff/(params.num_points_w-1):cutoff; % use different grid for pdf
df = zeros(length(theta),1);
X = 1/2*(1./abs(theta(1:D)))' * w; % Argument of gaussians
if dim > 100 %Need to bring the prefactor into the exponent for cancellations
    logGammaSer = dim/2*(log(dim/2)-1) - 1/2 * log(dim/4/pi) + 1/6/dim - 1/45/dim^3 + 8/315/dim^5 - 8/105/dim^7 + 128/297/dim^9;
    A = exp(-X.^2 + (dim-1)*log(X) - logGammaSer) .* ((1./abs(theta(1:D)))' * ones(1,length(w)));
else
    A = 1/my_gamma(dim/2) * exp(-X.^2 + (dim-1)*log(X)) .* ((1./abs(theta(1:D)))' * ones(1,length(w)));
end
f1 = theta(D+1:2*D) * A; % weighted sum of gaussians in w-wspace, i.e. the unnormalized pdf
normalization = 1./(eps + w(2) * sum(f1(f1>0))); % normalzation is integral of positive part
pdf = normalization * f1;
pdf(f1<0) = 0;
wx = w'*x; % this is the grid for the 2D integral.
Jwx = scaled_bessel(dim/2-1,wx).*((f1>0)'*ones(size(x))); % Here is the function we have to integrate against to do the inverse Fourier transform
f2 = (w(2)*(pdf * Jwx) - kernelfunc(x)); % This is the difference between our approximate kernel and the real one
f = 1/2 * x(2) * f2*f2'; % Here is the total cost, i.e. the squared L2-norm between our approximate kernel and the real one
fx = w(2)*(pdf * Jwx);
pdf = pdf * w(2);
end
