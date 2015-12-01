function [f, df] = approx_kernel_cost_grad(theta, params, kernelfunc)
% Calculates the cost and gradient of weighted sum of zero-centered
% Gaussians as it approximates kernelfunc. There is an additional
% restriction that the pdf be non-negative and the normalized, which we
% impose in Fourier space before transforming back and calculating the
% error in x-space.
%
% -theta(1:end/2) should be the estimate of std deviation (should be >0)
% -theta(end/2+1:end) should be corresponding weights (can be negative)
% -params is a structure containing the following fields:
%   -num_points_x: how many points to sample in x-space -num_points_w: how
%   many points to sample in w-space -eps: a cutoff parameter for
%   integrating the pdf -dim: the dimension of vectors
%
% Output: f is total cost, df is gradient wrt theta

dim = params.dim;
theta = theta'; % minFunc likes theta transposed
D = length(theta)/2; % Should be equal to num_gaussians

% The pdf is a sum of exp(-x^2)*x^n terms, which have a peak before
% decaying. The following line finds the position on the tail of the decay
% which has value < eps. Everything farther out should be smaller, so we
% can neglect it. We then weight each term and find the maximum one, which
% is how far we have to integrate before knowing there is nothing
% interesting left.
cutoff = max(find_cutoff(dim,eps./abs(theta(D+1:2*D)),abs(theta(1:D))));

% Grid in x-space
x = 0:2/(2*params.num_points_x-1):2; 

% Use different grid for pdf
w = 0:cutoff/(2*params.num_points_w-1):cutoff; 

% Initialize gradient vector
df = zeros(length(theta),1);

% Argument of gaussians
X = 1/2*(1./abs(theta(1:D)))' * w; 

% For large dim, evaluate the series to avoid overall. 
% Also bring the prefactor into the exponent for cancellations.
if dim > 100 
    logGammaSer = dim/2*(log(dim/2)-1) - 1/2 * log(dim/4/pi) + 1/6/dim ...
                  -1/45/dim^3 + 8/315/dim^5 - 8/105/dim^7 + 128/297/dim^9;
    A = exp(-X.^2 + (dim-1)*log(X) - logGammaSer) ...
        .* ((1./abs(theta(1:D)))' * ones(1,length(w)));
else
    A = 1/my_gamma(dim/2) * exp(-X.^2 + (dim-1)*log(X)) ...
        .* ((1./abs(theta(1:D)))' * ones(1,length(w)));
end

% Weighted sum of gaussians in w-space, i.e. the unnormalized pdf
f1 = theta(D+1:2*D) * A; 

% The integral of positive part: w(2) is grid spacing, i.e. 'dw'
normalization = 1./(eps + w(2) * sum(f1(f1>0))); 

pdf = normalization * f1;
pdf(f1<0) = 0;

% Because we made the pdf positive, it's no longer integrable analytically.
% But we need to integrate it to do the inverse Fourier transform, to get
% the predicted function in x-space (our approximate kernel). This is one
% numerical integral. After we get the approximate kernel, we want to
% calculate the L2-norm between it and the actual kernel. This is another
% numerical integral. So we have a 2D numerical integration to perform:

% The grid for the 2D integral.
wx = w'*x; 

% The integrand of the inverse Fourier transform
Jwx = scaled_bessel(dim/2-1,wx).*((f1>0)'*ones(size(x))); 

% The difference between our approximate kernel and the exact one
f2 = (w(2)*(pdf * Jwx) - kernelfunc(x));

% Total cost, i.e. squared L2-norm between approx. kernel and the exact one
f = 1/2 * x(2) * (f2*f2'); 

% Now we calculate some derivatives. Vectorizing this code made it really
% ugly, and surely it could be cleaned up / made more efficient.
ee = (theta(D+1:2*D)./abs(theta(1:D)))'*ones(size(w));
X2 = (-dim + 2*X.^2);
rr =  (-normalization.^2)*w(2)*X2.*A;
df(1:D) = w(2)*x(2)*(( (ee.*rr) * (f1>0)' * theta(D+1:2*D) * A ... 
            + ee.*X2.*(A*normalization)) * Jwx)*f2';
rr =  (-normalization.^2)*w(2)*A;
df(D+1:end) = w(2)*x(2)*(( (rr) * (f1>0)' * theta(D+1:2*D) * A ... 
                + (A*normalization)) * Jwx)*f2';
end
