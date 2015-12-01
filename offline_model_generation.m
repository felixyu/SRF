% Given kernel parameter a, b, dim, num_gaussians etc., generate
% the model file. Note that the model is independent of the data. 

addpath(genpath('./minFunc_2012'));
clear all;

%% Initialize parameters
% Kernel parameters. The kernel has the form:
% (1 - ||x - y||^2 / a^2 ) ^ b
params.dim = 128;
kernelparams.a = 4;
kernelparams.b = 3;
% Meta parameters for SRF
params.num_gaussians = 10; % Too many makes it hard to optimize, too few and it's not expressive enough.
params.num_points_x = 500; % More is better but will be slower.
params.num_points_w = 500; % Likewise.
params.eps = 1e-20;
kernelfunc = @(normz) (1 - (normz/kernelparams.a).^2 ).^kernelparams.b;

%% Optimization w/ minFunc (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
% It may help to multiple run times with different initializations in order
% to find a good minimum
rng(116)

options.Method = 'lbfgs';
options.MaxIter = 50;
options.MaxFunEvals = 100;
options.progTol = 1e-10;
options.optTol = 1e-10;
options.Corr = 500;
options.Display = 'iter';
theta0 = sqrt(1/params.num_gaussians)*(-1+2*(rand(params.num_gaussians*2,1)));
theta0(1:params.num_gaussians) = abs(theta0(1:params.num_gaussians));
[thetaOpt, val, ~, ~] = minFunc(@(theta) approx_kernel_cost_grad(theta, params, kernelfunc),theta0,options);

%% Evaluate result
params.num_points_w = params.num_points_w * 10; % Take a finer mesh for getting cdf. 
[fa, fx, pdf, cutoff] = approx_kernel_eval(thetaOpt, params, kernelfunc);
xx=0:2/(params.num_points_x - 1):2;
mean_abs_error = mean(abs(fx-kernelfunc(xx)))
max_abs_error = max(abs(fx-kernelfunc(xx)))
ww=0:cutoff/(params.num_points_w - 1):cutoff;
cdf = cumsum(pdf./sum(pdf));
[cdf, uniq_ind] = unique(cdf);
ww=ww(uniq_ind);

figure('Position', [200 200 450 300]);
hold on;
plot(xx,fx, '--b', 'LineWidth',2);
plot(xx,kernelfunc(xx), 'r', 'LineWidth',2);
xlabel('z');
ylabel('K(z)');
ylim([0 1]);
legend({'Original', 'Approximated'}, 'Location', 'NorthEast');
set(findall(gcf,'type','text'),'fontSize',13)

figure('Position', [200 200 450 300]);
plot(0:cutoff/(params.num_points_w - 1):cutoff, pdf, 'r', 'LineWidth',2)
xlabel('w');
ylabel('p(w)');
xlim([0, max(ww)]);
set(findall(gcf,'type','text'),'fontSize',13)

%% save
save(sprintf('model_a%d_b%d_dim%d_num_gaussians%d.mat', kernelparams.a, kernelparams.b, params.dim, params.num_gaussians));