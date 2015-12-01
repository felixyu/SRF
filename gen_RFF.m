function [ rff_x ] = gen_RFF(DATA, a, b, num_gaussians, CS_COL)
% Wrapper function to generate the RM

[N, dim] = size(DATA);

tic
params.dim = dim;
kernelparams.a = a;
kernelparams.b = b;
cdf = [];
model_file = sprintf('model_a%d_b%d_dim%d_num_gaussians%d.mat', kernelparams.a, kernelparams.b, params.dim, num_gaussians);
if (~exist(model_file, 'file'))
    error('Model file %s does not exisit. Run offline_model_generation to generate the file first.', model_file);
end
load (model_file);

W = interp1(cdf, ww, rand(CS_COL,1), 'linear', 0);
W = W';
w = normrnd(0,1,params.dim,CS_COL);
w = bsxfun(@times,1./sqrt(sum(w.^2)),w);
b = rand(CS_COL,1)*2*pi;

b = repmat(b', N, 1);
W = repmat(W, size(w, 1), 1) .* w;
rff_x = cos((DATA * W  + b)) * sqrt(2 / CS_COL);
toc

end