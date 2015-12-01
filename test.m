% Computer the nonlinear map based on SRF. 
clear all;

%% kernel parameter
a = 4;
b = 7;
num_gaussians = 10;

%% data
load('usps.mat');
num_train = size(data_train, 1);
num_test = size(data_test,1);
X = [data_train; data_test];

%% normalization
for i = 1:size(X,1)
    X(i,:) = X(i,:)./norm(X(i,:),2);
end

%% kernel apprximation and evaluation
N_sample = 10000;
mapdim_all = 2.^[9, 10, 11, 12, 13, 14];
if (size(X,1) > N_sample)
    rand_r = randperm(size(X,1));
    X = X(rand_r(1:N_sample), :);
end
kernelfunc = @(normz) (1 - (normz/a).^2 ).^b;
MSE = zeros(1, length(mapdim_all));
for ii = 1:length(mapdim_all)
    mapdim = mapdim_all(ii);
    fea = gen_RFF(X, a, b, num_gaussians, mapdim);
    kernel_gt =  kernelfunc(sqrt(2 - X*X'*2));
    kernel_diff = (kernel_gt - fea * fea').^2;
    MSE(ii) = mean(kernel_diff(:));
end
close all;
figure;
hold on;
plot(log(mapdim_all)/log(2), MSE);
xlabel('log(mapped dimensionality)');
ylabel('MSE')

