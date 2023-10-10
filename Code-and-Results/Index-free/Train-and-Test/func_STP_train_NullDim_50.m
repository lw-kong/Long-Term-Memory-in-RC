function [validation_performance,W_in,res_net,W_out,t_validate,x_real,x_validate] = ...
    func_STP_train_NullDim_50(udata,tp_train_set,dim_set,flag,...
    W_in_type,res_net_type,validation_type,print_progress_type)

% _50
% commented res_net=full(res_net); using sparse matrices
% using / instead of inv() or ^(-1) for regression
% has training noise


% use multiple trials of training to train one single result Wout
% Tp is affecting globally. Each node receives the same all control parameter
% W_in_type
%           1 : each node receives all dim of the input, a dense W_in
%           2 : each node receives one dim of the input
% res_net_type
%           1 : symmeric, normally distributed, with mean 0 and variance 1
%           2 : asymmeric, uniformly distributed between 0 and 1
% validation type
%           1 : max rmse among the tp_i
%           2 : success length
%           3 : prod of all tp_i rmse 
%           4 : average of all tp_i rmse
%           5 : rmse_set

% udata = zeros( trials, steps, dim + tp_dim )

%fprintf('in train %f\n',rand)
% flag_r_train = [n k eig_rho W_in_a a beta...
%                 0 train_r_step_length validate_r_step_length reservoir_tstep dim
%                 success_threshold];
n = flag(1); % number of nodes in res_net
k = flag(2); % mean degree of res_net
eig_rho = flag(3);
W_in_a = flag(4);
a = flag(5);
beta = flag(6);

train_length = flag(8);
validate_length = flag(9);

tstep = flag(10);
dim = flag(11);
noise_a = flag(12);

if validation_type == 2
    success_threshold = flag(13);
else
    success_threshold = 0;
end

tp_length = length(tp_train_set);
tp_dim = size(udata,3) - dim;

validate_start = train_length+2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define W_in
if W_in_type == 1
    W_in = W_in_a*(2*rand(n,dim+tp_dim)-1);
elseif W_in_type == 2
    % each node is inputed with with one dimenson of real data
    % and all the tuning parameters
    W_in=zeros(n,dim+tp_dim);
    n_win = n-mod(n,dim);
    index=randperm(n_win); index=reshape(index,n_win/dim,dim);
    for d_i=1:dim
        W_in(index(:,d_i),d_i)=W_in_a*(2*rand(n_win/dim,1)-1);
    end
    W_in(:,dim+1:dim+tp_dim) = W_in_a*(2*rand(n,tp_dim)-1);
elseif W_in_type == 3
    dim_sep = dim+tp_dim-1;
    W_in = zeros(n,dim_sep+1);
    n_win = n - mod(n,dim_sep);
    index = randperm(n_win); index=reshape(index,n_win/dim_sep,dim_sep);
    for d_i=1:dim_sep
        W_in(index(:,d_i),d_i) = W_in_a*(2*rand(n_win/dim_sep,1)-1);
    end
    W_in(:,end) = W_in_a*(2*rand(n,1)-1);
else
    fprintf('W_in type error\n');
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define reservoir_network
if res_net_type == 1
    res_net=sprandsym(n,k/n); % symmeric, normally distributed, with mean 0 and variance 1.
elseif res_net_type == 2
    k = round(k);
    index1=repmat(1:n,1,k)'; % asymmeric, uniformly distributed between 0 and 1
    index2=randperm(n*k)';
    index2(:,2)=repmat(1:n,1,k)';
    index2=sortrows(index2,1);
    index1(:,2)=index2(:,2);
    res_net=sparse(index1(:,1),index1(:,2),rand(size(index1,1),1),n,n); 
    clear index1 index2                                                    % clear
else
    fprintf('res_net type error\n');
    return
end
%res_net, adjacency matrix
%rescale eig
eig_D=eigs(res_net,1); %only use the biggest one. Warning about the others is harmless
res_net=(eig_rho/(abs(eig_D))).*res_net;
%res_net=full(res_net);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%disp('  training...')
r_reg = zeros(n,tp_length*(train_length-10));
y_reg = zeros(dim,tp_length*(train_length-10));
r_end = zeros(n,tp_length);
if print_progress_type == 2
    tic
end
for tp_i = 1:tp_length
    train_x = zeros(train_length,dim+tp_dim);
    train_y = zeros(train_length,dim+tp_dim);
    train_x(:,:)=udata(tp_i,1:train_length,:);
    train_y(:,:)=udata(tp_i,2:train_length+1,:);
    train_x = train_x';
    train_y = train_y';
    
    dim_system = dim_set(tp_i);
    train_x(1:dim_system,:) = train_x(1:dim_system,:) ...
        + noise_a * randn(size(train_x(1:dim_system,:)));                           %% noise
    
    
    %r_all=[];
    %r_all(:,1)=zeros(n,1);%2*rand(n,1)-1;%
    r_all = zeros(n,train_length+1);
    for ti=1:train_length
        r_all(:,ti+1)=(1-a)*r_all(:,ti) ...
            + a*tanh(res_net*r_all(:,ti)+W_in*train_x(:,ti));
    end
    r_out=r_all(:,12:end); % n * (train_length - 11)
    r_out(2:2:end,:)=r_out(2:2:end,:).^2;
    r_end(:,tp_i)=r_all(:,end); % n * 1
    
    r_reg(:, (tp_i-1)*(train_length-10) +1 : tp_i*(train_length-10) ) = r_out;
    y_reg(:, (tp_i-1)*(train_length-10) +1 : tp_i*(train_length-10) ) = train_y(1:dim,11:end); %no tp

    if print_progress_type == 2
        fprintf('%f of the training tps are done\n',tp_i/tp_length)
        toc
    elseif print_progress_type == 3
        if mod(tp_i,10)==0
            fprintf('%f of the training tps are done\n',tp_i/tp_length)
        end
    end
end

if print_progress_type == 2
    fprintf('echoing is done\n')
    toc
end

W_out= y_reg *r_reg'/(r_reg*r_reg'+beta*eye(n));

clear r_all r_out
clear y_reg r_reg
if print_progress_type == 2
    fprintf('regression is done\n')
    toc
end


%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% validate resnet model
%disp('validating...')
rmse_set = zeros(1,tp_length);
success_length_set = zeros(1,tp_length);

validate_predict_y_set = zeros(tp_length,validate_length,dim);
validate_real_y_set = zeros(tp_length,validate_length,dim);
for tp_i = 1:tp_length
    validate_real_y_set(tp_i,:,:) = udata(tp_i,validate_start:(validate_start+validate_length-1),1:dim);
    
    dim_system = dim_set(tp_i);
    
    r = r_end(:,tp_i);
    u = zeros(dim+tp_dim,1);
    u(1:dim_system) = udata(tp_i,train_length+1,1:dim_system); % first drive
    %u(dim+1:end) = udata(tp_i,train_length+1,dim+1:end);
    for t_i = 1:validate_length
        u(dim+1:end) = udata(tp_i,train_length+t_i,dim+1:end); %tp
        r = (1-a) * r + a * tanh(res_net*r+W_in*u);
        r_out = r;
        r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
        predict_y = W_out * r_out;
        validate_predict_y_set(tp_i,t_i,:) = predict_y;
        u(1:dim_system) = predict_y(1:dim_system);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    error = zeros(validate_length,dim_system);
    error(:,:) = validate_predict_y_set(tp_i,:,1:dim_system) - validate_real_y_set(tp_i,:,1:dim_system);
    %rmse_ts = sqrt( mean( abs(error).^2 ,2) );
    se_ts = sum( error.^2 ,2);
    
    success_length_set(tp_i) = validate_length * tstep;
    for t_i = 1:validate_length
        if se_ts(t_i) > success_threshold
            success_length_set(tp_i) = t_i * tstep;
            break;
        end
    end
    % if there is NaN in prediction, success length = 0 and rmse = 10
    if sum(isnan(validate_predict_y_set(:))) > 0
        success_length_set(tp_i) = 0;
        se_ts = 10;
    end
        
    rmse_set(tp_i) = sqrt(mean(se_ts));
end




if validation_type == 1
    validation_performance =  max(rmse_set);
elseif validation_type == 2
    success_length = min(success_length_set);
    %fprintf('attempt success_length = %f \n',success_length);
    validation_performance = success_length;
elseif validation_type == 3
    for tp_i = 1:tp_length
        rmse_set(tp_i) = max(rmse_set(tp_i),10^-3);
    end
    validation_performance = prod(rmse_set);
elseif validation_type == 4
    validation_performance =  mean(rmse_set);
elseif validation_type == 5
    validation_performance =  rmse_set';
else
    fprintf('validation type error');
    return
end


t_validate = tstep:tstep:tstep*validate_length;
x_validate = validate_predict_y_set;
x_real = validate_real_y_set;

end


