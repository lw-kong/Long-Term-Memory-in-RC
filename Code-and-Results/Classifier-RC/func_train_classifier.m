function [validation_performance,W_in,res_net,W_out,t_validate,x_real,x_validate] = ...
    func_train_classifier(train_input,train_target,flag,W_in_type,res_net_type,validation_type)
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
validate_cut = flag(10);

tstep = flag(11);
dim = flag(12);
noise_a = flag(13);



trial_num = size(train_target,1);
tp_num = size(train_target,3);

validate_start = train_length+2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define W_in
if W_in_type == 1
    W_in = W_in_a*(2*rand(n,dim)-1);
elseif W_in_type == 2
    % each node is inputed with with one dimenson of real data
    % and all the tuning parameters
    W_in=zeros(n,dim);
    n_win = n-mod(n,dim);
    index=randperm(n_win); index=reshape(index,n_win/dim,dim);
    for d_i=1:dim
        W_in(index(:,d_i),d_i)=W_in_a*(2*rand(n_win/dim,1)-1);
    end
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
r_reg = zeros(n,trial_num*(train_length-10));
y_reg = zeros(tp_num,trial_num*(train_length-10));
%r_end = zeros(n,trial_num);
for trial_i = 1:trial_num
    train_x = zeros(train_length,dim);
    train_y = zeros(train_length,tp_num);
    train_x(:,:)=train_input(trial_i,1:train_length,:);
    train_y(:,:)=train_target(trial_i,1:train_length,:);
    train_x = train_x';
    train_y = train_y';
    
    train_x = train_x + noise_a * randn(size(train_x));                           %% noise
    
    
    %r_all=[];
    %r_all(:,1)=zeros(n,1);%2*rand(n,1)-1;%
    r_all = zeros(n,train_length+1);
    for ti=1:train_length
        r_all(:,ti+1)=(1-a)*r_all(:,ti) ...
            + a*tanh(res_net*r_all(:,ti)+W_in*train_x(:,ti));
    end
    r_out=r_all(:,12:end); % n * (train_length - 11)
    r_out(2:2:end,:)=r_out(2:2:end,:).^2;
    %r_end(:,trial_i)=r_all(:,end); % n * 1
    
    r_reg(:, (trial_i-1)*(train_length-10) +1 : trial_i*(train_length-10) ) = r_out;
    y_reg(:, (trial_i-1)*(train_length-10) +1 : trial_i*(train_length-10) ) = train_y(:,11:end);
end
W_out= y_reg *r_reg'*(r_reg*r_reg'+beta*eye(n))^(-1);


%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% validate resnet model
%disp('validating...')
rmse_set = zeros(1,trial_num);

validate_predict_y_set = zeros(trial_num,validate_length,tp_num);
validate_real_y_set = zeros(trial_num,validate_length,tp_num);
for trial_i = 1:trial_num
    validate_real_y_set(trial_i,:,:) = train_target(trial_i,validate_start:(validate_start+validate_length-1),:);

    r = zeros(n,1);
    u = zeros(dim,1);
    for t_i = 1:validate_cut
        u(:) = train_input(trial_i,train_length+t_i,:);
        r = (1-a) * r + a * tanh(res_net*r+W_in*u);
    end
    for t_i = validate_cut+1:validate_cut+validate_length
        u(:) = train_input(trial_i,train_length+t_i,:);
        r = (1-a) * r + a * tanh(res_net*r+W_in*u);
        r_out = r;
        r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
        predict_y = W_out * r_out;
        validate_predict_y_set(trial_i,t_i-validate_cut,:) = predict_y;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    error = zeros(validate_length,tp_num);
    error(:,:) = validate_predict_y_set(trial_i,:,:) - validate_real_y_set(trial_i,:,:);
    %rmse_ts = sqrt( mean( abs(error).^2 ,2) );
    se_ts = sum( error.^2 ,2);
    
 
    rmse_set(trial_i) = sqrt(mean(se_ts));
end




if validation_type == 1
    validation_performance =  max(rmse_set);
%elseif validation_type == 2
%    success_length = min(success_length_set);
%    %fprintf('attempt success_length = %f \n',success_length);
%    validation_performance = success_length;
elseif validation_type == 3
    for trial_i = 1:trial_num
        rmse_set(trial_i) = max(rmse_set(trial_i),10^-3);
    end
    validation_performance = prod(rmse_set);
elseif validation_type == 4
    validation_performance =  mean(rmse_set);
else
    fprintf('validation type error');
    return
end


t_validate = tstep:tstep:tstep*validate_length;
x_validate = validate_predict_y_set;
x_real = validate_real_y_set;

end


