function predict_all = func_STP_predict_switch(x_warmup,tp_1,tp_2,W_in,res_net,P,flag)
% included warmup
% flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
n = flag(1); % number of nodes in res_net
dim = flag(2);
a = flag(3);
warmup_length = flag(4);
predict_length_1 = flag(5);
predict_length_2 = flag(6);

dim_tp = length(tp_1);

r = zeros(n,1); % hidden layer state n * 1
u = zeros(dim+dim_tp,1);
u(dim+1:end) = tp_1;
%% warm up
if warmup_length ~= 0
    x_warmup = x_warmup(1:warmup_length,:);        
    for t_i = 1:(warmup_length-1)
        u(1:dim) = x_warmup(t_i,:);
        r = (1-a) * r + a * tanh(res_net*r+W_in*u);
    end
else
    x_warmup = zeros(dim,1);
end

%% predicting No. 1
% disp('  predicting...')
predict_1 = zeros(predict_length_1,dim);
u(1:dim) = x_warmup(end,:);

for t_i=1:predict_length_1
    r = (1-a) * r + a * tanh( res_net*r+W_in*u );
    
    r_out = r;
    r_out(2:2:end) = r_out(2:2:end).^2; %even number -> squared
    
    predict_1(t_i,:) = P*r_out;
    u(1:dim) = predict_1(t_i,:);
end


%% predicting No. 2
% disp('  predicting...')
predict_2 = zeros(predict_length_2,dim);
u(dim+1:end) = tp_2;

for t_i=1:predict_length_2
    r = (1-a) * r + a * tanh( res_net*r+W_in*u );
    
    r_out = r;
    r_out(2:2:end) = r_out(2:2:end).^2; %even number -> squared
    
    predict_2(t_i,:) = P*r_out;
    u(1:dim) = predict_2(t_i,:);
end

predict_all = [predict_1; predict_2];


end