function [t_validate,x_validate] = ...
    func_test_classifier(test_input,W_in,W_r,W_out,flag)

n = flag(1);
a = flag(2);

validate_length = flag(3);
validate_cut = flag(4);

tstep = flag(5);
dim = flag(6);



trial_num = size(test_input,1);
tp_num = size(W_out,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% validate
validate_predict_y_set = zeros(trial_num,validate_length,tp_num);
for trial_i = 1:trial_num
    r = zeros(n,1);
    u = zeros(dim,1);
    for t_i = 1:validate_cut
        u(:) = test_input(trial_i,t_i,:);
        r = (1-a) * r + a * tanh(W_r*r+W_in*u);
    end
    for t_i = validate_cut+1:validate_cut+validate_length
        u(:) = test_input(trial_i,t_i,:);
        r = (1-a) * r + a * tanh(W_r*r+W_in*u);
        r_out = r;
        r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
        predict_y = W_out * r_out;
        validate_predict_y_set(trial_i,t_i-validate_cut,:) = predict_y;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

t_validate = tstep:tstep:tstep*validate_length;
x_validate = validate_predict_y_set;

end


