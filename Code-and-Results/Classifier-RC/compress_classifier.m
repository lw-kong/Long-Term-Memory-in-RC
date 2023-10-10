
classifier_W_in = W_in;
classifier_W_r = res_net;
classifier_W_out = P;
classifier_rand_start_len = rand_start_len;
classifier_n = n;
classifier_a = a;

save('save_classifier_S0_6_0.mat',...
    "classifier_n","classifier_rand_start_len","classifier_W_out",...
    "classifier_W_r","classifier_W_in","classifier_a")