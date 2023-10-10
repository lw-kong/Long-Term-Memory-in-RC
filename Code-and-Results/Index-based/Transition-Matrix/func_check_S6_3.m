function bool_result = func_check_S6_3(x,at_i)
% test if the time series x is similar to attractor No. tp_i
% x size: T*D

%{
attractor_bounds = zeros(6,3,3);
for tp_i = 1:6
    for plot_dim = 1:3
        s_max = max( data_all(tp_i,2e4+1:end,plot_dim) );
        s_min = min( data_all(tp_i,2e4+1:end,plot_dim) );
        attractor_bounds(tp_i,plot_dim,1) = s_max + 0.1*(s_max-s_min);
        attractor_bounds(tp_i,plot_dim,2) = s_min - 0.1*(s_max-s_min);

        attractor_bounds(tp_i,plot_dim,3) = (s_max+s_min)/2;

    end
end
%}
attractor_bounds(:,:,1) = [...
    1.6971    1.6970    1.6971
    1.8762    2.5466    2.0613
    2.8698    3.4607    3.3166
    2.5862    2.1887    8.5079
    2.9425    3.2321    3.4251
    5.8624    2.7988    1.9854];


attractor_bounds(:,:,2) =[...
   -1.6971   -1.6971   -1.6971
   -2.3186   -2.9960   -2.2569
   -2.9385   -3.5732   -3.0958
   -2.2027   -2.4333   -1.1065
   -1.7872   -1.7558   -1.3795
   -1.5392   -2.4326   -2.8046];

attractor_bounds(:,:,3) =[...
   -0.0000   -0.0000   -0.0000
   -0.2212   -0.2247   -0.0978
   -0.0343   -0.0562    0.1104
    0.1918   -0.1223    3.7007
    0.5776    0.7381    1.0228
    2.1616    0.1831   -0.4096];




bool_result = 1;
for d_i = 1:3
    if std(x(:,d_i)) < 0.7 || std(x(:,d_i)) > 1.3
        bool_result = 0;
        break
    end
    
    % why isn't a mean check?

    x_max = max(x(:,d_i));
    x_min = min(x(:,d_i));
    
    if x_max > attractor_bounds(at_i,d_i,1) || ...
       x_min  < attractor_bounds(at_i,d_i,2) || ...
       x_max < attractor_bounds(at_i,d_i,3) || ...
       x_min > attractor_bounds(at_i,d_i,3)
        
        bool_result = 0;
        break
    end
        

end

end

