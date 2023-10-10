%addpath('\klw\Research\Eco Synchro\19.9- different node dynamics\funcs dyna');
addpath('\klw\Research\Functions');

% no attractor, conservative

t_cut = 1000;
tmax = 1800 + t_cut;
tstep = 0.01;
t_ratio = 10;



rng('shuffle')
%{
figure()
set(gcf,'color','white')
for sys_i = 1:16

    x0 = 0.1*[rand; rand; rand];
    [t,x] = ode4(@(t,x) eq_Sprott_all(t,x,sys_i),0:tstep:tmax,x0);
    x = x(t_cut/tstep+1:end,:);
    x = x(1:t_ratio:end,:);

    subplot(4,4,sys_i)
    plot3(x(:,1),x(:,2),x(:,3))
    view([1 0.5 1])
end
%}


sys_i = 1;
x0 = 0.1*[rand; rand; rand];
[t,x] = ode4(@(t,x) eq_Sprott_all(t,x,sys_i),0:tstep:tmax,x0);
x = x(t_cut/tstep+1:end,:);
x = x(1:t_ratio:end,:);

figure()
set(gcf,'color','white')
plot3(x(:,1),x(:,2),x(:,3))
view([1 0.5 1])



%{
figure()
set(gcf,'color','white')

subplot(2,1,1)
plot(x(1:end-1,1))

subplot(2,1,2)
plot(x(1:200,1))
%}