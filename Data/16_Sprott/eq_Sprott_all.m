function dxdt = eq_Sprott_all(~,x_in,sys_indx)
% https://journals.aps.org/pre/pdf/10.1103/PhysRevE.50.R647

x = x_in(1);
y = x_in(2);
z = x_in(3);

switch sys_indx
    case {'a',1}
        dxdt = [ ...
            y;
            -x+y*z;
            1-y^2 ];
    case {'b',2}
        dxdt = [ ...
            y*z;
            x-y;
            1-x*y ];
    case {'c',3}
        dxdt = [ ...
            y*z;
            x-y;
            1-x^2 ];
    case {'d',4}
        dxdt = [ ...
            -y;
            x+z;
            x*z+3*y^2 ];
    case {'e',5}
        dxdt = [ ...
            y*z;
            x^2-y;
            1-4*x ];
    case {'f',6}
        dxdt = [ ...
            y+z;
            -x+y/2;
            x^2-z ];
    case {'g',7}
        dxdt = [ ...
            0.4*x+z;
            x*z-y;
            -x+y ];
    case {'h',8}
        dxdt = [ ...
            -y+z^2;
            x+y/2;
            x-z ];
    case {'i',9}
        dxdt = [ ...
            -0.2*y;
            x+z;
            x+y^2-z ];
    case {'j',10}
        dxdt = [ ...
            2*z;
            -2*y+z;
            -x+y+y^2 ];
    case {'k',11}
        dxdt = [ ...
            x*y-z;
            x-y;
            x+0.3*z ];
    case {'l',12}
        dxdt = [ ...
            y+3.9*z;
            0.9*x^2-y;
            1-x ];
    case {'m',13}
        dxdt = [ ...
            -z;
            -x^2-y;
            1.7+1.7*x+y ];
    case {'n',14}
        dxdt = [ ...
            -2*y;
            x+z^2;
            1+y-2*z ];
    case {'o',15}
        dxdt = [ ...
            y;
            x-z;
            x+x*z+2.7*y ];
    case {'p',16}
        dxdt = [ ...
            2.7*y+z;
            -x+y^2;
            x+y ];
end

end


