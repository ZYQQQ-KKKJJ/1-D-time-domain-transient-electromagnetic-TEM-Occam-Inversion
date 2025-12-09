function [dBdt, av_con] = fwd_TEM_stable(rho, thk, t, a, I)

u0 = 4*pi*1e-7;
alpha = 0.5;        
c = 3;           

rho = double(rho(:));       
thk = double(thk(:));
t   = double(t(:));     
Nt  = numel(t);

con = 1./rho;              
z = [0; cumsum(thk); inf]; 
L = numel(con);

av_con = mean(con) * ones(Nt,1);

for k = 1:30
    d = sqrt((c*t) ./ (av_con*u0)); 

    F = zeros(Nt, L);
    for i = 1:Nt
        for j = 1:L
            % 上边界响应
            if z(j) <= d(i)
                F1 = (2 - z(j)/d(i)) * (z(j)/d(i));
            else
                F1 = 1;
            end
            if z(j+1) <= d(i)
                F2 = (2 - z(j+1)/d(i)) * (z(j+1)/d(i));
            else
                F2 = 1;
            end
            F(i,j) = F2 - F1;  
        end
    end

    app_con = F * con;  

    av_con = alpha * app_con + (1 - alpha) * av_con;
end

x = a * sqrt((u0*av_con) ./ (4*t));  
K = zeros(Nt,1);
small = (x < 0.0018);          % 阈值可调：0.2~0.5 都可
xs = x(small);

if any(small)
    K(small) = (8/(5*sqrt(pi))) * xs.^5 .* (1 - (75/56)*xs.^2);
end

if any(~small)
    xl = x(~small);
    K(~small) = 3*erf(xl) - (2/sqrt(pi)) .* xl .* (3 + 2*xl.^2) .* exp(-xl.^2);
end
av_con = max(av_con, eps);
dBdt = (I ./ (av_con * (a^3))) .* K;
dBdt = dBdt';
end
