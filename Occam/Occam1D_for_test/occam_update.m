function [rho_new, thk_new] = occam_update(J, d, R, lambda, rho, thk, LBR, UBR, LBT, UBT)
    nlayer = length(rho);
    A = J' * J + lambda * (R' * R);
    b = J' * d';
    dm = A \ b;
    drho = dm(1:nlayer)'; 
    dthk = dm(nlayer+1:end)';
    rho_log = log10(rho) + drho;
    thk_log = log10(thk) + dthk;
    rho_new = max(min(10.^rho_log, UBR), LBR);
    thk_new = max(min(10.^thk_log, UBT), LBT);
end


