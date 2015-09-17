function psi_data=get_psi_data_weighted(d1square,d2square,gamma,temperature)
w1=1/(1+exp(temperature*(app_sqrt(d1square)-app_sqrt(d2square))));
w2=1-w1;
psi_data=w1.*psiDeriv(d1square)+gamma*w2.*psiDeriv(d2square);
end

function res=app_sqrt(s2)
res=sqrt(s2+eps^2);
end