function [q]=alform(k,m,s)
syms r
l=m+2*k+1;
P1=(-1).^m.*(1-r^2).^(m/2)*diff(legendreP(l,r),r,m);
P=subs(P1,r,sqrt(1-r^2));
q=subs(P,s);
end
