function [q]=alform2(k,m,l,s)
syms r
l1=m+2*k+1;
m1=-m+l;
P1=((-1)^m1/(2^l1*factorial(l1)))*(1-r.^2).^(m1/2)*diff((r.^2-1).^l1,r,l1+m1);
P=subs(P1,r,sqrt(1-r^2));
q=subs(P,s);
end
