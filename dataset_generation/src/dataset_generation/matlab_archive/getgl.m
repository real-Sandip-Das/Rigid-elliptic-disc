function [gval]=getgl(l,a,b)
N=100;c=-pi;d=pi;
[x,w]=lgwt(N,c,d);
gval1=(a*b*(a.^2*cos(x).^2+b.^2*sin(x).^2).^(-3/2).*w);
gval2=exp(-2*1i*x*l);
gval=1/(2*pi)*sum(gval1.*gval2);
end