function [P]=kernel(x,y,s,alpha,K,a,b,d)
gi=a*s.*cos(alpha);
eta=b*s.*sin(alpha);
R=sqrt((x-gi).^2+(y-eta).^2);
X=K*R;Y=K*d;
P=K^2*(2*Y./(X.^2+Y^2).^(3/2)+2./(X.^2+Y^2).^(1/2)+2*pi*1i*exp(-Y)*besselh(0,X)-4/pi*intgralv(Y,X));