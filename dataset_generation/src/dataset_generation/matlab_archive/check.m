function [P]=check(r,theta,s,alpha,d,K,a,b)
x=a*r.*cos(theta);
y=b*r.*sin(theta);
gi=a*s.*cos(alpha);
eta=b*s.*sin(alpha);
R=sqrt((x-gi).^2+(y-eta).^2);
X=K*R;z=-d;nu=-d;
Z=-K*(z+nu);
M=K.^3*(((2*Z-1)./(X.^2+Z.^2).^(3/2))+(3*Z.^2./(X.^2+Z.^2).^(5/2))...
    +1./(X.^2+Z.^2).^(1/2))+K.^3./sqrt(X.^2+Z.^2);
N=K.^3*(2*pi*1i*besselh(0,X)*exp(-Z)-4/pi*intgralv(Z,X));
P=M+N;
end