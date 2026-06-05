function [B]=hyperterm(l1,k,m,r,theta)
l=2*l1;
B=zeros(size(l));
 for i=1:numel(l)
 Clmk=(-1)^(l(i)+2*m)*(pi*2^(l(i)+2)/(l(i)^2-1))*(factorial(2*k-l(i)+1)...
      /factorial(2*k+2*m+l(i)+1))*(factorial(2*k+2*m+1)/factorial(2*k+1))...
      *(gamma(k+3/2)/gamma(k+m+1))*(gamma(k+l(i)/2+m+3/2)/gamma(k-l(i)/2+1));

 Elkm=(-1)^(l(i)+m)*(pi*2^(l(i)-2*m+2)/(l(i)^2-1))*(gamma(k+3/2)/gamma(k+m+1))*...
     (factorial(2*m+2*k+1)/factorial(2*k+1))*(gamma(k+l(i)/2+3/2)/gamma(m+k-l(i)/2+1))*...
     (factorial(2*m+2*k-l(i)+1)/factorial(l(i)+2*k+1));
 Xlkm=Clmk*(double(alform3(k,m,l(i),r))/sqrt(1-r^2))*exp(1i*l(i)*theta)*exp(1i*m*theta);
 Ylkm=Elkm*(double(alform2(k,m,l(i),r))/sqrt(1-r^2))*exp(1i*l(i)*theta)*exp(-1i*m*theta);
 B(i)=(Xlkm+Ylkm)/2;
 end

