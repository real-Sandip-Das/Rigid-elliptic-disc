function [value]=doubleintC(k,m,r,theta,depth,K,a,b)
c=0;d=1;
e=0;f=2*pi;
[x1,w1]=lgwt(100,c,d);
[x2,w2]=lgwt(100,e,f);
w=w1*w2';
p4=ones(size(w));
[x,y]=meshgrid(x1,x2);
p1=double(alform(k,m,x1)).*x1;
p2=cos(m*x2);
p3=check(r,theta,x(1:numel(w)),y(1:numel(w)),depth,K,a,b);
p4(1:numel(w))=p3(1:numel(w));
p=p1*p2.';
value=a*b*sum(sum(w.*p.*p4.'));
end
