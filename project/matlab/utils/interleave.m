% function interleaves to mxn matrices column wise. the result is a
% (mx(2n)) matrix z. 
function z = interleave(x,y)
m=size(x,1); n = size(x,2);
if size(x,1)~=size(y,1) || size(x,2)~=size(y,2)
    error('incompatible sizes');
end
z = zeros(m,2*n);
for i=1:n
    z(:,2*(i-1)+1) = x(:,i);
    z(:,2*i) = y(:,i);
end
end
