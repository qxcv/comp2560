function im=chistnorm(im)
for i=1:size(im,3)
    im(:,:,i)=histeq(im(:,:,i));
end
end