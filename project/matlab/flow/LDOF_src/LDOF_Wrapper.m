function [u,v] = LDOF_Wrapper(im1, im2)

% im1=imread(imfile1);
% im2=imread(imfile2);

verbose=0;
[p,q,~]=size(im1);
para=get_para_flow(p,q);


[F,~,~] = LDOF(im1,im2,para,verbose);

u = F(:,:,1); v = F(:,:,2);
% check_flow_correspondence(im1,im2,F);
%flow_warp(im1,im2,F,1)
%flow_view=flowToColor(F);
%imwrite(flow_view,[pwd '/view.png'],'png');
end