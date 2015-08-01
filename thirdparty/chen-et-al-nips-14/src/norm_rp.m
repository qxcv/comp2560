function norm_rp = norm_rp(iminfo, p1, p2, tsize)
% p1 - p2;
rp = iminfo.joints(p1, 1:2) - iminfo.joints(p2, 1:2);
norm_rp = rp ./ (2*[iminfo.scale_x, iminfo.scale_y]+1) .* tsize;
