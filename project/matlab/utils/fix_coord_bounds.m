function rt = fix_coord_bounds(rt, sz, coordtype)
rt(:,1) = max(rt(:,1),1);
rt(:,2) = max(rt(:,2),1);
if ~strcmp(coordtype,'x2y2')
    rt(:,3) = min(rt(:,1)+rt(:,3), sz(1))-rt(:,1);
    rt(:,4) = min(rt(:,2)+rt(:,4), sz(2))-rt(:,2);
else
    rt(:,3) = min(rt(:,3), sz(2));
    rt(:,4) = min(rt(:,4), sz(1));
end
end