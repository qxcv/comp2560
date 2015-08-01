function skelflow = get_motion_from_flow_Ex(skelpts, opticalflow, keyjoints, numpts_along_limb, displayposes)
%     pt = zeros(10000,2); cnt = 0; n = length(skelpts.regions{keyjoints(1)});
%     for i=1:length(keyjoints)
%         k = keyjoints(i);                      
%         pt(cnt+1:cnt+n,:) = skelpts.regions{keyjoints(k)};
%         cnt= cnt + n;
% %         for j=1:length(skelpts.regions(k).pts)
% %             cnt = cnt + 1;
% %             pt(cnt, :) = skelpts.regions(k).pts{j}';
% %         end
%     end    
%    pt = pt(1:cnt,:);  
    skelflow = zeros(size(skelpts));
    for k=keyjoints
        for s=1:numpts_along_limb
            valid_range = (k-1)*numpts_along_limb*2 + [(s-1)*2+1 : 2*s];
            skelflow(:, valid_range) = get_motion_from_flow(skelpts(:,valid_range), opticalflow, displayposes);
        end
    end
end%%
