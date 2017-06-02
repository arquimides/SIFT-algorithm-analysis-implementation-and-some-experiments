function [] = drawMaxAndMin( img, data)
% Function: Draw sift feature points
figure;
imshow(img);
hold on;
plot(extractfield(data, 'y'),extractfield(data, 'x'),'.');

end