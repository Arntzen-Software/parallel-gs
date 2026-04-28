pkg load signal;

luma_freq = 0;
chroma_freq = 1 / 16;

N = 256;

phases = (0 : N - 1);

bandpass = 2 * fir1(64, 1 / 48) .* cos(2 * pi * chroma_freq * (-32 : 32));
delay = [zeros(1, 32), 1, zeros(1, 32)];

% Prev line
line0_Y = 0.5;
line0_U = sin(2 * pi * chroma_freq * phases + (+0.5 * pi));
line0_V = cos(2 * pi * chroma_freq * phases + (+0.5 * pi));
line0 = line0_Y + line0_U - line0_V;

phases1 = 2 * pi * chroma_freq * phases + (0 * pi);

pal_modifier0 = 2.0 * sin(2.0 * (2 * pi * chroma_freq * (-32 : N - 1 + 32) + (+0.5 * pi)));
pal_modifier1 = 2.0 * sin(2.0 * (2 * pi * chroma_freq * (-32 : N - 1 + 32) + (+0.0 * pi)));
pal_modifier2 = 2.0 * sin(2.0 * (2 * pi * chroma_freq * (-32 : N - 1 + 32) + (-0.5 * pi)));

% Current line 
line1_Y = 0.5;
line1_U = sin(phases1);
line1_V = cos(phases1);
line1 = line1_Y + line1_U + line1_V;

% Next line 
line2_Y = 0.5;
line2_U = sin(2 * pi * chroma_freq * phases + (-0.5 * pi));
line2_V = cos(2 * pi * chroma_freq * phases + (-0.5 * pi));
line2 = line2_Y + line2_U - line2_V;

% PAL (figure 9.38)
function [Y, C] = pal_filter_9_38(line0, line1, line2, delay, bandpass, pal_modifier)
	line0 = conv(line0, bandpass) .* pal_modifier;
	line1 = conv(line1, delay);
	avg = 0.5 * (line0 + line1);

	avg = conv(avg, bandpass);
	line1 = conv(line1, delay);

	C = avg;
	Y = line1 - avg;
endfunction

% PAL (figure 9.39)
function [Y, C] = pal_filter_9_39(line0, line1, line2, delay, bandpass, pal_modifier)
	avg = 0.5 * (line2 - line0);
	avg = conv(avg, bandpass) .* pal_modifier;
	line1 = conv(line1, delay);

	avg = conv(avg, bandpass);
	line1 = conv(line1, delay);

	C = avg;
	Y = line1 - avg;
endfunction

function [Y, C] = pal_filter_9_44(line0, line1, line2, delay, bandpass, pal_modifier)
	% Luma output path
	Y_sub = line1 - 0.5 * (line0 + line2);
	line1 = conv(line1, delay);
	Y_sub = conv(Y_sub, bandpass);
	Y = line1 - Y_sub;

	% Chroma output path
	sub = 0.5 * (line2 - line0);
	sub = conv(sub, bandpass) .* pal_modifier;
	C = 0.5 * (sub + line1);

	Y = conv(delay, Y);
	C = conv(bandpass, C);
endfunction

[Y, C] = pal_filter_9_44(line0, line1, line2, delay, bandpass, pal_modifier2);
figure;
plot(Y);
hold on;
plot(C);

