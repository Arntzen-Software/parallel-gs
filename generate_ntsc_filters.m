pkg load signal;

bt601_rate = 13.5e6;

% The subclock of PS2 seems to be 4x BT.601 rate.
subpixel_rate = 4 * bt601_rate;
subpixel_nyquist = subpixel_rate / 2;

ntsc_subcarrier = 315e6 / 88;
pal_subcarrier = 4.43361875e6;

% Grabbed some sensible values from a paper that discusses a physical NTSC/PAL impl in 1997,
% so it looks plausible that PS2 would ship something similar.
luma_bw = 4.2e6;
luma_stop = bt601_rate / 2;
chroma_bw = 1.3e6;

% The chroma stop band + ntsc_subcarrier should stay inside the 13.5 MHz nyquist frequency of 6.75 MHz
chroma_stop = 2.2e6;

% First stage of downsampling from 54 MHz to 27 MHz.
% Assume some zero-order hold behavior coming out of the CRTC due to its subclock behavior.
% Most likely it's just holding pixels until the horizontal divider clocks enough times.
% 640 * 4 = 2560
% 512 * 5 = 2560
% 256 * 10 = 2560
% etc ...
downsampling_filter = firls(14, [0, luma_bw * 1.2, subpixel_nyquist * 0.7, subpixel_nyquist] / subpixel_nyquist, [1 1 0 0]);
%h = figure();
%set(h, 'Name', '54 MHz to 27 MHz downsampling filter');
%freqz(downsampling_filter);

% Next stage of filtering. Create a proper BT.601 signal that is sampled at 13.5 MHz.
luma_downsampling = firls(30, [0, luma_bw, luma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);
%h = figure();
%set(h, 'Name', '27 MHz to 13.5 MHz luma downsampling filter');
%freqz(luma_downsampling);

% Chroma filter at 13.5 MHz.
chroma_lp = firls(30, [0, chroma_bw, chroma_stop, 0.5 * bt601_rate] / (0.5 * bt601_rate), [1 1 0 0]);

% Luma decode filter at 27 MHz.
luma_lp_decode = fir1(30, 0.3689, 'low', kaiser(31, 8.0));

% Chroma decode filter at 27 MHz.
chroma_lp_decode = firls(26, [0, chroma_bw, chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);

% Suppress harmonics of subcarrier which arise when demodulating.
ntsc_subcarrier_w = ntsc_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * ntsc_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch / sum(subcarrier_notch);

chroma_lp_decode = conv(chroma_lp_decode, subcarrier_notch);

% Suppress the carrier.
subcarrier_notch = real(conv([1, -0.95 * subcarrier_zero], [1, -conj(0.95 * subcarrier_zero)]));
subcarrier_notch = subcarrier_notch / sum(subcarrier_notch);
luma_lp_decode = conv(luma_lp_decode, subcarrier_notch);

ntsc_subcarrier_w = 2.0 * ntsc_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * ntsc_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch / sum(subcarrier_notch);
chroma_lp_decode = conv(chroma_lp_decode, subcarrier_notch);

h = figure();
set(h, 'Name', 'NTSC UV filter @ 13.5 MHz');
freqz(chroma_lp);

h = figure();
set(h, 'Name', 'NTSC Y filter @ 27 MHz');
freqz(luma_lp_decode);

h = figure();
set(h, 'Name', 'NTSC UV filter @ 27 MHz');
freqz(chroma_lp_decode);

% Upsample chroma and luma.
upsampling = fir1(30, 0.3689, 'low', kaiser(31, 6.0));
h = figure();
set(h, 'Name', 'Upsampling filter: 13.5 MHz -> 27 MHz');
freqz(upsampling);
ylim([-100 0]);

% Test some signals.

F = 4.2e6;
y = 0.5 + 0.5 * sin((2 * pi * F / subpixel_rate) * [1 : 8 * 1024]);
u = 0.0 + 0.0 * sin((2 * pi * F / subpixel_rate) * [1 : 8 * 1024]);
v = 0.0 + 0.0 * sin((2 * pi * F / subpixel_rate) * [1 : 8 * 1024]);

% Filter and decimate.
y = conv(y, downsampling_filter)(:, 1:2:end);
u = conv(u, downsampling_filter)(:, 1:2:end);
v = conv(v, downsampling_filter)(:, 1:2:end);

y = conv(y, luma_downsampling)(:, 1:2:end);
u = conv(u, luma_downsampling)(:, 1:2:end);
v = conv(v, luma_downsampling)(:, 1:2:end);
u = conv(u, chroma_lp);
v = conv(v, chroma_lp);

% Modulating chroma before upsample would require a really sharp filter in upscale phase,
% so better to do it after.

% Now we're at 13.5 MHz. Interpolate up to 27 MHz.
y = kron(y, [2, 0]);
u = kron(u, [2, 0]);
v = kron(v, [2, 0]);

y = conv(y, upsampling);
u = conv(u, upsampling);
v = conv(v, upsampling);

u = u .* cos((2 * pi * ntsc_subcarrier / (2 * bt601_rate)) * [0 : length(u) - 1]);
v = v .* sin((2 * pi * ntsc_subcarrier / (2 * bt601_rate)) * [0 : length(v) - 1]);
c = u + v;

% Reconstruct YUV

composite_signal = [y, zeros(1, length(c) - length(y))] + c;

y = conv(luma_lp_decode, composite_signal);
u = 2 * conv(chroma_lp_decode, composite_signal .* cos((2 * pi * ntsc_subcarrier / (2 * bt601_rate)) * [0 : length(composite_signal) - 1]));
v = 2 * conv(chroma_lp_decode, composite_signal .* sin((2 * pi * ntsc_subcarrier / (2 * bt601_rate)) * [0 : length(composite_signal) - 1]));

h = figure();
set(h, 'Name', 'Composite signal')
plot(composite_signal);

h = figure();
set(h, 'Name', 'Reconstructed Y')
plot(y);

h = figure();
set(h, 'Name', 'Reconstructed U')
plot(u);

h = figure();
set(h, 'Name', 'Reconstructed V')
plot(v);

frequ = 20.0 * log10(abs(fft(u .* kaiser(length(u), 8.0)')));
freqv = 20.0 * log10(abs(fft(v .* kaiser(length(v), 8.0)')));
h = figure();
set(h, 'Name', 'U response')
plot((2 * bt601_rate / 1e6) * [0 : length(frequ) / 2 - 1] / length(frequ), frequ(:, 1 : length(frequ) / 2));
xlabel('Freq (MHz)');
ylabel('Response (dB)');



