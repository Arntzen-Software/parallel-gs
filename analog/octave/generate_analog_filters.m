% SPDX-FileCopyrightText: 2026 Arntzen Software AS
% SPDX-FileContributor: Hans-Kristian Arntzen
% SPDX-License-Identifier: LGPL-3.0+

% Filter design for parallel-gs NTSC/PAL filtering.
pkg load signal;

% As defined by BT.601.
bt601_rate = 13.5e6;

% The subclock of PS2 seems to be 4x BT.601 rate.
subpixel_rate = 4 * bt601_rate;
subpixel_nyquist = subpixel_rate / 2;

ntsc_subcarrier = 315e6 / 88;
pal_subcarrier = 4.43361875e6;

% Grabbed some sensible values from a paper that discusses a physical NTSC/PAL impl in 1997,
% so it looks plausible that PS2 would ship something similar.
ntsc_luma_bw = 4.2e6;
ntsc_luma_stop = 6.75e6;
ntsc_chroma_bw = 0.8e6;
ntsc_chroma_stop = 1.5e6;

% PAL-B which was used in Norway.
pal_luma_bw = 5e6;
pal_luma_stop = 6.75e6;
pal_chroma_bw = 1.0e6;
pal_chroma_stop = 2.0e6;

% BT.1358 calls for 11.0 MHz until falloff start, but that's for progressive scan which scales everything up.
% These sampling rates assume interlaced video.
% For progressive scan which BT.1358 is defined for, the BT.601 rate of 13.5 MHz is doubled to 27 MHz.
% Chroma is half BW of luma as expected since sampling rate is 4:2:2 pattern.
ypbpr_luma_bw = 5.5e6;
ypbpr_luma_stop = 8e6;
ypbpr_chroma_bw = 2.8e6;
ypbpr_chroma_stop = 4e6;

% First stage of downsampling from 54 MHz to 27 MHz.
% Assume some zero-order hold behavior coming out of the CRTC due to its subclock behavior.
% Most likely it's just holding pixels until the horizontal divider clocks enough times.
% 640 * 4 = 2560
% 512 * 5 = 2560
% 256 * 10 = 2560
% etc ...
% Allow some aliasing to go through to keep filter length reasonable.
% We'll filter that away in the next pass.
ntsc_downsampling_filter = firls(14, [0, ntsc_luma_bw * 1.1, subpixel_nyquist * 0.7, subpixel_nyquist] / subpixel_nyquist, [1 1 0 0]);
pal_downsampling_filter = firls(14, [0, pal_luma_bw * 1.1, subpixel_nyquist * 0.7, subpixel_nyquist] / subpixel_nyquist, [1 1 0 0]);

% Encode filters at 27 MHz. Pure low-pass.
%luma_lp_encode = fir1(30, 0.3689, 'low', kaiser(31, 8.0))'
ntsc_luma_lp_encode = firls(30, [0, ntsc_luma_bw, ntsc_luma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);
ntsc_chroma_lp_encode = firls(30, [0, ntsc_chroma_bw, ntsc_chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);
pal_luma_lp_encode = firls(30, [0, pal_luma_bw, ntsc_luma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);
pal_chroma_lp_encode = firls(30, [0, pal_chroma_bw, ntsc_chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);

% Decoding composite is much harder.

% NTSC decode
% Suppress the luma DC which ends up at subcarrier freq after demodulating.
ntsc_chroma_lp_decode = firls(64 - 6, [0, ntsc_chroma_bw, ntsc_chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0])';

ntsc_subcarrier_w = ntsc_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * ntsc_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch' / sum(subcarrier_notch);
ntsc_chroma_lp_decode = conv(ntsc_chroma_lp_decode, subcarrier_notch);

% Suppress the carrier when decoding luma. This leads to less detail in luma. IIR would likely be better, but that's a massive pain on GPUs.

% Declaring vectors in Octave is super annoying :(
ntsc_bands = [0, ntsc_subcarrier - 1.3e6, ntsc_subcarrier - 0.1e6, ntsc_subcarrier + 0.1e6, ntsc_subcarrier + 1.3e6, bt601_rate];
% Don't suppress to zero since it rings a lot
ntsc_luma_lp_decode = firls(30, ntsc_bands / bt601_rate, [1, 1, 0.25, 0.25, 1, 1]);

% Suppress harmonics of subcarrier which arise when demodulating.
ntsc_subcarrier_w = 2.0 * ntsc_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * ntsc_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch' / sum(subcarrier_notch);
ntsc_chroma_lp_decode = conv(ntsc_chroma_lp_decode, subcarrier_notch)';

% PAL decode
% Just use different carrier freqs.
pal_chroma_lp_decode = firls(64 - 6, [0, pal_chroma_bw, pal_chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0])';

pal_subcarrier_w = pal_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * pal_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch' / sum(subcarrier_notch);
pal_chroma_lp_decode = conv(pal_chroma_lp_decode, subcarrier_notch);

pal_bands = [0, pal_subcarrier - 1.3e6, pal_subcarrier - 0.1e6, pal_subcarrier + 0.1e6, pal_subcarrier + 1.3e6, bt601_rate];

pal_luma_lp_decode = firls(64 - 4, [0, pal_luma_bw * 1.1, pal_luma_stop * 1.1, bt601_rate] / bt601_rate, [1 1 0 0]);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch' / sum(subcarrier_notch);
pal_luma_lp_decode = firls(30, pal_bands / bt601_rate, [1 1 0.25 0.25 1 1]);

% Suppress harmonics of subcarrier which arise when demodulating.
pal_subcarrier_w = 2.0 * pal_subcarrier / (2 * bt601_rate);
subcarrier_zero = exp(2 * pi * j * pal_subcarrier_w);
subcarrier_notch = real(conv([1, -subcarrier_zero], [1, -conj(subcarrier_zero)]));
subcarrier_notch = subcarrier_notch' / sum(subcarrier_notch);
pal_chroma_lp_decode = conv(pal_chroma_lp_decode, subcarrier_notch)';

ntsc_luma_lp_decode = ntsc_luma_lp_decode / sum(ntsc_luma_lp_decode);
pal_luma_lp_decode = pal_luma_lp_decode / sum(pal_luma_lp_decode);
ntsc_chroma_lp_decode = ntsc_chroma_lp_decode / sum(ntsc_chroma_lp_decode);
pal_chroma_lp_decode = pal_chroma_lp_decode / sum(pal_chroma_lp_decode);

% Normalize the filters so that DC gain is exactly 0.0 dB.
ntsc_downsampling_filter = ntsc_downsampling_filter / sum(ntsc_downsampling_filter);
ntsc_luma_lp_encode = ntsc_luma_lp_encode / sum(ntsc_luma_lp_encode);
ntsc_chroma_lp_encode = ntsc_chroma_lp_encode / sum(ntsc_chroma_lp_encode);
ntsc_luma_lp_decode = ntsc_luma_lp_decode / sum(ntsc_luma_lp_decode);
ntsc_chroma_lp_decode = ntsc_chroma_lp_decode / sum(ntsc_chroma_lp_decode);

% Bandpass filters for Y / C separation.
chroma_bandpass = firls(30, [0, pal_chroma_bw * 0.5, pal_chroma_stop * 0.75, bt601_rate] / bt601_rate, [1 1 0 0])';
chroma_bandpass = chroma_bandpass / sum(chroma_bandpass);
ntsc_chroma_bandpass = 2.0 * chroma_bandpass .* cos((2 * pi * ntsc_subcarrier / (bt601_rate * 2)) * (-15 : 15));
pal_chroma_bandpass = 2.0 * chroma_bandpass .* cos((2 * pi * pal_subcarrier / (bt601_rate * 2)) * (-15 : 15));

%h = figure();
%set(h, 'Name', 'NTSC chroma bandpass filter');
%freqz(ntsc_chroma_bandpass);

%h = figure();
%set(h, 'Name', 'PAL chroma bandpass filter');
%freqz(pal_chroma_bandpass);


pal_downsampling_filter = pal_downsampling_filter / sum(pal_downsampling_filter);
pal_luma_lp_encode = pal_luma_lp_encode / sum(pal_luma_lp_encode);
pal_chroma_lp_encode = pal_chroma_lp_encode / sum(pal_chroma_lp_encode);
pal_luma_lp_decode = pal_luma_lp_decode / sum(pal_luma_lp_decode);
pal_chroma_lp_decode = pal_chroma_lp_decode / sum(pal_chroma_lp_decode);

%h = figure();
%set(h, 'Name', 'NTSC downsample filter');
%freqz(ntsc_downsampling_filter);

%h = figure();
%set(h, 'Name', 'NTSC Y encode filter @ 27 MHz');
%freqz(ntsc_luma_lp_encode);

%h = figure();
%set(h, 'Name', 'NTSC UV encode filter @ 27 MHz');
%freqz(ntsc_chroma_lp_encode);

h = figure();
set(h, 'Name', 'NTSC Y decode filter @ 27 MHz');
freqz(ntsc_luma_lp_decode);

%h = figure();
%set(h, 'Name', 'NTSC UV decode filter @ 27 MHz');
%freqz(ntsc_chroma_lp_decode);

%h = figure();
%set(h, 'Name', 'PAL downsample filter @ 27 MHz');
%freqz(pal_downsampling_filter);

%h = figure();
%set(h, 'Name', 'PAL Y encode filter @ 27 MHz');
%freqz(pal_luma_lp_encode);

%h = figure();
%set(h, 'Name', 'PAL UV encode filter @ 27 MHz');
%freqz(pal_chroma_lp_encode);

h = figure();
set(h, 'Name', 'PAL Y decode filter @ 27 MHz');
freqz(pal_luma_lp_decode);

%h = figure();
%set(h, 'Name', 'PAL UV decode filter @ 27 MHz');
%freqz(pal_chroma_lp_decode);

ypbpr_downsampling_filter = firls(14, [0, ypbpr_luma_bw, subpixel_nyquist * 0.6, subpixel_nyquist] / subpixel_nyquist, [1 1 0 0]);
ypbpr_luma_filter = firls(30, [0, ypbpr_luma_bw, ypbpr_luma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);
ypbpr_chroma_filter = firls(30, [0, ypbpr_chroma_bw, ypbpr_chroma_stop, bt601_rate] / bt601_rate, [1 1 0 0]);

%h = figure();
%set(h, 'Name', 'YPbPr downsample filter');
%freqz(ypbpr_downsampling_filter);

%h = figure();
%set(h, 'Name', 'YPbPr luma filter @ 27 MHz');
%freqz(ypbpr_luma_filter);

%h = figure();
%set(h, 'Name', 'YPbPr chroma filter @ 27 MHz');
%freqz(ypbpr_chroma_filter);


assert(length(ntsc_downsampling_filter) == 15, "Length must be 15");
assert(length(ntsc_luma_lp_encode) == 31, "Length must be 31");
assert(length(ntsc_chroma_lp_encode) == 31, "Length must be 31");
assert(length(ntsc_luma_lp_decode) == 31, "Length must be 31");
assert(length(ntsc_chroma_lp_decode) == 63, "Length must be 63");
assert(length(ntsc_chroma_bandpass) == 31, "Length must be 31");

assert(length(pal_downsampling_filter) == 15, "Length must be 15");
assert(length(pal_luma_lp_encode) == 31, "Length must be 31");
assert(length(pal_chroma_lp_encode) == 31, "Length must be 31");
assert(length(pal_luma_lp_decode) == 31, "Length must be 63");
assert(length(pal_chroma_lp_decode) == 63, "Length must be 31");
assert(length(pal_chroma_bandpass) == 31, "Length must be 31");

assert(length(ypbpr_downsampling_filter) == 15, "Length must be 15");
assert(length(ypbpr_luma_filter) == 31, "Length must be 31");
assert(length(ypbpr_chroma_filter) == 31, "Length must be 31");

printf("// Autogenerated by generate_analog_filters.m\n");
printf("const float DownsamplingKernelNTSC[16] = float[](\n");
printf("    0.0,\n");
printf("    %.10f,\n", ntsc_downsampling_filter(1:14));
printf("    %.10f);\n", ntsc_downsampling_filter(15));
printf("\n");
printf("const float LumaEncodeNTSC[31] = float[](\n");
printf("    %.10f,\n", ntsc_luma_lp_encode(1:30));
printf("    %.10f);\n", ntsc_luma_lp_encode(31));
printf("\n");
printf("const float ChromaEncodeNTSC[31] = float[](\n");
printf("    %.10f,\n", ntsc_chroma_lp_encode(1:30));
printf("    %.10f);\n", ntsc_chroma_lp_encode(31));
printf("\n");
printf("const float LumaDecodeNTSC[31] = float[](\n");
printf("    %.10f,\n", ntsc_luma_lp_decode(1:30));
printf("    %.10f);\n", ntsc_luma_lp_decode(31));
printf("\n");
printf("const float ChromaDecodeNTSC[63] = float[](\n");
printf("    %.10f,\n", ntsc_chroma_lp_decode(1:62));
printf("    %.10f);\n", ntsc_chroma_lp_decode(63));
printf("\n");
printf("const float DownsamplingKernelPAL[16] = float[](\n");
printf("    0.0,\n");
printf("    %.10f,\n", pal_downsampling_filter(1:14));
printf("    %.10f);\n", pal_downsampling_filter(15));
printf("\n");
printf("const float LumaEncodePAL[31] = float[](\n");
printf("    %.10f,\n", pal_luma_lp_encode(1:30));
printf("    %.10f);\n", pal_luma_lp_encode(31));
printf("\n");
printf("const float ChromaEncodePAL[31] = float[](\n");
printf("    %.10f,\n", pal_chroma_lp_encode(1:30));
printf("    %.10f);\n", pal_chroma_lp_encode(31));
printf("\n");
printf("const float LumaDecodePAL[31] = float[](\n");
printf("    %.8f,\n", pal_luma_lp_decode(1:30));
printf("    %.8f);\n", pal_luma_lp_decode(31));
printf("\n");
printf("const float ChromaDecodePAL[63] = float[](\n");
printf("    %.8f,\n", pal_chroma_lp_decode(1:62));
printf("    %.8f);\n", pal_chroma_lp_decode(63));
printf("\n");
printf("const float ChromaBandpassNTSC[31] = float[](\n");
printf("    %.10f,\n", ntsc_chroma_bandpass(1:30));
printf("    %.10f);\n", ntsc_chroma_bandpass(31));
printf("\n");
printf("const float ChromaBandpassPAL[31] = float[](\n");
printf("    %.10f,\n", pal_chroma_bandpass(1:30));
printf("    %.10f);\n", pal_chroma_bandpass(31));
printf("const float DownsamplingKernelYPbPr[16] = float[](\n");
printf("    0.0,\n");
printf("    %.10f,\n", ypbpr_downsampling_filter(1:14));
printf("    %.10f);\n", ypbpr_downsampling_filter(15));
printf("\n");
printf("const float LumaEncodeYPbPr[31] = float[](\n");
printf("    %.10f,\n", ypbpr_luma_filter(1:30));
printf("    %.10f);\n", ypbpr_luma_filter(31));
printf("\n");
printf("const float ChromaEncodeYPbPr[31] = float[](\n");
printf("    %.10f,\n", ypbpr_chroma_filter(1:30));
printf("    %.10f);\n", ypbpr_chroma_filter(31));
printf("\n");

printf("\n");

printf("\n");

printf("////////\n");

