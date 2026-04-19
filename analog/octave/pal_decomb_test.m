% SPDX-FileCopyrightText: 2026 Arntzen Software AS
% SPDX-FileContributor: Hans-Kristian Arntzen
% SPDX-License-Identifier: LGPL-3.0+

N = 2000;
lines = 625;

pal_subcarrier = 4.43361875e6;
sample_rate = 27e6;

samples_per_cycle = sample_rate / pal_subcarrier;

horiz_w = 2 * pi / samples_per_cycle;
phases = horiz_w * conv2(0 : N - 1, ones(lines, 1));
phase_offset_per_line = 0.75 + 1.0 / 625.0;
phase_offsets = 2 * pi * phase_offset_per_line * conv2([0 : lines - 1]', ones(1, N));
phases = phases + phase_offsets;

% Sample signal
U_amp = 1.0;
V_amp = 1.0;
U = U_amp * sin(phases);
V = V_amp * cos(phases);
% Phase alternation
V(2 : 2 : end, :) = -V(2 : 2 : end, :);

C = U + V;

sampling_offset = (1.0 - phase_offset_per_line) * samples_per_cycle;
horiz_lerp = sampling_offset - floor(sampling_offset);
e = 0.041 / 4;

%horiz_filt = [-0.5 * horiz_lerp, -0.5 * (1 - horiz_lerp), 1, -0.5 * (1 - horiz_lerp), -0.5 * horiz_lerp];
horiz_filt = [-0.25 - e, -0.25 + e, 1, -0.25 + e, -0.25 - e];
vert_filt = [-0.5; 1; -0.5];
filt = conv2(horiz_filt, vert_filt);

Cs = conv2(C, filt);
Cs = Cs(3 : end - 2, 5 : end - 4);

Combined = U + V;
Combined = Combined(2 : end - 1, 3 : end - 2);

Error = Combined - Cs;
MaxAbsError = max(max(abs(Error)))
StdDevError = sqrt(sum(sum(Error .* Error)) / (size(Combined)(1) * size(Combined)(2)))

Us = conv2(U, filt);
Us = Us(3 : end - 2, 5 : end - 4);
ErrorU = U(2 : end - 1, 3 : end - 2) - Us;
MaxAbsErrorU = max(max(abs(ErrorU)))
StdDevErrorU = sqrt(sum(sum(ErrorU .* ErrorU)) / (size(Us)(1) * size(Us)(2)))

Vs = conv2(V, filt);
Vs = Vs(3 : end - 2, 5 : end - 4);
ErrorV = V(2 : end - 1, 3 : end - 2) - Vs;
MaxAbsErrorV = max(max(abs(ErrorV)))
StdDevErrorV = sqrt(sum(sum(ErrorV .* ErrorV)) / (size(Vs)(1) * size(Vs)(2)))

