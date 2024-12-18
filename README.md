# paraLLEl-GS

paraLLEl-GS emulates the PlayStation 2 Graphics Synthesizer using Vulkan compute shaders.
It is similar in spirit to paraLLEl-RDP, with different implementation trade-offs.

The end goal is to be a no-compromises PS2 graphics emulation,
i.e., retain the accuracy of a CPU software renderer while supporting upscaling / super-sampling and be
fast enough to do so on modest GPU hardware.
Unfortunately, while N64 has the extremely accurate Angrylion implementation that can be tested against,
I don't believe GSdx's software renderer is bit-wise accurate with real GS hardware, so
paraLLEl-GS currently does not aim for hardware bit-wise accuracy on varying interpolation.
Extremely detailed hardware tests would need to be written to reverse the exact behavior.
To my knowledge, such tests don't exist, at least not publicly.

It is a completely standalone implementation from scratch, and does not use GSdx from PCSX2.
The GS dump format is used to make debugging and triaging issues easier, but that is only relevant for development.

## Features

- 2x / 4x / 8x / 16x SSAA. More than 8x is arguably complete overkill, but it's there.
- Weave de-interlacer (could certainly be better)
- Auto-promotion to progressive scan for FFMD = 0
- CRTC field blending (and ability to turn the blur off)
- AA1 handling
- Lots of mitigation for bad up-sampling behavior

Generally, I tend to prefer super-sampling over straight up-scaling on SD content.
The mixing of SD UI elements and ultra-sharp polygon edges looks quite jarring to me.
Super-sampling also works much better with CRT emulation.
To upscale the anti-aliased content to screen, AMD FSR1 + RCAS can be used, and does a decent job here.

### Known missing features

- AA1 implementation is questionable. There are many details which are unknown to me how it's supposed to work exactly.

## Implementation details

This is best left to blog posts.

## Tested GPU / driver combos.

- RADV on RX 7600/6800.
- AMDVLK on RX 7600/6800.
- amdgpu-pro on RX 7600/6800.
- Steam Deck on SteamOS 3.6.6 / 3.7.
- RTX 4070 on Linux and Windows.
- RTX 3060 mobile on a Windows 11 laptop.
- Intel UHD 620 on Windows 11 (but obviously too slow for it to be practical).
- Arc A770/B580 on Mesa 24.3.1.
- Arc A770/B580 on Windows 10.

## Required driver features

- `descriptorIndexing`
- `timelineSemaphore`
- `storageBuffer8BitAccess`
- `storageBuffer16BitAccess`
- `shaderInt16`
- `scalarBlockLayout`
- Full subgroup support (minus clustered)
- Subgroup size control with full groups between 16 and 64 threads per subgroup

This should not be a problem for any desktop driver or somewhat modern mobile GPU.

## Contributors

- Hans-Kristian "themaister" Arntzen
- Runar Heyer

Runar's contributions were done as paid work for my company Arntzen Software AS as an employee.
He did the early study, studied game behavior, wrote a lot of tests,
implemented most of the PS2-specific details in `ubershader.comp`,
and implemented most of VRAM upload / texture caching shaders.

Most of the non-shader code was rewritten after the initial prototype implementation with a lot of hindsight from that earlier work.

## PCSX2 integration

The PCSX2 integration is early days and very experimental / hacky. An Arch Linux PKGBUILD can be found in `misc/`.
To build with Visual Studio, apply the Git patches manually and checkout parallel-gs in the correct folder (`pcsx2/GS/parallel-gs`).

There is very basic UI integration. As API, paraLLEl-GS can be chosen. The super-sampling rate can be modified.
Under display settings, some options are honored:

- Bilinear filtering. The sharp bilinear option uses FSR1 + RCAS.
- Anti-Blur
- Screen Offsets
- Show Overscan
- Integer Scaling

Save states seem to work, and GS dumps also works.
OSD does *not* work in the current integration.

## Dump format

The primary way to debug paraLLEl-GS is through dumps.

With `parallel-gs-replayer`, a PCSX2 generated dump (current upstream, version 8) can be replayed.
RenderDoc should be attached, and a RenderDoc capture covering the entire dump will be made automatically.

With `parallel-gs-stream`, a raw dump format that skips the header can be used.
See `misc/` for a hacky patch for PCSX2 that allows the use of `mkfifo` to test the renderer in complete isolation in real-time.
E.g.:

```
mkfifo /tmp/gs.stream
parallel-gs-stream /tmp/gs.stream

# Run some game and parallel-gs-stream should start pumping out frames when PCSX2 starts generating GS commands.
GS_STREAM=/tmp/gs.stream pcsx2
```

`parallel-gs-stream` can pause the emulation, step frames, and trigger captures on its own when RenderDoc is attached.

## Debugging

paraLLEl-GS can emit labels and regions which makes it easy to step through primitives being drawn.
The primary debugger is RenderDoc.

## License

The license for current code is LGPLv3+, but dual-licensing may be possible.

## Contributions

External contributions are currently not accepted. This may be relaxed eventually.
This repository is public only to facilitate testing and debug for the time being.
