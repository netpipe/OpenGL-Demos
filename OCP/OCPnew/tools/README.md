# OCP logo checkers

The band shape lives in `../ocp_logo_geom.h` and nowhere else. `../test.cpp`
and both checkers here include that header, so a measurement taken here is a
measurement of what the demo actually draws. Keeping a second, offline copy of
the geometry is what let f602603 ship a notch that measured open offline and
rendered sealed on screen.

## Does it match the client logo?

```sh
powershell -File topgm.ps1 -Path ../../ocp-2.jpg -Out ref.pgm
cl /EHsc /O2 /D_CRT_SECURE_NO_WARNINGS verify_logo.cpp
verify_logo ref.pgm diff.ppm render.ppm
```

Rasterises the logo pose (bands on sides 0 / 5 / 5) at the reference image's
scale and diffs it against `ocp-2.jpg`. Report `structural mismatch` is the
number that matters: it counts disagreeing pixels that are *not* explained by
an edge landing one pixel over, so a wrong stem or notch shows up there while
antialiasing does not.

Current geometry: 243 px raw mismatch, **4 px structural**, out of 21775 ink
pixels. The rest is the reference's own irregularity — its eight sides sit at
apothems spanning 0.37 px (see `radial.ps1`), so no regular octagon can match
it exactly.

`diff.ppm` marks logo-only pixels red and render-only pixels blue.
`render.ppm` is the raw rasterised shape. Convert either with `ppm2png.ps1`.

## Can the dot still get out?

```sh
cl /EHsc /O2 verify_escape.cpp
verify_escape
```

The logo's notch is not a radial slot: the corridor between the stem and the
bottom bar runs parallel to local side 7's flat, so the way out is a dogleg
through the C opening and then along the corridor. This flood-fills a grid with
the dot's radius to confirm it fits through that turn, and asserts that all 8
aligned poses escape while all 56 misaligned poses stay sealed.

## Is the binary on screen actually built from this source?

A screenshot of a stale build looks exactly like a geometry bug, so check the
binary before touching the shape:

```sh
powershell -File fromshot.ps1 -Path shot.png -Out live.pgm `
    -CropX0 200 -CropY0 85 -CropX1 850 -CropY1 554
powershell -File diffpgm.ps1 -A ref.pgm -B live.pgm -Out diff_live.png
```

`fromshot.ps1` finds the logo's ink box inside the crop (pass `-Profile` to
print per-row ink extents and pick a crop that excludes the HUD) and resamples
it into the reference's 195x195 frame, so the result is directly comparable to
`ref.pgm` and to `render.ppm`. `diffpgm.ps1` reports the same raw/structural
split as `verify_logo` and writes a red/blue overlay.

If `live.pgm` disagrees with `render.ppm` from the current source, the running
binary is not this source. That is what the 2026-09-27 report turned out to be.

## Measuring a reference image

- `runs.ps1` — black pixel runs per row, for reading off band edges
- `edge.ps1` — subpixel 50% crossings along one row or column
- `radial.ps1` — apothem of every band boundary along a given direction
- `zoom.ps1` — nearest-neighbour blow-up with a pixel grid
- `compose.ps1` — side-by-side sheet from several images
- `ppm2png.ps1` — binary PPM or PGM to PNG, with optional nearest-neighbour scale
