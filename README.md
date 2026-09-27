gbcamextract
============

Extracts photos from Game Boy Camera / Pocket Camera saves. Frames can be preserved. The Hello Kitty camera is supported too.

## Usage

```console
gbcamextract [-r rom.gb] [--flat-pad-policy baseline|lookahead] \
  [--motion-candidate first-match|top-k8] -s save.sav
```

This will produce 30 AVIF files containing your photos. It is optional to specify the rom; this will allow the picture frames to be extracted too.

Each file is a lossless 8-bit monochrome AVIF still using
the four Game Boy grey levels (0, 85, 170, 255). Each file holds two
`av01` items in an `altr` group: the original-fidelity 160x144 raster
plus an 8x nearest-neighbor upscale (1280x1152) as the primary item, so
viewers display the large crisp version by default instead of
blurry-interpolating the tiny original. Both rasters are written by
a purpose-built lossless AV1 encoder (`src/mono.rs`): cost-based 64x64/32x32/16x16
partitions (NONE+palette, NONE+IntraBC copy where available, or SPLIT with
recursively chosen children, ranked by fractional-bit estimates; 16x16 leaves
compare palette vs copy, including an exact-byte lookup for genuinely uniform
blocks), each block coded `skip` with DC prediction plus a small luma
palette holding the block's exact colors (reusing the neighbours' palette
cache), or — where an exact copy exists in already-decoded area — as an
IntraBC block copy with an integer motion vector and no residuals. CDFs
adapt from the spec defaults. The unchanged greedy baseline is kept as a
whole-image fallback (smaller complete AVIF kept, baseline on ties). On typical photos the pair lands well under
the old 2-bit PNG sizes. Layered output (border
as background, photo as overlay) is not used: an `iovl` overlay is not
valid AVIF, and AVIF grids require tiles of at least 64x64px, so neither
renders in standard viewers.

Set `GBCAMEXTRACT_RDO_STATS=1` to print per-output and process-wide RDO,
IntraBC search, fallback, selection, error, and uniform-cache memory counts.
Per-output deltas are reported with a one-thread Rayon pool; aggregate counts
are reported at any thread count.

`--motion-candidate top-k8` is an opt-in lossless search candidate for the
large raster's RDO path. On nonuniform 16x16 blocks in a verified 8x image,
it prices up to eight legal exact IntraBC copies from the same incoming
coding state and emits the lowest-cost copy when it beats palette coding.
Every copy has zero residual. The default `first-match` policy preserves the
existing search. With `GBCAMEXTRACT_RDO_STATS=1`, motion shortlist probes,
priced matches, and committed alternate-match selections are reported.

Flat palette blocks keep the current padding rule by default. The optional
`--flat-pad-policy lookahead` mode tests the existing pad, cached colors, and
gray levels present in the rendered image, restricted to pads that keep the
used shade at the same numeric palette index as the baseline for that block's
incoming cache state. It scores the current block plus the next sibling
partition subtree, including any above/left palette reuse.
Every candidate is replayed from the same cost-only encoder snapshot; the
preview recomputes partition and copy choices and uses the baseline pad rule
to keep the horizon bounded. This is a one-step fractional-bit estimate, not
a global size optimum. The existing pad is the deterministic tie-break. Use
`GBCAMEXTRACT_RDO_STATS=1` to include committed flat-block counts, look-ahead
search counts, and estimated-cost outcomes.

## Building

```console
cargo build --release
```

The binary is at `target/release/gbcamextract`. No system libraries are
needed: the container is written with the pure-Rust `gamut-isobmff`
crate, the arithmetic coder comes from pure-Rust `gamut-bitstream`, and
argument parsing uses `clap`.


## License

Licensed under the expat license, which is sometimes called the MIT license.
See the `LICENSE` file for details.
