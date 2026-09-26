use clap::Parser;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process;

mod mono;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 144;
const SAVE_SIZE: usize = 128 * 1024;
const ROM_SIZE: usize = 1024 * 1024;
const BANK_SIZE: usize = 0x4000;

// Output is one AVIF still per photo with two `av01` items in an `altr`
// group: the 8x nearest-neighbor upscale (1280x1152, primary, displayed by
// default) plus the original-fidelity 160x144 raster.
// (A 'grid' of small tiles would isolate border from photo, but MIAF
// requires grid tiles to be at least 64x64px, and an 'iovl' overlay is
// not valid AVIF at all, so neither renders in standard viewers.)
const CANVAS_PIXELS: usize = WIDTH as usize * HEIGHT as usize;

// Fixed upscale factor for the display-size item. 8 divides the 16px
// coding block, so upscale block boundaries stay aligned to source pixels.
const SCALE: u32 = 8;
const LARGE_W: u32 = WIDTH * SCALE;
const LARGE_H: u32 = HEIGHT * SCALE;

const PHOTO_X: u32 = 16;
const PHOTO_Y: u32 = 16;
const PHOTO_W: u32 = 128;
const PHOTO_H: u32 = 112;

// AV1 has no 2-bit depth (8-bit minimum), so the four Game Boy shades are
// stored as evenly spaced 8-bit gray levels (R=G=B). Encoding is lossless,
// so these exact values survive the round trip.
const GRAY_8BIT: [u8; 4] = [0, 85, 170, 255];

const HELLO_KITTY_FRAME_OFFSETS: [[usize; 2]; 25] = [
    [0xC6C70, 0xCF5D0],
    [0xC3B80, 0xCF548],
    [0xCBEC0, 0xCF4C0],
    [0xC5F10, 0xCF658],
    [0xCF210, 0xCF7F0],
    [0xC73A0, 0xCF768],
    [0xB7420, 0xCF6E0],
    [0xBE3E0, 0xCF438],
    [0xB3CD0, 0xC7EF0],
    [0xB2B80, 0xCF3B0],
    [0x8FD50, 0xC7F78],
    [0xC3800, 0xD7800],
    [0xBDC00, 0xD3F70],
    [0xD7F70, 0xD7888],
    [0xC5C00, 0xD7998],
    [0xB7C20, 0xD7910],
    [0xC3ED0, 0xD3D50],
    [0x33F80, 0xD3CC8],
    [0xDB800, 0xD3DD8],
    [0xB2200, 0xD3EE8],
    [0xB34D0, 0xD3E60],
    [0xB3030, 0xD7A20],
    [0x93E00, 0xD7D50],
    [0x77FE0, 0xCFCB8],
    [0x77FF0, 0xCFDC4],
];

#[derive(Parser)]
#[command(
    name = "gbcamextract",
    version = "1.1",
    about = "Extracts photos from Game Boy Camera / Pocket Camera saves"
)]
struct Args {
    #[arg(short = 's', value_name = "save.sav")]
    save: PathBuf,
    #[arg(short = 'r', value_name = "rom.gb")]
    rom: Option<PathBuf>,
}

fn is_gb_rom(data: &[u8]) -> bool {
    data.len() >= 0x108 && data[0x104..0x108] == [0xce, 0xed, 0x66, 0x66]
}

fn is_hk_rom(rom: &[u8]) -> bool {
    rom.len() >= 0x134 + 15 && rom[0x134..0x134 + 15] == *b"POCKETCAMERA_SN"
}

fn get_pic_num_for_slot_num(save: &[u8], slot_num: usize) -> Option<u32> {
    if !(1..=30).contains(&slot_num) {
        return None;
    }
    let vec_base = 0x11b2usize;
    let pic_num = *save.get(vec_base + slot_num - 1)?;
    match pic_num {
        0..=29 => Some(pic_num as u32 + 1),
        _ => None,
    }
}

fn slot_num_to_base_address(slot_num: usize) -> usize {
    (slot_num + 1) * 0x1000
}

struct FrameInfo {
    addr: usize,
    idx: usize,
    is_hk: bool,
}

fn frame_base_address(rom: &[u8], frame_number: i32) -> (usize, usize) {
    if is_hk_rom(rom) {
        let idx = if (0..25).contains(&frame_number) {
            frame_number as usize
        } else {
            24
        };
        (HELLO_KITTY_FRAME_OFFSETS[idx][0], idx)
    } else {
        let idx = if (0..18).contains(&frame_number) {
            frame_number as usize
        } else {
            13
        };
        let addr = if idx < 9 {
            BANK_SIZE * 0x34 + idx * 0x688
        } else {
            BANK_SIZE * 0x35 + (idx - 9) * 0x688
        };
        (addr, idx)
    }
}

/// Decode one 8x8 Game Boy 2bpp tile into 64 shade values (0-3).
/// The bytes are inverted on load: a raw value of 0 (lightest) becomes 3.
fn decode_gb_tile(raw: &[u8; 16]) -> [u8; 64] {
    let mut px = [0u8; 64];
    for row in 0..8 {
        let low = raw[row * 2];
        let high = raw[row * 2 + 1];
        for col in 0..8 {
            let lb = (low >> (7 - col)) & 1;
            let hb = (high >> (7 - col)) & 1;
            px[row * 8 + col] = 3 - ((hb << 1) | lb);
        }
    }
    px
}

/// Raw 16 bytes for the border 8x8 block at canvas block coords
/// (`bx` in 0..20, `by` in 0..18), or `None` for the photo area.
/// Corners belong to the top/bottom strips, matching the historical draw
/// order (sides first, top/bottom over them).
fn border_raw_tile(rom: &[u8], frame: &FrameInfo, bx: usize, by: usize) -> Option<[u8; 16]> {
    let x = bx * 8;
    let y = by * 8;
    let map_off = if (16..128).contains(&y) && !(16..144).contains(&x) {
        // Side strips.
        let y_tile = (y - 16) / 8;
        let z = (bx % 2) + if x >= 144 { 2 } else { 0 };
        if frame.is_hk {
            HELLO_KITTY_FRAME_OFFSETS[frame.idx][1] + 0x50 + y_tile * 4 + z
        } else {
            frame.addr + 0x650 + y_tile * 4 + z
        }
    } else if !(16..128).contains(&y) {
        // Top/bottom strips (corners included).
        let x_tile = bx;
        let z = if y < 16 { by } else { by - 14 };
        if frame.is_hk {
            HELLO_KITTY_FRAME_OFFSETS[frame.idx][1] + x_tile + 0x14 * z
        } else {
            frame.addr + 0x600 + x_tile + 0x14 * z
        }
    } else {
        return None;
    };
    let tile_num = *rom.get(map_off)? as usize;
    let tile_off = frame.addr + tile_num * 16;
    let bytes = rom.get(tile_off..tile_off + 16)?;
    let mut raw = [0u8; 16];
    raw.copy_from_slice(bytes);
    Some(raw)
}

/// 2-bit pixels of the 8x8 canvas block at (`bx`, `by`).
/// Missing data (no ROM, or a ROM lookup outside the file) is black,
/// matching the historical behaviour of leaving the zeroed buffer alone.
fn canvas_block(
    rom: Option<(&[u8], &FrameInfo)>,
    save: &[u8],
    base_address: usize,
    bx: usize,
    by: usize,
) -> [u8; 64] {
    let x = (bx * 8) as u32;
    let y = (by * 8) as u32;
    if (PHOTO_X..PHOTO_X + PHOTO_W).contains(&x) && (PHOTO_Y..PHOTO_Y + PHOTO_H).contains(&y) {
        let i = ((x - PHOTO_X) / 8) as usize;
        let j = ((y - PHOTO_Y) / 8) as usize;
        let off = base_address + j * 256 + i * 16;
        match save.get(off..off + 16) {
            Some(b) => decode_gb_tile(b.try_into().unwrap()),
            None => [0u8; 64],
        }
    } else if let Some((rom_data, frame)) = rom {
        match border_raw_tile(rom_data, frame, bx, by) {
            Some(raw) => decode_gb_tile(&raw),
            None => [0u8; 64],
        }
    } else {
        [0u8; 64]
    }
}

/// Full-canvas 8-bit gray pixels, row-major.
fn render_photo(rom: Option<(&[u8], &FrameInfo)>, save: &[u8], base_address: usize) -> Vec<u8> {
    let mut out = vec![0u8; CANVAS_PIXELS];
    for by in 0..18 {
        for bx in 0..20 {
            let block = canvas_block(rom, save, base_address, bx, by);
            for r in 0..8 {
                for c in 0..8 {
                    out[(by * 8 + r) * WIDTH as usize + bx * 8 + c] =
                        GRAY_8BIT[block[r * 8 + c] as usize];
                }
            }
        }
    }
    out
}

/// Nearest-neighbor upscale by an integer `scale` (each source pixel
/// becomes a `scale`x`scale` block of the same value).
fn upscale_nearest(gray: &[u8], w: u32, h: u32, scale: u32) -> Vec<u8> {
    let (w, h, scale) = (w as usize, h as usize, scale as usize);
    let mut out = vec![0u8; w * scale * h * scale];
    for y in 0..h {
        for x in 0..w {
            let v = gray[y * w + x];
            for dy in 0..scale {
                let base = (y * scale + dy) * w * scale + x * scale;
                out[base..base + scale].fill(v);
            }
        }
    }
    out
}

/// Every setting or property that changes the serialized AVIF identity for
/// one rendered raster. The complete border/photo pixels are compared
/// separately, byte-for-byte, after the hash lookup.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct EncodeIdentity {
    small_w: u32,
    small_h: u32,
    scale: u32,
    large_w: u32,
    large_h: u32,
    gray_levels: [u8; 4],
    intrabc: bool,
    rdo: bool,
    brands: [[u8; 4]; 3],
    primary_item_id: u32,
    large_item_id: u32,
    small_item_id: u32,
    alternative_group_type: [u8; 4],
    alternative_group_id: u32,
}

impl EncodeIdentity {
    fn current() -> Self {
        Self {
            small_w: WIDTH,
            small_h: HEIGHT,
            scale: SCALE,
            large_w: LARGE_W,
            large_h: LARGE_H,
            gray_levels: GRAY_8BIT,
            intrabc: mono::USE_INTRABC,
            rdo: true,
            brands: [*b"avif", *b"mif1", *b"miaf"],
            primary_item_id: 1,
            large_item_id: 1,
            small_item_id: 2,
            alternative_group_type: *b"altr",
            alternative_group_id: 10,
        }
    }
}

struct RasterGroup {
    identity: EncodeIdentity,
    gray: Vec<u8>,
    slots: Vec<usize>,
}

fn raster_hash(identity: &EncodeIdentity, gray: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    identity.hash(&mut hasher);
    gray.hash(&mut hasher);
    hasher.finish()
}

fn group_rendered_by<F>(rasters: Vec<(usize, EncodeIdentity, Vec<u8>)>, hash: F) -> Vec<RasterGroup>
where
    F: Fn(&EncodeIdentity, &[u8]) -> u64,
{
    let mut groups: Vec<RasterGroup> = Vec::new();
    let mut buckets: HashMap<(EncodeIdentity, u64), Vec<usize>> = HashMap::new();
    for (slot, identity, gray) in rasters {
        let digest = hash(&identity, &gray);
        let key = (identity, digest);
        let existing = buckets.get(&key).and_then(|indices| {
            indices
                .iter()
                .copied()
                .find(|&index| groups[index].gray == gray)
        });
        if let Some(index) = existing {
            groups[index].slots.push(slot);
        } else {
            let index = groups.len();
            groups.push(RasterGroup {
                identity,
                gray,
                slots: vec![slot],
            });
            buckets.entry(key).or_default().push(index);
        }
    }
    groups
}

fn group_rendered(rasters: Vec<(usize, Vec<u8>)>) -> Vec<RasterGroup> {
    let identity = EncodeIdentity::current();
    group_rendered_by(
        rasters
            .into_iter()
            .map(|(slot, gray)| (slot, identity, gray))
            .collect(),
        raster_hash,
    )
}

fn run(args: Args) -> Result<(), String> {
    let report_rdo_stats = std::env::var_os("GBCAMEXTRACT_RDO_STATS").is_some();
    let save = std::fs::read(&args.save).map_err(|e| {
        format!(
            "couldn't open save '{}' for reading: {e}",
            args.save.display()
        )
    })?;
    if save.len() != SAVE_SIZE {
        return Err(format!(
            "save '{}' has weird size: expected {SAVE_SIZE} bytes, got {}",
            args.save.display(),
            save.len()
        ));
    }
    if is_gb_rom(&save) {
        return Err(format!(
            "save '{}': save expected, but rom was given",
            args.save.display()
        ));
    }

    let rom: Option<Vec<u8>> = match &args.rom {
        Some(path) => {
            let data = std::fs::read(path)
                .map_err(|e| format!("couldn't open rom '{}' for reading: {e}", path.display()))?;
            if data.len() != ROM_SIZE {
                return Err(format!(
                    "rom '{}' has weird size: expected {ROM_SIZE} bytes, got {}",
                    path.display(),
                    data.len()
                ));
            }
            if !is_gb_rom(&data) {
                return Err(format!(
                    "rom '{}' doesn't look like a real rom",
                    path.display()
                ));
            }
            Some(data)
        }
        None => None,
    };
    let rom_ref: Option<&[u8]> = rom.as_deref();

    // Precompute all output filenames before encoding anything. Every
    // in-range byte in the slot-number table is otherwise trusted as a
    // unique output number, so two physical slots sharing one metadata
    // number target the same filename and `std::fs::write` silently
    // truncates the earlier slot's photo while reporting success.
    let filenames: Vec<String> = (1..=30usize)
        .map(|slot_num| match get_pic_num_for_slot_num(&save, slot_num) {
            Some(n) => format!("IMG_{:02}.avif", n),
            None => format!("DEL_{:02}.avif", slot_num),
        })
        .collect();

    // Intra-run collisions are fatal: damaged or uninitialized metadata
    // must not lose photos silently. Report every duplicated name with
    // the slots that share it, then exit before encoding or writing
    // anything (no partial outputs).
    {
        use std::collections::HashMap;
        let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, name) in filenames.iter().enumerate() {
            by_name.entry(name.as_str()).or_default().push(idx + 1);
        }
        let mut duplicates: Vec<(&str, Vec<usize>)> = by_name
            .into_iter()
            .filter(|(_, slots)| slots.len() > 1)
            .collect();
        if !duplicates.is_empty() {
            duplicates.sort_unstable();
            let mut msg = String::new();
            for (name, slots) in &duplicates {
                if !msg.is_empty() {
                    msg.push_str("; ");
                }
                msg.push_str(&format!(
                    "duplicate output filename {} from slots {}: \
                     photo numbers in save are not unique; \
                     refusing to overwrite extracted slots in the same run",
                    name,
                    slots
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                ));
            }
            return Err(msg);
        }
    }
    // Policy for files left over from previous runs: overwrite them in
    // place via `std::fs::write` below. That is distinct from the
    // intra-run collision check above, which always fails instead of
    // overwriting a file written earlier in the same run.

    // Render every slot first, then group identical full-canvas grayscale
    // rasters. The grouping is extraction-local; its hash bucket is always
    // checked with exact pixel equality before sharing encoded bytes.
    let mut rendered: Vec<(usize, Vec<u8>)> = (1..31usize)
        .into_par_iter()
        .map(|slot_num| {
            let base_address = slot_num_to_base_address(slot_num);
            let frame_number = *save.get(base_address + 0xfb0).ok_or_else(|| {
                format!(
                    "save '{}': slot {slot_num} frame offset {:#x} out of bounds (save len {})",
                    args.save.display(),
                    base_address + 0xfb0,
                    save.len()
                )
            })? as i32;
            let frame = rom_ref.map(|r| {
                let (addr, idx) = frame_base_address(r, frame_number);
                FrameInfo {
                    addr,
                    idx,
                    is_hk: is_hk_rom(r),
                }
            });
            let rom_tuple: Option<(&[u8], &FrameInfo)> = match (&rom_ref, &frame) {
                (Some(r), Some(f)) => Some((r, f)),
                _ => None,
            };

            Ok::<_, String>((slot_num, render_photo(rom_tuple, &save, base_address)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    rendered.sort_unstable_by_key(|(slot_num, _)| *slot_num);
    let groups = group_rendered(rendered);
    eprintln!(
        "gbcamextract: {} unique rendered rasters for 30 slots ({} duplicate slots reused)",
        groups.len(),
        30 - groups.len()
    );
    // Per-file deltas are exact only when groups cannot interleave. The
    // process-wide totals below remain exact with a larger Rayon pool.
    let report_file_stats = report_rdo_stats && rayon::current_num_threads() == 1;
    let paired_encode_nanos = std::sync::atomic::AtomicU64::new(0);

    // Encoding remains bounded by the configured Rayon pool. Every group is
    // encoded once and the resulting bytes are written to all of its
    // pre-validated, distinct output names.
    groups.into_par_iter().try_for_each(|group| {
        let first_slot = group.slots[0];
        let large = upscale_nearest(&group.gray, WIDTH, HEIGHT, SCALE);
        let before = report_file_stats.then(|| {
            (
                mono::rdo_stats(),
                mono::rdo_errors(),
                mono::pattern_cache_stats(),
                mono::pattern_cache_stats_speculative(),
                mono::uniform_search_stats(),
            )
        });
        let encode_started = report_rdo_stats.then(std::time::Instant::now);
        let avif = mono::encode_gray_pair(&group.gray, WIDTH, HEIGHT, &large, LARGE_W, LARGE_H)
            .map_err(|e| {
                format!(
                    "couldn't encode slot {first_slot} ({}): {e}",
                    filenames[first_slot - 1]
                )
            })?;
        let paired_encode_ms = encode_started.map(|started| {
            let elapsed = started.elapsed();
            paired_encode_nanos.fetch_add(elapsed.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            elapsed.as_secs_f64() * 1000.0
        });
        if let Some((rdo0, errors0, pattern0, pattern_spec0, uniform0)) = before {
            let (rdo1, errors1) = (mono::rdo_stats(), mono::rdo_errors());
            let (pattern1, pattern_spec1) = (
                mono::pattern_cache_stats(),
                mono::pattern_cache_stats_speculative(),
            );
            let uniform1 = mono::uniform_search_stats();
            let files = group
                .slots
                .iter()
                .map(|&slot| filenames[slot - 1].as_str())
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "rdo-stats: files={files} bytes={} paired_ms={:.3} rdo_wins={} baseline_wins={} rdo_candidates={} rdo_errors={} pattern_committed={}/{} pattern_speculative={}/{} uniform_searches={} uniform_committed_searches={} uniform_speculative_searches={} uniform_work={} uniform_speculative_work={} uniform_fallbacks={} uniform_legal_matches={} uniform_selected={} uniform_errors={} uniform_cache_max_bytes={}",
                avif.len(),
                paired_encode_ms.unwrap_or_default(),
                rdo1.0 - rdo0.0,
                rdo1.1 - rdo0.1,
                rdo1.2 - rdo0.2,
                errors1 - errors0,
                pattern1.0 - pattern0.0,
                pattern1.1 - pattern0.1,
                pattern_spec1.0 - pattern_spec0.0,
                pattern_spec1.1 - pattern_spec0.1,
                uniform1.searches - uniform0.searches,
                uniform1.committed_searches - uniform0.committed_searches,
                uniform1.speculative_searches - uniform0.speculative_searches,
                uniform1.search_work - uniform0.search_work,
                uniform1.speculative_work - uniform0.speculative_work,
                uniform1.fallbacks - uniform0.fallbacks,
                uniform1.legal_matches - uniform0.legal_matches,
                uniform1.selected_copies - uniform0.selected_copies,
                uniform1.errors - uniform0.errors,
                uniform1.cache_bytes,
            );
        }
        debug_assert_eq!(group.identity, EncodeIdentity::current());
        for slot_num in group.slots {
            let filename = &filenames[slot_num - 1];
            std::fs::write(filename, &avif)
                .map_err(|e| format!("couldn't write '{filename}': {e}"))?;
        }
        Ok::<(), String>(())
    })?;
    if report_rdo_stats {
        let rdo = mono::rdo_stats();
        let (pattern_committed, pattern_fallbacks) = mono::pattern_cache_stats();
        let (pattern_speculative, pattern_speculative_fallbacks) =
            mono::pattern_cache_stats_speculative();
        let uniform = mono::uniform_search_stats();
        eprintln!(
            "rdo-stats-total: rayon_threads={} paired_ms_sum={:.3} rdo_wins={} baseline_wins={} rdo_candidates={} rdo_errors={} pattern_committed={pattern_committed}/{pattern_fallbacks} pattern_speculative={pattern_speculative}/{pattern_speculative_fallbacks} uniform_searches={} uniform_committed_searches={} uniform_speculative_searches={} uniform_work={} uniform_speculative_work={} uniform_fallbacks={} uniform_legal_matches={} uniform_selected={} uniform_errors={} uniform_cache_max_bytes={}",
            rayon::current_num_threads(),
            paired_encode_nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0,
            rdo.0,
            rdo.1,
            rdo.2,
            mono::rdo_errors(),
            uniform.searches,
            uniform.committed_searches,
            uniform.speculative_searches,
            uniform.search_work,
            uniform.speculative_work,
            uniform.fallbacks,
            uniform.legal_matches,
            uniform.selected_copies,
            uniform.errors,
            uniform.cache_bytes,
        );
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("gbcamextract: {e}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_grouping_checks_exact_pixels_and_encoder_identity() {
        let identity = EncodeIdentity::current();
        let mut other_identity = identity;
        other_identity.scale += 1;
        let groups = group_rendered_by(
            vec![
                (1, identity, vec![0, 85, 170, 255]),
                (2, identity, vec![0, 85, 170, 255]),
                (3, identity, vec![0, 85, 170, 254]),
                (4, other_identity, vec![0, 85, 170, 255]),
            ],
            |_, _| 0, // Force a collision to exercise exact equality.
        );
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].slots, [1, 2]);
        assert_eq!(groups[1].slots, [3]);
        assert_eq!(groups[2].slots, [4]);
    }
}
