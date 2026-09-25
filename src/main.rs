use clap::Parser;
use gamut_avif::AvifEncoder;
use gamut_core::{Dimensions, EncodeImage, ImageRef, Rgb8};
use std::io;
use std::path::PathBuf;
use std::process;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 144;
const SAVE_SIZE: usize = 128 * 1024;
const ROM_SIZE: usize = 1024 * 1024;
const BANK_SIZE: usize = 0x4000;

// Output is a single flat 160x144 lossless AVIF still per photo.
// (A 'grid' of small tiles would isolate border from photo, but MIAF
// requires grid tiles to be at least 64x64px, and an 'iovl' overlay is
// not valid AVIF at all, so neither renders in standard viewers.)
const CANVAS_PIXELS: usize = WIDTH as usize * HEIGHT as usize;

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
    let pic_num = save[vec_base + slot_num - 1];
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
fn border_raw_tile(
    rom: &[u8],
    frame: &FrameInfo,
    bx: usize,
    by: usize,
) -> Option<[u8; 16]> {
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
    if (PHOTO_X..PHOTO_X + PHOTO_W).contains(&x) && (PHOTO_Y..PHOTO_Y + PHOTO_H).contains(&y)
    {
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

/// Full-canvas RGB pixels (gray, R=G=B), row-major.
fn render_photo(
    rom: Option<(&[u8], &FrameInfo)>,
    save: &[u8],
    base_address: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; CANVAS_PIXELS * 3];
    for by in 0..18 {
        for bx in 0..20 {
            let block = canvas_block(rom, save, base_address, bx, by);
            for r in 0..8 {
                for c in 0..8 {
                    let v = GRAY_8BIT[block[r * 8 + c] as usize];
                    let o = ((by * 8 + r) * WIDTH as usize + bx * 8 + c) * 3;
                    out[o] = v;
                    out[o + 1] = v;
                    out[o + 2] = v;
                }
            }
        }
    }
    out
}

/// Lossless AVIF still. The encoder is bit-exact: decoding yields the
/// input pixels unchanged.
fn encode_avif(encoder: &AvifEncoder, rgb: &[u8]) -> io::Result<Vec<u8>> {
    let image = ImageRef::<Rgb8>::new(rgb, Dimensions {
        width: WIDTH,
        height: HEIGHT,
    })
    .map_err(|e| io::Error::other(format!("image wrap: {e:?}")))?;
    encoder
        .encode_to_vec(image)
        .map_err(|e| io::Error::other(format!("avif encode: {e:?}")))
}

fn main() {
    let args = Args::parse();

    let save = match std::fs::read(&args.save) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("gbcamextract: couldn't open save for reading: {}", e);
            process::exit(1);
        }
    };
    if save.len() != SAVE_SIZE {
        eprintln!("gbcamextract: savegame has weird size");
        process::exit(1);
    }
    if is_gb_rom(&save) {
        eprintln!("gbcamextract: save expected, but rom was given");
        process::exit(1);
    }

    let rom: Option<Vec<u8>> = match &args.rom {
        Some(path) => {
            let data = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("gbcamextract: couldn't open rom for reading: {}", e);
                    process::exit(1);
                }
            };
            if data.len() != ROM_SIZE {
                eprintln!("gbcamextract: rom has weird size");
                process::exit(1);
            }
            if !is_gb_rom(&data) {
                eprintln!("gbcamextract: rom given doesn't look like a real rom");
                process::exit(1);
            }
            Some(data)
        }
        None => None,
    };
    let rom_ref: Option<&[u8]> = rom.as_deref();
    let encoder = AvifEncoder::new();

    for slot_num in 1..=30usize {
        let pic_num = get_pic_num_for_slot_num(&save, slot_num);
        let base_address = slot_num_to_base_address(slot_num);
        let frame_number = save[base_address + 0xfb0] as i32;
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

        let rgb = render_photo(rom_tuple, &save, base_address);
        let avif = match encode_avif(&encoder, &rgb) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("gbcamextract: couldn't encode {}: {}", slot_num, e);
                process::exit(1);
            }
        };
        let filename = match pic_num {
            Some(n) => format!("IMG_{:02}.avif", n),
            None => format!("DEL_{:02}.avif", slot_num),
        };
        if let Err(e) = std::fs::write(&filename, &avif) {
            eprintln!("gbcamextract: couldn't write {}: {}", filename, e);
            process::exit(1);
        }
    }
}
