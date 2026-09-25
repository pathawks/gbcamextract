use clap::Parser;
use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::process;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 144;
const ROW_SIZE: usize = 40;
const PIXEL_BUFFER_SIZE: usize = ROW_SIZE * HEIGHT as usize;
const SAVE_SIZE: usize = 128 * 1024;
const ROM_SIZE: usize = 1024 * 1024;
const BANK_SIZE: usize = 0x4000;

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

fn interleave_bytes(low: u8, high: u8) -> u16 {
    let mut result: u16 = 0;
    result |= low as u16 & 1;
    result |= (high as u16 & 1) << 1;
    result |= (low as u16 & 2) << 1;
    result |= (high as u16 & 2) << 2;
    result |= (low as u16 & 4) << 2;
    result |= (high as u16 & 4) << 3;
    result |= (low as u16 & 8) << 3;
    result |= (high as u16 & 8) << 4;
    result |= (low as u16 & 16) << 4;
    result |= (high as u16 & 16) << 5;
    result |= (low as u16 & 32) << 5;
    result |= (high as u16 & 32) << 6;
    result |= (low as u16 & 64) << 6;
    result |= (high as u16 & 64) << 7;
    result |= (low as u16 & 128) << 7;
    result |= (high as u16 & 128) << 8;
    result
}

fn draw_span(pixel_buffer: &mut [u8; PIXEL_BUFFER_SIZE], tile: &[u8], x: usize, y: usize) {
    let col = x / 4;
    for row in 0..8 {
        let p = col + (y + row) * ROW_SIZE;
        let low = !tile[row * 2];
        let high = !tile[row * 2 + 1];
        let interleaved = interleave_bytes(low, high);
        pixel_buffer[p] = (interleaved >> 8) as u8;
        pixel_buffer[p + 1] = (interleaved & 0xff) as u8;
    }
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

fn convert(
    rom: Option<&[u8]>,
    save: &[u8],
    pixel_buffer: &mut [u8; PIXEL_BUFFER_SIZE],
    slot_num: usize,
) {
    let base_address = slot_num_to_base_address(slot_num);
    let frame_number = save[base_address + 0xfb0] as i32;

    let (frame_addr, eff_frame) = match rom {
        Some(r) => {
            let (a, i) = frame_base_address(r, frame_number);
            (Some(a), Some(i))
        }
        None => (None, None),
    };
    let is_hk = rom.map(is_hk_rom).unwrap_or(false);

    for y_tile in 0..14usize {
        let y = 16 + y_tile * 8;
        for (i, x) in (16..=(8 * 17)).step_by(8).enumerate() {
            let off = base_address + y_tile * 256 + i * 16;
            let tile = &save[off..off + 16];
            draw_span(pixel_buffer, tile, x, y);
        }

        if let (Some(r), Some(faddr), Some(eff)) = (rom, frame_addr, eff_frame) {
            let y = 16 + y_tile * 8;
            for z in 0..4usize {
                let tile_num = if is_hk {
                    match r.get(HELLO_KITTY_FRAME_OFFSETS[eff][1] + 0x50 + y_tile * 4 + z) {
                        Some(&b) => b as usize,
                        None => continue,
                    }
                } else {
                    match r.get(faddr + 0x650 + y_tile * 4 + z) {
                        Some(&b) => b as usize,
                        None => continue,
                    }
                };
                let tile_off = faddr + tile_num * 16;
                let tile = match r.get(tile_off..tile_off + 16) {
                    Some(t) => t,
                    None => continue,
                };
                let x = ((z & 1 != 0) as usize) * 8 + ((z & 2 != 0) as usize) * HEIGHT as usize;
                draw_span(pixel_buffer, tile, x, y);
            }
        }
    }

    if let (Some(r), Some(faddr), Some(eff)) = (rom, frame_addr, eff_frame) {
        for x_tile in 0..20usize {
            for z in 0..4usize {
                let tile_num = if is_hk {
                    match r.get(HELLO_KITTY_FRAME_OFFSETS[eff][1] + x_tile + 0x14 * z) {
                        Some(&b) => b as usize,
                        None => continue,
                    }
                } else {
                    match r.get(faddr + 0x600 + x_tile + 0x14 * z) {
                        Some(&b) => b as usize,
                        None => continue,
                    }
                };
                let tile_off = faddr + tile_num * 16;
                let tile = match r.get(tile_off..tile_off + 16) {
                    Some(t) => t,
                    None => continue,
                };
                let x = x_tile * 8;
                let y = ((z & 1 != 0) as usize) * 8 + ((z & 2 != 0) as usize) * 128;
                draw_span(pixel_buffer, tile, x, y);
            }
        }
    }
}

fn write_image_file(pixel_buffer: &[u8; PIXEL_BUFFER_SIZE], filename: &str) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut encoder = png::Encoder::new(file, WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Two);
    encoder.set_compression(png::Compression::High);
    encoder
        .add_text_chunk("Source".to_string(), "Nintendo Gameboy Camera".to_string())
        .map_err(io::Error::other)?;
    encoder
        .add_text_chunk("Software".to_string(), "gbcamextract".to_string())
        .map_err(io::Error::other)?;
    let mut writer = encoder.write_header().map_err(io::Error::other)?;
    writer
        .write_image_data(&pixel_buffer[..])
        .map_err(io::Error::other)
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

    let mut pixel_buffer = [0u8; PIXEL_BUFFER_SIZE];

    for slot_num in 1..=30usize {
        let pic_num = get_pic_num_for_slot_num(&save, slot_num);
        convert(rom_ref, &save, &mut pixel_buffer, slot_num);
        let filename = match pic_num {
            Some(n) => format!("IMG_{:02}.png", n),
            None => format!("DEL_{:02}.png", slot_num),
        };
        if let Err(e) = write_image_file(&pixel_buffer, &filename) {
            eprintln!("gbcamextract: couldn't write {}: {}", filename, e);
            process::exit(1);
        }
    }
}
