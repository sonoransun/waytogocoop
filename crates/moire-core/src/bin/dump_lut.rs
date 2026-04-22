//! Regenerate the cross-stack colormap LUT JSON consumed by the Python stack.
//!
//! Usage: `cargo run -p moire-core --bin dump_lut -- <out-path>`
//! Default out-path: `src/waytogocoop/components/colormaps_data.json`.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use moire_core::colormap::{colormap_lut, ColormapName};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let out: PathBuf = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| {
            // When invoked from the workspace root (`cargo run -p moire-core`),
            // the CWD is the workspace root, so this relative path works.
            "src/waytogocoop/components/colormaps_data.json".to_string()
        })
        .into();

    let mut names: Vec<ColormapName> = ColormapName::ALL.to_vec();
    names.sort_by_key(|n| n.as_str());

    let mut file = BufWriter::new(File::create(&out)?);
    writeln!(file, "{{")?;
    writeln!(file, "  \"_meta\": {{")?;
    writeln!(
        file,
        "    \"generator\": \"cargo run -p moire-core --bin dump_lut\","
    )?;
    writeln!(file, "    \"samples\": 256,")?;
    writeln!(file, "    \"format\": \"rgb 0-255\"")?;
    writeln!(file, "  }},")?;
    for (i, name) in names.iter().enumerate() {
        let lut = colormap_lut(*name);
        writeln!(file, "  \"{}\": [", name.as_str())?;
        for j in 0..256 {
            let r = lut[j * 4];
            let g = lut[j * 4 + 1];
            let b = lut[j * 4 + 2];
            let comma = if j == 255 { "" } else { "," };
            writeln!(file, "    [{}, {}, {}]{}", r, g, b, comma)?;
        }
        let trail = if i + 1 == names.len() { "" } else { "," };
        writeln!(file, "  ]{}", trail)?;
    }
    writeln!(file, "}}")?;
    file.flush()?;

    eprintln!("wrote {}", out.display());
    Ok(())
}
