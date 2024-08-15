use std::{
    io::Cursor,
    sync::{
        atomic::{AtomicU64, Ordering},
        LazyLock,
    },
};

use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use color_eyre::{eyre::eyre, Result};
use indicatif::ParallelProgressIterator;
use nsfw::{
    create_model, examine,
    model::{Classification, Metric},
    Model,
};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use walkdir::WalkDir;

const EXTENSIONS: &[&str] = &["png", "jpeg", "jpg", "webp", "jpe", "gif"];
const MODEL: LazyLock<Model> = LazyLock::new(|| {
    let model = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model.onnx"));
    let model = Cursor::new(model);
    create_model(model).expect("failed to create model")
});

#[derive(Parser, Debug)]
struct Args {
    /// Does not actually move any files, just prints out what it would do
    #[clap(long)]
    pub dry_run: bool,

    /// Number of threads (defaults to number of CPUs)
    #[clap(long)]
    pub num_threads: Option<usize>,

    /// Path to put the NSFW images
    #[clap(short = 'd', long = "destination")]
    pub nsfw_folder: Utf8PathBuf,

    /// Folder to get the images from
    pub source_folder: Utf8PathBuf,
}

fn is_nsfw(classifications: &[Classification]) -> bool {
    const METRICS: [Metric; 3] = [Metric::Hentai, Metric::Porn, Metric::Sexy];
    let score_max = classifications
        .iter()
        .filter(|c| METRICS.contains(&c.metric))
        .max_by_key(|m| OrderedFloat(m.score))
        .map(|s| s.score)
        .unwrap_or(0.0);

    score_max > 0.5
}

fn collect_paths(source: &Utf8Path) -> Result<Vec<Utf8PathBuf>> {
    let mut paths = vec![];
    for entry in WalkDir::new(source) {
        match entry {
            Ok(e) => {
                let path = Utf8Path::from_path(e.path()).expect("path must be utf-8");
                if let Some(extension) = path.extension() {
                    if EXTENSIONS.contains(&extension) {
                        paths.push(path.to_owned());
                    }
                }
            }
            Err(e) => eprintln!("failed to open file: {e}"),
        }
    }

    Ok(paths)
}

fn find_non_conflicting_file_name(dir: &Utf8Path, file_name: &str) -> Result<String> {
    let mut dest = dir.join(file_name);
    if dest.is_file() {
        let mut counter = 1;
        while dest.is_file() {
            let ext = dest.extension().expect("must have extension");
            let stem = dest.file_name().expect("must have file stem");
            let file_name = format!("{stem} ({counter}).{ext}");
            dest.set_file_name(file_name);
            counter += 1;
        }

        Ok(dest.file_name().unwrap().to_string())
    } else {
        Ok(dest.file_name().unwrap().to_string())
    }
}

fn copy_image_if_nsfw(path: &Utf8Path, destination_dir: &Utf8Path, dry_run: bool) -> Result<bool> {
    let image = image::open(path)?;
    let image = image.into_rgba8();

    let result = examine(&MODEL, &image).map_err(|e| eyre!("failed to examine image: {e}"))?;
    let is_nsfw = is_nsfw(&result);

    if is_nsfw {
        let file_name = find_non_conflicting_file_name(
            destination_dir,
            path.file_name().expect("file must have file name"),
        )?;
        let dest = destination_dir.join(file_name);
        if !dry_run {
            std::fs::copy(path, dest)?;
        }
    }

    Ok(is_nsfw)
}

fn main() -> Result<()> {
    let args = Args::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads.unwrap_or(num_cpus::get()))
        .build_global()?;

    std::fs::create_dir_all(&args.nsfw_folder)?;
    let image_paths = collect_paths(&args.source_folder)?;
    let len = image_paths.len() as u64;
    println!("found {len} files in {}", args.source_folder);
    let nsfw_count = AtomicU64::new(0);

    image_paths
        .into_par_iter()
        .progress_count(len)
        .for_each(
            |path| match copy_image_if_nsfw(&path, &args.nsfw_folder, args.dry_run) {
                Ok(true) => {
                    nsfw_count.fetch_add(1, Ordering::SeqCst);
                }
                Ok(false) => {}
                Err(e) => eprintln!("failed to check or move file {path}: {e}"),
            },
        );

    println!(
        "Processed {} images, {} were NSFW and moved to {}",
        len,
        nsfw_count.load(Ordering::SeqCst),
        args.nsfw_folder
    );

    Ok(())
}

#[cfg(test)]
mod tests {}
