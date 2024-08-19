use std::{
    collections::HashMap,
    io::Cursor,
    sync::{
        atomic::{AtomicU64, Ordering},
        LazyLock,
    },
    time::Instant,
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
const METRICS: [Metric; 3] = [Metric::Hentai, Metric::Porn, Metric::Sexy];

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

    /// Flatten the destination folders into a single folder.
    #[clap(short, long)]
    pub flatten: bool,

    /// Threshold for detecting something as NSFW. Can be between 0 and 1,
    /// 1 being 100% certain that it's NSFW.
    #[clap(short, long, default_value = "0.5")]
    pub threshold: f32,

    /// Folder to get the images from
    pub source_folder: Utf8PathBuf,
}

fn is_nsfw(classifications: &[Classification], threshold: f32) -> bool {
    let score_max = classifications
        .iter()
        .filter(|c| METRICS.contains(&c.metric))
        .max_by_key(|m| OrderedFloat(m.score))
        .map(|s| s.score)
        .unwrap_or(0.0);

    score_max > threshold
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

fn classify_image(path: impl AsRef<Utf8Path>) -> Result<Vec<Classification>> {
    let image = image::open(path.as_ref())?;
    let image = image.into_rgba8();
    let result = examine(&MODEL, &image).map_err(|e| eyre!("failed to examine image: {e}"))?;
    Ok(result)
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

fn main() -> Result<()> {
    let args = Args::parse();
    let n_threads = args.num_threads.unwrap_or(num_cpus::get());
    println!("Running with {n_threads} threads.");
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()?;
    let start = Instant::now();

    std::fs::create_dir_all(&args.nsfw_folder)?;
    let image_paths = collect_paths(&args.source_folder)?;
    let len = image_paths.len() as u64;
    println!("found {len} files in {}", args.source_folder);
    let nsfw_count = AtomicU64::new(0);

    let results: HashMap<_, _> = image_paths
        .into_par_iter()
        .progress_count(len)
        .filter_map(|path| {
            let results = classify_image(&path);
            match results {
                Ok(results) => Some((path, results)),
                Err(e) => {
                    eprintln!("failed to classify image {path}: {e}");
                    None
                }
            }
        })
        .collect();

    for (path, classifications) in &results {
        let is_nsfw = is_nsfw(&classifications, args.threshold);
        if is_nsfw {
            nsfw_count.fetch_add(1, Ordering::SeqCst);
            let original_file_name = path.file_name().expect("file must have file name");
            let dest = if args.flatten {
                let file_name =
                    find_non_conflicting_file_name(&args.nsfw_folder, original_file_name)?;
                args.nsfw_folder.join(file_name)
            } else {
                let path_segments = path
                    .strip_prefix(&args.source_folder)
                    .expect("must be a prefix");
                args.nsfw_folder.join(path_segments)
            };

            if !args.dry_run {
                println!("Moving '{path}' -> '{dest}'");
                std::fs::copy(&path, &dest)?;
            } else {
                println!("Would move '{path}' -> '{dest}'");
            }
        }
    }

    write_markdown_report(&args.source_folder, &results)?;

    let elapsed = start.elapsed();

    println!(
        "Processed {} images, {} of which were classified as NSFW and moved to destination '{}'",
        len,
        nsfw_count.load(Ordering::SeqCst),
        args.nsfw_folder
    );
    println!("Elapsed time: {elapsed:?}");

    Ok(())
}

fn write_markdown_report(
    source_folder: &Utf8Path,
    results: &HashMap<Utf8PathBuf, Vec<Classification>>,
) -> Result<()> {
    let mut output = String::new();
    output.push_str("# NSFW Report\n\n");
    output.push_str(&format!("## Source Folder: {source_folder}\n\n"));
    // write a markdown table with headers for file path and the different classification types
    // then write the file path and the classification scores (as percentages) for each image
    let metrics = METRICS
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    let header = format!("| File | {} |\n", metrics);
    output += &header;

    let separator = format!("| --- | {:-<1$} |\n", "", metrics.len() * 5);
    output += &separator;

    for (path, classifications) in results {
        let classifications = classifications
            .iter()
            .map(|c| format!("{:.2}%", c.score * 100.0))
            .collect::<Vec<_>>()
            .join(" | ");
        let row = format!("| {path} | {} |\n", classifications);
        output += &row;
    }

    std::fs::write("nsfw_report.md", output)?;

    Ok(())
}

#[cfg(test)]
mod tests {}
