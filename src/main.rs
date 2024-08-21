use std::{
    collections::BTreeMap,
    hash::Hash,
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
use nsfw::{create_model, examine, model::Metric, Model};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HashableMetric {
    /// safe for work drawings (including anime)
    Drawings,
    /// hentai and pornographic drawings
    Hentai,
    /// safe for work neutral images
    Neutral,
    /// pornographic images, sexual acts
    Porn,
    /// sexually explicit images, not pornography
    Sexy,
}

impl std::fmt::Display for HashableMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            HashableMetric::Drawings => "Drawings",
            HashableMetric::Hentai => "Hentai",
            HashableMetric::Neutral => "Neutral",
            HashableMetric::Porn => "Porn",
            HashableMetric::Sexy => "Sexy",
        };

        write!(f, "{string}")
    }
}

impl From<Metric> for HashableMetric {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Drawings => HashableMetric::Drawings,
            Metric::Hentai => HashableMetric::Hentai,
            Metric::Neutral => HashableMetric::Neutral,
            Metric::Porn => HashableMetric::Porn,
            Metric::Sexy => HashableMetric::Sexy,
        }
    }
}

#[derive(Debug)]
pub struct FileResult {
    pub path: Utf8PathBuf,
    pub classifications: BTreeMap<HashableMetric, f32>,
}

impl FileResult {
    pub fn is_nsfw(&self, threshold: f32) -> bool {
        let score_max = self
            .classifications
            .values()
            .copied()
            .max_by_key(|m| OrderedFloat(*m))
            .unwrap_or(0.0);

        score_max > threshold
    }
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

fn classify_image(path: impl AsRef<Utf8Path>) -> Result<FileResult> {
    let image = image::open(path.as_ref())?;
    let image = image.into_rgba8();
    let result = examine(&MODEL, &image).map_err(|e| eyre!("failed to examine image: {e}"))?;

    let classifications = result
        .iter()
        .filter(|c| METRICS.contains(&c.metric))
        .map(|c| (HashableMetric::from(c.metric.clone()), c.score))
        .collect();

    Ok(FileResult {
        path: path.as_ref().to_owned(),
        classifications,
    })
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

fn write_markdown_report(source_folder: &Utf8Path, results: &[FileResult]) -> Result<()> {
    let mut output = String::new();
    output.push_str("# NSFW Report\n\n");
    output.push_str(&format!("## Source Folder: {source_folder}\n\n"));

    if results.len() > 0 {
        let metrics = results[0]
            .classifications
            .keys()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" | ");

        let header = format!("| File | {} |\n", metrics);
        output += &header;

        let separator = (0..METRICS.len())
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | ");
        output += &format!("| --- | {separator} |\n");

        for result in results {
            let classifications = result
                .classifications
                .values()
                .map(|s| format!("{:.2}%", s * 100.0))
                .collect::<Vec<_>>()
                .join(" | ");

            let row = format!("| {} | {classifications} |\n", result.path);
            output += &row;
        }
    }

    std::fs::write("nsfw_report.md", output)?;

    Ok(())
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

    let results: Vec<_> = image_paths
        .into_par_iter()
        .progress_count(len)
        .filter_map(|path| match classify_image(&path) {
            Ok(result) => Some(result),
            Err(e) => {
                eprintln!("failed to classify image {path}: {e}");
                None
            }
        })
        .collect();

    for result in &results {
        let is_nsfw = result.is_nsfw(args.threshold);
        let path = &result.path;
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
