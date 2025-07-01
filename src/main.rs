use std::{
    collections::BTreeMap, fs, hash::Hash, io::Cursor, process::Command, sync::LazyLock,
    time::Instant,
};

use camino::{Utf8Path, Utf8PathBuf};
use clap::{Parser, ValueEnum};
use color_eyre::{
    eyre::eyre,
    owo_colors::{AnsiColors, OwoColorize},
    Result,
};
use image::DynamicImage;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use nsfw::{create_model, examine, model::Metric, Model};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpeg", "jpg", "webp", "jpe", "gif"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "avi", "mov", "flv", "wmv", "webm"];

static MODEL: LazyLock<Model> = LazyLock::new(|| {
    let model = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model.onnx"));
    let model = Cursor::new(model);
    create_model(model).expect("failed to create model")
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, ValueEnum)]
pub enum GroupingStrategy {
    /// Group by the NSFW metric, one folder per type
    Category,

    /// SFW or NSFW, one folder for each, this is the default
    #[default]
    NsfwOrSfw,
}

#[derive(Parser, Debug)]
struct Args {
    /// Does not actually move any files, just prints out what it would do
    #[clap(long)]
    pub dry_run: bool,

    /// Number of threads (defaults to number of CPUs)
    #[clap(long)]
    pub num_threads: Option<usize>,

    #[clap(short = 'd', long = "destination")]
    pub destination: Utf8PathBuf,

    #[clap(short = 'g', long = "grouping")]
    pub grouping_strategy: GroupingStrategy,

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

    // compute a running average of the classifications
    pub fn merge(&mut self, result: FileResult) {
        for (metric, score) in &mut self.classifications {
            if let Some(new_score) = result.classifications.get(metric) {
                *score = (*score + new_score) / 2.0;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Image,
    Video,
}

fn collect_paths(source: &Utf8Path) -> Result<Vec<(Utf8PathBuf, FileType)>> {
    let mut paths = vec![];
    for entry in WalkDir::new(source) {
        match entry {
            Ok(e) => {
                let path = Utf8Path::from_path(e.path()).expect("path must be utf-8");
                if path.is_dir() {
                    continue;
                }

                if let Some(extension) = path.extension() {
                    if VIDEO_EXTENSIONS.contains(&extension) {
                        paths.push((path.to_owned(), FileType::Video));
                    } else if IMAGE_EXTENSIONS.contains(&extension) {
                        paths.push((path.to_owned(), FileType::Image));
                    } else {
                        eprintln!(
                            "skipping file with unsupported extension: {}",
                            path.file_name().unwrap().bold()
                        );
                    }
                }
            }
            Err(e) => eprintln!("failed to open file: {e}"),
        }
    }

    Ok(paths)
}

fn classify_image_at_path(path: impl AsRef<Utf8Path>) -> Result<FileResult> {
    let image = image::open(path.as_ref())?;
    classify_image(path, image)
}

fn classify_image(path: impl AsRef<Utf8Path>, image: DynamicImage) -> Result<FileResult> {
    let image = image.into_rgba8();
    let result = examine(&MODEL, &image).map_err(|e| eyre!("failed to examine image: {e}"))?;

    let classifications = result
        .iter()
        .map(|c| (HashableMetric::from(c.metric.clone()), c.score))
        .collect();

    Ok(FileResult {
        path: path.as_ref().to_owned(),
        classifications,
    })
}

fn get_video_length(path: impl AsRef<Utf8Path>) -> Result<f32> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.as_ref().as_str(),
        ])
        .output()?;

    if output.status.success() {
        let duration_str = String::from_utf8_lossy(&output.stdout);
        duration_str
            .trim()
            .parse::<f32>()
            .map_err(|e| eyre!("failed to parse video duration: {e}, output: {duration_str}"))
    } else {
        Err(eyre!(
            "ffprobe failed with error: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn extract_frame_from_video(path: impl AsRef<Utf8Path>, timestamp: f32) -> Result<DynamicImage> {
    let output = Command::new("ffmpeg")
        .args([
            "-ss",
            &timestamp.to_string(),
            "-i",
            path.as_ref().as_str(),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()?;
    if output.status.success() {
        image::load_from_memory(&output.stdout)
            .map_err(|e| eyre!("failed to load image from ffmpeg output: {e}"))
    } else {
        Err(eyre!(
            "ffmpeg failed with error: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn classify_video(path: impl AsRef<Utf8Path>) -> Result<FileResult> {
    let duration = get_video_length(&path)?;

    let mut results = FileResult {
        path: path.as_ref().to_owned(),
        classifications: BTreeMap::new(),
    };
    // take 5 samples from the video and average the results
    for i in 0..5 {
        let timestamp = i as f32 * duration / 5.0;
        let image = extract_frame_from_video(&path, timestamp)?;
        let result = classify_image(&path, image)?;
        results.merge(result);
    }

    Ok(results)
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

    if !results.is_empty() {
        let metrics = results[0]
            .classifications
            .keys()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" | ");

        let header = format!("| File | {metrics} |\n");
        output += &header;

        // let separator = (0..METRICS.len())
        //     .map(|_| "---")
        //     .collect::<Vec<_>>()
        //     .join(" | ");
        // output += &format!("| --- | {separator} |\n");

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

    match args.grouping_strategy {
        GroupingStrategy::Category => {
            println!("Grouping by category, one folder per type.");
            for category in [
                Metric::Drawings,
                Metric::Hentai,
                Metric::Neutral,
                Metric::Porn,
                Metric::Sexy,
            ] {
                let folder = args.destination.join(category.to_string());
                if !folder.exists() {
                    std::fs::create_dir_all(&folder)?;
                }
            }
        }
        GroupingStrategy::NsfwOrSfw => {
            let nsfw_folder = args.destination.join("nsfw");
            let sfw_folder = args.destination.join("sfw");

            fs::create_dir_all(&nsfw_folder)?;
            fs::create_dir_all(&sfw_folder)?;
        }
    }

    let mut image_paths = collect_paths(&args.source_folder)?;
    image_paths.sort_by_key(|(path, _)| path.to_string());
    let len = image_paths.len() as u64;
    println!(
        "found {} files in {}",
        len.bold(),
        args.source_folder.bold()
    );

    let results: Vec<_> = image_paths
        .into_par_iter()
        //.progress_count(len)
        .progress_with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] (eta {eta}, {per_sec}) {wide_bar} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
        )
        .filter_map(|(path, file_type)| match file_type {
            FileType::Image => match classify_image_at_path(&path) {
                Ok(result) => Some(result),
                Err(e) => {
                    eprintln!("failed to classify image {path}: {e}");
                    None
                }
            },
            FileType::Video => match classify_video(&path) {
                Ok(result) => Some(result),
                Err(e) => {
                    eprintln!("failed to classify video {path}: {e}");
                    None
                }
            },
        })
        .collect();

    for result in &results {
        let is_nsfw = result.is_nsfw(args.threshold);
        let path = &result.path;
        match args.grouping_strategy {
            GroupingStrategy::Category => {
                let Some((metric, confidence)) = result
                    .classifications
                    .iter()
                    .max_by_key(|(_, v)| OrderedFloat(**v))
                else {
                    continue;
                };

                let destination = args.destination.join(metric.to_string());
                let destination = {
                    let original_file_name = path.file_name().expect("file must have file name");
                    let file_name =
                        find_non_conflicting_file_name(&destination, original_file_name)?;
                    destination.join(file_name)
                };

                if args.dry_run {
                    println!(
                        "Classified '{}' as {} (confidence: {:.2}%), would move to '{}'",
                        path.strip_prefix(&args.source_folder).unwrap().bold(),
                        metric.bold().color(match metric {
                            HashableMetric::Drawings => AnsiColors::Green,
                            HashableMetric::Hentai => AnsiColors::Red,
                            HashableMetric::Neutral => AnsiColors::Blue,
                            HashableMetric::Porn => AnsiColors::Yellow,
                            HashableMetric::Sexy => AnsiColors::Magenta,
                        }),
                        confidence * 100.0,
                        destination.bold(),
                    );
                } else {
                    fs::copy(&result.path, &destination)?;
                    println!(
                        "Copied '{}' to '{}'",
                        path.bold(),
                        destination.bold().color(match metric {
                            HashableMetric::Drawings => AnsiColors::Green,
                            HashableMetric::Hentai => AnsiColors::Red,
                            HashableMetric::Neutral => AnsiColors::Blue,
                            HashableMetric::Porn => AnsiColors::Yellow,
                            HashableMetric::Sexy => AnsiColors::Magenta,
                        })
                    );
                }
            }
            GroupingStrategy::NsfwOrSfw => {
                let destination = if is_nsfw {
                    args.destination.join("nsfw")
                } else {
                    args.destination.join("sfw")
                };

                if args.dry_run {
                    println!("Would move {} -> {}", path.bold(), destination.bold());
                } else {
                    fs::copy(&result.path, &destination)?;
                }
            }
        }
    }

    // write_markdown_report(&args.source_folder, &results)?;

    let elapsed = start.elapsed();
    println!("Elapsed time: {elapsed:?}");

    Ok(())
}
