//! The transcription catalogue, discovered from the library author's Hugging
//! Face organisation rather than hand-maintained.
//!
//! transcribe.cpp runs a whole family of speech models -- whisper, parakeet,
//! canary, moonshine and more -- and the author publishes a GGUF of each under
//! one org. `list` fetches that set so Settings can offer all of them; the
//! app's own three whisper models stay in `commands::models::catalogue` so
//! onboarding still works with no network.
//!
//! The list is immutable for a run (the models rarely change) and cached, so
//! the size-fetch fan-out happens once. Download state is read from disk on
//! every call, because that does change.

use crate::error::{Error, Result};
use crate::model::{ModelInfo, ModelKind, ModelState};
use serde::Deserialize;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

/// How many size lookups run at once. The list is one call, but sizes are one
/// call per repo; a whole org at once gets rate-limited (429) and the models
/// whose lookup failed would vanish, so they go out in small batches instead.
const SIZE_BATCH: usize = 8;

const ORG_URL: &str =
    "https://huggingface.co/api/models?author=handy-computer&full=true&limit=1000";

/// Preferred quantisations, best size/quality first. The catalogue offers one
/// file per repo; a smaller K-quant is the right default and the larger ones
/// are there for anyone who fetches their own via the custom-model setting.
const QUANTS: &[&str] = &["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"];

#[derive(Deserialize)]
struct Repo {
    id: String,
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: u64,
}

/// One catalogue model, without the download state that has to come from disk.
#[derive(Clone)]
struct Entry {
    id: String,
    quant: String,
    size_bytes: u64,
    url: String,
}

static CACHE: Mutex<Option<Vec<Entry>>> = Mutex::new(None);

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent("parallax")
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Other(format!("could not reach the model catalogue: {e}")))
}

/// The chosen GGUF for a repo: the first preferred quant present, and the file
/// name that carries it. `None` when the repo publishes no usable GGUF.
fn pick(siblings: &[Sibling]) -> Option<(String, String)> {
    let ggufs: Vec<&str> = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    for quant in QUANTS {
        if let Some(f) = ggufs.iter().find(|f| f.contains(quant)) {
            return Some((quant.to_string(), (*f).to_string()));
        }
    }
    // A repo with a GGUF but no recognised quant tag still gets offered.
    ggufs.first().map(|f| (String::new(), (*f).to_string()))
}

/// `handy-computer/parakeet-ctc-0.6b-gguf` -> `parakeet-ctc-0.6b`, which is the
/// name the file takes on disk and the id the setting stores.
fn id_for(repo: &str) -> String {
    repo.rsplit('/')
        .next()
        .unwrap_or(repo)
        .trim_end_matches("-gguf")
        .to_string()
}

/// The byte size of one file in a repo, or `None` if the lookup fails. Retried
/// once, since a rate-limited call is transient and losing the size hides the
/// figure a user needs before spending a gigabyte.
fn size_of(client: &reqwest::blocking::Client, repo: &str, file: &str) -> Option<u64> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    for attempt in 0..2 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(400));
        }
        if let Ok(tree) = client
            .get(&url)
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.json::<Vec<TreeEntry>>())
        {
            return tree.into_iter().find(|t| t.path == file).map(|t| t.size);
        }
    }
    None
}

/// Every published model is offered -- one-shot and streaming families alike,
/// since `stt` drives whichever path a model needs. Only gated repos are held
/// back, because they cannot be fetched without credentials the app has not.
fn wanted(repo: &Repo) -> bool {
    repo.gated == serde_json::Value::Bool(false)
}

/// Fetches, filters, and sizes the whole catalogue. Cached for the run.
fn entries() -> Result<Vec<Entry>> {
    if let Some(cached) = CACHE.lock().unwrap_or_else(|p| p.into_inner()).clone() {
        return Ok(cached);
    }

    let client = client()?;
    let repos: Vec<Repo> = client
        .get(ORG_URL)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.json())
        .map_err(|e| Error::Other(format!("could not read the model catalogue: {e}")))?;

    // Repo id, its on-disk id, chosen quant, and the file to fetch a size for.
    let picks: Vec<(String, String, String, String)> = repos
        .iter()
        .filter(|r| wanted(r))
        .filter_map(|r| pick(&r.siblings).map(|(quant, file)| (r.id.clone(), id_for(&r.id), quant, file)))
        .collect();

    // Sizes are one call per repo, in small batches to stay under the rate
    // limit. A model is never dropped for a failed lookup -- it is shown with
    // size 0 (unknown), still selectable and downloadable -- because a missing
    // size is a worse bug than a blank one.
    let mut built: Vec<Entry> = Vec::with_capacity(picks.len());
    for batch in picks.chunks(SIZE_BATCH) {
        let sizes: Vec<u64> = std::thread::scope(|scope| {
            let handles: Vec<_> = batch
                .iter()
                .map(|(repo, _id, _quant, file)| {
                    let client = client.clone();
                    scope.spawn(move || size_of(&client, repo, file).unwrap_or(0))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or(0)).collect()
        });
        for ((repo, id, quant, file), size) in batch.iter().zip(sizes) {
            built.push(Entry {
                id: id.clone(),
                quant: quant.clone(),
                size_bytes: size,
                url: format!("https://huggingface.co/{repo}/resolve/main/{file}"),
            });
        }
    }

    built.sort_by(|a, b| a.id.cmp(&b.id));
    *CACHE.lock().unwrap_or_else(|p| p.into_inner()) = Some(built.clone());
    Ok(built)
}

/// State from disk: a present `{id}.gguf` is complete, since downloads land in
/// `.part` and are renamed only when whole.
fn state_on_disk(models_dir: &Path, id: &str) -> ModelState {
    let file = models_dir.join(format!("{id}.gguf"));
    if file.is_file() {
        return ModelState::Ready;
    }
    match std::fs::metadata(crate::model::download::part_path(&file)) {
        Ok(part) => ModelState::Downloading {
            received_bytes: part.len(),
            total_bytes: 0,
        },
        Err(_) => ModelState::NotDownloaded,
    }
}

/// The full transcription catalogue, each entry carrying its current download
/// state.
pub fn list(models_dir: &Path) -> Result<Vec<ModelInfo>> {
    Ok(entries()?
        .into_iter()
        .map(|e| ModelInfo {
            state: state_on_disk(models_dir, &e.id),
            kind: ModelKind::Transcription,
            name: e.id.clone(),
            params: String::new(),
            quantization: e.quant,
            size_bytes: e.size_bytes,
            recommended_ram_bytes: 0,
            id: e.id,
            url: e.url,
        })
        .collect())
}
