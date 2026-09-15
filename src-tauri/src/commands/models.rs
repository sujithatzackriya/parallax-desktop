//! What this machine is, and which models it has.
//!
//! §9.4 -- downloading gates nothing. Capture and transcription work without a
//! reasoning model, and the question simply arrives when one lands.

use crate::error::{Error, Result};
use crate::model::download::Pace;
use crate::model::{ModelInfo, ModelKind, ModelState, SystemProfile};
use crate::state::AppState;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};

/// A 2.5GB file in 64KB chunks is 38,000 callbacks; the bar does not need them.
const PROGRESS_EVERY: std::time::Duration = std::time::Duration::from_millis(250);

/// Drives the onboarding default: the largest model this machine can hold.
#[tauri::command]
pub fn get_system_profile() -> SystemProfile {
    use sysinfo::System;

    // Not new_all: that enumerates every process on the machine, and this
    // runs on the onboarding screen.
    let mut system = System::new();
    system.refresh_memory();
    system.refresh_cpu_all();

    let cpu = system.cpus().first();

    SystemProfile {
        total_ram_bytes: system.total_memory(),
        available_ram_bytes: system.available_memory(),
        cpu_name: cpu
            .map(|c| c.brand().trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "unknown".into()),
        cpu_cores: system.cpus().len() as u32,
        // Not read from the system yet. Reporting a card without knowing its
        // VRAM would make the onboarding default worse than admitting nothing:
        // the recommendation is sized against memory it cannot see.
        gpu_name: None,
        vram_bytes: None,
    }
}

/// The catalogue. Sizes are the real download sizes; the RAM figure is the
/// least a model is worth recommending at, since onboarding picks the largest
/// one that fits.
///
/// Measured: Qwen3-4B is the floor for classification. At 1.7B the JSON is
/// always valid and the content is not -- wrong register, a summary that
/// misstates the note, and the move phrase copied out of the prompt, so 1.7B is
/// not offered at all; the official repo ships no Q4_K_M for it either. 4B's
/// figure is what it actually needs rather than a round number: 2.4GB of
/// weights plus a KV cache is comfortable in 12GB, and a 16GB machine that
/// reports 14.9 would otherwise be handed the model known not to work.
///
/// Figures are bytes rather than whole gigabytes, and set against what a machine
/// of that size actually reports, not what is printed on the box. Firmware takes its share before the OS
/// sees anything, so a round number here silently excludes the machine it was
/// chosen for.
fn catalogue() -> Vec<ModelInfo> {
    vec![
        whisper("whisper-tiny", "tiny", "39M", 43_600_000, 1_800_000_000),
        whisper("whisper-base", "base", "74M", 58_900_000, 3_700_000_000),
        whisper("whisper-small", "small", "244M", 171_600_000, 7_500_000_000),
        // RAM only has to hold it: 2.5GB of weights plus KV cache and buffers is
        // about 3.3GB resident, which an 8GB machine has. VRAM decides whether
        // that takes 1.8s or 9s, and is not this number.
        qwen(
            "qwen3-4b-q4",
            "Qwen3 4B",
            "4B",
            "Qwen3-4B",
            2_500_000_000,
            7_500_000_000,
        ),
        embedding(
            "bge-small-en-v1.5",
            "BGE small",
            "33M",
            36_806_944,
            "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        ),
        embedding(
            "all-minilm-l6-v2",
            "MiniLM L6",
            "22M",
            25_008_064,
            "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf",
        ),
        embedding(
            "nomic-embed-text-v1.5",
            "Nomic embed",
            "137M",
            146_146_432,
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf",
        ),
        qwen(
            "qwen3-8b-q4",
            "Qwen3 8B",
            "8B",
            "Qwen3-8B",
            5_030_000_000,
            15_000_000_000,
        ),
    ]
}

/// GGUF conversions by the author of the transcription library, which is the
/// format it loads; the whisper.cpp releases are ggml and will not open.
fn whisper(id: &str, name: &str, params: &str, size_bytes: u64, ram: u64) -> ModelInfo {
    let slug = id.trim_start_matches("whisper-");
    model(
        id,
        ModelKind::Transcription,
        name,
        params,
        "Q4_K_M",
        size_bytes,
        ram,
        &format!(
            "https://huggingface.co/handy-computer/whisper-{slug}-gguf/resolve/main/whisper-{slug}-Q4_K_M.gguf"
        ),
    )
}

fn qwen(id: &str, name: &str, params: &str, repo: &str, size_bytes: u64, ram: u64) -> ModelInfo {
    model(
        id,
        ModelKind::Reasoning,
        name,
        params,
        "Q4_K_M",
        size_bytes,
        ram,
        &format!("https://huggingface.co/Qwen/{repo}-GGUF/resolve/main/{repo}-Q4_K_M.gguf"),
    )
}

#[allow(clippy::too_many_arguments)]
/// Measured on the sixteen fixtures, in the architecture that ships -- inside
/// the topic shelf, capped at eight -- all three score identically. They
/// differ only in context window, which is what decides a long spoken note: MiniLM
/// truncates at 256 tokens, bge at 512, nomic at 8192, and truncation is
/// silent. So this is a size-against-length choice, not a quality ladder.
fn embedding(id: &str, name: &str, params: &str, size_bytes: u64, url: &str) -> ModelInfo {
    model(
        id,
        ModelKind::Embedding,
        name,
        params,
        "Q8_0",
        size_bytes,
        // It runs on the CPU beside everything else and is measured in tens of
        // megabytes, so any machine that can transcribe can hold it.
        2_000_000_000,
        url,
    )
}

// One positional field per `ModelInfo` column; a struct wrapper would just
// move the same eight names one level out for a function only called nearby.
#[allow(clippy::too_many_arguments)]
fn model(
    id: &str,
    kind: ModelKind,
    name: &str,
    params: &str,
    quantization: &str,
    size_bytes: u64,
    recommended_ram_bytes: u64,
    url: &str,
) -> ModelInfo {
    ModelInfo {
        id: id.into(),
        kind,
        name: name.into(),
        params: params.into(),
        quantization: quantization.into(),
        size_bytes,
        recommended_ram_bytes,
        state: ModelState::NotDownloaded,
        url: url.into(),
    }
}

/// A file well under its expected size is a partial download, not a model.
/// Loading a truncated gguf fails in a way that reads as the app being broken
/// rather than the file being incomplete, so it reports as still arriving.
///
/// The tolerance is because published sizes and bytes on disk rarely agree
/// exactly, not because a tenth of a model is acceptable.
const COMPLETE_ENOUGH: f64 = 0.9;

fn state_for(expected_bytes: u64, on_disk: Option<u64>, partial: Option<u64>) -> ModelState {
    // A .part is a download in flight. The bytes are on disk under a name
    // nothing loads, so without looking for it a download in progress reads as
    // never started -- and the status bar said "not downloaded" at 22%.
    let on_disk = match on_disk {
        Some(bytes) => bytes,
        None => {
            return match partial {
                Some(bytes) => ModelState::Downloading {
                    received_bytes: bytes,
                    total_bytes: expected_bytes,
                },
                None => ModelState::NotDownloaded,
            }
        }
    };
    match Some(on_disk) {
        None => ModelState::NotDownloaded,
        Some(bytes) if (bytes as f64) >= expected_bytes as f64 * COMPLETE_ENOUGH => {
            ModelState::Ready
        }
        Some(bytes) => ModelState::Downloading {
            received_bytes: bytes,
            total_bytes: expected_bytes,
        },
    }
}

/// State comes from the disk rather than a record of what was asked for: a
/// half-finished download that was interrupted is not a model, and a file
/// copied in by hand is.
#[tauri::command]
pub fn list_models(state: State<AppState>) -> Vec<ModelInfo> {
    models_on_disk(&state.models_dir())
}

#[tauri::command]
pub fn models_location(state: State<AppState>) -> Result<String> {
    let dir = state.models_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir.display().to_string())
}

/// Whether first run is behind this machine: a reasoning model was chosen, and
/// every model the settings name is on disk.
///
/// A valid custom file counts as its role being present, same as a catalogue
/// download that finished -- someone who only ever points the app at their own
/// file must not be sent back to onboarding for a catalogue model they never
/// asked for.
pub fn set_up(settings: &crate::model::Settings, models: &[ModelInfo]) -> bool {
    let ready = |id: &str| {
        models
            .iter()
            .any(|m| m.id == id && matches!(m.state, ModelState::Ready))
    };
    let has_custom = |path: &Option<String>| {
        path.as_deref()
            .is_some_and(|p| crate::state::resolve_custom_gguf(Some(p)).is_some())
    };

    let reasoning_ok = has_custom(&settings.custom_reasoning_model_path)
        || settings.model_id.as_deref().is_some_and(ready);
    if !reasoning_ok {
        return false;
    }

    let transcription_ok = has_custom(&settings.custom_transcription_model_path)
        || ready(&settings.transcription_model);

    // The embedder is optional -- connections work without one -- so only a
    // chosen one has to be there.
    transcription_ok && settings.embedding_model_id.as_deref().is_none_or(ready)
}

/// Native "choose file" for a custom GGUF model, filtered to that extension.
/// `None` when the dialog is cancelled.
#[tauri::command]
pub async fn pick_model_file(app: AppHandle) -> Result<Option<String>> {
    let path = crate::commands::archive::chosen(
        crate::commands::archive::dialog(&app)
            .set_title("Choose a model file")
            .add_filter("GGUF model", &["gguf"])
            .blocking_pick_file(),
    );
    Ok(path.map(|p| p.display().to_string()))
}

fn models_on_disk(dir: &std::path::Path) -> Vec<ModelInfo> {
    catalogue()
        .into_iter()
        .map(|mut info| {
            let path = dir.join(format!("{}.gguf", info.id));
            let part = crate::model::download::part_path(&path);
            info.state = state_for(
                info.size_bytes,
                std::fs::metadata(&path).ok().map(|m| m.len()),
                std::fs::metadata(&part).ok().map(|m| m.len()),
            );
            info
        })
        .collect()
}

/// Whether to open on onboarding. Asked of the disk, like `list_models`, so a
/// download interrupted by closing the app brings onboarding back to finish it.
#[tauri::command]
pub fn setup_complete(state: State<AppState>) -> Result<bool> {
    let settings = crate::db::settings::get(&state.db())?;
    Ok(set_up(&settings, &models_on_disk(&state.models_dir())))
}

#[cfg(test)]
mod set_up_tests {
    use super::*;
    use crate::model::Settings;

    fn on_disk(ids: &[&str]) -> Vec<ModelInfo> {
        catalogue()
            .into_iter()
            .map(|mut m| {
                m.state = if ids.contains(&m.id.as_str()) {
                    ModelState::Ready
                } else {
                    ModelState::NotDownloaded
                };
                m
            })
            .collect()
    }

    fn chosen() -> Settings {
        Settings {
            model_id: Some("qwen3-4b-q4".into()),
            transcription_model: "whisper-base".into(),
            embedding_model_id: Some("bge-small-en-v1.5".into()),
            ..Settings::default()
        }
    }

    /// The installed app, opened again: nothing left to set up, so no
    /// onboarding. The build had been replaying it on every launch.
    #[test]
    fn every_chosen_model_on_disk_is_set_up() {
        let models = on_disk(&["qwen3-4b-q4", "whisper-base", "bge-small-en-v1.5"]);
        assert!(set_up(&chosen(), &models));
    }

    #[test]
    fn nothing_chosen_is_not_set_up() {
        let models = on_disk(&["qwen3-4b-q4", "whisper-base", "bge-small-en-v1.5"]);
        assert!(!set_up(&Settings::default(), &models));
    }

    /// Onboarding saves the choice before the download finishes. Closing the
    /// app then must bring onboarding back, because it is what fetches the rest.
    #[test]
    fn a_model_still_downloading_is_not_set_up() {
        let mut models = on_disk(&["whisper-base", "bge-small-en-v1.5"]);
        for m in &mut models {
            if m.id == "qwen3-4b-q4" {
                m.state = ModelState::Downloading {
                    received_bytes: 1,
                    total_bytes: 2,
                };
            }
        }
        assert!(!set_up(&chosen(), &models));
        assert!(!set_up(
            &chosen(),
            &on_disk(&["qwen3-4b-q4", "bge-small-en-v1.5"])
        ));
    }

    /// Connections work without an embedder, so not having chosen one is not
    /// unfinished setup.
    #[test]
    fn no_embedder_chosen_does_not_hold_it_back() {
        let settings = Settings {
            embedding_model_id: None,
            ..chosen()
        };
        assert!(set_up(
            &settings,
            &on_disk(&["qwen3-4b-q4", "whisper-base"])
        ));
    }

    fn temp_gguf(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "parallax-set-up-test-{}-{name}",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, b"stand-in, never loaded").unwrap();
        path
    }

    /// A user who only ever pointed the app at their own file must not be
    /// bounced back to onboarding for a catalogue model they never chose.
    #[test]
    fn a_valid_custom_reasoning_model_counts_as_present() {
        let file = temp_gguf("mine.gguf");
        let settings = Settings {
            model_id: None,
            custom_reasoning_model_path: Some(file.to_str().unwrap().into()),
            transcription_model: "whisper-base".into(),
            embedding_model_id: None,
            ..Settings::default()
        };
        assert!(set_up(&settings, &on_disk(&["whisper-base"])));
        let _ = std::fs::remove_file(&file);
    }

    /// Same for transcription: a custom whisper file stands in for the
    /// catalogue download that gates recording.
    #[test]
    fn a_valid_custom_transcription_model_counts_as_present() {
        let file = temp_gguf("my-whisper.gguf");
        let settings = Settings {
            custom_transcription_model_path: Some(file.to_str().unwrap().into()),
            embedding_model_id: None,
            ..chosen()
        };
        assert!(set_up(&settings, &on_disk(&["qwen3-4b-q4"])));
        let _ = std::fs::remove_file(&file);
    }

    /// A custom path that does not resolve (missing, or the wrong extension)
    /// must not be read as present -- onboarding still has real work to do.
    #[test]
    fn an_invalid_custom_reasoning_path_does_not_count() {
        let settings = Settings {
            model_id: None,
            custom_reasoning_model_path: Some("E:/nowhere/ghost.gguf".into()),
            ..Settings::default()
        };
        assert!(!set_up(&settings, &on_disk(&["whisper-base"])));
    }

    /// With no custom path set at all, behaviour must be exactly what it was
    /// before this feature existed.
    #[test]
    fn no_custom_path_behaves_exactly_as_before() {
        let models = on_disk(&["qwen3-4b-q4", "whisper-base", "bge-small-en-v1.5"]);
        assert!(set_up(&chosen(), &models));
        assert!(!set_up(&Settings::default(), &models));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_that_is_not_there_is_not_downloaded() {
        assert!(matches!(
            state_for(1_000, None, None),
            ModelState::NotDownloaded
        ));
    }

    #[test]
    fn a_complete_file_is_ready() {
        assert!(matches!(
            state_for(1_000, Some(1_000), None),
            ModelState::Ready
        ));
    }

    /// Published sizes and bytes on disk rarely agree exactly, so a small
    /// shortfall is still a model.
    #[test]
    fn a_slightly_smaller_file_is_still_ready() {
        assert!(matches!(
            state_for(1_000, Some(950), None),
            ModelState::Ready
        ));
    }

    /// The case this exists for: an interrupted download left on disk would
    /// otherwise load as a corrupt model and read as the app being broken.
    #[test]
    fn a_truncated_download_reports_as_still_arriving() {
        match state_for(1_000, Some(400), None) {
            ModelState::Downloading {
                received_bytes,
                total_bytes,
            } => {
                assert_eq!(received_bytes, 400);
                assert_eq!(total_bytes, 1_000);
            }
            other => panic!("expected downloading, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_file_is_not_a_model() {
        assert!(matches!(
            state_for(1_000, Some(0), None),
            ModelState::Downloading { .. }
        ));
    }

    /// The catalogue drives the onboarding default, so its shape matters:
    /// something to transcribe with and something to reason with.
    #[test]
    fn the_catalogue_offers_both_kinds() {
        let all = catalogue();
        assert!(all.iter().any(|m| m.kind == ModelKind::Transcription));
        assert!(all.iter().any(|m| m.kind == ModelKind::Reasoning));
        assert!(all.iter().all(|m| m.size_bytes > 0));
        assert!(all.iter().all(|m| m.recommended_ram_bytes > m.size_bytes));
    }

    /// Ids are the filename on disk, so a collision would make two models the
    /// same file.
    /// Bytes on disk under a name nothing loads. Without this the status bar
    /// said "not downloaded" while a 2.5GB download sat at 22%.
    #[test]
    fn a_part_file_reports_as_downloading() {
        match state_for(1_000, None, Some(220)) {
            ModelState::Downloading {
                received_bytes,
                total_bytes,
            } => {
                assert_eq!(received_bytes, 220);
                assert_eq!(total_bytes, 1_000);
            }
            other => panic!("expected downloading, got {other:?}"),
        }
    }

    /// The finished file wins: a .part left behind must not mask it.
    #[test]
    fn a_complete_file_beats_a_leftover_part() {
        assert!(matches!(
            state_for(1_000, Some(1_000), Some(220)),
            ModelState::Ready
        ));
    }

    #[test]
    fn catalogue_ids_are_unique() {
        let mut ids: Vec<String> = catalogue().into_iter().map(|m| m.id).collect();
        let count = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    /// 4B is the measured floor for classification, so it has to be reachable
    /// on an ordinary laptop. A 16GB machine reports about 14.9GB once
    /// firmware has taken its share, and recommending against the sticker
    /// number hands that machine the model known not to work.
    #[test]
    fn a_sixteen_gigabyte_laptop_is_recommended_the_model_that_works() {
        let reported = 14_900_000_000u64;
        let best = catalogue()
            .into_iter()
            .filter(|m| m.kind == ModelKind::Reasoning)
            .filter(|m| m.recommended_ram_bytes <= reported)
            .max_by_key(|m| m.size_bytes)
            .unwrap();

        assert_eq!(
            best.id, "qwen3-4b-q4",
            "1.7B was measured as not good enough"
        );
    }

    /// An 8GB machine reports about 7.8, and still holds 4B: RAM carries the
    /// model, VRAM only decides how fast it answers.
    #[test]
    fn an_eight_gigabyte_machine_is_still_offered_a_model() {
        let reported = 7_800_000_000u64;
        let offered: Vec<ModelInfo> = catalogue()
            .into_iter()
            .filter(|m| m.recommended_ram_bytes <= reported)
            .collect();

        assert!(offered.iter().any(|m| m.kind == ModelKind::Transcription));
        let reasoning: Vec<&ModelInfo> = offered
            .iter()
            .filter(|m| m.kind == ModelKind::Reasoning)
            .collect();
        assert_eq!(reasoning.len(), 1, "4B and only 4B");
        assert_eq!(reasoning[0].id, "qwen3-4b-q4");
    }
}

/// Fetches a catalogue model, reporting through `model://progress`.
///
/// Downloading gates nothing (§9.4): a failure is announced and returned, and
/// capture still works without it.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    url: Option<String>,
) -> Result<()> {
    // A `url` marks a model from the fetched transcription catalogue
    // (`list_transcription_catalog`): the frontend already knows where it lives
    // and the file is `{id}.gguf` like any other. Size is unknown, which the
    // atomic `.part`->final rename makes fine -- a present file is complete.
    // Without a url the id must name a built-in, so onboarding is unchanged.
    let info = match url {
        Some(url) => ModelInfo {
            id: model_id.clone(),
            kind: ModelKind::Transcription,
            name: model_id.clone(),
            params: String::new(),
            quantization: String::new(),
            size_bytes: 0,
            recommended_ram_bytes: 0,
            state: ModelState::NotDownloaded,
            url,
        },
        None => catalogue()
            .into_iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| Error::NotFound(format!("no model called {model_id}")))?,
    };
    let dest = state.models_dir().join(format!("{}.gguf", info.id));

    // §9.4 decides the lane, not the caller. Transcription gates recording and
    // the embedder is tens of megabytes in front of gigabytes, so both take the
    // line; the reasoning model starts at once and yields to them, because a
    // 5GB bar showing nothing for the whole of onboarding reads as broken while
    // costing the small ones nothing to avoid.
    let pace = match info.kind {
        ModelKind::Transcription | ModelKind::Embedding => Pace::now(state.downloads.clone()),
        ModelKind::Reasoning => Pace::when_idle(state.downloads.clone()),
    };

    // Already here: announce it and stop. Re-onboarding, or asking twice, must
    // not spend 2.5GB of someone's connection on a file they already have.
    if let Ok(on_disk) = std::fs::metadata(&dest) {
        if matches!(
            state_for(info.size_bytes, Some(on_disk.len()), None),
            ModelState::Ready
        ) {
            let mut ready = info.clone();
            ready.state = ModelState::Ready;
            let _ = app.emit("model://progress", ready);
            return Ok(());
        }
    }

    // Off the runtime thread: this runs for minutes and `fetch` is blocking.
    tauri::async_runtime::spawn_blocking(move || run_download(app, info, dest, pace))
        .await
        .map_err(|e| Error::Other(format!("the download task did not finish: {e}")))?
}

fn run_download(app: AppHandle, info: ModelInfo, dest: PathBuf, pace: Pace) -> Result<()> {
    let announce = |state: ModelState| {
        let mut snapshot = info.clone();
        snapshot.state = state;
        let _ = app.emit("model://progress", snapshot);
    };

    let mut last = std::time::Instant::now()
        .checked_sub(PROGRESS_EVERY)
        .unwrap_or_else(std::time::Instant::now);
    let mut report = |received: u64, total: u64| {
        if last.elapsed() < PROGRESS_EVERY && received != total {
            return;
        }
        last = std::time::Instant::now();
        announce(ModelState::Downloading {
            received_bytes: received,
            total_bytes: total,
        });
    };

    let outcome = crate::model::download::fetch_paced(&info.url, &dest, &mut report, pace);

    announce(match &outcome {
        Ok(()) => ModelState::Ready,
        Err(e) => ModelState::Failed {
            error: e.to_string(),
        },
    });
    outcome
}

/// The full transcription catalogue -- every transcribe.cpp model the library
/// author publishes -- fetched from Hugging Face. Kept apart from `list_models`
/// so onboarding stays offline and static; only Settings pays the network cost.
#[tauri::command]
pub async fn list_transcription_catalog(state: State<'_, AppState>) -> Result<Vec<ModelInfo>> {
    let dir = state.models_dir();
    tauri::async_runtime::spawn_blocking(move || crate::model::remote::list(&dir))
        .await
        .map_err(|e| Error::Other(format!("the catalogue lookup did not finish: {e}")))?
}

/// Removes a downloaded model and any partial, freeing the space and returning
/// the entry to NotDownloaded. `model_id` is the on-disk name; it is validated
/// as a bare filename so this can never reach outside the models directory.
#[tauri::command]
pub fn delete_model(state: State<AppState>, model_id: String) -> Result<()> {
    let safe = !model_id.is_empty()
        && model_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !model_id.contains("..");
    if !safe {
        return Err(Error::NotFound(format!("no model called {model_id}")));
    }
    let file = state.models_dir().join(format!("{model_id}.gguf"));
    let part = crate::model::download::part_path(&file);
    for path in [file, part] {
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}
