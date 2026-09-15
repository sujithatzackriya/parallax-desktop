//! Process-wide state, and where the corpus lives on disk.

use crate::audio::recorder::Recording;
use crate::db;
use crate::error::Result;
use crate::llm::LlmProvider;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri_plugin_global_shortcut::Shortcut;

/// A recording thrown away but not yet gone. §4 keeps it for a minute rather
/// than asking "are you sure", because a confirmation dialog on every discard
/// is worse than an undo nobody uses.
pub struct Discarded {
    pub pcm: Vec<f32>,
    pub duration_ms: i64,
    pub at: std::time::Instant,
}

/// A take and the window it belongs to.
///
/// The owner rides with the recording rather than in a field beside it, so the
/// two cannot disagree: a stale owner would route the stop to a window that
/// is not recording, which is the bug this exists to fix.
pub struct InFlight {
    pub take: Recording,
    pub owner: String,
}

pub struct AppState {
    /// The window's connection. Every command it reads through is short, and
    /// they run on the main thread, so nothing slow may ever hold this lock.
    pub conn: Mutex<Connection>,
    /// Background work's own connection: enrichment, embedding, and a question
    /// someone asked for. These hold it across model calls that take seconds,
    /// which on the shared connection froze the app for as long as 21.9s. WAL
    /// lets the two read side by side, and a writer waits on the other rather
    /// than failing.
    background: Mutex<Connection>,
    /// Kept so the app can say where its data is rather than making the user
    /// guess, and so the audio directory hangs off the same root.
    pub root: PathBuf,
    /// At most one recording at a time -- there is one microphone and one
    /// hotkey, and a second concurrent take has no meaning.
    pub recording: Mutex<Option<InFlight>>,
    pub discarded: Mutex<Option<Discarded>>,
    /// The record hotkey actually registered with the OS -- parsed once at
    /// open from the saved setting (or the factory default), and kept current
    /// by `shortcuts::rebind_hotkey` so a rebind knows exactly what to
    /// unregister.
    pub hotkey: Mutex<Shortcut>,
    /// Armed only while a recording is in flight; `None` whenever
    /// nothing is recording, which must always be true when idle -- the popup
    /// must not steal a key from every other app the rest of the time.
    pub discard_shortcut: Mutex<Option<Shortcut>>,
    /// Where the bundled llama-server sits. `None` in a dev build that has not
    /// fetched it; an explicit setting still overrides either way.
    pub llama_dir: Mutex<Option<PathBuf>>,
    /// Started on first use and kept for as long as the residency setting says,
    /// then released -- see `release_idle_at`.
    pub llama: Mutex<Option<crate::llm::llama_server::LlamaServer>>,
    /// When the reasoning model last finished a call.
    pub llama_used: Mutex<Option<std::time::Instant>>,
    /// Its own process: a generative llama-server refuses the embedding
    /// endpoint outright, and `--embedding` disables generation, so the two
    /// cannot share one. Tens of megabytes on the CPU, so keeping it costs
    /// little and starting it per capture would cost a second every time.
    pub embedder: Mutex<Option<crate::embed::llama::LlamaEmbedder>>,
    /// How many downloads that gate something are in flight. Process-wide
    /// because the reasoning model's download yields to them, and it can only
    /// do that by observing the same counter they raise.
    pub downloads: Arc<crate::model::download::Gate>,
    /// Entries with an enrichment pass in flight.
    ///
    /// Opening a note now asks for one if it never got it, and a note can be
    /// opened repeatedly while the first pass is still running -- each of which
    /// would otherwise append its own question, since `questions` has no
    /// uniqueness constraint the way edges do.
    pub enriching: Mutex<std::collections::HashSet<String>>,
    /// An upload read and waiting for its merge-or-replace answer.
    pub pending_upload: Mutex<Option<crate::mdx::archive::Contents>>,
}

impl AppState {
    pub fn open(root: PathBuf) -> Result<Self> {
        let conn = db::open(&root.join("corpus.db"))?;
        // Second, so the first has already migrated and this finds it current.
        let background = db::open(&root.join("corpus.db"))?;
        std::fs::create_dir_all(root.join("audio"))?;
        // Read up front so the OS registration `run()` performs reflects
        // whatever was last saved, not the factory default -- a rebind that
        // survived to the next launch used to silently revert to it, because
        // nothing here ever consulted the setting at all.
        let saved_hotkey = db::settings::get(&conn)?.hotkey;
        let hotkey =
            crate::shortcuts::parse(&saved_hotkey).unwrap_or_else(crate::shortcuts::default_hotkey);
        Ok(Self {
            conn: Mutex::new(conn),
            background: Mutex::new(background),
            root,
            recording: Mutex::new(None),
            discarded: Mutex::new(None),
            hotkey: Mutex::new(hotkey),
            discard_shortcut: Mutex::new(None),
            llama_dir: Mutex::new(None),
            llama: Mutex::new(None),
            llama_used: Mutex::new(None),
            embedder: Mutex::new(None),
            downloads: crate::model::download::Gate::new(),
            enriching: Mutex::new(std::collections::HashSet::new()),
            pending_upload: Mutex::new(None),
        })
    }

    /// A panic inside one command poisons the mutex, and every later command
    /// would then panic on a lock it could otherwise have used. The connection
    /// itself is not left inconsistent -- SQLite rolls back an incomplete
    /// statement -- so recovering is better than bricking the session.
    pub fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The lock itself, for work that takes it per step rather than holding it.
    pub fn background_conn(&self) -> &Mutex<Connection> {
        &self.background
    }

    /// The connection background work reads and writes through.
    pub fn background_db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.background
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The window a take belongs to, or `None` when nothing is recording.
    pub fn recording_owner(&self) -> Option<String> {
        self.recording
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|in_flight| in_flight.owner.clone())
    }

    pub fn audio_dir(&self) -> PathBuf {
        self.root.join("audio")
    }

    /// Matches --fit's own floor, so the fit never has to shrink it.
    pub const REASONING_CONTEXT: u32 = 4096;

    /// The reasoning model to actually load: a valid custom file first, the
    /// one onboarding chose otherwise.
    pub fn reasoning_model(&self, custom: Option<&str>, model_id: Option<&str>) -> Option<PathBuf> {
        resolve_reasoning_model(custom, model_id, &self.models_dir())
    }

    /// The embedding model, if one was chosen and its file is there.
    ///
    /// `None` is an ordinary state, not an error: topics propose candidates on
    /// their own and cosine only reorders them, so an absent embedder costs
    /// ranking quality and nothing else.
    pub fn embedding_model(&self, model_id: Option<&str>) -> Option<PathBuf> {
        let path = self.models_dir().join(format!("{}.gguf", model_id?));
        path.is_file().then_some(path)
    }

    /// The llama-server binary, if there is one to find.
    pub fn llama_binary(&self) -> Option<PathBuf> {
        let explicit = db::settings::get(&self.db())
            .ok()
            .and_then(|s| s.llama_server_path)
            .map(PathBuf::from);
        let bundled = self
            .llama_dir
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        crate::llm::binary::resolve(explicit.as_deref(), bundled.as_deref())
            .or_else(crate::llm::binary::on_path)
    }

    /// Starts the reasoning server if it is not already up, and runs `f` against
    /// it. Returns `Ok(None)` when there is no binary or no model: enrichment is
    /// allowed to be absent (§9.4), and absent is not an error.
    /// Whether an enrichment pass would find a model to run against, without
    /// starting one to find out.
    ///
    /// The same two conditions `with_reasoning` checks before it spawns
    /// anything. Asked so the sample can say plainly that it will stay as
    /// spoken until the model lands, rather than looking like that is all the
    /// sample is.
    pub fn reasoning_available(&self) -> bool {
        let Some(_) = self.llama_binary() else {
            return false;
        };
        let Ok(settings) = db::settings::get(&self.db()) else {
            return false;
        };
        self.reasoning_model(
            settings.custom_reasoning_model_path.as_deref(),
            settings.model_id.as_deref(),
        )
        .is_some()
    }

    pub fn with_reasoning<T>(
        &self,
        f: impl FnOnce(&dyn crate::llm::LlmProvider) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(binary) = self.llama_binary() else {
            return Ok(None);
        };
        let settings = db::settings::get(&self.db())?;
        let Some(model) = self.reasoning_model(
            settings.custom_reasoning_model_path.as_deref(),
            settings.model_id.as_deref(),
        ) else {
            return Ok(None);
        };

        let mut slot = self.llama.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_none_or(|s| !s.ready()) {
            let devices = crate::llm::binary::devices(&binary);
            let device = crate::llm::binary::best_device(&devices).map(|d| d.id.clone());
            *slot = Some(crate::llm::llama_server::LlamaServer::spawn(
                &binary,
                &model,
                settings.reasoning_backend,
                device.as_deref(),
                Self::REASONING_CONTEXT,
            )?);
        }
        let server = slot.as_ref().expect("just started");
        let result = f(server).map(Some);
        // Stamped on the way out, still under the slot's lock, so the sweeper
        // can never see a finished call as older than it is.
        *self.llama_used.lock().unwrap_or_else(|p| p.into_inner()) =
            Some(std::time::Instant::now());
        result
    }

    /// Stops the reasoning model once it has sat unused past what the
    /// residency setting keeps it for. Returns whether it stopped one.
    pub fn release_idle_at(&self, now: std::time::Instant) -> bool {
        // `with_reasoning` holds this lock for the whole of a call, so failing
        // to take it means a question is being answered right now.
        let Ok(mut slot) = self.llama.try_lock() else {
            return false;
        };
        if slot.is_none() {
            return false;
        }
        let Some(last) = *self.llama_used.lock().unwrap_or_else(|p| p.into_inner()) else {
            return false;
        };
        let keep = db::settings::get(&self.db())
            .map(|s| s.residency)
            .unwrap_or(crate::model::Residency::Warm)
            .keep_for();
        if now.saturating_duration_since(last) < keep {
            return false;
        }
        if let Some(server) = slot.take() {
            server.stop();
        }
        true
    }

    /// Stops the reasoning server outright, independent of how long it has
    /// sat idle -- the file it would keep serving is no longer the one
    /// settings name, so keeping it warm would answer the next call with the
    /// wrong model. Blocks on the same lock `with_reasoning` holds for a
    /// whole call, so this waits for one in flight rather than cutting it off.
    pub fn stop_reasoning(&self) -> bool {
        let mut slot = self.llama.lock().unwrap_or_else(|p| p.into_inner());
        match slot.take() {
            Some(server) => {
                server.stop();
                true
            }
            None => false,
        }
    }

    /// Runs `f` against the embedder, starting it if it is not up.
    ///
    /// `Ok(None)` whenever there is no embedder to run -- no binary, no model
    /// chosen, or the file not downloaded yet. That is an ordinary state, not
    /// a failure: topics propose candidates on their own and cosine only
    /// reorders them, so an absent embedder costs ranking and nothing else.
    pub fn with_embedder<T>(
        &self,
        f: impl FnOnce(&dyn crate::embed::Embedder) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(binary) = self.llama_binary() else {
            return Ok(None);
        };
        let settings = db::settings::get(&self.db())?;
        let Some(id) = settings.embedding_model_id else {
            return Ok(None);
        };
        let Some(model) = self.embedding_model(Some(&id)) else {
            return Ok(None);
        };

        let mut slot = self.embedder.lock().unwrap_or_else(|p| p.into_inner());
        // Restarted when the chosen model changes, or the vectors it writes
        // would be compared against a space they do not belong to.
        if slot
            .as_ref()
            .is_none_or(|e| !e.ready() || crate::embed::Embedder::model_id(e) != id)
        {
            if let Some(old) = slot.take() {
                old.stop();
            }
            *slot = Some(crate::embed::llama::LlamaEmbedder::spawn(
                &binary, &model, &id,
            )?);
        }
        let embedder = slot.as_ref().expect("just started");
        f(embedder).map(Some)
    }

    /// Unlinks a recording, and only ever one inside the corpus.
    ///
    /// A stored path is data; an imported one is data from a stranger. Reads
    /// have been checked since the start, and deleting on an unchecked path is
    /// strictly worse -- a crafted export naming `audio/../../something` had
    /// replace delete it. Silent on refusal: a path that does not resolve into
    /// the corpus has no file of ours behind it to report on.
    pub fn remove_audio(&self, relative: &str) {
        let (Ok(full), Ok(audio_dir)) = (
            self.root.join(relative).canonicalize(),
            self.audio_dir().canonicalize(),
        ) else {
            return;
        };
        if full.starts_with(&audio_dir) {
            let _ = std::fs::remove_file(full);
        }
    }

    /// The row goes first: a failed unlink orphans a file, while the reverse
    /// destroys the recording of an entry that still exists.
    pub fn delete_entry(&self, id: &str) -> Result<()> {
        let audio = {
            let conn = self.db();
            let path = db::entries::audio_path(&conn, id)?;
            db::entries::delete(&conn, id)?;
            path
        };
        if let Some(relative) = audio {
            self.remove_audio(&relative);
        }
        Ok(())
    }

    /// Everything that outlives a destructor. Called from the exit handler
    /// rather than `Drop`, because Tauri exits through `std::process::exit`.
    pub fn shutdown(&self) {
        // Explicit, because Tauri exits through std::process::exit and Drop
        // never runs -- which is how a llama-server child got orphaned before.
        if let Some(server) = self.llama.lock().unwrap_or_else(|p| p.into_inner()).take() {
            server.stop();
        }
        if let Some(embedder) = self
            .embedder
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
        {
            embedder.stop();
        }
        // A recording in flight is dropped, which stops the stream. Nothing is
        // written: an app being quit mid-sentence did not ask for a note.
        self.recording
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// The transcription model the user chose, if it is actually there.
    ///
    /// Named rather than enumerated: with tiny and base both present, taking
    /// whichever the filesystem yields first would silently ignore the setting.
    /// `None` is a normal state, not an error -- capture works without it, and
    /// the audio is the record the transcript is derived from.
    pub fn transcription_model(&self, custom: Option<&str>, chosen: &str) -> Option<PathBuf> {
        resolve_transcription_model(custom, chosen, &self.models_dir())
    }
}

/// A custom path counts only when it is a real file with a `.gguf`
/// extension (case-insensitive) -- anything else, including a typo or a
/// moved file, must fall back exactly as if nothing had been set, not fail
/// silently with reasoning or transcription simply absent.
pub fn resolve_custom_gguf(custom: Option<&str>) -> Option<PathBuf> {
    let trimmed = custom?.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    let is_gguf = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    (is_gguf && path.is_file()).then_some(path)
}

/// A valid custom file always wins; otherwise the catalogue file onboarding
/// chose, exactly as before this setting existed.
pub fn resolve_reasoning_model(
    custom: Option<&str>,
    model_id: Option<&str>,
    models_dir: &Path,
) -> Option<PathBuf> {
    resolve_custom_gguf(custom).or_else(|| {
        let path = models_dir.join(format!("{}.gguf", model_id?));
        path.is_file().then_some(path)
    })
}

/// Same rule for speech-to-text: a valid custom GGUF wins, otherwise the
/// catalogue model the settings name (`chosen` is its id, i.e. `{id}.gguf`).
pub fn resolve_transcription_model(
    custom: Option<&str>,
    chosen: &str,
    models_dir: &Path,
) -> Option<PathBuf> {
    resolve_custom_gguf(custom).or_else(|| {
        let path = models_dir.join(format!("{chosen}.gguf"));
        path.is_file().then_some(path)
    })
}

/// A `data` directory beside the executable wins if it exists, which is the
/// whole portable-install mechanism: make the folder and the corpus follows
/// the binary onto a stick. Otherwise the platform's app-data directory, which
/// is where an installed app belongs and survives an upgrade.
///
/// Discoverable on purpose. A tool holding your private thinking should not
/// hide where it keeps it.
pub fn resolve_root(app_data: &Path) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let portable = dir.join("data");
            if portable.is_dir() {
                return portable;
            }
        }
    }
    app_data.to_path_buf()
}

#[cfg(test)]
mod custom_model_resolution_tests {
    use super::*;

    /// A real, empty file is enough to exist for these tests -- resolution
    /// never reads the model, only checks it is there.
    fn gguf_file(dir: &std::path::Path, name: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, b"not a real model, just needs to exist").unwrap();
        path
    }

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "parallax-custom-model-{tag}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn a_valid_custom_reasoning_file_wins_over_the_catalogue() {
        let dir = scratch("reasoning-wins");
        let models_dir = dir.join("models");
        let catalogue = gguf_file(&models_dir, "qwen3-4b-q4.gguf");
        let custom = gguf_file(&dir, "mine.gguf");

        let resolved = resolve_reasoning_model(
            Some(custom.to_str().unwrap()),
            Some("qwen3-4b-q4"),
            &models_dir,
        );
        assert_eq!(resolved, Some(custom));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = catalogue; // exists only to prove it was not the one picked
    }

    #[test]
    fn none_falls_back_to_the_catalogue_unchanged() {
        let dir = scratch("none-falls-back");
        let models_dir = dir.join("models");
        let catalogue = gguf_file(&models_dir, "qwen3-4b-q4.gguf");

        assert_eq!(
            resolve_reasoning_model(None, Some("qwen3-4b-q4"), &models_dir),
            Some(catalogue)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_custom_file_falls_back_to_the_catalogue() {
        let dir = scratch("missing-falls-back");
        let models_dir = dir.join("models");
        let catalogue = gguf_file(&models_dir, "qwen3-4b-q4.gguf");
        let ghost = dir.join("gone.gguf");

        assert_eq!(
            resolve_reasoning_model(
                Some(ghost.to_str().unwrap()),
                Some("qwen3-4b-q4"),
                &models_dir
            ),
            Some(catalogue)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_gguf_custom_file_falls_back_to_the_catalogue() {
        let dir = scratch("wrong-ext-falls-back");
        let models_dir = dir.join("models");
        let catalogue = gguf_file(&models_dir, "qwen3-4b-q4.gguf");
        let wrong_ext = gguf_file(&dir, "mine.bin");

        assert_eq!(
            resolve_reasoning_model(
                Some(wrong_ext.to_str().unwrap()),
                Some("qwen3-4b-q4"),
                &models_dir
            ),
            Some(catalogue)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_custom_file_and_no_catalogue_choice_resolves_to_nothing() {
        let dir = scratch("nothing");
        let models_dir = dir.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        assert_eq!(resolve_reasoning_model(None, None, &models_dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_valid_custom_transcription_file_wins_over_the_catalogue() {
        let dir = scratch("transcription-wins");
        let models_dir = dir.join("models");
        gguf_file(&models_dir, "whisper-base.gguf");
        let custom = gguf_file(&dir, "my-whisper.gguf");

        let resolved = resolve_transcription_model(
            Some(custom.to_str().unwrap()),
            "whisper-base",
            &models_dir,
        );
        assert_eq!(resolved, Some(custom));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transcription_none_falls_back_to_the_catalogue_unchanged() {
        let dir = scratch("transcription-none");
        let models_dir = dir.join("models");
        let catalogue = gguf_file(&models_dir, "whisper-base.gguf");

        assert_eq!(
            resolve_transcription_model(None, "whisper-base", &models_dir),
            Some(catalogue)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stop_reasoning_on_an_empty_slot_is_a_harmless_no_op() {
        let dir = scratch("stop-empty");
        let state = AppState::open(dir.clone()).unwrap();
        assert!(!state.stop_reasoning());
        drop(state);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn stop_reasoning_stops_a_running_server() {
        let dir = scratch("stop-running");
        let state = AppState::open(dir.clone()).unwrap();
        let child = std::process::Command::new("cmd")
            .args(["/C", "ping -n 60 127.0.0.1 > nul"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        *state.llama.lock().unwrap() = Some(crate::llm::llama_server::stand_in(child));

        assert!(state.stop_reasoning());
        assert!(state.llama.lock().unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(
        residency: crate::model::Residency,
        idle: std::time::Duration,
    ) -> (AppState, PathBuf, u32) {
        let dir = std::env::temp_dir().join(format!("parallax-residency-{}", uuid::Uuid::new_v4()));
        let state = AppState::open(dir.clone()).unwrap();
        {
            let conn = state.db();
            let mut settings = db::settings::get(&conn).unwrap();
            settings.residency = residency;
            db::settings::set(&conn, &settings).unwrap();
        }
        let child = std::process::Command::new("cmd")
            .args(["/C", "ping -n 60 127.0.0.1 > nul"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();
        *state.llama.lock().unwrap() = Some(crate::llm::llama_server::stand_in(child));
        *state.llama_used.lock().unwrap() = Some(std::time::Instant::now());
        let _ = idle;
        (state, dir, pid)
    }

    /// Found in the packaged app: while a note was being enriched, reading one
    /// note's file took 21.9s. A pass held the only connection across every
    /// model call it made -- eight judge calls in `propose` alone -- and the
    /// window's commands run on the main thread, so the whole app waited on it.
    /// Background work holds a connection of its own, and what it writes is
    /// what the window reads.
    #[test]
    fn background_work_never_holds_the_connection_the_window_uses() {
        let dir =
            std::env::temp_dir().join(format!("parallax-background-{}", uuid::Uuid::new_v4()));
        let state = AppState::open(dir).unwrap();

        // Held the way a pass holds it, for as long as the model takes.
        let background = state.background_db();
        let window = state.conn.try_lock();
        assert!(
            window.is_ok(),
            "the window's connection is taken by background work"
        );
        let window = window.unwrap();

        let id = db::create::create(
            &background,
            db::create::NewEntry {
                transcript: "written by a pass".into(),
                duration_ms: 1_000,
                fingerprint: vec![],
                parent_entry_id: None,
                local_only: None,
                typed: true,
            },
        )
        .unwrap()
        .id;
        assert!(db::entries::get(&window, &id).unwrap().is_some());
    }

    fn running(pid: u32) -> bool {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }

    /// Measured on the RTX 3050: a resident model holds the card in D0 at about
    /// 6W for as long as the app runs, and released it goes to D3, powered off.
    /// Cold is the setting that says to give it back.
    #[test]
    fn a_cold_model_left_idle_is_stopped() {
        use std::time::{Duration, Instant};
        let (state, dir, pid) = state_with(crate::model::Residency::Cold, Duration::ZERO);

        let later = Instant::now() + Duration::from_secs(120);
        assert!(state.release_idle_at(later), "an idle cold model was kept");
        assert!(state.llama.lock().unwrap().is_none());
        assert!(!running(pid), "the server was forgotten but not stopped");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A note recorded a minute ago is a session in progress, and warm is for
    /// exactly that.
    #[test]
    fn a_warm_model_survives_a_short_pause() {
        use std::time::{Duration, Instant};
        let (state, dir, pid) = state_with(crate::model::Residency::Warm, Duration::ZERO);

        let later = Instant::now() + Duration::from_secs(120);
        assert!(
            !state.release_idle_at(later),
            "warm released after two minutes"
        );
        assert!(state.llama.lock().unwrap().is_some());
        state.shutdown();
        assert!(!running(pid));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The whole residency setting did nothing before this. Warm still gives
    /// the card back once the session is plainly over.
    #[test]
    fn a_warm_model_is_released_once_the_session_is_over() {
        use std::time::{Duration, Instant};
        let (state, dir, _pid) = state_with(crate::model::Residency::Warm, Duration::ZERO);

        let later = Instant::now() + Duration::from_secs(60 * 60);
        assert!(state.release_idle_at(later));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `with_reasoning` holds the slot's lock for a whole call. A call in
    /// flight is the one thing the sweeper must never cut short.
    #[test]
    fn a_model_answering_a_question_is_never_stopped() {
        use std::time::{Duration, Instant};
        let (state, dir, _pid) = state_with(crate::model::Residency::Cold, Duration::ZERO);

        let held = state.llama.lock().unwrap();
        let later = Instant::now() + Duration::from_secs(60 * 60);
        assert!(!state.release_idle_at(later), "stopped mid-call");
        drop(held);
        state.shutdown();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn app_data_is_the_default() {
        let app_data = Path::new("C:/app-data/parallax");
        assert_eq!(resolve_root(app_data), app_data.to_path_buf());
    }

    /// An import supplies its own audio paths, so a crafted export must not be
    /// able to name a file outside the corpus and have replace delete it. The
    /// read path has been checked since the start; the unlink was not.
    #[test]
    fn a_crafted_audio_path_is_not_deleted() {
        let dir = std::env::temp_dir().join(format!("parallax-unlink-{}", uuid::Uuid::new_v4()));
        let state = AppState::open(dir.clone()).unwrap();

        let outside = dir
            .parent()
            .unwrap()
            .join(format!("hostage-{}.wav", uuid::Uuid::new_v4()));
        std::fs::write(&outside, b"not yours").unwrap();
        let escape = format!(
            "audio/../../{}",
            outside.file_name().unwrap().to_string_lossy()
        );

        state.remove_audio(&escape);
        assert!(outside.is_file(), "an unlink escaped the corpus");

        let inside = state.audio_dir().join("e1.wav");
        std::fs::write(&inside, b"mine").unwrap();
        state.remove_audio("audio/e1.wav");
        assert!(
            !inside.exists(),
            "a recording inside the corpus must still go"
        );

        let _ = std::fs::remove_file(outside);
        drop(state);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn opening_creates_the_corpus_and_the_audio_directory() {
        let dir = std::env::temp_dir().join(format!("parallax-test-{}", uuid::Uuid::new_v4()));
        let state = AppState::open(dir.clone()).unwrap();

        assert!(dir.join("corpus.db").is_file());
        assert!(state.audio_dir().is_dir());

        drop(state);
        let _ = std::fs::remove_dir_all(dir);
    }
}
