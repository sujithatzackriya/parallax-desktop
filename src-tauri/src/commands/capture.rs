//! Recording: start, stop, discard, undo.
//!
//! §4 -- the hotkey starts recording immediately and the panel appears second,
//! so `start_recording` must not wait on anything visual.

use crate::audio::{fingerprint, recorder, wav};
use crate::db;
use crate::error::{Error, Result};
use crate::model::Entry;
use crate::state::AppState;
use tauri::{Emitter, Manager, State};

/// §4 -- a discard is undoable for a minute, because escape meaning "throw it
/// away" while recording and "leave it" once stopped is a muscle-memory trap.
pub const UNDO_WINDOW_MS: u64 = 60_000;

/// Takes the calling window, because the take belongs to it until it ends: the
/// hotkey is global and routes the stop back to whoever started it.
#[tauri::command]
pub fn start_recording(
    app: tauri::AppHandle,
    window: tauri::Window,
    state: State<AppState>,
) -> Result<()> {
    let mut slot = state.recording.lock().unwrap_or_else(|p| p.into_inner());
    if slot.is_some() {
        return Err(Error::Other("already recording".into()));
    }
    *slot = Some(crate::state::InFlight {
        take: recorder::start()?,
        owner: window.label().to_string(),
    });
    drop(slot);
    // Armed for exactly the lifetime of this recording -- the popup
    // must not steal a key from every other app the rest of the time. A
    // settings read failure must not cost a recording that is already
    // running, so this falls back to the default rather than propagating.
    let discard_hotkey = db::settings::get(&state.db())
        .map(|s| s.discard_hotkey)
        .unwrap_or_else(|_| crate::model::Settings::default().discard_hotkey);
    crate::shortcuts::arm_discard(&app, state.inner(), &discard_hotkey);
    Ok(())
}

/// The live level for the equalizer bars. Polled rather than pushed: the
/// panel asks while it is on screen, and nothing has to be torn down when it
/// is not.
#[tauri::command]
pub fn recording_level(state: State<AppState>) -> f32 {
    state
        .recording
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| r.take.level())
        .unwrap_or(0.0)
}

/// The last ten seconds at 16kHz.
const PREVIEW_SAMPLES: usize = 160_000;

/// The panel shows nine words, three or four seconds of speech. Ten seconds
/// fills them with context to spare, and a word severed at the window's start
/// falls outside the nine shown.
fn preview_window(samples: &[f32]) -> &[f32] {
    &samples[samples.len().saturating_sub(PREVIEW_SAMPLES)..]
}

/// The transcript so far, for the panel to show while you are still talking.
///
/// Polled for the same reason the level is: the panel asks while it is on
/// screen, and nothing has to be torn down when it is not. Each call transcribes
/// only the recent audio: transcribing everything so far made every refresh a
/// full pass, so a long note fell further behind and held the CPU for as long as
/// it ran. The saved transcript is still one pass over the whole take. Empty
/// rather than an error when there is no model, too little audio, or no
/// recording.
#[tauri::command]
pub async fn partial_transcript(state: State<'_, AppState>) -> Result<String> {
    const MIN_SAMPLES: usize = 16_000;

    let samples = {
        let slot = state.recording.lock().unwrap_or_else(|p| p.into_inner());
        match slot.as_ref() {
            Some(in_flight) => in_flight.take.samples(),
            None => return Ok(String::new()),
        }
    };
    if samples.len() < MIN_SAMPLES {
        return Ok(String::new());
    }

    let settings = {
        let conn = state.db();
        db::settings::get(&conn)?
    };
    let Some(model) = state.transcription_model(
        settings.custom_transcription_model_path.as_deref(),
        &settings.transcription_model,
    ) else {
        return Ok(String::new());
    };

    // Always on the CPU: the reasoning model has first claim on VRAM, and this
    // runs repeatedly while a recording is in flight.
    Ok(crate::stt::transcribe(
        &model,
        preview_window(&samples),
        crate::model::ComputeBackend::Cpu,
    )?
    .text
    .trim()
    .to_string())
}

/// Stops, writes the audio, transcribes, and lands an entry.
///
/// Async so the transcription does not run on the thread pumping the window.
/// A long recording takes seconds even at 35x realtime, and doing that inline
/// freezes the UI and the hotkey with it.
///
/// Enrichment does not happen here. What this owes the caller is a row that
/// exists and a place on the canvas; classification and the question arrive
/// after, so a slow or absent model cannot cost someone their recording.
#[tauri::command]
pub async fn stop_recording(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    parent_edge: Option<String>,
    question_id: Option<String>,
) -> Result<Entry> {
    let recording = state
        .recording
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .ok_or_else(|| Error::Other("not recording".into()))?
        .take;
    // The recording is no longer in flight, so discard's global shortcut has
    // nothing left to mean -- disarmed here rather than left armed through
    // transcription, which would let it fire on whatever key it was bound to.
    crate::shortcuts::disarm_discard(&app, state.inner());

    let duration_ms = recording.elapsed_ms();
    let pcm = recording.stop();
    let entry = finish(&state, pcm, duration_ms, parent_edge, question_id)?;
    enrich_later(&app, entry.id.clone());
    Ok(entry)
}

/// One model call per candidate, so this is the cost of a capture. Eight is
/// the number Task 10 settled on: enough that a real connection is usually in
/// the set, few enough that the pass stays behind a single recording.
const PROPOSE_CAP: usize = 8;

/// Enrichment runs after the entry is safe on disk, never before. It is allowed
/// to be slow, absent or wrong, and none of that may cost a recording (§9.4).
///
/// Both ends of the pass are announced, and the settled event fires on every
/// exit including failure. A single event on success only would leave the
/// indicator spinning forever on the paths that are most likely to be taken --
/// no model installed, or a model that threw.
pub fn enrich_later(app: &tauri::AppHandle, entry_id: String) {
    enrich_in_order(app, vec![entry_id]);
}

/// Several notes at once -- the sample, an import -- read one at a time, oldest
/// first, the way they would have arrived had they been recorded.
///
/// Found in the packaged app: run side by side, the passes finished in whatever
/// order they happened to, so a note was compared against notes recorded after
/// it, and the shelves the early notes should have seeded were coined by
/// whichever pass got there first.
pub fn enrich_in_order(app: &tauri::AppHandle, ids: Vec<String>) {
    let app = app.clone();
    let queued: Vec<String> = {
        // One pass per note at a time. Opening a note asks for one if it never
        // got a pass, and a note can be opened again while the first is still
        // running -- each of which would append its own question, since
        // `questions` carries no uniqueness constraint the way edges do.
        let state = app.state::<AppState>();
        let mut in_flight = state.enriching.lock().unwrap_or_else(|p| p.into_inner());
        ids.into_iter()
            .filter(|id| in_flight.insert(id.clone()))
            .collect()
    };
    if queued.is_empty() {
        return;
    }
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let ordered = if queued.len() > 1 {
            let found = db::entries::oldest_first(&state.background_db(), &queued);
            found.unwrap_or_else(|e| {
                eprintln!("could not order the queue, reading it as given: {e}");
                queued.clone()
            })
        } else {
            queued.clone()
        };
        for entry_id in &ordered {
            pass(&app, &state, entry_id);
        }
        // A note deleted while it waited is not in `ordered`, and would
        // otherwise stay claimed and never be read again under that id.
        // Only those: a note already read may have been claimed again since.
        let mut in_flight = state.enriching.lock().unwrap_or_else(|p| p.into_inner());
        for id in queued.iter().filter(|id| !ordered.contains(id)) {
            in_flight.remove(id);
        }
    });
}

fn pass(app: &tauri::AppHandle, state: &AppState, entry_id: &str) {
    // Emitted before with_reasoning, which is where a cold llama-server is
    // started: the first pass after launch spends most of its time there, and
    // that wait is exactly what needs saying.
    let _ = app.emit("entry://enriching", entry_id);
    // Every step below holds a connection across model calls, so each uses
    // background work's own and never the one the window reads through.
    let done = state.with_reasoning(|provider| {
        crate::enrich::run::run(&state.background_db(), provider, entry_id)
    });

    // Separate from enrichment and after it, because it is ranking rather than
    // eligibility: topics already decided who this note can be compared
    // against, and the vector only orders them. An absent or failing embedder
    // therefore costs ordering and no connections at all.
    if let Err(e) = state.with_embedder(|embedder| {
        crate::embed::embed_now(&state.background_db(), embedder, entry_id)
    }) {
        eprintln!("embedding failed for {entry_id}: {e}");
    }
    // After embedding, because candidates are ordered by cosine and this note's
    // own vector has to exist for that to mean anything. And before the settled
    // event, or the indicator clears while the judge is still working -- one
    // model call per candidate is the slowest part of the whole pass.
    if let Err(e) = state.with_reasoning(|provider| {
        crate::enrich::propose::propose(&state.background_db(), provider, entry_id, PROPOSE_CAP)
    }) {
        eprintln!("proposing failed for {entry_id}: {e}");
    }

    match done {
        // No binary or no model yet: the question arrives when one lands.
        Ok(None) => {}
        Ok(Some(enriched)) => {
            if enriched.question_id.is_none() {
                println!("classified {entry_id}, no question");
            }
        }
        Err(e) => eprintln!("enrichment failed for {entry_id}: {e}"),
    }
    state
        .enriching
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(entry_id);
    let _ = app.emit("entry://enriched", entry_id);
}

/// Everything after the microphone stops, separated so the order in which a
/// recording becomes durable is testable without a microphone.
pub fn finish(
    state: &AppState,
    pcm: Vec<f32>,
    duration_ms: i64,
    parent_edge: Option<String>,
    question_id: Option<String>,
) -> Result<Entry> {
    // The recording cannot be recreated, so it is staged before anything that
    // can fail -- reading settings included.
    stage(state, &pcm, duration_ms);

    let settings = {
        let conn = state.db();
        db::settings::get(&conn)?
    };

    // On disk before transcription, not after: a model that fails to load must
    // cost the transcript and never the recording.
    let id = uuid::Uuid::new_v4().to_string();
    let full = state.root.join(format!("audio/{id}.wav"));
    let bytes = wav::write(&pcm, &full)?;

    let transcript = match state.transcription_model(
        settings.custom_transcription_model_path.as_deref(),
        &settings.transcription_model,
    ) {
        Some(model) => crate::stt::transcribe(&model, &pcm, settings.transcription_backend)?.text,
        // No model yet is not a lost recording: the audio is the record and
        // the transcript is derived from it, so it can be filled in later.
        None => String::new(),
    };

    let mut entry = land(
        state,
        id,
        &full,
        bytes,
        pcm,
        transcript,
        duration_ms,
        parent_edge,
    )?;

    let conn = state.db();
    if let Some(question_id) = question_id {
        db::questions::mark_answered(&conn, &question_id)?;
        // Which question, not only which note: a note can carry several.
        conn.execute(
            "UPDATE entries SET answers_question_id = ?2 WHERE id = ?1",
            rusqlite::params![entry.id, question_id],
        )?;
        entry.answers_question_id = Some(question_id);
    }

    // Only now is there another copy.
    state
        .discarded
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take();
    Ok(entry)
}

fn stage(state: &AppState, pcm: &[f32], duration_ms: i64) {
    *state.discarded.lock().unwrap_or_else(|p| p.into_inner()) = Some(crate::state::Discarded {
        pcm: pcm.to_vec(),
        duration_ms,
        at: std::time::Instant::now(),
    });
}

/// The row for audio that is already on disk. The file is written first so a
/// full disk cannot leave an entry on the canvas pointing at nothing.
#[allow(clippy::too_many_arguments)]
fn land(
    state: &AppState,
    id: String,
    full: &std::path::Path,
    bytes: u64,
    pcm: Vec<f32>,
    transcript: String,
    duration_ms: i64,
    parent_edge: Option<String>,
) -> Result<Entry> {
    let conn = state.db();
    let entry = db::create::create_with_id(
        &conn,
        id,
        db::create::NewEntry {
            transcript,
            duration_ms,
            fingerprint: fingerprint::downsample(&pcm),
            parent_entry_id: parent_edge,
            local_only: None,
            typed: false,
        },
    );

    match entry {
        Ok(entry) => {
            conn.execute(
                "UPDATE audio SET byte_size = ?2 WHERE entry_id = ?1",
                rusqlite::params![entry.id, bytes as i64],
            )?;
            Ok(entry)
        }
        Err(e) => {
            // No row, so the file is an orphan. Leaving it would accumulate
            // silently in the audio directory.
            let _ = std::fs::remove_file(full);
            Err(e)
        }
    }
}

/// §4 -- discard belongs in the recording state, not after it. You know it is
/// junk before you stop.
#[tauri::command]
pub fn discard_recording(app: tauri::AppHandle, state: State<AppState>) -> Result<()> {
    let recording = state
        .recording
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .ok_or_else(|| Error::Other("not recording".into()))?
        .take;
    // Same reasoning as stop_recording: nothing is in flight for the key to
    // discard any more.
    crate::shortcuts::disarm_discard(&app, state.inner());

    let duration_ms = recording.elapsed_ms();
    let pcm = recording.stop();

    // Held, not dropped. Nothing is written to the corpus, but the samples
    // stay recoverable for the length of the window.
    *state.discarded.lock().unwrap_or_else(|p| p.into_inner()) = Some(crate::state::Discarded {
        pcm,
        duration_ms,
        at: std::time::Instant::now(),
    });
    Ok(())
}

/// Async for the same reason as stop_recording: this transcribes too.
#[tauri::command]
pub async fn undo_discard(state: State<'_, AppState>) -> Result<Option<Entry>> {
    // Read, do not take. The staged samples are the only copy, and consuming
    // them before the replacement is written would let a failure downstream
    // destroy exactly the recording the undo window exists to protect.
    let staged = {
        let slot = state.discarded.lock().unwrap_or_else(|p| p.into_inner());
        match slot.as_ref() {
            Some(d) if d.at.elapsed().as_millis() as u64 <= UNDO_WINDOW_MS => {
                Some((d.pcm.clone(), d.duration_ms))
            }
            _ => None,
        }
    };
    let Some((pcm, duration_ms)) = staged else {
        return Ok(None);
    };

    // Same pipeline as a fresh stop, so undo cannot drift from it. Parent and
    // question are still lost here -- the staging slot does not carry them.
    Ok(Some(finish(&state, pcm, duration_ms, None, None)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `finish` takes its PCM as an argument precisely so this needs no
    /// microphone. A few hundred samples stand in for a take.
    fn a_take() -> Vec<f32> {
        (0..800).map(|i| (i as f32 * 0.02).sin() * 0.4).collect()
    }

    /// Distinct values, so the test can tell which end was kept.
    fn counting(seconds: usize) -> Vec<f32> {
        (0..seconds * 16_000).map(|i| i as f32).collect()
    }

    #[test]
    fn a_short_recording_is_previewed_whole() {
        let samples = counting(4);
        assert_eq!(preview_window(&samples), &samples[..]);
    }

    /// Each poll transcribed everything recorded so far, so a three-minute note
    /// cost a full pass per refresh to show nine words.
    #[test]
    fn a_long_recording_is_previewed_from_its_last_ten_seconds() {
        let samples = counting(180);
        let window = preview_window(&samples);
        assert_eq!(window.len(), PREVIEW_SAMPLES);
        assert_eq!(window, &samples[samples.len() - PREVIEW_SAMPLES..]);
    }

    fn corpus(tag: &str) -> (AppState, std::path::PathBuf) {
        let root =
            std::env::temp_dir().join(format!("parallax-capture-{tag}-{}", uuid::Uuid::new_v4()));
        let state = AppState::open(root.clone()).expect("a corpus");
        (state, root)
    }

    /// §9.4 -- the audio is the record and the transcript is derived from it, so
    /// a corpus with no speech model installed still keeps the take. This is the
    /// state every install is in before the first download finishes.
    #[test]
    fn a_take_lands_with_no_speech_model_installed() {
        let (state, root) = corpus("no-model");

        let entry = finish(&state, a_take(), 4_200, None, None).expect("the take landed");

        assert_eq!(entry.transcript, "", "no model, so nothing was transcribed");
        assert_eq!(entry.duration_ms, 4_200);
        let wav = root.join(entry.audio_path.as_ref().expect("a recording"));
        assert!(wav.is_file(), "the recording is not on disk");
        assert!(
            std::fs::metadata(&wav).unwrap().len() > 0,
            "the recording is empty"
        );

        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The whole reason staging comes first. A recording cannot be re-made, so
    /// anything that fails after the microphone stops has to leave it
    /// recoverable -- the four defects fixed on 10 September were all this
    /// shape. Failure is induced by putting a file where the audio directory
    /// belongs: `wav::write` calls `create_dir_all` first, so simply deleting
    /// the directory is not a failure at all -- it is recreated.
    #[test]
    fn a_failure_after_the_microphone_stops_still_leaves_the_take() {
        let (state, root) = corpus("staged");
        let audio = state.audio_dir();
        std::fs::remove_dir_all(&audio).expect("no audio directory");
        std::fs::write(&audio, b"not a directory").expect("a file in its place");

        let failed = finish(&state, a_take(), 4_200, None, None);
        assert!(failed.is_err(), "writing the WAV should have failed");

        {
            let slot = state.discarded.lock().unwrap_or_else(|p| p.into_inner());
            let staged = slot.as_ref().expect("the take was not staged");
            assert_eq!(staged.pcm.len(), a_take().len(), "the take was truncated");
            assert_eq!(staged.duration_ms, 4_200);
        }

        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The answer records which question it answered, not only which note --
    /// found missing when the mock was audited against this path.
    #[test]
    fn an_answer_records_the_question_it_answers() {
        let (state, root) = corpus("answers-question");
        let parent = {
            let conn = state.db();
            let parent = db::create::create(
                &conn,
                db::create::NewEntry {
                    transcript: "Standups are theatre.".into(),
                    duration_ms: 40_000,
                    fingerprint: vec![0.3],
                    parent_entry_id: None,
                    local_only: None,
                    typed: false,
                },
            )
            .unwrap();
            db::questions::insert(
                &conn,
                &crate::model::Question {
                    id: "q1".into(),
                    entry_id: parent.id.clone(),
                    text: "Where does this stop holding?".into(),
                    span: None,
                    answered: false,
                    dismissed: false,
                    provider_name: "test".into(),
                    created_at: "2026-09-14T00:00:00Z".into(),
                },
                &parent.transcript,
            )
            .unwrap();
            parent
        };

        let answer = finish(
            &state,
            a_take(),
            4_200,
            Some(parent.id.clone()),
            Some("q1".into()),
        )
        .expect("the answer landed");

        assert_eq!(answer.answers_question_id.as_deref(), Some("q1"));
        let stored = db::entries::get(&state.db(), &answer.id).unwrap().unwrap();
        assert_eq!(stored.answers_question_id.as_deref(), Some("q1"));

        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Staging is a safety net, not a second copy kept forever: once the entry
    /// is durable the slot is released, so an undo cannot resurrect a take that
    /// is already on the canvas.
    #[test]
    fn a_landed_take_is_no_longer_staged() {
        let (state, root) = corpus("released");

        finish(&state, a_take(), 4_200, None, None).expect("the take landed");

        assert!(
            state
                .discarded
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .is_none(),
            "the staging slot outlived the entry"
        );

        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }
}
