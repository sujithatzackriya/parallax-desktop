pub mod audio;
pub mod commands;
pub mod db;
pub mod embed;
// Public for `tests/end_to_end.rs`, which drives the enrichment pass directly.
// It was private, so that whole file had stopped compiling and none of those
// tests had been running.
pub mod enrich;
pub mod error;
pub mod llm;
pub mod mdx;
pub mod model;
pub mod scene;
pub mod shortcuts;
pub mod state;
pub mod stt;
pub mod text;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, WindowEvent};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

/// The canvas. Hidden rather than closed, so the app outlives its window.
const MAIN: &str = "main";
/// The capture panel. Defined in tauri.conf.json, hidden until it has something
/// to show.
const PANEL: &str = "panel";

/// Gap from the corner of the work area, in logical pixels.
const PANEL_MARGIN: f64 = 16.0;

/// §4 — the hotkey starts recording immediately; the panel appears second.
///
/// Capture happens where you are looking. With the canvas focused the recording
/// belongs to it, and the entry opens when it lands. Anywhere else — the whole
/// point of a global shortcut — the panel takes it and the app stays where it
/// was, in the background.
///
/// Only the event goes out from here. The panel window shows itself once
/// recording is actually under way, so the two can never come apart: a webview
/// that is slow to start, or throws before it reaches the recorder, used to
/// leave an empty transparent window on screen swallowing clicks.
/// Which window a hotkey press belongs to.
///
/// A recording belongs to the window that started it until it ends. Routing on
/// focus alone sent the *stop* to whichever window happened to be focused, and
/// that window's store reads idle, so it called start instead; Rust refused it
/// as "already recording" and the take could no longer be stopped from the
/// keyboard at all. Focus decides only who takes a *new* recording.
fn hotkey_target(recording_owner: Option<&str>, canvas_focused: bool) -> &'static str {
    match recording_owner {
        Some(PANEL) => PANEL,
        // Anything else in flight belongs to the canvas: those are the only
        // two windows, and an unknown label is safer sent to the one the user
        // can see.
        Some(_) => MAIN,
        None if canvas_focused => MAIN,
        None => PANEL,
    }
}

fn on_hotkey(app: &AppHandle) {
    let in_canvas = app
        .get_webview_window(MAIN)
        .and_then(|w| w.is_focused().ok())
        .unwrap_or(false);

    let owner = app
        .try_state::<state::AppState>()
        .and_then(|state| state.recording_owner());

    let target = hotkey_target(owner.as_deref(), in_canvas);
    let _ = app.emit_to(target, "capture://hotkey", ());
}

/// Discard's global shortcut is armed only while something is
/// recording, so the owner is always Some in practice -- a stray event with
/// nothing recording is a race with the recording just having ended, and is
/// silently ignored the same way `on_hotkey` ignores an unknown window.
fn on_discard(app: &AppHandle) {
    if let Some(owner) = app
        .try_state::<state::AppState>()
        .and_then(|s| s.recording_owner())
    {
        let _ = app.emit_to(owner, "capture://discard", ());
    }
}

/// Bring the canvas forward, from the tray. `show` comes first: a window
/// sitting in the tray cannot take focus until it is on screen again.
fn reveal_main(app: &AppHandle) {
    if let Some(main) = app.get_webview_window(MAIN) {
        let _ = main.show();
        let _ = main.unminimize();
        let _ = main.set_focus();
    }
}

/// Recording has started, so there is now something to show.
///
/// Bottom-right of the work area, clear of the taskbar and out of whatever you
/// are actually reading — you hit the key mid-paragraph and keep reading, which
/// a panel in the middle of the screen makes impossible. Recomputed on every
/// show, because the display you are working on is the one it belongs on.
///
/// Deliberately no set_focus: taking the keyboard out from under whatever you
/// were typing in is the interruption §4 exists to avoid. Start and stop both
/// arrive on the global shortcut, so the panel has no use for focus.
#[tauri::command]
fn show_capture(app: AppHandle) {
    let Some(panel) = app.get_webview_window(PANEL) else {
        return;
    };

    if let (Ok(Some(monitor)), Ok(size)) = (panel.current_monitor(), panel.outer_size()) {
        let area = monitor.work_area();
        let margin = (PANEL_MARGIN * monitor.scale_factor()).round() as i32;
        let x = area.position.x + area.size.width as i32 - size.width as i32 - margin;
        let y = area.position.y + area.size.height as i32 - size.height as i32 - margin;
        let _ = panel.set_position(PhysicalPosition::new(x, y));
    }

    let _ = panel.show();
}

/// Capture is over. The panel leaves without summoning anything: a recording
/// made while you were reading something else should cost you nothing but the
/// keypress, and the entry is already on the canvas for whenever you next open
/// it. Transcription carries on behind this.
#[tauri::command]
fn hide_capture(app: AppHandle) {
    if let Some(panel) = app.get_webview_window(PANEL) {
        let _ = panel.hide();
    }
}

/// The tray is the app's real home: capture is global, so the process has to
/// outlive the canvas window for the hotkey to mean anything (§4).
fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open canvas", true, None::<&str>)?;
    let capture = MenuItem::with_id(app, "capture", "Start capture", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &capture, &separator, &quit])?;

    TrayIconBuilder::with_id("tray")
        .icon(
            app.default_window_icon()
                .cloned()
                .expect("bundled app icon"),
        )
        .tooltip("Parallax — Ctrl+Shift+Space to capture")
        .menu(&menu)
        // Windows convention: left click opens the window, right click opens
        // the menu. The default puts the menu on both.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "open" => reveal_main(app),
            "capture" => on_hotkey(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                reveal_main(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                // Read from `AppState` on every event rather than closing over a
                // fixed combo: the record hotkey can be rebound from Settings at
                // any time (see `shortcuts::rebind_hotkey`), and discard is
                // registered and torn down around each recording, so neither one
                // is a constant this closure could capture up front.
                .with_handler(|app, triggered, event| {
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }
                    let Some(state) = app.try_state::<state::AppState>() else {
                        return;
                    };
                    let is_hotkey =
                        *state.hotkey.lock().unwrap_or_else(|p| p.into_inner()) == *triggered;
                    if is_hotkey {
                        on_hotkey(app);
                        return;
                    }
                    let is_discard = state
                        .discard_shortcut
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .is_some_and(|d| d == *triggered);
                    if is_discard {
                        on_discard(app);
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            show_capture,
            hide_capture,
            commands::archive::export_archive,
            commands::archive::export_transcripts,
            commands::archive::pick_upload,
            commands::archive::apply_upload,
            commands::corpus::list_entries,
            commands::capture::start_recording,
            commands::capture::stop_recording,
            commands::capture::discard_recording,
            commands::capture::undo_discard,
            commands::capture::recording_level,
            commands::capture::partial_transcript,
            commands::corpus::create_entry,
            commands::corpus::get_entry,
            commands::corpus::read_audio,
            commands::corpus::list_children,
            commands::corpus::move_entry,
            commands::corpus::delete_entry,
            commands::corpus::resolve_entry,
            commands::corpus::reopen_entry,
            commands::corpus::correct_transcript,
            commands::corpus::set_register,
            commands::corpus::set_entry_type,
            commands::corpus::entry_mdx,
            commands::corpus::ensure_enriched,
            commands::corpus::import_corpus,
            commands::corpus::ask_recall,
            commands::enrichment::get_question,
            commands::enrichment::ask_question,
            commands::enrichment::run_probe,
            commands::enrichment::list_questions,
            commands::enrichment::dismiss_question,
            commands::enrichment::list_edges,
            commands::enrichment::list_proposed_edges,
            commands::enrichment::accept_edge,
            commands::enrichment::dismiss_edge,
            commands::enrichment::create_manual_edge,
            commands::enrichment::list_action_items,
            commands::enrichment::set_action_item_done,
            commands::corpus::search_entries,
            commands::corpus::load_sample_corpus,
            commands::corpus::clear_sample_corpus,
            commands::models::get_system_profile,
            commands::models::list_models,
            commands::models::setup_complete,
            commands::models::download_model,
            commands::models::list_transcription_catalog,
            commands::models::delete_model,
            commands::models::models_location,
            commands::models::pick_model_file,
            commands::system::get_settings,
            commands::system::set_settings,
            commands::system::corpus_location,
            commands::types::list_types,
            commands::types::create_type,
            commands::types::update_type,
            commands::types::delete_type,
        ])
        .setup(move |app| {
            // Resolved before anything else: every other subsystem hangs off
            // this root, and a failure here has to stop the app rather than
            // leave it running against nothing.
            let app_data = app.path().app_data_dir()?;
            let root = state::resolve_root(&app_data);
            println!("corpus at {}", root.display());
            let app_state = state::AppState::open(root)?;
            // Before anything spawns, so a server orphaned by a crash is gone
            // before this run tries to fit its own model beside it.
            llm::llama_server::reap_orphans(&app_state.models_dir());
            // Bundled beside the installed app; in a dev build it is still in
            // the source tree, which is why both are tried.
            for candidate in [
                app.path()
                    .resource_dir()
                    .ok()
                    .map(|d| d.join("binaries/llama")),
                Some(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries/llama")),
            ]
            .into_iter()
            .flatten()
            {
                if candidate.is_dir() {
                    println!("llama runtime at {}", candidate.display());
                    *app_state.llama_dir.lock().unwrap() = Some(candidate);
                    break;
                }
            }
            // Read before the state is moved into `manage`: whatever was parsed
            // from settings at `AppState::open` (the saved hotkey, or the
            // factory default if none was saved or it failed to parse) is what
            // gets registered, not a hardcoded combo -- a rebind that survived
            // to the next launch used to silently revert to Ctrl+Shift+Space.
            let hotkey = *app_state.hotkey.lock().unwrap_or_else(|p| p.into_inner());
            app.manage(app_state);

            // Gives the graphics card back once the reasoning model has sat
            // unused for as long as the residency setting keeps it. A loaded
            // model holds the card powered on for as long as it is loaded, and
            // this was the only thing that ever let it go -- the setting existed
            // and nothing read it.
            let handle = app.handle().clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(15));
                if let Some(state) = handle.try_state::<state::AppState>() {
                    state.release_idle_at(std::time::Instant::now());
                }
            });

            // Not fatal. Another instance, or any other app holding the same
            // combination, makes this fail -- and aborting setup means the
            // whole app dies at launch over a convenience. Observed: a second
            // instance panicked with "HotKey already registered" and never
            // reached a window. Capture is still reachable from the tray.
            if let Err(e) = app.global_shortcut().register(hotkey) {
                eprintln!("global hotkey unavailable, continuing without it: {e}");
            }
            build_tray(app.handle())?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the canvas sends the app to the tray instead of ending
            // it. A capture tool that only runs while its window is open is not
            // available from anywhere, which is the whole premise.
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == MAIN {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building application")
        .run(|app, event| {
            // Tauri exits with std::process::exit, which runs no destructors,
            // so nothing in managed state is ever dropped -- a llama-server
            // child would be orphaned on every quit, holding gigabytes.
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                if let Some(state) = app.try_state::<state::AppState>() {
                    state.shutdown();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug: start in the canvas, click away, press the hotkey. The stop
    /// went to the panel, which thought nothing was recording.
    #[test]
    fn a_recording_is_stopped_by_the_window_that_started_it() {
        assert_eq!(hotkey_target(Some(MAIN), false), MAIN);
        assert_eq!(hotkey_target(Some(PANEL), true), PANEL);
    }

    /// Focus still decides who takes a new one -- capture happens where you
    /// are looking (§4).
    #[test]
    fn with_nothing_in_flight_focus_decides() {
        assert_eq!(hotkey_target(None, true), MAIN);
        assert_eq!(hotkey_target(None, false), PANEL);
    }
}
