/**
 * TauriBridge — the only backend (§9.4). Each method maps to a
 * #[tauri::command] in src-tauri/src/commands/.
 */

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type {
  ActionItem,
  Edge,
  Entry,
  ModelInfo,
  Question,
  Register,
  Settings,
  Span,
  SystemProfile,
} from '@/lib/types';
import type {
  Bridge,
  CorpusImport,
  Exported,
  ImportMode,
  UploadPreview,
  NewEntryDraft,
  NewTypeDraft,
  SampleLoad,
  SearchHit,
  TypePatchDraft,
  Unsubscribe,
} from './index';
import type { TypeDefinition } from '@/lib/scene/classification';

/**
 * Tauri event listeners register asynchronously, so the unsubscribe function
 * has to survive being called before registration completes.
 */
function subscribe<T>(event: string, cb: (payload: T) => void): Unsubscribe {
  let stop: (() => void) | null = null;
  let cancelled = false;

  void listen<T>(event, (e) => cb(e.payload)).then((unlisten) => {
    if (cancelled) unlisten();
    else stop = unlisten;
  });

  return () => {
    cancelled = true;
    stop?.();
  };
}

export class TauriBridge implements Bridge {
  readonly kind = 'tauri' as const;

  // -- corpus -------------------------------------------------------------

  listEntries(): Promise<Entry[]> {
    return invoke('list_entries');
  }

  getEntry(id: string): Promise<Entry | null> {
    return invoke('get_entry', { id });
  }

  listChildren(entryId: string): Promise<Entry[]> {
    return invoke('list_children', { entryId });
  }

  listEdges(): Promise<Edge[]> {
    return invoke('list_edges');
  }

  createEntry(draft: NewEntryDraft): Promise<Entry> {
    return invoke('create_entry', { draft });
  }

  moveEntry(id: string, x: number, y: number): Promise<Entry> {
    return invoke('move_entry', { id, x, y });
  }

  deleteEntry(id: string): Promise<void> {
    return invoke('delete_entry', { id });
  }

  correctTranscript(entryId: string, transcript: string): Promise<Entry> {
    return invoke('correct_transcript', { entryId, transcript });
  }

  setRegister(entryId: string, register: Register): Promise<Entry> {
    return invoke('set_register', { entryId, register });
  }

  setEntryType(entryId: string, typeId: string): Promise<Entry> {
    return invoke('set_entry_type', { entryId, typeId });
  }

  entryMdx(entryId: string): Promise<string> {
    return invoke('entry_mdx', { entryId });
  }

  ensureEnriched(entryId: string): Promise<boolean> {
    return invoke('ensure_enriched', { entryId });
  }

  // -- search ---------------------------------------------------------------

  searchEntries(query: string): Promise<SearchHit[]> {
    return invoke('search_entries', { query });
  }

  askRecall(query: string): Promise<{ answer: string; hits: Entry[] }> {
    return invoke('ask_recall', { query });
  }

  // -- capture ------------------------------------------------------------

  startRecording(): Promise<void> {
    return invoke('start_recording');
  }

  stopRecording(parentEdge: string | null = null, questionId: string | null = null): Promise<Entry> {
    return invoke('stop_recording', { parentEdge, questionId });
  }

  discardRecording(): Promise<void> {
    return invoke('discard_recording');
  }

  undoDiscard(): Promise<Entry | null> {
    return invoke('undo_discard');
  }

  /** cpal capture emits these; the equalizer is the only animation in the app (§8). */
  partialTranscript(): Promise<string> {
    return invoke('partial_transcript');
  }

  onAmplitude(cb: (level: number) => void): Unsubscribe {
    // Polled, not pushed. recording_level exists for exactly this and nothing
    // ever emitted capture://amplitude, so the bars sat still and a live mic
    // looked muted. 33ms is the refresh the equalizer was built against.
    let stopped = false;
    const timer = setInterval(() => {
      void invoke<number>('recording_level')
        .then((level) => {
          if (!stopped) cb(Math.min(1, Math.max(0, level)));
        })
        .catch(() => {});
    }, 33);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }

  // -- enrichment ---------------------------------------------------------

  getQuestion(entryId: string): Promise<Question | null> {
    return invoke('get_question', { entryId });
  }

  listQuestions(): Promise<Question[]> {
    return invoke('list_questions');
  }

  askQuestion(entryId: string, span?: Span | null): Promise<Question> {
    return invoke('ask_question', { entryId, span: span ?? null });
  }

  runProbe(entryId: string, probeId: string, span?: Span | null): Promise<Question> {
    return invoke('run_probe', { entryId, probeId, span: span ?? null });
  }

  dismissQuestion(entryId: string, questionId: string): Promise<void> {
    return invoke('dismiss_question', { entryId, questionId });
  }

  listProposedEdges(entryId: string): Promise<Edge[]> {
    return invoke('list_proposed_edges', { entryId });
  }

  dismissEdge(edgeId: string): Promise<void> {
    return invoke('dismiss_edge', { edgeId });
  }

  acceptEdge(edgeId: string): Promise<void> {
    return invoke('accept_edge', { edgeId });
  }

  createManualEdge(a: string, b: string, relation: Edge['relation']): Promise<Edge> {
    return invoke('create_manual_edge', { entryA: a, entryB: b, relation });
  }

  // -- action items -------------------------------------------------------

  listActionItems(): Promise<ActionItem[]> {
    return invoke('list_action_items');
  }

  setActionItemDone(id: string, done: boolean): Promise<void> {
    return invoke('set_action_item_done', { id, done });
  }

  // -- resolution ---------------------------------------------------------

  resolveEntry(entryId: string, text: string): Promise<Entry> {
    return invoke('resolve_entry', { entryId, text });
  }

  reopenEntry(entryId: string): Promise<Entry> {
    return invoke('reopen_entry', { entryId });
  }

  // -- system -------------------------------------------------------------

  getSystemProfile(): Promise<SystemProfile> {
    return invoke('get_system_profile');
  }

  listModels(): Promise<ModelInfo[]> {
    return invoke('list_models');
  }

  listTranscriptionCatalog(): Promise<ModelInfo[]> {
    return invoke('list_transcription_catalog');
  }

  downloadModel(modelId: string, url?: string): Promise<void> {
    return invoke('download_model', { modelId, url });
  }

  deleteModel(modelId: string): Promise<void> {
    return invoke('delete_model', { modelId });
  }

  setupComplete(): Promise<boolean> {
    return invoke('setup_complete');
  }

  onModelProgress(cb: (m: ModelInfo) => void): Unsubscribe {
    return subscribe<ModelInfo>('model://progress', cb);
  }

  onEntryEnriching(cb: (entryId: string) => void): Unsubscribe {
    return subscribe<string>('entry://enriching', cb);
  }

  onEntryEnriched(cb: (entryId: string) => void): Unsubscribe {
    return subscribe<string>('entry://enriched', cb);
  }

  async readAudio(entryId: string): Promise<ArrayBuffer | null> {
    // A typed entry rejects rather than returning null, and that is not an error
    // worth surfacing: the pill simply has nothing to play.
    try {
      return await invoke<ArrayBuffer>('read_audio', { entryId });
    } catch {
      return null;
    }
  }

  getSettings(): Promise<Settings> {
    return invoke('get_settings');
  }

  setSettings(patch: Partial<Settings>): Promise<Settings> {
    return invoke('set_settings', { patch });
  }

  pickModelFile(): Promise<string | null> {
    return invoke('pick_model_file');
  }

  // -- sample corpus ------------------------------------------------------

  loadSampleCorpus(): Promise<SampleLoad> {
    return invoke('load_sample_corpus');
  }

  clearSampleCorpus(): Promise<void> {
    return invoke('clear_sample_corpus');
  }

  importCorpus(data: CorpusImport, mode: ImportMode): Promise<void> {
    return invoke('import_corpus', { data, mode });
  }

  // -- export and upload ---------------------------------------------------

  exportArchive(withAudio: boolean): Promise<Exported | null> {
    return invoke('export_archive', { withAudio });
  }

  exportTranscripts(markdown: string): Promise<string | null> {
    return invoke('export_transcripts', { contents: markdown });
  }

  pickUpload(): Promise<UploadPreview | null> {
    return invoke('pick_upload');
  }

  applyUpload(mode: ImportMode): Promise<void> {
    return invoke('apply_upload', { mode });
  }

  // -- types ----------------------------------------------------------------

  listTypes(): Promise<TypeDefinition[]> {
    return invoke('list_types');
  }

  createType(draft: NewTypeDraft): Promise<TypeDefinition> {
    return invoke('create_type', { draft });
  }

  updateType(id: string, patch: TypePatchDraft): Promise<TypeDefinition> {
    return invoke('update_type', { id, patch });
  }

  deleteType(id: string): Promise<void> {
    return invoke('delete_type', { id });
  }
}
