/**
 * The bridge — the one seam between UI and everything below it (§9.1): this
 * interface is both what the UI codes against and the spec for Rust's future
 * command surface. Import only via getBridge(), never ./tauri directly.
 */

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
import type { TypeDefinition } from '@/lib/scene/classification';

/** Unsubscribe. Every stream returns one; call it on unmount. */
export type Unsubscribe = () => void;

/** One match from searchEntries. Offsets are into the transcript;
 *  snippetStart/snippetEnd are the same match re-based into `snippet`. */
export interface SearchHit {
  entryId: string;
  start: number;
  end: number;
  /** ~90 chars of surrounding transcript, for display. */
  snippet: string;
  snippetStart: number;
  snippetEnd: number;
}

export interface NewEntryDraft {
  transcript: string;
  durationMs: number;
  fingerprint: number[];
  /** Set when this is an answer to a question — makes it a thread layer (§6.2). */
  parentEdge?: string | null;
  localOnly?: boolean;
  /** true for the typed-entry path (§4). Nobody dictates a list. */
  typed?: boolean;
}

export interface Bridge {
  readonly kind: 'tauri';

  // -- corpus -------------------------------------------------------------
  listEntries(): Promise<Entry[]>;
  getEntry(id: string): Promise<Entry | null>;
  /** Children of an entry — the layers behind its rings (§6.2). */
  listChildren(entryId: string): Promise<Entry[]>;
  listEdges(): Promise<Edge[]>;
  createEntry(draft: NewEntryDraft): Promise<Entry>;
  /** Manual placement override (§5.1). Auto-placement stays the default; this
   *  just overwrites the frozen position, it never re-solves the field. */
  moveEntry(id: string, x: number, y: number): Promise<Entry>;
  /** Removes one entry and any edges touching it. Children are orphaned, not
   *  deleted — an answer is still something you said (§6.2). */
  deleteEntry(id: string): Promise<void>;
  /**
   * Fixes what the speech-to-text heard wrong. The *only* edit a note takes:
   * a note is the verbatim record of what was said, so there is no append and
   * no rewrite — a note you can reword is a note you cannot cite.
   *
   * Returns the entry because the correction re-anchors it. Spans, questions
   * and action items are keyed on offsets into the transcript, so Rust re-finds
   * each one by its stored quote and drops the spans whose words are genuinely
   * gone; the caller must take the entry that comes back rather than patching
   * the transcript locally and keeping the old offsets. Rejects a blank
   * transcript — emptying a note is a delete, not a correction.
   */
  correctTranscript(entryId: string, transcript: string): Promise<Entry>;

  /**
   * Overrules the classifier on one note's register.
   *
   * §3.2 gives the invoked path to the user because the risk is theirs to
   * spend; this is the same argument one step earlier. The model guesses what
   * is at stake in a note, and the person who spoke it knows.
   *
   * Returns the entry so the caller takes what was stored rather than assuming
   * the write landed as sent.
   */
  setRegister(entryId: string, register: Register): Promise<Entry>;

  /**
   * Assigns a type by hand (§3.6 -- the editor's hint has always said to
   * write "manual" and tag entries yourself; nothing did until now).
   *
   * Locks the type against the next re-classification on the Rust side, so a
   * transcript correction or `ensureEnriched` catching up an old note cannot
   * silently take the choice back. Returns the entry so the caller takes what
   * was actually stored.
   */
  setEntryType(entryId: string, typeId: string): Promise<Entry>;

  /**
   * One note as the file it would be exported to: JSON frontmatter, then the
   * transcript verbatim.
   *
   * A viewer, not an editor. SQLite stays authoritative and offsets are frozen
   * at insert (§5.1), so this renders what a file *would* contain rather than
   * offering a way to write one. Built from the same assembly the exporter
   * uses, so the two cannot drift.
   */
  entryMdx(entryId: string): Promise<string>;

  /**
   * Asks for an enrichment pass on a note that never got one.
   *
   * §9.4 lets the reasoning model arrive late, and "late" used to mean
   * "never" for anything captured before it landed: the pass fires once, at
   * capture, and nothing retried. Opening a note is when someone is actually
   * looking at it, so it is when the gap is worth closing.
   *
   * Resolves true when a pass started. False means there was nothing to do —
   * already classified, or still no model to do it with.
   */
  ensureEnriched(entryId: string): Promise<boolean>;

  // -- search ---------------------------------------------------------------
  /**
   * Case-insensitive search over transcripts. Substring by default, because
   * most searches are for a phrase half-remembered. A query wrapped in double
   * quotes matches whole words only — the way to ask for a subject rather than
   * any word containing it (`ai` otherwise finds maintain and explaining).
   */
  searchEntries(query: string): Promise<SearchHit[]>;

  /**
   * Generates a conversational answer using the local LLM and semantic search.
   */
  askRecall(query: string): Promise<{ answer: string; hits: Entry[] }>;

  // -- capture ------------------------------------------------------------
  /**
   * §4: the hotkey starts recording immediately and the panel appears second.
   * Panel-first means two actions and a moment looking at UI before speaking.
   */
  startRecording(): Promise<void>;
  /** Returns the entry once transcription lands (~2s). `parentEdge` makes it
   *  an answer — a layer on that entry rather than its own node (§6.2). */
  stopRecording(parentEdge?: string | null, questionId?: string | null): Promise<Entry>;
  /** Discard belongs in the recording state, not after — you know it's junk
   *  before you stop (§4). Backed by a ~60s undo window, not a dialog. */
  discardRecording(): Promise<void>;
  undoDiscard(): Promise<Entry | null>;
  /** The transcript so far, while still recording. Empty when there is no model
   *  yet, too little audio, or nothing recording. */
  partialTranscript(): Promise<string>;
  /** Live amplitude for the equalizer bars. The only thing that animates (§8). */
  onAmplitude(cb: (level: number) => void): Unsubscribe;

  // -- enrichment ---------------------------------------------------------
  /**
   * Auto post-recording question; resolves to null when any of the three
   * facets suppresses it — not a position, live register, someone else's
   * words, or under ~10s (§3.2). A missed question beats a bad probe.
   */
  getQuestion(entryId: string): Promise<Question | null>;
  /**
   * Every question in the corpus, in one call. `getQuestion` returns only the
   * oldest open one, so a load built on it drops every answered and dismissed
   * question from the record and from the export — the accumulation §3.4 is
   * about. The load path falls back to `getQuestion` without it.
   */
  listQuestions?(): Promise<Question[]>;
  /**
   * User-invoked question (§3.6). Which probe fits is the model's call — the
   * UI offers one door, not a menu of techniques. Register does not gate here
   * — §3.2 gives the invoked path to the user — but role and provenance do,
   * because a fact, a list and someone else's sentence offer nothing to push on.
   */
  askQuestion(entryId: string, span?: Span | null): Promise<Question>;
  /** The primitive askQuestion picks from. Kept for replay and evaluation;
   *  no UI path names a probe. */
  runProbe(entryId: string, probeId: string, span?: Span | null): Promise<Question>;
  /** Strikes a question out. It stays on the entry; it stops being open. */
  dismissQuestion(entryId: string, questionId: string): Promise<void>;
  /** Proposed connections, shown as dismissible cards below the transcript (§6.1). */
  listProposedEdges(entryId: string): Promise<Edge[]>;
  /** Dismissals are training signal, not just UI (§6.1). */
  dismissEdge(edgeId: string): Promise<void>;
  acceptEdge(edgeId: string): Promise<void>;
  /** §5.4 — with a high threshold the app will miss real connections, and
   *  naming one yourself is the step the research says carries the benefit. */
  createManualEdge(a: string, b: string, relation: Edge['relation']): Promise<Edge>;

  // -- action items (§1.2) ------------------------------------------------
  listActionItems(): Promise<ActionItem[]>;
  /** State on the span, not a mutation of the text. */
  setActionItemDone(id: string, done: boolean): Promise<void>;

  // -- resolution (§6.3) --------------------------------------------------
  /** Requires stating what the resolution *is*. A bare flag gives the app nothing. */
  resolveEntry(entryId: string, text: string): Promise<Entry>;
  reopenEntry(entryId: string): Promise<Entry>;

  // -- system / onboarding ------------------------------------------------
  /** Drives the recommended-model default so onboarding stays one screen. */
  getSystemProfile(): Promise<SystemProfile>;
  listModels(): Promise<ModelInfo[]>;
  /** The full transcription catalogue — every transcribe.cpp model the host
   *  publishes — fetched over the network. Kept apart from listModels so
   *  onboarding stays offline; only Settings pays the fetch. */
  listTranscriptionCatalog(): Promise<ModelInfo[]>;
  /** Downloads in the background; gates nothing. Capture and transcription
   *  work without it, and the question surfaces when the model lands (§9.4).
   *  `url` is passed for catalogue models the host lists dynamically; built-in
   *  ids resolve without it, so onboarding calls this unchanged. */
  downloadModel(modelId: string, url?: string): Promise<void>;
  /** Removes a downloaded model file and any partial, freeing the space. */
  deleteModel(modelId: string): Promise<void>;
  /** Whether onboarding is behind this machine: a reasoning model chosen and
   *  every chosen model on disk. False brings onboarding back to finish. */
  setupComplete(): Promise<boolean>;
  onModelProgress(cb: (m: ModelInfo) => void): Unsubscribe;
  /** Fires when an enrichment pass starts. Paired with `onEntryEnriched`,
   *  which fires on every exit including failure -- an indicator that only
   *  clears on success is an indicator that sticks. */
  onEntryEnriching(cb: (entryId: string) => void): Unsubscribe;
  /** Fires when classification and the question have landed on an entry.
   *  Enrichment runs after capture returns, so without this the canvas keeps
   *  showing the placeholder title and the question never appears. */
  onEntryEnriched(cb: (entryId: string) => void): Unsubscribe;
  /** The recording itself, for playback. `null` when the entry was typed or the
   *  file is gone -- the caller falls back to a simulated clock. */
  readAudio(entryId: string): Promise<ArrayBuffer | null>;

  getSettings(): Promise<Settings>;
  setSettings(patch: Partial<Settings>): Promise<Settings>;
  /** Native "choose file" filtered to `*.gguf`, for either custom model
   *  setting. `null` when the dialog is cancelled. */
  pickModelFile(): Promise<string | null>;

  // -- sample corpus ------------------------------------------------------
  /** Offered from the empty state, never forced. Sample entries stay marked
   *  so they can never be mistaken for the user's own. */
  loadSampleCorpus(): Promise<SampleLoad>;
  clearSampleCorpus(): Promise<void>;

  /** Restores a previously exported corpus. 'merge' keeps existing ids. */
  importCorpus(data: CorpusImport, mode: ImportMode): Promise<void>;

  // -- export and upload ---------------------------------------------------
  /** The corpus as one zip of MDX notes, with the recordings when asked for.
   *  Asks where to save it; null when that is cancelled. */
  exportArchive(withAudio: boolean): Promise<Exported | null>;
  /** Saves Markdown the page rendered, asking where; null when cancelled. */
  exportTranscripts(markdown: string): Promise<string | null>;
  /** Asks for a file and reads it without applying it, so the merge-or-replace
   *  choice can say what it holds. null when cancelled; rejects on a file that
   *  does not read. */
  pickUpload(): Promise<UploadPreview | null>;
  /** Applies what pickUpload read. */
  applyUpload(mode: ImportMode): Promise<void>;

  // -- types (§3.6) ---------------------------------------------------------
  /** Built-ins and custom types, from the one table both the classifier's
   *  enum and the gate's tier lookup read. */
  listTypes(): Promise<TypeDefinition[]>;
  /** `id` is the slug the editor already shows before submitting — computed
   *  client-side so what the confirmation showed is what gets stored. */
  createType(draft: NewTypeDraft): Promise<TypeDefinition>;
  /** A built-in's id and identity are not the user's to take over, so this
   *  patches everything but those. */
  updateType(id: string, patch: TypePatchDraft): Promise<TypeDefinition>;
  /** Notes carrying this type fall back to their own role's built-in id, in
   *  the same transaction as the delete. */
  deleteType(id: string): Promise<void>;
}

export interface NewTypeDraft {
  id: string;
  label: string;
  match: string;
  prompt: string | null;
  tier: TypeDefinition['tier'];
  role: TypeDefinition['role'];
  mark: TypeDefinition['mark'];
}

export type TypePatchDraft = Omit<NewTypeDraft, 'id'>;

export interface Exported {
  path: string;
  notes: number;
  audio: number;
  /** Notes whose recording was not on disk to put in. */
  missingAudio: number;
}

export interface UploadPreview {
  fileName: string;
  notes: number;
  edges: number;
  questions: number;
  recordings: number;
}

let instance: Bridge | null = null;

export function isTauri(): boolean {
  return typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI_IPC__' in window || 'isTauri' in window);
}

export function getBridge(): Bridge {
  if (instance) return instance;
  throw new Error('Bridge not initialised — call initBridge() first.');
}

/**
 * There is one backend, the Rust one. The in-browser fixture backend was
 * removed after it drifted from Rust and let a missing feature pass every
 * browser check, so outside the desktop shell this refuses rather than fakes.
 */
export async function initBridge(): Promise<Bridge> {
  if (instance) return instance;
  const tauri = isTauri();

  // Opt-in, and deliberately not keyed to NODE_ENV: the case worth diagnosing
  // is a packaged build failing to inject its IPC, which is a production build
  // by definition, so a dev-only guard would switch this off exactly where it
  // is needed. Off unless asked for, because the title bar is the app's name to
  // the person using it, not a place to leave instrumentation.
  const diagnose = process.env.NEXT_PUBLIC_BRIDGE_DIAG === '1';
  const report = (which: string) => {
    if (!diagnose) return;
    const diag = `isTauri=${tauri} t_internals=${typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window} t_ipc=${typeof window !== 'undefined' && '__TAURI_IPC__' in window} t_isTauri=${typeof window !== 'undefined' && 'isTauri' in window}`;
    console.log(`[bridge] → ${which}`, diag);
    if (typeof document !== 'undefined') document.title = `Parallax [${which}] ${diag}`;
  };

  if (!tauri) {
    report('none');
    throw new Error('Parallax runs inside its desktop shell: use `npm run tauri:dev`.');
  }
  const { TauriBridge } = await import('./tauri');
  instance = new TauriBridge();
  report('TauriBridge');
  return instance;
}

/** Test seam — lets a story or a test pin a specific implementation. */
export function __setBridge(b: Bridge): void {
  instance = b;
}

export type ImportMode = 'merge' | 'replace';

/** What loading the sample set in motion, so the app can say so. */
export interface SampleLoad {
  /** Notes inserted by this call. Zero on a second load. */
  inserted: number;
  /**
   * False when there is no reasoning model to read them back yet. The notes are
   * there and searchable either way; without one they keep the title derived
   * from their first words, which otherwise looks like all the sample is.
   */
  enriching: boolean;
}

export interface CorpusImport {
  entries: Entry[];
  edges: Edge[];
  questions: Question[];
}
