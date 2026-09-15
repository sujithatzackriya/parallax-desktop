/**
 * Domain types — mirrors the §7 schema (titles §5.2, summaries §7.1, action items §1.2).
 * Also the wire types for the bridge; the Rust side serialises to this exact shape.
 */

// ---------------------------------------------------------------------------
// Classification — §3.6, two axes that scale differently.
// ---------------------------------------------------------------------------

/**
 * Facet 1 — role. What the entry *does*. Drives letterform and retrieval,
 * never the gate. Three values: the canvas budget (§5.3) is a ceiling, not a
 * quota, and `position` absorbs the old claim/rant split because that
 * difference was always resolved-vs-unresolved, which `resolved` already
 * carries deliberately.
 */
export type Role = 'position' | 'evidence' | 'note';

/**
 * Facet 2 — register. Whole-entry and binary, never span-level: the one study
 * that tried a span-scoped affect layer alongside an argument layer got
 * αU 0.30 on affect against 0.48 on role and dropped it. Gates the
 * *automatic* question only (§3.2); the invoked path is the user's to spend.
 *
 * Defaults to `live` under uncertainty. A false `live` costs a missed
 * question; a false `neutral` costs the thing that cannot be taken back.
 */
export type Register = 'live' | 'neutral';

/**
 * The registry key. Equals the role id for built-ins; a user-defined type
 * (§3.6) puts its own id here and binds to a role for its letterform.
 */
export type TypeId = string;

export type ProbeTier = 'silent' | 'safe' | 'heavy' | 'retrieval';

// ---------------------------------------------------------------------------
// Relations — §5.4. An edge requires a nameable relation. If the best the
// model can produce is "related" or "same subject", we draw nothing.
//
// `echoes` is gone: it meant the same as `same move` and nothing in the corpus
// told them apart, which diluted the one relation the move vector exists to
// find. It was also "related" under a better name, which is the escape hatch
// §5.4 was written to close.
// ---------------------------------------------------------------------------

export type Relation =
  | 'contradicts'
  | 'same move'
  | 'returns to'
  | 'questions'
  | 'extends'
  | 'example of'
  | 'answers'
  | 'related';

/**
 * What the model is allowed to emit. `related` is deliberately outside it: the
 * rule that stops the app drawing a line it cannot name does not apply to a
 * person who knows two notes belong together and cannot yet say why. Refusing
 * them the link only loses the connection.
 */
export const MODEL_RELATIONS: Relation[] = [
  'contradicts',
  'same move',
  'returns to',
  'questions',
  'extends',
  'example of',
];

/**
 * §5.4/§6.2 resolution: no parent→child line exists (§6.2's child entries
 * thicken the parent's ring instead), so fact-vs-guess renders via
 * --edge-fact (accepted/manual) vs --edge (proposed).
 */
export type EdgeStatus = 'proposed' | 'accepted' | 'dismissed' | 'manual';

export interface Edge {
  id: string;
  entryA: string;
  entryB: string;
  relation: Relation;
  /** Optional question carried by the connection — opened from the entry view. */
  question: string | null;
  status: EdgeStatus;
  createdAt: string;
}

// ---------------------------------------------------------------------------
// Spans — §7.3. Attribution is separated so the move vector can exclude
// quoted material; without it, a note quoting three people gets connected on
// *their* thinking.
// ---------------------------------------------------------------------------

export interface Span {
  start: number;
  end: number;
  /**
   * Facet 3 — provenance. true = someone else's words, false = the user's own.
   * Span-level because speech fuses the two in one breath: a note about a book
   * carries the author's argument and the speaker's own position together.
   * The app may only *push* on an `own` span; attributed spans can still be
   * quoted, connected and cited (§7.3).
   */
  attributed: boolean;
}

/**
 * §1.2 — an inert type. Never initiates a question, never nags. Ticking is
 * state on the span, not a mutation of the text: the record accumulates,
 * nothing is overwritten.
 */
export interface ActionItem {
  id: string;
  entryId: string;
  /** Anchored to the span it came from, so ticking never edits the transcript. */
  span: Span;
  text: string;
  done: boolean;
}

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

export interface Entry {
  id: string;
  /** null => typed entry. No fingerprint, which is a free visual distinction (§4). */
  audioPath: string | null;
  /** The record. Verbatim, never rewritten, never summarised over (§1.1, §2). */
  transcript: string;
  createdAt: string;

  /** Frozen at insert and never recomputed (§5.1). Position encodes *when*. */
  x: number;
  y: number;

  /** Set => this entry is an answer to the entry named here. */
  parentEdge: string | null;
  /** Which question it answers. Without it, an answer can only be paired to a
   *  question by order, which guesses as soon as an entry has two open. */
  answersQuestionId: string | null;

  /** Facet 1 — what this entry does. Display and retrieval only. */
  role: Role;
  /** Facet 2 — emotionally live? Gates the automatic question, not the invoked one. */
  register: Register;
  /** Registry key: a role id, or a user-defined type's id (§3.6). */
  typeId: TypeId;

  /** User-declared only. The AI never decides you're done thinking (§6.3). */
  resolved: boolean;
  resolutionText: string | null;

  // --- enrichment; added to the record, never replacing it ---

  /** 3–4 words, from the user's own phrasing where possible (§5.2). */
  title: string;
  /**
   * §7.1 — embedded for similarity, and displayable, but never above the
   * transcript. null for felt entries: "reflects on a relationship that ended"
   * is worse than useless (§1.1).
   */
  summary: string | null;

  durationMs: number;
  /** 7–9 samples downsampled from actual amplitude (§5.2). Empty if typed. */
  fingerprint: number[];

  /** §5.3 dashed stroke. From the user's own hedges — regex, not a model. */
  unfinished: boolean;
  /** §9.4 — set by the user at capture, never inferred by a classifier. */
  localOnly: boolean;

  spans: Span[];
  actionItems: ActionItem[];

  /** Marks a seeded sample entry so it can never be mistaken for the user's own. */
  isSample?: boolean;
}

/**
 * The one voice in the app that isn't the user's (§8). Rendered ~15px serif so
 * it visibly isn't theirs. Every analytical claim quotes a span (§3.4), which
 * is what makes it checkable rather than authoritative.
 */
export interface Question {
  id: string;
  entryId: string;
  text: string;
  /** The span the question is anchored to. Unanchored output is not allowed. */
  span: Span | null;
  answered: boolean;
  /**
   * Struck out, not deleted. §3.4 bans a regenerate button — rerolling until
   * the question is agreeable is the echo chamber by the back door — but a
   * genuinely bad question still needs somewhere to go. Dismissing keeps it in
   * the record and stops it counting as open; replacing it would let you
   * escape the one that stung.
   */
  dismissed: boolean;
  /** Shown in the UI — the user always knows who answered (§9.4). */
  providerName: string;
  createdAt: string;
}

// ---------------------------------------------------------------------------
// System / settings — drives onboarding and §9.4's model management.
// ---------------------------------------------------------------------------

export interface SystemProfile {
  totalRamBytes: number;
  availableRamBytes: number;
  cpuName: string;
  cpuCores: number;
  gpuName: string | null;
  vramBytes: number | null;
}

export type ModelState =
  | { kind: 'not-downloaded' }
  | { kind: 'downloading'; receivedBytes: number; totalBytes: number }
  | { kind: 'ready' }
  | { kind: 'failed'; error: string };

/** Transcription gates recording; reasoning does not (§9.4). */
export type ModelKind = 'transcription' | 'reasoning' | 'embedding';

export interface ModelInfo {
  id: string;
  kind: ModelKind;
  name: string;
  params: string;
  quantization: string;
  sizeBytes: number;
  /** Minimum RAM we'd recommend this at — drives the onboarding default. */
  recommendedRamBytes: number;
  state: ModelState;
  /** Shown before downloading, so nothing is fetched the user cannot see. */
  url: string;
}

/** §9.4 — the residency fork. Only the post-recording question is latency-sensitive. */
export type Residency = 'warm' | 'cold';

/**
 * `system` tracks `prefers-color-scheme` and is the only one that can change
 * without the app being told. Persisted (§ Settings) so the capture panel — a
 * second webview with its own JS runtime and its own copy of the UI store —
 * opens on the choice already showing in the main window instead of always
 * starting on the store's own default.
 */
export type Theme = 'system' | 'light' | 'dark';

/**
 * What a model runs on. `auto` resolves against the machine; `gpu` forces
 * acceleration and leaves the backend open. The named backends are an explicit
 * override — a broken driver, a second card — and fail rather than falling
 * back, so a forced choice stays forced. Only backends the build and machine
 * actually have are offered.
 */
export type ComputeBackend =
  | 'auto'
  | 'cpu'
  | 'gpu'
  | 'cuda'
  | 'vulkan'
  | 'metal'
  | 'rocm';

export interface Settings {
  hotkey: string;
  discardHotkey: string;
  modelId: string | null;
  residency: Residency;
  providerName: string;
  /** Global default for the per-entry local-only flag (§9.4). */
  defaultLocalOnly: boolean;
  /** A transcription catalogue id (also the file name on disk, `{id}.gguf`) —
   *  any whisper size or transcribe.cpp model, not a fixed set. */
  transcriptionModel: string;
  /**
   * `null` until chosen. Connections still work without it — topics propose on
   * their own and cosine only reorders them — so this is an upgrade to the
   * ordering, never a dependency.
   */
  embeddingModelId: string | null;
  /**
   * Under `auto` the reasoning model has first claim on VRAM: it is the one a
   * person is waiting on, while transcription runs at ~35x realtime on CPU.
   */
  transcriptionBackend: ComputeBackend;
  reasoningBackend: ComputeBackend;
  /**
   * Whether the live register is consulted at all (§3.2). On by default,
   * because suppressing a question is the recoverable direction.
   *
   * Off is for the case it exists for: a note that argues an impersonal point
   * *through* a personal example reads as live to the classifier, so the note
   * that most wanted a question is the one that silently never gets one.
   * Turning it off stops the facet being read; it rewrites nothing, so turning
   * it back on restores what the classifier decided.
   */
  liveRegister: boolean;
  theme: Theme;
  /**
   * A user's own chat model, checked before the catalogue choice. `null` for
   * everyone who has not set one -- `modelId` then works exactly as before.
   */
  customReasoningModelPath: string | null;
  /** Same idea for speech-to-text; must be a whisper GGUF, same format as
   *  the catalogue's own whisper files. */
  customTranscriptionModelPath: string | null;
}
