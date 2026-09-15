'use client';

import { useEffect, useState } from 'react';
import { getBridge } from '@/lib/bridge';
import { useApp } from '@/lib/store';
import type { ModelInfo, Settings } from '@/lib/types';
import styles from './SettingsPanel.module.css';
import TypeEditor from './TypeEditor';

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)}GB`;
const mb = (bytes: number) => `${Math.round(bytes / 1e6)}MB`;
const size = (bytes: number) => (bytes >= 1e9 ? gb(bytes) : mb(bytes));


type Rebindable = 'hotkey' | 'discardHotkey';

/**
 * Right-hand sheet, same pattern as EntryView: fixed, own background, esc to
 * close (handled by the app-wide overlay Escape handling in app/page.tsx).
 */
export default function SettingsPanel({
  settings,
  onChange,
}: {
  settings: Settings;
  onChange(next: Settings): void;
}) {
  const close = useApp((s) => s.closeOverlay);
  const entryCount = useApp((s) => s.order.length);
  const loadSample = useApp((s) => s.loadSample);
  const clearSample = useApp((s) => s.clearSample);
  const setSampleLoaded = useApp((s) => s.setSampleLoaded);

  const [models, setModels] = useState<ModelInfo[]>([]);
  // The transcription catalogue is fetched from the model host, so it is kept
  // apart from the built-in list and carries its own loading/error state.
  const [speech, setSpeech] = useState<ModelInfo[]>([]);
  const [speechLoading, setSpeechLoading] = useState(true);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState<Rebindable | null>(null);
  const [modelPathError, setModelPathError] = useState<{
    field: 'customReasoningModelPath' | 'customTranscriptionModelPath';
    message: string;
  } | null>(null);

  useEffect(() => {
    const bridge = getBridge();
    void bridge.listModels().then(setModels);
    void bridge
      .listTranscriptionCatalog()
      .then((m) => {
        setSpeech(m);
        setSpeechError(null);
      })
      .catch((e: unknown) => setSpeechError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSpeechLoading(false));
    return bridge.onModelProgress((m) => {
      setModels((prev) => prev.map((x) => (x.id === m.id ? m : x)));
      setSpeech((prev) => prev.map((x) => (x.id === m.id ? m : x)));
    });
  }, []);

  // Delete removes the file; refetch both lists so the row falls back to
  // "download". The host list is cached backend-side, so this is cheap.
  async function removeModel(id: string) {
    await getBridge().deleteModel(id);
    void getBridge().listModels().then(setModels);
    void getBridge().listTranscriptionCatalog().then(setSpeech).catch(() => {});
  }

  // What a custom path would be overriding, named for the "none — using…" line.
  const reasoningCatalogueName =
    models.find((m) => m.id === settings.modelId)?.name ?? 'no catalogue model chosen';
  const transcriptionCatalogueName =
    speech.find((m) => m.id === settings.transcriptionModel)?.name ?? settings.transcriptionModel;

  // Capture the next chord for whichever field is rebinding — same logic as
  // onboarding's hotkey capture, generalised to either field here.
  useEffect(() => {
    if (!capturing) return;
    const field = capturing;
    const onKeyDown = (e: KeyboardEvent) => {
      e.preventDefault();
      if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
      const parts = [
        e.ctrlKey && 'Ctrl',
        e.shiftKey && 'Shift',
        e.altKey && 'Alt',
        e.code === 'Space' ? 'Space' : e.key.toUpperCase(),
      ].filter(Boolean);
      const chord = parts.join('+');
      void update(field === 'hotkey' ? { hotkey: chord } : { discardHotkey: chord });
      setCapturing(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [capturing]);

  async function update(patch: Partial<Settings>) {
    const next = await getBridge().setSettings(patch);
    onChange(next);
  }

  // Rejection is a real outcome here -- a moved or renamed file -- not just a
  // slow round trip, so it is shown inline against the field rather than
  // dropped as an unhandled promise rejection.
  async function updateModelPath(
    field: 'customReasoningModelPath' | 'customTranscriptionModelPath',
    value: string | null,
  ) {
    try {
      await update({ [field]: value } as Partial<Settings>);
      setModelPathError(null);
    } catch (e) {
      setModelPathError({ field, message: e instanceof Error ? e.message : String(e) });
    }
  }

  async function chooseModel(field: 'customReasoningModelPath' | 'customTranscriptionModelPath') {
    const path = await getBridge().pickModelFile();
    if (path === null) return; // cancelled
    void updateModelPath(field, path);
  }

  return (
    <aside className={styles.sheet}>
      <header className={styles.header}>
        <span className={styles.title}>settings</span>
        <button type="button" className={styles.close} onClick={close} aria-label="Close">
          esc
        </button>
      </header>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>capture</h2>

        <div className={styles.row}>
          <span className={styles.label}>record hotkey</span>
          <div className={styles.control}>
            <kbd className={styles.kbd}>
              {capturing === 'hotkey' ? 'press a combination…' : settings.hotkey}
            </kbd>
            <button type="button" className={styles.inline} onClick={() => setCapturing('hotkey')}>
              rebind
            </button>
          </div>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>discard key</span>
          <div className={styles.control}>
            <kbd className={styles.kbd}>
              {capturing === 'discardHotkey' ? 'press a combination…' : settings.discardHotkey}
            </kbd>
            <button
              type="button"
              className={styles.inline}
              onClick={() => setCapturing('discardHotkey')}
            >
              rebind
            </button>
          </div>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>live register</span>
          <div className={styles.control}>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={settings.liveRegister}
                onChange={(e) => void update({ liveRegister: e.target.checked })}
              />
              <span>leave notes with something personal at stake alone</span>
            </label>
          </div>
        </div>
        {/* Says what turning it off costs: the default is the safe direction
            and the trade is not obvious from the label alone. */}
        <p className={styles.helper}>
          {settings.liveRegister
            ? 'A note the classifier reads as live is never opened on its own, and carries no summary. Selecting a sentence still asks.'
            : 'Every note is treated as neutral. Nothing filed is rewritten, so turning this back on restores what was decided.'}
        </p>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>transcription</h2>

        {speechError && (
          <p className={styles.reject}>couldn’t load the model catalogue: {speechError}</p>
        )}
        {speechLoading && speech.length === 0 && (
          <p className={styles.helper}>loading models…</p>
        )}
        <div className={styles.models}>
          {speech.map((m) => (
            <div
              key={m.id}
              className={m.id === settings.transcriptionModel ? styles.modelRowOn : styles.modelRow}
            >
              <button
                type="button"
                className={styles.modelSelect}
                disabled={m.state.kind !== 'ready'}
                onClick={() => void update({ transcriptionModel: m.id })}
              >
                <span className={styles.modelName}>{m.name}</span>
                <span className={styles.modelMeta}>
                  {[m.quantization, m.sizeBytes ? size(m.sizeBytes) : null]
                    .filter(Boolean)
                    .join(' · ')}
                </span>
              </button>

              <div className={styles.modelState}>
                {m.state.kind === 'ready' && (
                  <>
                    {m.id === settings.transcriptionModel && (
                      <span className={styles.ready}>in use</span>
                    )}
                    <button
                      type="button"
                      className={styles.inline}
                      onClick={() => void removeModel(m.id)}
                    >
                      delete
                    </button>
                  </>
                )}
                {m.state.kind === 'failed' && (
                  <button
                    type="button"
                    className={styles.download}
                    onClick={() => void getBridge().downloadModel(m.id, m.url)}
                  >
                    retry
                  </button>
                )}
                {m.state.kind === 'not-downloaded' && (
                  <button
                    type="button"
                    className={styles.download}
                    onClick={() => void getBridge().downloadModel(m.id, m.url)}
                  >
                    download
                  </button>
                )}
                {m.state.kind === 'downloading' && (
                  <div className={styles.progressTrack} aria-hidden>
                    <div
                      className={styles.progressFill}
                      style={{
                        width: `${m.state.totalBytes ? Math.round((m.state.receivedBytes / m.state.totalBytes) * 100) : 0}%`,
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.row}>
          <span className={styles.label}>your own transcription model</span>
          <div className={styles.control}>
            <span className={styles.value}>
              {settings.customTranscriptionModelPath ||
                `none — using ${transcriptionCatalogueName}`}
            </span>
            <button
              type="button"
              className={styles.inline}
              onClick={() => void chooseModel('customTranscriptionModelPath')}
            >
              choose file…
            </button>
            {settings.customTranscriptionModelPath && (
              <button
                type="button"
                className={styles.inline}
                onClick={() => void updateModelPath('customTranscriptionModelPath', null)}
              >
                clear
              </button>
            )}
          </div>
        </div>
        {modelPathError?.field === 'customTranscriptionModelPath' && (
          <p className={styles.reject}>{modelPathError.message}</p>
        )}
        <p className={styles.helper}>a transcribe.cpp GGUF in the same format as the built-in ones</p>

        {/* Fact, not a setting — no control here on purpose. */}
        <p className={styles.statement}>Audio and transcription never leave this machine.</p>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>reasoning model</h2>

        <div className={styles.models}>
          {models
            .filter((m) => m.kind === 'reasoning')
            .map((model) => (
            <div key={model.id} className={model.id === settings.modelId ? styles.modelRowOn : styles.modelRow}>
              <button
                type="button"
                className={styles.modelSelect}
                onClick={() => void update({ modelId: model.id })}
              >
                <span className={styles.modelName}>{model.name}</span>
                <span className={styles.modelMeta}>
                  {model.quantization} · {gb(model.sizeBytes)}
                </span>
              </button>

              <div className={styles.modelState}>
                {model.state.kind === 'ready' && (
                  <>
                    <span className={styles.ready}>ready</span>
                    <button
                      type="button"
                      className={styles.inline}
                      onClick={() => void removeModel(model.id)}
                    >
                      delete
                    </button>
                  </>
                )}
                {model.state.kind === 'failed' && <span className={styles.failed}>failed</span>}
                {model.state.kind === 'not-downloaded' && (
                  <button
                    type="button"
                    className={styles.download}
                    onClick={() => void getBridge().downloadModel(model.id)}
                  >
                    download
                  </button>
                )}
                {model.state.kind === 'downloading' && (
                  <div className={styles.progressTrack} aria-hidden>
                    <div
                      className={styles.progressFill}
                      style={{
                        width: `${Math.round((model.state.receivedBytes / model.state.totalBytes) * 100)}%`,
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.row}>
          <span className={styles.label}>your own reasoning model</span>
          <div className={styles.control}>
            <span className={styles.value}>
              {settings.customReasoningModelPath || `none — using ${reasoningCatalogueName}`}
            </span>
            <button
              type="button"
              className={styles.inline}
              onClick={() => void chooseModel('customReasoningModelPath')}
            >
              choose file…
            </button>
            {settings.customReasoningModelPath && (
              <button
                type="button"
                className={styles.inline}
                onClick={() => void updateModelPath('customReasoningModelPath', null)}
              >
                clear
              </button>
            )}
          </div>
        </div>
        {modelPathError?.field === 'customReasoningModelPath' && (
          <p className={styles.reject}>{modelPathError.message}</p>
        )}
        <p className={styles.helper}>
          any GGUF chat model llama.cpp can run; quality is not measured
        </p>

        <div className={styles.row}>
          <span className={styles.label}>residency</span>
          <div className={styles.options}>
            {(['warm', 'cold'] as const).map((r) => (
              <button
                key={r}
                type="button"
                className={r === settings.residency ? styles.optionOn : styles.option}
                onClick={() => void update({ residency: r })}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
        <p className={styles.helper}>
          Warm keeps the model loaded for ten minutes after your last note, so each question lands in
          about two seconds. Cold gives the graphics card back thirty seconds after, which saves battery,
          and the next question takes about four seconds longer while the model loads.
        </p>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>appearance</h2>
        <div className={styles.row}>
          <span className={styles.label}>theme</span>
          <div className={styles.options}>
            {(['system', 'light', 'dark'] as const).map((t) => (
              <button
                key={t}
                type="button"
                className={t === settings.theme ? styles.optionOn : styles.option}
                onClick={() => void update({ theme: t })}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>types</h2>
        <TypeEditor />
        <p className={styles.helper}>
          A type you add fires only on a position, only on your own words, and never on its
          own when an entry reads as live. Ask it yourself and it will answer anything.
          The mark is yours; the gate is not.
        </p>
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>corpus</h2>

        <div className={styles.row}>
          <span className={styles.label}>entries</span>
          <span className={styles.value}>{entryCount}</span>
        </div>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.inline}
            onClick={async () => {
              await loadSample();
              setSampleLoaded(true);
            }}
          >
            load sample corpus
          </button>
          <button
            type="button"
            className={styles.inline}
            onClick={async () => {
              await clearSample();
              setSampleLoaded(false);
            }}
          >
            clear sample corpus
          </button>
        </div>
      </section>
    </aside>
  );
}
