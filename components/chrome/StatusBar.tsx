'use client';

import { useEffect, useState } from 'react';
import { getBridge } from '@/lib/bridge';
import { useApp } from '@/lib/store';
import type { ModelInfo, Settings } from '@/lib/types';
import styles from './StatusBar.module.css';

const mb = (bytes: number) => `${Math.round(bytes / 1e6)}MB`;

/** Everything about a model, for the hover. The strip stays short; this is
 *  where you look when a transcript came back worse than you expected. */
function detail(model: ModelInfo | undefined): string | undefined {
  if (!model) return undefined;
  return `${model.name} · ${model.params} params · ${model.quantization} · ${mb(model.sizeBytes)}`;
}

/** The file's own name, so a custom model reads as itself rather than as a
 *  full path in the strip that also carries the entry count and the
 *  thinking indicator. */
function baseName(path: string): string {
  const name = path.replace(/\\/g, '/').split('/').pop() ?? path;
  return name.replace(/\.gguf$/i, '');
}

function stateLabel(model: ModelInfo | undefined): string {
  if (!model) return 'none';
  switch (model.state.kind) {
    case 'ready':
      return 'ready';
    case 'downloading':
      return `${Math.round((model.state.receivedBytes / model.state.totalBytes) * 100)}%`;
    case 'failed':
      return 'failed';
    case 'not-downloaded':
      return 'not downloaded';
  }
}

/**
 * Bottom chrome strip. Both models are shown because they mean different
 * things: speech gates recording, the question does not (§9.4).
 */
export default function StatusBar({ settings }: { settings: Settings }) {
  const entryCount = useApp((s) => s.order.length);
  // A note caught from the panel lands on the canvas without opening, so the
  // entry's own indicator is not on screen. This strip is.
  const thinkingCount = useApp((s) => s.enriching.size);
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    const bridge = getBridge();
    void bridge.listModels().then(setModels);
    return bridge.onModelProgress((m) => {
      setModels((prev) => prev.map((x) => (x.id === m.id ? m : x)));
    });
  }, []);

  // `models` here is the built-in list (whisper); a transcribe.cpp model
  // chosen from the fetched catalogue is not in it, so it is `undefined` and
  // shown by its id below rather than the family-and-size line.
  const speech = models.find(
    (m) => m.kind === 'transcription' && m.id === settings.transcriptionModel,
  );
  const reasoning = models.find((m) => m.id === settings.modelId);
  // A custom path overrides its catalogue counterpart the moment it is set
  // (§ resolve_reasoning_model / resolve_transcription_model), so the strip
  // has to say which file is actually the one in use.
  const customSpeechName = settings.customTranscriptionModelPath
    ? baseName(settings.customTranscriptionModelPath)
    : null;
  const customReasoningName = settings.customReasoningModelPath
    ? baseName(settings.customReasoningModelPath)
    : null;
  // A chosen model that is not a built-in is a catalogue model, which Settings
  // only lets you select once it is downloaded -- so treat it as ready.
  const speechReady =
    customSpeechName !== null ||
    speech?.state.kind === 'ready' ||
    (!speech && settings.transcriptionModel !== '');

  return (
    <footer className={styles.bar}>
      <span>
        {entryCount} {entryCount === 1 ? 'entry' : 'entries'}
      </span>

      {thinkingCount > 0 && (
        <span className={styles.thinking} role="status">
          <span className={styles.thinkingDot} aria-hidden />
          reading {thinkingCount === 1 ? 'a note' : `${thinkingCount} notes`} back
        </span>
      )}

      <span className={styles.models}>
        {/* Both models are identified, not just stated ready. There are three
            transcription models and "base" alone names none of them, so it gets
            the same family-and-size treatment the reasoning model already had —
            which one ran is the first thing you want when a transcript comes
            back worse than you expected (§9.5). */}
        <span
          className={speechReady ? styles.model : styles.modelPending}
          title={settings.customTranscriptionModelPath ?? detail(speech)}
        >
          {customSpeechName ??
            (speech
              ? `whisper ${speech.name} ${speech.params}`
              : settings.transcriptionModel || 'no speech model')}{' '}
          {customSpeechName ? 'ready' : speech ? stateLabel(speech) : 'ready'}
        </span>
        <span className={styles.divider} aria-hidden>
          ·
        </span>
        <span className={styles.model} title={settings.customReasoningModelPath ?? detail(reasoning)}>
          {customReasoningName ?? reasoning?.name ?? 'no model'}{' '}
          {customReasoningName ? 'ready' : stateLabel(reasoning)}
        </span>
      </span>
    </footer>
  );
}
