//! Transcription, via transcribe.cpp.
//!
//! Loading is mmap'd below the Rust API, so a model "loads" in about 80ms and
//! the OS can evict its pages under pressure. That is what makes §9.3's
//! load-per-use right rather than a compromise: nothing is held resident, and
//! the second load is warm anyway.
//!
//! Measured on the target machine: whisper-tiny transcribes 11 seconds of
//! audio in 315ms on CPU, which is ~35x realtime. Transcription is not a
//! bottleneck and does not need the GPU -- and on a 4GB card it must not have
//! it, because the reasoning model is the thing a person is waiting on.

use crate::error::{Error, Result};
use crate::model::ComputeBackend;
use std::path::Path;
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, StreamOptions};

pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
}

/// `Auto` resolves here rather than at build time, so one binary suits a
/// machine with no GPU and a machine with plenty. Under `Auto` transcription
/// stays on CPU: it already finishes before anyone notices, and the VRAM is
/// worth more to the model that answers questions.
fn backend_for(preference: ComputeBackend) -> Backend {
    match preference {
        ComputeBackend::Auto | ComputeBackend::Cpu => Backend::Cpu,
        ComputeBackend::Gpu => Backend::Auto,
        ComputeBackend::Cuda => Backend::Cuda,
        ComputeBackend::Vulkan => Backend::Vulkan,
        ComputeBackend::Metal => Backend::Metal,
        ComputeBackend::Rocm => Backend::Rocm,
    }
}

pub fn transcribe(
    model_path: &Path,
    pcm: &[f32],
    preference: ComputeBackend,
) -> Result<Transcript> {
    if pcm.is_empty() {
        return Ok(Transcript {
            text: String::new(),
            language: None,
        });
    }

    let options = ModelOptions {
        backend: backend_for(preference),
        ..Default::default()
    };

    let model = Model::load_with(model_path, &options)
        .map_err(|e| Error::Other(format!("could not load the transcription model: {e}")))?;
    let mut session = model
        .session()
        .map_err(|e| Error::Other(format!("could not start transcription: {e}")))?;

    // One-shot decode where the model supports it. A streaming-only model
    // (moonshine-streaming, nemotron-streaming, ...) refuses `run`; feed it the
    // whole recording as a stream and finalise instead -- the same
    // record-then-transcribe result, reached through the streaming API.
    let text = match session.run(pcm, &RunOptions::default()) {
        Ok(result) => result.text,
        Err(run_err) => {
            if !model.capabilities().supports_streaming {
                return Err(Error::Other(format!("transcription failed: {run_err}")));
            }
            transcribe_streaming(&model, pcm)?
        }
    };

    Ok(Transcript {
        text: text.trim().to_string(),
        language: None,
    })
}

/// Runs a streaming-only model over a finished recording: the audio is fed in
/// one-second chunks and finalised, yielding the same complete transcript the
/// one-shot path gives every other model.
fn transcribe_streaming(model: &Model, pcm: &[f32]) -> Result<String> {
    // One second of 16 kHz mono per feed -- transcribe.cpp buffers internally,
    // so the chunk size only trades call count against latency, and this is a
    // finished recording where neither matters.
    const CHUNK: usize = 16_000;
    let mut session = model
        .session()
        .map_err(|e| Error::Other(format!("could not start transcription: {e}")))?;
    let mut stream = session
        .stream(&RunOptions::default(), &StreamOptions::default())
        .map_err(|e| Error::Other(format!("could not start streaming transcription: {e}")))?;
    for chunk in pcm.chunks(CHUNK) {
        stream
            .feed(chunk)
            .map_err(|e| Error::Other(format!("transcription failed: {e}")))?;
    }
    stream
        .finalize()
        .map_err(|e| Error::Other(format!("transcription failed: {e}")))?;
    Ok(stream.text().full)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing recorded is not an error: the discard path and a hotkey pressed
    /// twice by accident both land here, and neither should surface a failure.
    #[test]
    fn empty_audio_transcribes_to_nothing() {
        let out = transcribe(Path::new("does-not-exist.gguf"), &[], ComputeBackend::Auto).unwrap();
        assert!(out.text.is_empty());
    }

    /// Under Auto the GPU belongs to the reasoning model.
    #[test]
    fn auto_keeps_transcription_on_the_cpu() {
        assert_eq!(backend_for(ComputeBackend::Auto), Backend::Cpu);
        assert_eq!(backend_for(ComputeBackend::Cuda), Backend::Cuda);
    }
}
