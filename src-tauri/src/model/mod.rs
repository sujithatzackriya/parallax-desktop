//! Wire types. These mirror `lib/types.ts`, which is the contract.

pub mod download;
pub mod edge;
pub mod entry;
pub mod question;
pub mod remote;
pub mod settings;
pub mod type_def;

pub use edge::{Edge, EdgeStatus, Relation};
pub use entry::{ActionItem, Entry, Register, Role, Span};
pub use question::Question;
pub use settings::{
    ComputeBackend, ModelInfo, ModelKind, ModelState, Residency, Settings, SystemProfile, Theme,
};
pub use type_def::{Mark, NewType, ProbeTier, TypeDef, TypePatch};

#[cfg(test)]
mod contract_tests {
    //! The frontend reads these fields by name, so a rename is not a compile
    //! error on either side — it is a silent `undefined`. The key set is
    //! asserted rather than assumed.

    use super::*;
    use serde_json::json;

    fn sample_entry() -> Entry {
        Entry {
            id: "e1".into(),
            audio_path: Some("audio/e1.wav".into()),
            transcript: "I don't think free will requires that.".into(),
            created_at: "2024-01-14T09:38:00.000Z".into(),
            x: 12.5,
            y: -40.0,
            parent_entry_id: None,
            answers_question_id: None,
            role: Role::Position,
            register: Register::Neutral,
            type_id: "position".into(),
            resolved: false,
            resolution_text: None,
            title: "our own reasoning".into(),
            summary: Some("Free will as your own reasoning.".into()),
            duration_ms: 31_000,
            fingerprint: vec![0.2, 0.8, 0.5],
            unfinished: false,
            local_only: false,
            spans: vec![Span {
                start: 0,
                end: 12,
                attributed: false,
            }],
            action_items: vec![],
            is_sample: None,
        }
    }

    #[test]
    fn entry_serialises_to_the_wire_contract() {
        let v = serde_json::to_value(sample_entry()).unwrap();
        let obj = v.as_object().unwrap();

        let expected = [
            "id",
            "audioPath",
            "transcript",
            "createdAt",
            "x",
            "y",
            "parentEdge",
            "answersQuestionId",
            "role",
            "register",
            "typeId",
            "resolved",
            "resolutionText",
            "title",
            "summary",
            "durationMs",
            "fingerprint",
            "unfinished",
            "localOnly",
            "spans",
            "actionItems",
        ];
        for key in expected {
            assert!(obj.contains_key(key), "missing wire field: {key}");
        }

        // Renaming it inside Rust must not change the wire shape.
        assert!(obj.contains_key("parentEdge"));
        assert!(!obj.contains_key("parentEntryId"));

        // Optional on the wire: absent, not null.
        assert!(!obj.contains_key("isSample"));

        assert_eq!(obj["role"], json!("position"));
        assert_eq!(obj["register"], json!("neutral"));
    }

    #[test]
    fn entry_round_trips() {
        let before = sample_entry();
        let json = serde_json::to_string(&before).unwrap();
        let after: Entry = serde_json::from_str(&json).unwrap();
        assert_eq!(
            serde_json::to_value(&before).unwrap(),
            serde_json::to_value(&after).unwrap()
        );
    }

    /// Spaces on the wire; getting this wrong breaks every edge label.
    #[test]
    fn relations_keep_their_spaces() {
        assert_eq!(
            serde_json::to_value(Relation::SameMove).unwrap(),
            json!("same move")
        );
        assert_eq!(
            serde_json::to_value(Relation::ReturnsTo).unwrap(),
            json!("returns to")
        );
        assert_eq!(
            serde_json::to_value(Relation::ExampleOf).unwrap(),
            json!("example of")
        );
    }

    #[test]
    fn model_may_not_emit_answers_or_related() {
        assert!(!Relation::Answers.is_model_emittable());
        assert!(!Relation::Related.is_model_emittable());
        assert!(Relation::Contradicts.is_model_emittable());
        assert_eq!(Relation::MODEL_RELATIONS.len(), 6);
    }

    #[test]
    fn model_state_is_internally_tagged() {
        assert_eq!(
            serde_json::to_value(ModelState::NotDownloaded).unwrap(),
            json!({ "kind": "not-downloaded" })
        );
        assert_eq!(
            serde_json::to_value(ModelState::Downloading {
                received_bytes: 5,
                total_bytes: 10
            })
            .unwrap(),
            json!({ "kind": "downloading", "receivedBytes": 5, "totalBytes": 10 })
        );
        assert_eq!(
            serde_json::to_value(ModelState::Failed {
                error: "disk full".into()
            })
            .unwrap(),
            json!({ "kind": "failed", "error": "disk full" })
        );
    }

    #[test]
    fn settings_defaults_are_what_the_ui_expects() {
        let s = Settings::default();
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["hotkey"], json!("Ctrl+Shift+Space"));
        assert_eq!(v["discardHotkey"], json!("Ctrl+Shift+Backspace"));
        assert_eq!(v["modelId"], json!(null));
        assert_eq!(v["residency"], json!("warm"));
        assert_eq!(v["providerName"], json!("llama-server"));
        assert_eq!(v["defaultLocalOnly"], json!(false));
        assert_eq!(v["transcriptionModel"], json!("whisper-base"));
        assert_eq!(v["transcriptionBackend"], json!("auto"));
        assert_eq!(v["reasoningBackend"], json!("auto"));
        assert_eq!(v["theme"], json!("system"));
        assert_eq!(v["customReasoningModelPath"], json!(null));
        assert_eq!(v["customTranscriptionModelPath"], json!(null));
    }
}
