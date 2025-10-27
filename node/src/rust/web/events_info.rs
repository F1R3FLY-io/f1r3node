use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
    routing::get,
    Router,
};
use futures::StreamExt;
use serde_json::{json, Value};

use shared::rust::shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::EventStream};
use tracing::error;

use crate::rust::web::shared_handlers::AppState;

pub struct EventsInfo;

impl EventsInfo {
    pub fn create_router() -> Router<AppState> {
        Router::new().route("/", get(events_info_handler))
    }

    async fn handle_websocket(mut socket: WebSocket, mut event_stream: EventStream) {
        while let Some(event) = event_stream.next().await {
            if let Err(e) = Self::send_event_to_websocket(&mut socket, &event).await {
                error!("Failed to send event to WebSocket: {}", e);
            }
        }
    }

    async fn send_event_to_websocket(
        socket: &mut WebSocket,
        event: &F1r3flyEvent,
    ) -> Result<(), String> {
        let json = Self::transform_f1r3fly_event(event)
            .map_err(|e| format!("Failed to transform event: {}", e))?;

        let message_str =
            serde_json::to_string(&json).map_err(|e| format!("Failed to serialize JSON: {}", e))?;

        let message = Message::Text(message_str.into());
        socket
            .send(message)
            .await
            .map_err(|e| format!("Failed to send message: {}", e))?;

        Ok(())
    }

    // Transforms an F1r3flyEvent into a JSON structure matching the Scala implementation
    // This converts the discriminated union to a structure with:
    // - "event": the event type
    // - "schema-version": 1
    // - "payload": the rest of the fields
    pub fn transform_f1r3fly_event(event: &F1r3flyEvent) -> Result<Value, serde_json::Error> {
        let serialized = serde_json::to_value(event)?;

        let event_type = serialized
            .get("event")
            .cloned()
            .unwrap_or_else(|| json!("unknown"));

        let payload = match serialized {
            Value::Object(mut obj) => {
                obj.remove("event");
                Value::Object(obj)
            }
            _ => serialized,
        };

        Ok(json!({
            "event": event_type,
            "schema-version": 1,
            "payload": payload
        }))
    }
}

#[utoipa::path(
        get,
        path = "/ws/events",
        responses(
            (status = 101, description = "WebSocket upgrade successful"),
            (status = 426, description = "Upgrade Required"),
        ),
        tag = "System"
    )]
pub async fn events_info_handler(
    ws: WebSocketUpgrade,
    State(app_state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| {
        EventsInfo::handle_websocket(socket, app_state.event_stream.new_subscribe())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_transform_block_created_event() {
        let event = F1r3flyEvent::block_created(
            "hash123".to_string(),
            vec!["parent1".to_string()],
            vec![("j1".to_string(), "j2".to_string())],
            vec!["deploy1".to_string()],
            "creator1".to_string(),
            42,
        );

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "block-created");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash123");
        assert_eq!(payload["parent-hashes"], json!(["parent1"]));
        assert_eq!(payload["seq-number"], 42);
    }

    #[test]
    fn test_transform_block_added_event() {
        let event = F1r3flyEvent::block_added(
            "hash456".to_string(),
            vec!["parent2".to_string()],
            vec![("j3".to_string(), "j4".to_string())],
            vec!["deploy2".to_string()],
            "creator2".to_string(),
            100,
        );

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "block-added");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash456");
    }

    #[test]
    fn test_transform_block_finalised_event() {
        let event = F1r3flyEvent::block_finalised("hash789".to_string());

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "block-finalised");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash789");
    }

    #[test]
    fn test_transform_sent_unapproved_block_event() {
        let event = F1r3flyEvent::sent_unapproved_block("hash-unauthorized".to_string());

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "sent-unapproved-block");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash-unauthorized");
    }

    #[test]
    fn test_transform_sent_approved_block_event() {
        let event = F1r3flyEvent::sent_approved_block("hash-authorized".to_string());

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "sent-approved-block");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash-authorized");
    }

    #[test]
    fn test_transform_block_approval_received_event() {
        let event = F1r3flyEvent::block_approval_received(
            "hash-approved".to_string(),
            "sender123".to_string(),
        );

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "block-approval-received");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash-approved");
        assert_eq!(payload["sender"], "sender123");
    }

    #[test]
    fn test_transform_approved_block_received_event() {
        let event = F1r3flyEvent::approved_block_received("hash-received".to_string());

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "approved-block-received");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "hash-received");
    }

    #[test]
    fn test_transform_entered_running_state_event() {
        let event = F1r3flyEvent::entered_running_state("running-hash".to_string());

        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        assert_eq!(result["event"], "entered-running-state");
        assert_eq!(result["schema-version"], 1);

        let payload = &result["payload"];
        assert_eq!(payload["block-hash"], "running-hash");
    }

    #[test]
    fn test_transformation_has_correct_structure() {
        let event = F1r3flyEvent::block_finalised("test-hash".to_string());
        let result = EventsInfo::transform_f1r3fly_event(&event).unwrap();

        // Verify the structure has exactly 3 top-level keys
        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert_eq!(obj.len(), 3);
        assert!(obj.contains_key("event"));
        assert!(obj.contains_key("schema-version"));
        assert!(obj.contains_key("payload"));

        // Verify payload doesn't contain the "event" field
        let payload = &result["payload"];
        assert!(payload.is_object());
        let payload_obj = payload.as_object().unwrap();
        assert!(!payload_obj.contains_key("event"));
    }
}
