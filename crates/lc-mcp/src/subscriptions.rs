//! MCP resource subscriptions (2026-07-28 tracks: stateless & streamable).
//!
//! Lets a client subscribe to resource changes. The client registers interest
//! with `resources/subscribe` and drops it with `resources/unsubscribe`; when a
//! subscribed resource changes, the server pushes
//! `notifications/resources/list_changed` to the interested subscribers.
//!
//! This module supplies the protocol method names, the request/response param
//! types, and an in-memory [`ResourceSubscriptions`] registry that turns a
//! change into the JSON-RPC notification envelope for exactly the interested
//! subscribers. Delivery of that envelope is left to the transport layer (SSE,
//! WebSocket, a channel, …) — this module is pure type/state logic with no real
//! network push, so it is fully unit-testable offline.

use crate::protocol::{notification_message, MCPError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// `resources/subscribe` JSON-RPC method name.
pub const RESOURCES_SUBSCRIBE: &str = "resources/subscribe";
/// `resources/unsubscribe` JSON-RPC method name.
pub const RESOURCES_UNSUBSCRIBE: &str = "resources/unsubscribe";
/// `notifications/resources/list_changed` notification method name (server→client).
pub const NOTIFICATIONS_RESOURCES_LIST_CHANGED: &str = "notifications/resources/list_changed";
/// `notifications/resources/updated` notification method name (server→client):
/// the *content of a subscribed resource* changed — distinct from
/// `list_changed`, which is about the resource *list* itself changing.
pub const NOTIFICATIONS_RESOURCES_UPDATED: &str = "notifications/resources/updated";

/// `resources/subscribe` / `resources/unsubscribe` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceSubscribeParams {
    /// URI of the resource to subscribe to (or stop subscribing to).
    pub uri: String,
}

/// Receives a `notifications/resources/list_changed` envelope for one
/// interested subscriber once a subscribed resource changes.
///
/// The transport implements this to physically deliver the envelope (SSE
/// write, WebSocket frame, channel send, …). The server calls it *after* the
/// registry has produced the envelope, so delivery is real — the only thing
/// this module no longer does is fabricate network I/O it cannot observe.
pub trait ResourcePushSink: Send + Sync {
    /// Deliver `envelope` to `subscriber`. Errors are logged/ignored by the
    /// caller (a delivery failure must not take down the server); the sink
    /// decides whether a delivery failure is fatal for its connection.
    fn deliver(&self, subscriber: &str, envelope: &Value);
}

/// Tracks which subscribers are interested in which resource URIs, so a change
/// to one resource becomes a `notifications/resources/list_changed` delivery to
/// exactly the subscribers who care.
///
/// Pure in-memory state keyed by `uri` → set of subscriber ids (a session id,
/// a connection id, an opaque client token). Interior-mutable via `Mutex` so an
/// [`Arc`](std::sync::Arc)-shared instance can be updated from the server without
/// leaking a lock across an `await`.
#[derive(Debug)]
pub struct ResourceSubscriptions {
    /// `uri` → set of subscriber ids registered for that uri.
    subscribers: Mutex<HashMap<String, HashSet<String>>>,
}

impl Default for ResourceSubscriptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSubscriptions {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            subscribers: Mutex::new(HashMap::new()),
        }
    }

    /// Registers `subscriber`'s interest in `uri`. Returns `Ok(())`; fails with
    /// `invalid_params` when `uri` is empty (a subscribe must name a resource).
    pub fn subscribe(&self, uri: &str, subscriber: impl Into<String>) -> Result<(), MCPError> {
        let uri = uri.trim();
        if uri.is_empty() {
            return Err(MCPError::invalid_params(
                "resources/subscribe uri must not be empty",
            ));
        }
        let subscriber = subscriber.into();
        let mut map = self.subscribers.lock().expect("subscription lock");
        map.entry(uri.to_string()).or_default().insert(subscriber);
        Ok(())
    }

    /// Removes `subscriber`'s interest in `uri`. A no-op when the subscriber (or
    /// the uri) was never registered; returns `Ok(())` regardless so an
    /// unsubscribe is idempotent on the wire.
    pub fn unsubscribe(&self, uri: &str, subscriber: impl Into<String>) -> Result<(), MCPError> {
        let subscriber = subscriber.into();
        let mut map = self.subscribers.lock().expect("subscription lock");
        if let Some(set) = map.get_mut(uri.trim()) {
            set.remove(&subscriber);
            if set.is_empty() {
                map.remove(uri.trim());
            }
        }
        Ok(())
    }

    /// Whether `subscriber` is currently subscribed to `uri`.
    pub fn is_subscribed(&self, uri: &str, subscriber: &str) -> bool {
        let map = self.subscribers.lock().expect("subscription lock");
        map.get(uri)
            .map(|set| set.contains(subscriber))
            .unwrap_or(false)
    }

    /// The ids currently subscribed to `uri` (order is unspecified). Returns an
    /// empty vector when nothing is subscribed.
    pub fn subscribers_of(&self, uri: &str) -> Vec<String> {
        let map = self.subscribers.lock().expect("subscription lock");
        map.get(uri)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether *anyone* is subscribed to `uri`.
    pub fn is_watched(&self, uri: &str) -> bool {
        let map = self.subscribers.lock().expect("subscription lock");
        map.get(uri).map(|set| !set.is_empty()).unwrap_or(false)
    }

    /// Emits the `notifications/resources/list_changed` envelope for `uri` when
    /// at least one subscriber is watching it; returns `None` when nothing is
    /// subscribed (nothing to notify — an honest no-op, no traffic).
    ///
    /// The caller is responsible for delivering the returned [`Value`] to the
    /// interested subscribers (e.g. via [`Self::subscribers_of`]).
    pub fn emit_list_changed(&self, uri: &str) -> Option<Value> {
        if !self.is_watched(uri) {
            return None;
        }
        Some(notification_message(
            NOTIFICATIONS_RESOURCES_LIST_CHANGED,
            Some(json!({ "uri": uri })),
        ))
    }

    /// Emits the `notifications/resources/updated` envelope for `uri` when at
    /// least one subscriber is watching it; returns `None` when nothing is
    /// subscribed. This is the per-resource update the spec delivers to a
    /// subscriber of a specific resource — distinct from
    /// [`Self::emit_list_changed`], which announces the resource *list* itself
    /// changing.
    pub fn emit_updated(&self, uri: &str) -> Option<Value> {
        if !self.is_watched(uri) {
            return None;
        }
        Some(notification_message(
            NOTIFICATIONS_RESOURCES_UPDATED,
            Some(json!({ "uri": uri })),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_constants() {
        assert_eq!(RESOURCES_SUBSCRIBE, "resources/subscribe");
        assert_eq!(RESOURCES_UNSUBSCRIBE, "resources/unsubscribe");
        assert_eq!(
            NOTIFICATIONS_RESOURCES_LIST_CHANGED,
            "notifications/resources/list_changed"
        );
        assert_eq!(
            NOTIFICATIONS_RESOURCES_UPDATED,
            "notifications/resources/updated"
        );
    }

    #[test]
    fn subscribe_then_is_subscribed() {
        let subs = ResourceSubscriptions::new();
        subs.subscribe("file:///a.txt", "sess-1").unwrap();
        assert!(subs.is_subscribed("file:///a.txt", "sess-1"));
        assert!(!subs.is_subscribed("file:///a.txt", "sess-2"));
        assert!(!subs.is_subscribed("file:///b.txt", "sess-1"));
        assert!(subs.is_watched("file:///a.txt"));
    }

    #[test]
    fn subscribe_rejects_empty_uri() {
        let subs = ResourceSubscriptions::new();
        assert!(subs.subscribe("  ", "sess-1").is_err());
        assert!(!subs.is_watched(""));
    }

    #[test]
    fn unsubscribe_is_idempotent_and_removes_watch() {
        let subs = ResourceSubscriptions::new();
        subs.subscribe("file:///a.txt", "sess-1").unwrap();
        subs.unsubscribe("file:///a.txt", "sess-1").unwrap();
        // Removing the last subscriber forgets the uri entirely.
        assert!(!subs.is_watched("file:///a.txt"));
        assert_eq!(subs.subscribers_of("file:///a.txt"), Vec::<String>::new());
        // Unsubscribing never registered (or already removed) is a safe no-op.
        subs.unsubscribe("file:///a.txt", "ghost").unwrap();
        subs.unsubscribe("file:///never", "sess-1").unwrap();
    }

    #[test]
    fn multiple_subscribers_survive_until_last_leaves() {
        let subs = ResourceSubscriptions::new();
        subs.subscribe("file:///a.txt", "sess-1").unwrap();
        subs.subscribe("file:///a.txt", "sess-2").unwrap();
        assert_eq!(subs.subscribers_of("file:///a.txt").len(), 2);
        subs.unsubscribe("file:///a.txt", "sess-1").unwrap();
        assert!(subs.is_watched("file:///a.txt"), "sess-2 still watching");
        subs.unsubscribe("file:///a.txt", "sess-2").unwrap();
        assert!(!subs.is_watched("file:///a.txt"));
    }

    #[test]
    fn emit_list_changed_only_when_watched() {
        let subs = ResourceSubscriptions::new();
        // Nothing subscribed → no notification to emit.
        assert!(subs.emit_list_changed("file:///a.txt").is_none());

        subs.subscribe("file:///a.txt", "sess-1").unwrap();
        let envelope = subs.emit_list_changed("file:///a.txt").expect("emits");
        // Correct method, JSON-RPC notification shape (no id), params carry uri.
        assert_eq!(envelope["method"], NOTIFICATIONS_RESOURCES_LIST_CHANGED);
        assert_eq!(envelope["jsonrpc"], "2.0");
        assert!(envelope.get("id").is_none(), "notifications carry no id");
        assert_eq!(envelope["params"]["uri"], "file:///a.txt");
    }

    #[test]
    fn emit_updated_is_per_resource_notification() {
        let subs = ResourceSubscriptions::new();
        // Not watched → no updated frame.
        assert!(subs.emit_updated("file:///a.txt").is_none());

        subs.subscribe("file:///a.txt", "sess-1").unwrap();
        let envelope = subs.emit_updated("file:///a.txt").expect("emits");
        // Distinct method from list_changed — content change, not list change.
        assert_eq!(envelope["method"], NOTIFICATIONS_RESOURCES_UPDATED);
        assert!(envelope.get("id").is_none(), "notifications carry no id");
        assert_eq!(envelope["params"]["uri"], "file:///a.txt");

        // A *different* unwatched resource still only fires when subscribed.
        assert!(subs.emit_updated("file:///other.txt").is_none());
    }

    #[test]
    fn params_roundtrip() {
        let p = ResourceSubscribeParams {
            uri: "file:///a.txt".to_string(),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"uri\":\"file:///a.txt\""), "{json}");
        let back: ResourceSubscribeParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.uri, "file:///a.txt");
    }
}
