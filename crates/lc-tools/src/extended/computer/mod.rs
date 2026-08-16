//! Computer Use tool for screen interaction via Anthropic API or native mode.

pub mod actions;
pub mod screen;

pub use actions::{ComputerMode, ComputerUseInput, ComputerUseOutput};
pub use screen::ComputerUseTool;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::tools::ToolError;
    use lc_core::BaseTool;

    fn tool() -> ComputerUseTool {
        ComputerUseTool::new_anthropic("test-key", 1024, 768)
    }

    #[test]
    fn test_name() {
        let t = tool();
        assert_eq!(t.name(), "computer_use");
    }

    #[test]
    fn test_description_contains_actions() {
        let t = tool();
        let desc = t.description();
        for action in &["screenshot", "click", "type", "scroll", "key_press", "wait"] {
            assert!(
                desc.contains(action),
                "description should mention '{}'",
                action
            );
        }
    }

    #[test]
    fn test_args_schema_has_action_field() {
        let t = tool();
        let schema = t.args_schema().expect("schema should be present");
        let props = schema
            .get("properties")
            .expect("schema should have properties");
        assert!(
            props.get("action").is_some(),
            "schema should define 'action' property"
        );
    }

    #[tokio::test]
    async fn test_run_invalid_json() {
        let t = tool();
        let result = t.run("not json".to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_run_unknown_action() {
        let t = tool();
        let input = r#"{"action":"fly"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("Unknown action"),
            "error should mention unknown action, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_click_missing_coordinate() {
        let t = tool();
        let input = r#"{"action":"click"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("coordinate"),
            "error should mention missing coordinate, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_type_missing_text() {
        let t = tool();
        let input = r#"{"action":"type"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("text"),
            "error should mention missing text, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_key_press_missing_keys() {
        let t = tool();
        let input = r#"{"action":"key_press"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("keys"),
            "error should mention missing keys, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_scroll_missing_direction() {
        let t = tool();
        let input = r#"{"action":"scroll","coordinate":[100,200]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("direction"),
            "error should mention missing direction, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_coordinate_out_of_bounds() {
        let t = tool();
        let input = r#"{"action":"click","coordinate":[2000,500]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("out of bounds"),
            "error should mention out of bounds, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_coordinate_wrong_length() {
        let t = tool();
        let input = r#"{"action":"click","coordinate":[100]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("exactly [x, y]"),
            "error should mention coordinate format, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_empty_api_key() {
        let t = ComputerUseTool::new_anthropic("", 1024, 768);
        let input = r#"{"action":"screenshot"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("API key"),
            "error should mention API key, got: {}",
            err
        );
    }

    #[test]
    fn test_build_tool_input_screenshot() {
        let t = tool();
        let input = ComputerUseInput {
            action: "screenshot".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "screenshot");
    }

    #[test]
    fn test_build_tool_input_click() {
        let t = tool();
        let input = ComputerUseInput {
            action: "click".to_string(),
            coordinate: Some(vec![100, 200]),
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "mouse_click");
        assert_eq!(result["coordinate"], serde_json::json!([100, 200]));
        assert_eq!(result["text"], "left");
    }

    #[test]
    fn test_build_tool_input_click_right_button() {
        let t = tool();
        let input = ComputerUseInput {
            action: "click".to_string(),
            coordinate: Some(vec![50, 75]),
            text: Some("right".to_string()),
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["text"], "right");
    }

    #[test]
    fn test_build_tool_input_type() {
        let t = tool();
        let input = ComputerUseInput {
            action: "type".to_string(),
            coordinate: None,
            text: Some("hello world".to_string()),
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "type");
        assert_eq!(result["text"], "hello world");
    }

    #[test]
    fn test_build_tool_input_scroll() {
        let t = tool();
        let input = ComputerUseInput {
            action: "scroll".to_string(),
            coordinate: Some(vec![500, 300]),
            text: None,
            keys: None,
            direction: Some("up".to_string()),
            amount: Some(5),
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "scroll");
        assert_eq!(result["coordinate"], serde_json::json!([500, 300]));
        assert_eq!(result["direction"], "up");
        assert_eq!(result["amount"], 5);
    }

    #[test]
    fn test_build_tool_input_key_press() {
        let t = tool();
        let input = ComputerUseInput {
            action: "key_press".to_string(),
            coordinate: None,
            text: None,
            keys: Some(vec!["ctrl".to_string(), "c".to_string()]),
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "key");
        assert_eq!(result["text"], "ctrl+c");
    }

    #[test]
    fn test_build_tool_input_wait() {
        let t = tool();
        let input = ComputerUseInput {
            action: "wait".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: Some(2000),
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "wait");
        assert_eq!(result["duration_ms"], 2000);
    }

    #[test]
    fn test_build_tool_input_unknown_action() {
        let t = tool();
        let input = ComputerUseInput {
            action: "fly".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown action"));
    }

    #[test]
    fn test_default() {
        let t = ComputerUseTool::default();
        assert_eq!(t.name(), "computer_use");
        assert_eq!(t.display_width, 1024);
        assert_eq!(t.display_height, 768);
    }

    #[test]
    fn test_with_base_url() {
        let t = ComputerUseTool::new_anthropic("key", 1920, 1080)
            .with_base_url("https://custom.api.com");
        assert_eq!(t.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_mode_returns_anthropic() {
        let t = tool();
        assert!(matches!(t.mode(), ComputerMode::AnthropicApi));
    }

    #[test]
    fn test_build_anthropic_request_contains_model() {
        let t = tool();
        let input = ComputerUseInput {
            action: "screenshot".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let body = t.build_anthropic_request(&input).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["type"], "computer_20250124");
    }
}
