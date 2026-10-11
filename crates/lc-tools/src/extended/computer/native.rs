//! Computer Use `Native` mode — *real local* screen interaction.
//!
//! Unlike the `AnthropicApi` mode (which forwards actions to a cloud API and
//! acts as if on a remote display), `Native` drives the actual local machine:
//! it captures a real screenshot and moves a real cursor / types / scrolls /
//! presses real keys. This is what the "computer use local execution loop"
//! (v0.27 Track 3) means — the annotated action becomes a live local side
//! effect instead of a simulated one.
//!
//! **Safety gate.** This is intentionally *opt-in* on two axes:
//!   1. Distribution (compile time): the heavy native deps (`enigo`, `xcap`)
//!      are only pulled in under the `computer-native` cargo feature. Building
//!      the published default split does not link them.
//!   2. Construction (runtime): a native tool only exists if you explicitly
//!      build one via [`ComputerUseTool::new_native`]. There is no implicit
//!      native fallback from an Anthropic-key tool — you must opt in.
//!
//! Because actions run on the caller's own display, this module treats every
//! action as a deliberate, user-authorized effect. It never starts with any
//! privilege beyond what the OS already grants the process (a GUI session that
//! can take a screenshot can equally click), and it performs no network I/O.
//!
//! **Feedback loop** (v0.27 Track 3, item 206). Screenshotting alone does not
//! close the loop an agent drives: after acting (clicking, typing, scrolling,
//! pressing a key) it must *see* the changed screen to decide the next action.
//! So every state-changing native action — `click` / `type` / `scroll` /
//! `key_press` / `wait` — captures a fresh screenshot and attaches it to the
//! same output, turning the sequence into an observe-act-observe loop. The
//! refresh is best-effort: a failed capture does not undo a click that already
//! happened; it only notes the failure (see [`close_feedback`]).

use lc_core::tools::ToolError;

use super::actions::{ComputerUseInput, ComputerUseOutput};

/// The `Native` executor, gated behind the `computer-native` feature. It is a
/// separate type so the dependency surface stays invisible to builds that
/// disable the feature.
pub(super) struct NativeComputer {
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}

impl NativeComputer {
    /// Perform one native action. This may visibly drive the live desktop.
    pub(super) async fn execute(
        &self,
        input: &ComputerUseInput,
    ) -> Result<ComputerUseOutput, ToolError> {
        use enigo::{Enigo, Keyboard, Mouse, Settings};

        // All input effects come from a single thread-safe handle.
        let mut enigo = Enigo::new(&Settings::default())
            .map_err(|e| ToolError::ExecutionFailed(format!("native input init failed: {e}")))?;

        // Resolve the *actual* primary display size (T2 §4): coordinates are
        // validated against the real desktop, not the `w/h` the caller passed
        // to `new_native`, which can disagree with the display the cursor will
        // actually move on (multi-monitor mixed sizes / DPI scale).
        let (phys_w, phys_h) = primary_display_size()?;

        // Validate before any side effect: a malformed action must not leave a
        // partial cursor/type/scroll sequence applied to the real desktop.
        validate_native(input, phys_w, phys_h)?;

        match input.action.as_str() {
            "screenshot" => {
                let png = capture_primary_screenshot()?;
                Ok(ComputerUseOutput {
                    action: "screenshot".to_string(),
                    result: format!("Captured display {phys_w}x{phys_h}"),
                    screenshot_base64: Some(b64(&png)),
                })
            }
            "click" => {
                let (x, y) = coord(input)?;
                let button = if input.text.as_deref() == Some("right") {
                    enigo::Button::Right
                } else {
                    enigo::Button::Left
                };
                enigo
                    .move_mouse(x, y, enigo::Coordinate::Abs)
                    .map_err(native_err("move_mouse"))?;
                enigo
                    .button(button, enigo::Direction::Click)
                    .map_err(native_err("button"))?;
                with_screen_feedback(ComputerUseOutput {
                    action: "click".to_string(),
                    result: format!("Clicked {} at ({x}, {y})", name(button)),
                    screenshot_base64: None,
                })
            }
            "type" => {
                let text = input.text.as_deref().unwrap_or("");
                enigo.text(text).map_err(native_err("type"))?;
                with_screen_feedback(ComputerUseOutput {
                    action: "type".to_string(),
                    result: format!("Typed {} char(s)", text.chars().count()),
                    screenshot_base64: None,
                })
            }
            "scroll" => {
                let amount = input.amount.unwrap_or(3);
                // Positive length = up, negative = down on Axis::Vertical.
                let dy = if input.direction.as_deref() == Some("up") {
                    amount
                } else {
                    -amount
                };
                enigo
                    .scroll(dy, enigo::Axis::Vertical)
                    .map_err(native_err("scroll"))?;
                with_screen_feedback(ComputerUseOutput {
                    action: "scroll".to_string(),
                    result: format!(
                        "Scrolled by {} ({})",
                        dy,
                        input.direction.as_deref().unwrap_or("down")
                    ),
                    screenshot_base64: None,
                })
            }
            "key_press" => {
                let names = input.keys.as_deref().unwrap_or(&[]);
                let keys = names
                    .iter()
                    .map(|n| key(n))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|u| ToolError::InvalidInput(format!("unknown key: {u}")))?;
                for k in &keys {
                    enigo
                        .key(*k, enigo::Direction::Press)
                        .map_err(native_err("key"))?;
                }
                for k in keys.iter().rev() {
                    enigo
                        .key(*k, enigo::Direction::Release)
                        .map_err(native_err("key"))?;
                }
                with_screen_feedback(ComputerUseOutput {
                    action: "key_press".to_string(),
                    result: format!("Pressed chord: {}", names.join("+")),
                    screenshot_base64: None,
                })
            }
            "wait" => {
                let ms = input.duration_ms.unwrap_or(1000).min(60_000);
                // B3 (framework review): `std::thread::sleep` blocked the whole
                // worker thread this future runs on, stalling every other task
                // scheduled on that thread. An async `wait` must yield, not park
                // the thread.
                tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                with_screen_feedback(ComputerUseOutput {
                    action: "wait".to_string(),
                    result: format!("Waited {ms} ms"),
                    screenshot_base64: None,
                })
            }
            other => Err(ToolError::InvalidInput(format!(
                "Unknown action: '{other}'"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (no side effect until it passes)
// ---------------------------------------------------------------------------

fn coord(input: &ComputerUseInput) -> Result<(i32, i32), ToolError> {
    let c = input
        .coordinate
        .as_deref()
        .ok_or_else(|| ToolError::InvalidInput("coordinate required".to_string()))?;
    validate_coord(c, 0, i32::MAX)?;
    Ok((c[0], c[1]))
}

fn validate_coord(c: &[i32], min: i32, max: i32) -> Result<(), ToolError> {
    if c.len() != 2 {
        return Err(ToolError::InvalidInput(
            "coordinate must be exactly [x, y]".to_string(),
        ));
    }
    for (idx, v) in c.iter().enumerate() {
        if *v < min || *v > max {
            return Err(ToolError::InvalidInput(format!(
                "coordinate[{idx}] {v} out of bounds ({min}-{max})"
            )));
        }
    }
    Ok(())
}

fn validate_native(input: &ComputerUseInput, width: u32, height: u32) -> Result<(), ToolError> {
    let valid = ["screenshot", "click", "type", "scroll", "key_press", "wait"];
    if !valid.contains(&input.action.as_str()) {
        return Err(ToolError::InvalidInput(format!(
            "Unknown action: '{}'. Valid actions: {:?}",
            input.action, valid
        )));
    }
    let w = i32::try_from(width).unwrap_or(i32::MAX);
    let h = i32::try_from(height).unwrap_or(i32::MAX);
    match input.action.as_str() {
        "click" => {
            let c = input
                .coordinate
                .as_deref()
                .ok_or_else(|| ToolError::InvalidInput("coordinate required".to_string()))?;
            validate_coord(c, 0, w)?;
            validate_coord(c, 0, h)?;
        }
        "scroll" => {
            let c = input
                .coordinate
                .as_deref()
                .ok_or_else(|| ToolError::InvalidInput("coordinate required".to_string()))?;
            validate_coord(c, 0, w)?;
            validate_coord(c, 0, h)?;
            if input.direction.is_none() {
                return Err(ToolError::InvalidInput(
                    "'scroll' action requires 'direction' field".to_string(),
                ));
            }
        }
        "type" if input.text.is_none() => {
            return Err(ToolError::InvalidInput(
                "'type' action requires 'text' field".to_string(),
            ));
        }
        "key_press" if input.keys.is_none() => {
            return Err(ToolError::InvalidInput(
                "'key_press' action requires 'keys' field".to_string(),
            ));
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Primary display + screenshot (xcap)
// ---------------------------------------------------------------------------

/// The *primary* display's physical size in pixels.
///
/// Multi-monitor setups can mix resolutions/DPI, so this reflects the real
/// desktop the cursor moves on rather than any caller-reported `new_native(w,h)`.
fn primary_display_size() -> Result<(u32, u32), ToolError> {
    let monitor = primary_monitor()?;
    let w = monitor
        .width()
        .map_err(|e| ToolError::ExecutionFailed(format!("monitor width failed: {e}")))?;
    let h = monitor
        .height()
        .map_err(|e| ToolError::ExecutionFailed(format!("monitor height failed: {e}")))?;
    Ok((w as u32, h as u32))
}

/// Pick the *primary* monitor (falling back to the first listed).
///
/// T2 §4: `Monitor::all().first()` was not guaranteed to be the primary one,
/// so a laptop with an external monitor could screenshot the wrong display.
fn primary_monitor() -> Result<xcap::Monitor, ToolError> {
    use xcap::Monitor;

    let monitors = Monitor::all()
        .map_err(|e| ToolError::ExecutionFailed(format!("list monitors failed: {e}")))?;
    monitors
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .or_else(|| Monitor::all().ok().and_then(|ms| ms.into_iter().next()))
        .ok_or_else(|| ToolError::ExecutionFailed("no display available".to_string()))
}

fn capture_primary_screenshot() -> Result<Vec<u8>, ToolError> {
    use std::io::Cursor;

    let monitor = primary_monitor()?;
    let img = monitor
        .capture_image()
        .map_err(|e| ToolError::ExecutionFailed(format!("capture failed: {e}")))?;

    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| ToolError::ExecutionFailed(format!("png encode failed: {e}")))?;
    Ok(out.into_inner())
}

// ---------------------------------------------------------------------------
// Feedback loop (click / type / scroll / key_press / wait → refreshed screen)
// ---------------------------------------------------------------------------

/// Best-effort feedback loop (v0.27 T3 item 206): attach a freshly-captured
/// screenshot to a *state-changing* action's output so the driving agent sees
/// what the action did and can decide the next move.
///
/// Pure and testable: the refresh `Vec<u8>` is injected, so the message/base64
/// composition is verified offline without a real display. Degrades gracefully —
/// the action already happened before this is called, so a failed or empty
/// refresh must never turn that success into an error; it only annotates the
/// result. A `screenshot` action's own return value *is* the feedback, so it is
/// left untouched to avoid a redundant second capture.
fn close_feedback(out: ComputerUseOutput, refresh: Result<Vec<u8>, ToolError>) -> ComputerUseOutput {
    let mut out = out;
    if out.action == "screenshot" {
        return out;
    }
    match refresh {
        Ok(png) if !png.is_empty() => {
            out.result = format!("{} | screen refreshed ({} bytes)", out.result, png.len());
            out.screenshot_base64 = Some(b64(&png));
        }
        Ok(_) => out.result = format!("{} | screen refresh empty", out.result),
        Err(e) => out.result = format!("{} | screen refresh failed: {e}", out.result),
    }
    out
}

/// Capture the current screen and close the feedback loop over `out`.
///
/// Always returns `Ok`; a capture failure is folded into the result message by
/// [`close_feedback`] rather than fail a successful action.
fn with_screen_feedback(out: ComputerUseOutput) -> Result<ComputerUseOutput, ToolError> {
    Ok(close_feedback(out, capture_primary_screenshot()))
}

fn b64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for ch in bytes.chunks(3) {
        let b0 = ch[0] as u32;
        let b1 = *ch.get(1).unwrap_or(&0) as u32;
        let b2 = *ch.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if ch.len() > 1 {
            A[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if ch.len() > 2 {
            A[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Key mapping (subset → enigo `Key`)
// ---------------------------------------------------------------------------

fn key(name: &str) -> Result<enigo::Key, String> {
    use enigo::Key;
    Ok(match name.to_lowercase().as_str() {
        "ctrl" | "control" => Key::Control,
        "shift" => Key::Shift,
        "alt" => Key::Alt,
        "meta" | "cmd" | "win" | "super" => Key::Meta,
        "enter" | "return" => Key::Return,
        "tab" => Key::Tab,
        "esc" | "escape" => Key::Escape,
        "space" => Key::Space,
        "backspace" => Key::Backspace,
        "del" | "delete" => Key::Delete,
        "up" => Key::UpArrow,
        "down" => Key::DownArrow,
        "left" => Key::LeftArrow,
        "right" => Key::RightArrow,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" => Key::PageUp,
        "pagedown" => Key::PageDown,
        single if single.chars().count() == 1 => Key::Unicode(single.chars().next().unwrap()),
        other => return Err(other.to_string()),
    })
}

fn name(b: enigo::Button) -> &'static str {
    match b {
        enigo::Button::Right => "right",
        enigo::Button::Middle => "middle",
        _ => "left",
    }
}

fn native_err(ctx: &'static str) -> impl Fn(enigo::InputError) -> ToolError {
    move |e| ToolError::ExecutionFailed(format!("native {ctx} failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // REAL desktop smoke test (v0.27 Track 3). This is deliberately `#[ignore]`d:
    // it drives the actual local display (captures a screenshot, moves a pointer),
    // so it must only run by hand on a real interactive session, never in CI.
    //   CARGO_TARGET_DIR=/d/rt-tools cargo test -p lc-tools \
    //     --features computer-native --lib -- --ignored native_real_desktop_smoke
    #[cfg(feature = "computer-native")]
    #[tokio::test]
    #[ignore]
    async fn native_real_desktop_smoke() {
        use lc_core::{tools::ToolError, BaseTool};
        let t = super::super::ComputerUseTool::new_native(4, 4);
        // 1) Real screenshot: must produce a non-trivial PNG. `new_native(4,4)`
        //    just seeds a tiny display size; the screenshot captures the real desktop.
        //    (Height/width are only used for coordinate validation, not for size.)
        let shot: Result<String, ToolError> = t.run(r#"{"action":"screenshot"}"#.to_string()).await;
        match shot {
            Ok(out) => {
                assert!(
                    out.len() > 2000,
                    "a real desktop screenshot PNG should be >2KB, got {} bytes (headless?)",
                    out.len()
                );
            }
            Err(e) => panic!("real screenshot failed (no interactive display?): {e}"),
        }
        // 2) Pointer: enigo must at least initialise on a real desktop. Moving to
        //    the origin is a harmless visual move; we do not click anything.
        use enigo::{Coordinate, Mouse};
        let mut enigo = enigo::Enigo::new(&enigo::Settings::default())
            .expect("enigo must initialise on a real Windows desktop");
        let mv = enigo.move_mouse(0, 0, Coordinate::Abs);
        assert!(
            mv.is_ok(),
            "enigo move_mouse failed on a real desktop: {mv:?}"
        );
    }

    #[test]
    fn b64_encodes_png_like_bytes() {
        // length-0 → ""
        assert_eq!(b64(b""), "");
        // "abc" → standard base64 "YWJj"
        assert_eq!(b64(b"abc"), "YWJj");
        // length 2 → "YWI="
        assert_eq!(b64(b"ab"), "YWI=");
    }

    #[test]
    fn key_maps_names_and_single_chars() {
        assert!(matches!(key("ctrl"), Ok(enigo::Key::Control)));
        assert!(matches!(key("SHIFT"), Ok(enigo::Key::Shift)));
        assert!(matches!(key("return"), Ok(enigo::Key::Return)));
        assert!(matches!(key("pageup"), Ok(enigo::Key::PageUp)));
        assert!(matches!(key("a"), Ok(enigo::Key::Unicode('a'))));
        assert!(key("not-a-key").is_err());
        assert!(key("ab").is_err());
    }

    #[test]
    fn validate_rejects_bad_coord() {
        let input = ComputerUseInput {
            action: "click".into(),
            coordinate: Some(vec![1]), // wrong length
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        assert!(validate_native(&input, 1024, 768).is_err());
    }

    #[test]
    fn validate_rejects_out_of_bounds() {
        let input = ComputerUseInput {
            action: "click".into(),
            coordinate: Some(vec![5000, 5000]),
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        assert!(validate_native(&input, 1024, 768).is_err());
    }

    fn base_click() -> ComputerUseOutput {
        ComputerUseOutput {
            action: "click".to_string(),
            result: "Clicked left at (10, 20)".to_string(),
            screenshot_base64: None,
        }
    }

    #[test]
    fn close_feedback_attaches_refreshed_screenshot() {
        let out = close_feedback(base_click(), Ok(vec![1, 2, 3]));
        assert_eq!(out.screenshot_base64.as_deref(), Some("AQID"));
        assert!(out.result.contains("screen refreshed (3 bytes)"));
        assert_eq!(out.action, "click");
    }

    #[test]
    fn close_feedback_skips_the_screenshot_action() {
        let shot = ComputerUseOutput {
            action: "screenshot".to_string(),
            result: "Captured 1024x768".to_string(),
            screenshot_base64: Some("already-the-feedback".to_string()),
        };
        // Save the fields before handing the struct to `close_feedback` (it takes
        // ownership; `ComputerUseOutput` is not `Clone`).
        let expected_result = shot.result.clone();
        let expected_shot = shot.screenshot_base64.clone();
        // Even with a refresh injected, a screenshot action is left untouched to
        // avoid capturing a redundant second frame of the same screen.
        let out = close_feedback(shot, Ok(vec![9, 9, 9]));
        assert_eq!(out.screenshot_base64, expected_shot);
        assert_eq!(out.result, expected_result);
    }

    #[test]
    fn close_feedback_degrades_on_empty_or_failed_refresh() {
        // Empty PNG: no screenshot attached, message notes it, action stays Ok.
        let empty = close_feedback(base_click(), Ok(Vec::new()));
        assert_eq!(empty.screenshot_base64, None);
        assert!(empty.result.contains("screen refresh empty"));

        // Capture failure: same graceful behavior, message notes the cause.
        let failed = close_feedback(
            base_click(),
            Err(ToolError::ExecutionFailed("no display".to_string())),
        );
        assert_eq!(failed.screenshot_base64, None);
        assert!(
            failed.result.contains("screen refresh failed:") && failed.result.contains("no display"),
            "message should surface the capture failure, got: {}",
            failed.result
        );
    }

    #[test]
    fn validate_accepts_valid_click() {
        let input = ComputerUseInput {
            action: "click".into(),
            coordinate: Some(vec![500, 400]),
            text: Some("left".into()),
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        assert!(validate_native(&input, 1920, 1080).is_ok());
    }

    #[test]
    fn validate_requires_type_text() {
        let input = ComputerUseInput {
            action: "type".into(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        assert!(validate_native(&input, 1920, 1080).is_err());
    }

    #[test]
    fn validate_requires_keys() {
        let input = ComputerUseInput {
            action: "key_press".into(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        assert!(validate_native(&input, 1920, 1080).is_err());
    }
}
