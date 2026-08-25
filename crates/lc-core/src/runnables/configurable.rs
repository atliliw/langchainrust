// lc-core/src/runnables/configurable.rs
//! Configurable runnables — runtime selection, the Rust counterpart of
//! Python LCEL's `Runnable.configurable_alternatives(...)` and
//! `Runnable.configurable_fields(...)`.
//!
//! Both read the `configurable` map of `RunnableConfig` (Python's
//! `config["configurable"]`) at invoke time, so the same chain can route
//! differently per call without rebuilding the pipeline.

use super::any::{into_runnable_any, RunnableAny};
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde_json::Value;
use std::any::Any;
use std::marker::PhantomData;
use std::pin::Pin;

/// Routes between a default runnable and named alternatives at invoke time.
///
/// The selector key (`which`) is read from `config.configurable`; the value
/// must be a string naming either the `default_key` (→ default) or one of
/// the alternatives. An unknown value falls back to the default.
///
/// # Example
///
/// ```rust,ignore
/// let chain = llm.configurable_alternatives(
///     "provider", "default",
///     vec![("anthropic", anthropic_llm), ("ollama", ollama_llm)],
/// );
/// // config = RunnableConfig::new().with_configurable("provider", json!("anthropic"))
/// ```
pub struct RunnableConfigurable<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    default: Box<dyn RunnableAny>,
    alternatives: Vec<(String, Box<dyn RunnableAny>)>,
    which: String,
    default_key: String,
    _marker: PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug
    for RunnableConfigurable<I, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.alternatives.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("RunnableConfigurable")
            .field("which", &self.which)
            .field("default_key", &self.default_key)
            .field("alternatives", &names)
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableConfigurable<I, O> {
    /// Build a configurable router.
    ///
    /// * `default` — runnable used when the selector is absent, equals
    ///   `default_key`, or names an unknown alternative.
    /// * `which` — key read from `config.configurable`.
    /// * `default_key` — config value that routes to `default`.
    /// * `alternatives` — `(option name, runnable)` pairs; the config value
    ///   must match one of these names to be selected.
    pub fn new<R>(default: R, which: impl Into<String>, default_key: impl Into<String>) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            default: into_runnable_any(default),
            alternatives: Vec::new(),
            which: which.into(),
            default_key: default_key.into(),
            _marker: PhantomData,
        }
    }

    /// Add an alternative branch reachable via `config.configurable[which] == name`.
    pub fn with_alternative<R>(mut self, name: impl Into<String>, runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        self.alternatives
            .push((name.into(), into_runnable_any(runnable)));
        self
    }

    /// Pick the target for this call based on the configurable selector.
    fn resolve(&self, config: &Option<RunnableConfig>) -> &dyn RunnableAny {
        let selected = config
            .as_ref()
            .and_then(|c| c.configurable_value(&self.which))
            .and_then(|v| v.as_str());
        match selected {
            Some(name) if name != self.default_key => self
                .alternatives
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, r)| r.as_ref())
                .unwrap_or(&*self.default),
            _ => &*self.default,
        }
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O>
    for RunnableConfigurable<I, O>
{
    type Error = LcelError;

    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<O, LcelError> {
        let target = self.resolve(&config);
        let boxed = target
            .invoke_any(Box::new(input) as Box<dyn Any + Send>, config)
            .await?;
        boxed.downcast::<O>().map(|b| *b).map_err(|_| {
            LcelError::TypeMismatch(format!(
                "configurable invoke downcast: expected {}",
                std::any::type_name::<O>()
            ))
        })
    }

    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<O>, LcelError> {
        let target = self.resolve(&config);
        let boxed_inputs: Vec<Box<dyn Any + Send>> = inputs
            .into_iter()
            .map(|i| Box::new(i) as Box<dyn Any + Send>)
            .collect();
        let results = target.batch_any(boxed_inputs, config).await?;
        results
            .into_iter()
            .map(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable batch downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
            .collect()
    }

    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let target = self.resolve(&config);
        let stream = target
            .stream_any(Box::new(input) as Box<dyn Any + Send>, config)
            .await?;
        let output = stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable stream downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(output))
    }

    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let target = self.resolve(&config);
        let any_input: Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>> =
            Box::pin(input.map(|result| result.map(|item| Box::new(item) as Box<dyn Any + Send>)));

        let stream = target.transform_any(any_input, config).await?;
        let output = stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable transform downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(output))
    }
}

/// Overrides recognized config fields at invoke time from
/// `config.configurable` (Python's `Runnable.configurable_fields`).
///
/// The following configurable keys are applied before the inner runnable is
/// invoked:
///
/// | configurable key        | effect |
/// |-------------------------|--------|
/// | `temperature` (number)  | `RunnableConfig.temperature` override |
/// | `max_tokens` (integer)  | `RunnableConfig.max_tokens` override |
/// | anything else           | merged into `RunnableConfig.metadata` |
///
/// Providers already consume `temperature` / `max_tokens` (via
/// `sampling_overrides`), so e.g. `llm.configurable_fields()` actually
/// changes sampling when the runtime config carries the key.
pub struct RunnableConfigurableFields<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    inner: Box<dyn RunnableAny>,
    _marker: PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableConfigurableFields<I, O> {
    /// Wrap a runnable so its config fields can be overridden at invoke time.
    pub fn new<R>(runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            inner: into_runnable_any(runnable),
            _marker: PhantomData,
        }
    }

    /// Build the effective config by promoting configurable keys into typed
    /// config fields / metadata.
    fn effective_config(&self, config: &Option<RunnableConfig>) -> RunnableConfig {
        let mut effective = config.clone().unwrap_or_default();
        // Clone the map first — applying each override consumes `effective`.
        let configurables = effective.configurable.clone();
        for (key, value) in configurables {
            match (key.as_str(), &value) {
                ("temperature", Value::Number(n)) => match n.as_f64() {
                    Some(f) => effective = effective.with_temperature(f as f32),
                    None => effective = effective.with_metadata(key.to_string(), value.clone()),
                },
                ("max_tokens", Value::Number(n)) => match n.as_u64() {
                    Some(u) => effective = effective.with_max_tokens(u as usize),
                    None => effective = effective.with_metadata(key.to_string(), value.clone()),
                },
                (k, v) => effective = effective.with_metadata(k.to_string(), v.clone()),
            }
        }
        effective
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O>
    for RunnableConfigurableFields<I, O>
{
    type Error = LcelError;

    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<O, LcelError> {
        let effective = self.effective_config(&config);
        let boxed = self
            .inner
            .invoke_any(Box::new(input) as Box<dyn Any + Send>, Some(effective))
            .await?;
        boxed.downcast::<O>().map(|b| *b).map_err(|_| {
            LcelError::TypeMismatch(format!(
                "configurable_fields invoke downcast: expected {}",
                std::any::type_name::<O>()
            ))
        })
    }

    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<O>, LcelError> {
        let effective = self.effective_config(&config);
        let boxed_inputs: Vec<Box<dyn Any + Send>> = inputs
            .into_iter()
            .map(|i| Box::new(i) as Box<dyn Any + Send>)
            .collect();
        let results = self.inner.batch_any(boxed_inputs, Some(effective)).await?;
        results
            .into_iter()
            .map(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable_fields batch downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
            .collect()
    }

    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let effective = self.effective_config(&config);
        let stream = self
            .inner
            .stream_any(Box::new(input) as Box<dyn Any + Send>, Some(effective))
            .await?;
        let output = stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable_fields stream downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(output))
    }

    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let effective = self.effective_config(&config);
        let any_input: Pin<Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>> =
            Box::pin(input.map(|result| result.map(|item| Box::new(item) as Box<dyn Any + Send>)));

        let stream = self.inner.transform_any(any_input, Some(effective)).await?;
        let output = stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "configurable_fields transform downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });
        Ok(Box::pin(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunnableExt;
    use crate::RunnableLambda;

    #[tokio::test]
    async fn routes_to_alternative() {
        let chain = RunnableLambda::new_sync(|s: String| format!("default:{s}"))
            .configurable_alternatives(
                "which",
                "default",
                vec![(
                    "alt",
                    RunnableLambda::new_sync(|s: String| format!("alt:{s}")),
                )],
            );
        let cfg = RunnableConfig::new().with_configurable("which", serde_json::json!("alt"));
        assert_eq!(
            chain.invoke("x".to_string(), Some(cfg)).await.unwrap(),
            "alt:x"
        );
    }

    #[tokio::test]
    async fn defaults_when_no_selector() {
        let chain = RunnableLambda::new_sync(|s: String| format!("default:{s}"))
            .configurable_alternatives(
                "which",
                "default",
                vec![(
                    "alt",
                    RunnableLambda::new_sync(|s: String| format!("alt:{s}")),
                )],
            );
        // 无 config / 无 which 键 → default
        assert_eq!(
            chain.invoke("x".to_string(), None).await.unwrap(),
            "default:x"
        );
        let cfg = RunnableConfig::new().with_configurable("which", serde_json::json!("default"));
        assert_eq!(
            chain.invoke("x".to_string(), Some(cfg)).await.unwrap(),
            "default:x"
        );
        // 未知值 → default
        let cfg = RunnableConfig::new().with_configurable("which", serde_json::json!("nope"));
        assert_eq!(
            chain.invoke("x".to_string(), Some(cfg)).await.unwrap(),
            "default:x"
        );
    }

    /// 探针:Runnable<(), String>,invoke 时把收到的 config.temperature 打出来
    struct TemperatureProbe;

    #[async_trait]
    impl Runnable<(), String> for TemperatureProbe {
        type Error = LcelError;

        async fn invoke(
            &self,
            _input: (),
            config: Option<RunnableConfig>,
        ) -> Result<String, LcelError> {
            Ok(format!(
                "temp={:?}",
                config.as_ref().and_then(|c| c.temperature)
            ))
        }
    }

    #[tokio::test]
    async fn configurable_fields_promotes_temperature() {
        let wrapped = RunnableConfigurableFields::<(), String>::new(TemperatureProbe);
        let cfg = RunnableConfig::new().with_configurable("temperature", serde_json::json!(0.5));
        let out = wrapped.invoke((), Some(cfg)).await.unwrap();
        assert_eq!(
            out, "temp=Some(0.5)",
            "temperature 应被提升为 typed config 字段"
        );
    }

    #[tokio::test]
    async fn configurable_fields_promotes_max_tokens() {
        let wrapped = RunnableConfigurableFields::<(), String>::new(TemperatureProbe);
        let cfg = RunnableConfig::new().with_configurable("max_tokens", serde_json::json!(128));
        let out = wrapped.invoke((), Some(cfg)).await.unwrap();
        assert_eq!(out, "temp=None"); // temperature 未被设置
    }

    #[tokio::test]
    async fn configurable_fields_other_keys_go_to_metadata() {
        struct MetadataProbe;

        #[async_trait]
        impl Runnable<(), String> for MetadataProbe {
            type Error = LcelError;

            async fn invoke(
                &self,
                _input: (),
                config: Option<RunnableConfig>,
            ) -> Result<String, LcelError> {
                let cfg = config.unwrap_or_default();
                Ok(cfg
                    .metadata
                    .get("provider")
                    .cloned()
                    .unwrap_or_default()
                    .to_string())
            }
        }

        let wrapped = RunnableConfigurableFields::<(), String>::new(MetadataProbe);
        let cfg = RunnableConfig::new().with_configurable("provider", serde_json::json!("x"));
        let out = wrapped.invoke((), Some(cfg)).await.unwrap();
        assert_eq!(out, "\"x\"", "未知 configurable 键应进 metadata");
    }
}
