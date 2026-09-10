use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::model::{default_input_modes, default_output_modes, default_protocol_version};

/// A2A v1.0.1 protocol version (the first stable line; Agent Cards restructured
/// to `supportedInterfaces[]` — incompatible with v0.3's single-declaration shape).
pub const A2A_VERSION_V101: &str = "1.0.1";

/// Transport binding of one agent interface (v1.0.1: three bindings,
/// cumulative — a card may advertise several).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum A2ATransport {
    /// JSON-RPC 2.0 over HTTP.
    JsonRpc,
    /// Plain HTTP+JSON.
    HttpJson,
    /// gRPC (see `a2a.proto`; the规范源 for all bindings).
    Grpc,
}

/// One interface of an agent (v1.0.1 `supportedInterfaces[]` entry).
///
/// v1.0.1 breaks with v0.3: instead of a single protocol declaration on the
/// card, each interface carries its own `protocolVersion`, transport binding,
/// URL, and (enterprise multi-tenancy) optional tenant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentInterface {
    /// Protocol version spoken on THIS interface (e.g. "1.0.1").
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Transport binding of this interface.
    pub transport: A2ATransport,
    /// Endpoint URL for this interface.
    pub url: String,
    /// Enterprise multi-tenancy: tenant this interface serves (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl AgentInterface {
    /// Creates an interface entry.
    pub fn new(
        protocol_version: impl Into<String>,
        transport: A2ATransport,
        url: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: protocol_version.into(),
            transport,
            url: url.into(),
            tenant: None,
        }
    }

    /// Sets the tenant (enterprise multi-tenancy).
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }
}

/// A skill that an agent can perform.
///
/// Aligned with the structured skill objects required by the A2A v0.3
/// Agent Card specification (a flat string list is not sufficient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    /// Stable identifier for the skill.
    pub id: String,
    /// Human-readable skill name.
    pub name: String,
    /// Description of what the skill does.
    pub description: String,
}

impl AgentSkill {
    /// Create a new skill.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Agent metadata card, served at `/.well-known/agent-card.json`.
///
/// Describes an agent's identity, endpoint, and capabilities so that
/// other agents can discover and interact with it. Aligned with the
/// A2A v0.3 Agent Card specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Human-readable agent name.
    pub name: String,
    /// Description of what the agent does.
    pub description: String,
    /// Base URL where the agent accepts A2A requests.
    pub url: String,
    /// Structured list of skills the agent can perform.
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
    /// A2A protocol version supported by this agent.
    #[serde(default = "default_protocol_version", rename = "protocolVersion")]
    pub protocol_version: String,
    /// Security schemes the agent supports (e.g. `{"bearerAuth": {...}}`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "securitySchemes")]
    pub security_schemes: Option<Value>,
    /// Interfaces the agent exposes (e.g. `{"sse": true}`).
    ///
    /// Deprecated by v1.0.1's `supported_interfaces` (typed); kept for
    /// v0.3-era readers. New cards should populate `supported_interfaces`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interfaces: Option<Value>,
    /// v1.0.1 typed interface list: each entry carries its own
    /// `protocolVersion`, transport binding, URL and optional tenant.
    ///
    /// A v1.0.1-compliant card MUST have at least one interface here; the
    /// legacy top-level `protocol_version`/`url` remain for v0.3 readers.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        rename = "supportedInterfaces"
    )]
    pub supported_interfaces: Vec<AgentInterface>,
    /// Provider/organization name (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Documentation URL (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    /// Authentication schemes supported (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication: Option<Vec<String>>,
    /// Default input modes (e.g. ["text", "image"]).
    #[serde(default = "default_input_modes")]
    pub default_input_modes: Vec<String>,
    /// Default output modes (e.g. ["text"]).
    #[serde(default = "default_output_modes")]
    pub default_output_modes: Vec<String>,
    /// Digital signature over the canonical card content (P1-3 / P2-5).
    ///
    /// When present, clients SHOULD verify it against the agent's public key
    /// before trusting the card (see `lc_a2a::security`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Data classification this agent deals with (e.g. "public", "internal",
    /// "confidential"). Used for data-boundary / federation policy (P2-7/P2-8).
    #[serde(skip_serializing_if = "Option::is_none", rename = "dataClass")]
    pub data_class: Option<String>,
    /// Jurisdiction(s) this agent operates under (e.g. "US", "EU"). Used for
    /// compliance-aware routing in federations (P2-8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jurisdiction: Option<String>,
    /// Optional protocol capabilities the agent can negotiate (P2-8).
    ///
    /// Backward-compatible extension points, e.g. `"tasks/runWorkflow"`,
    /// `"streaming-sse"`, `"input-required-resume"`. Unknown entries are
    /// ignored by clients that do not understand them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

impl AgentCard {
    /// Create a new agent card.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            url: url.into(),
            skills: Vec::new(),
            protocol_version: default_protocol_version(),
            security_schemes: None,
            interfaces: None,
            supported_interfaces: Vec::new(),
            provider: None,
            documentation_url: None,
            authentication: None,
            default_input_modes: default_input_modes(),
            default_output_modes: default_output_modes(),
            signature: None,
            data_class: None,
            jurisdiction: None,
            capabilities: Vec::new(),
        }
    }

    /// Add a skill.
    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Set the A2A protocol version this agent supports.
    pub fn with_protocol_version(mut self, version: impl Into<String>) -> Self {
        self.protocol_version = version.into();
        self
    }

    /// Set the security schemes advertised on the card.
    pub fn with_security_schemes(mut self, schemes: Value) -> Self {
        self.security_schemes = Some(schemes);
        self
    }

    /// Set the interfaces advertised on the card.
    pub fn with_interfaces(mut self, interfaces: Value) -> Self {
        self.interfaces = Some(interfaces);
        self
    }

    /// Advertises a typed v1.0.1 interface.
    pub fn with_supported_interface(mut self, interface: AgentInterface) -> Self {
        self.supported_interfaces.push(interface);
        self
    }

    /// v1.0.1 negotiation: picks the protocol version of the first supported
    /// interface whose transport matches `transport` and whose
    /// `protocol_version` is in `client_versions` (order = card priority).
    /// Errors when nothing matches.
    pub fn negotiate(
        &self,
        transport: A2ATransport,
        client_versions: &[&str],
    ) -> Result<AgentInterface, String> {
        self.supported_interfaces
            .iter()
            .find(|i| {
                i.transport == transport && client_versions.contains(&i.protocol_version.as_str())
            })
            .cloned()
            .ok_or_else(|| {
                format!(
                    "no mutually supported interface: transport={transport:?} client_versions={client_versions:?}"
                )
            })
    }

    /// Whether the card is v1.0.1-shaped (has at least one typed interface).
    pub fn is_v101(&self) -> bool {
        !self.supported_interfaces.is_empty()
    }

    /// Set the provider/organization name.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set the documentation URL.
    pub fn with_documentation_url(mut self, url: impl Into<String>) -> Self {
        self.documentation_url = Some(url.into());
        self
    }

    /// Set the authentication schemes.
    pub fn with_authentication(mut self, schemes: Vec<String>) -> Self {
        self.authentication = Some(schemes);
        self
    }

    /// Set the digital signature over the card content (P1-3).
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Set the data classification of this agent (P2-8).
    pub fn with_data_class(mut self, class: impl Into<String>) -> Self {
        self.data_class = Some(class.into());
        self
    }

    /// Set the jurisdiction(s) this agent operates under (P2-8).
    pub fn with_jurisdiction(mut self, jurisdiction: impl Into<String>) -> Self {
        self.jurisdiction = Some(jurisdiction.into());
        self
    }

    /// Advertise an optional protocol capability (P2-8).
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }
}
