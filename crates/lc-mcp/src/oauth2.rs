//! OAuth 2.1 support for the Streamable HTTP track (B1, 0.22.4).
//!
//! Remote MCP servers protect their Streamable HTTP endpoint as an OAuth 2.1
//! **resource server** (RFC 9728): an unauthenticated request gets
//! `401 Unauthorized` with a `WWW-Authenticate: Bearer resource_metadata="…"`
//! challenge, the client discovers the authorization server via protected
//! resource metadata, runs an authorization-code + PKCE (often DCR) flow in a
//! browser, and then sends `Authorization: Bearer <token>` on every POST.
//!
//! This module provides the machine-to-machine building blocks; the part that
//! needs a human (opening the authorization URL, receiving the redirect) stays
//! with the embedding application:
//!
//! - [`BearerTokenProvider`] — per-request token injection with one
//!   invalidate-and-retry cycle on 401 (token refresh seam);
//! - [`StaticBearerToken`] — the trivial fixed-token implementation;
//! - [`OAuthChallenge`] — parsed `WWW-Authenticate` challenge (resource
//!   metadata URL, realm, scopes);
//! - [`discover_protected_resource`] / [`discover_authorization_server`] —
//!   RFC 9728 / RFC 8414 metadata discovery;
//! - [`OAuthTokenClient`] — token-endpoint exchange helpers
//!   (authorization code + PKCE verifier, refresh token, client credentials).
//!
//! No new crypto dependency is pulled in: the PKCE `code_challenge` is
//! computed by the caller together with the verifier it keeps; only the
//! verifier reaches the token exchange.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::protocol::{MCPError, MCP_ERROR_UNAUTHORIZED};
use crate::sandbox::EgressPolicy;

/// Timeout for metadata discovery and token-endpoint calls.
const OAUTH_HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// Supplies the bearer token attached to outgoing Streamable HTTP requests.
///
/// Implement for static config, an in-memory token cache, or a full
/// refresh-token manager. The transport calls [`BearerTokenProvider::token`]
/// before each POST and, on a 401,
/// [`BearerTokenProvider::invalidate`] exactly once before retrying with a
/// fresh token; a second 401 surfaces to the caller.
#[async_trait]
pub trait BearerTokenProvider: Send + Sync {
    /// Returns the current access token (without the `Bearer ` prefix).
    async fn token(&self) -> Result<String, MCPError>;

    /// Reports that `token` was rejected by the server (401); the next
    /// [`BearerTokenProvider::token`] call should rotate/refresh. Default no-op
    /// for providers without refresh semantics.
    async fn invalidate(&self, _token: &str) {}
}

/// A fixed bearer token (tests, dev sidecars, long-lived PATs).
#[derive(Debug, Clone)]
pub struct StaticBearerToken(pub String);

#[async_trait]
impl BearerTokenProvider for StaticBearerToken {
    async fn token(&self) -> Result<String, MCPError> {
        Ok(self.0.clone())
    }
}

/// Parsed OAuth challenge from a `WWW-Authenticate` response header.
///
/// Example: `Bearer realm="MCP",resource_metadata="https://host/.well-known/oauth-protected-resource"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OAuthChallenge {
    /// Auth scheme token (normally `"Bearer"`).
    pub scheme: String,
    /// `realm` parameter, if present.
    pub realm: Option<String>,
    /// Full URL of the RFC 9728 protected-resource metadata document.
    #[serde(rename = "resource_metadata")]
    pub resource_metadata: Option<String>,
    /// `scope` parameter split on whitespace, if present.
    pub scopes: Option<Vec<String>>,
    /// The raw header value (diagnostics).
    pub raw: String,
}

impl OAuthChallenge {
    /// Extracts the first challenge for `scheme` (case-insensitive), or the
    /// first challenge of any scheme when `scheme` is `None`.
    ///
    /// RFC 9110 separates both multiple challenges and the auth-params within
    /// one challenge with commas, so a plain split is ambiguous. The
    /// disambiguating rule used here: after a parsed `key=value` pair, a
    /// following `<token>=` is another param, while a bare `<token>` (no `=`
    /// before the next comma) starts a new challenge. Quoted-string values are
    /// skipped while scanning, so commas/equals inside quotes never count.
    pub fn parse(header: &str, want_scheme: Option<&str>) -> Option<OAuthChallenge> {
        parse_all_challenges(header)
            .into_iter()
            .find(|c| want_scheme.is_none_or(|want| c.scheme.eq_ignore_ascii_case(want)))
    }

    /// Restores a challenge from the `data` member of the -32001 error the
    /// transport returns on 401.
    pub fn from_error(error: &MCPError) -> Option<Self> {
        let data = error.data.as_ref()?;
        serde_json::from_value(data.clone()).ok()
    }
}

/// One parsed challenge.
struct RawChallenge {
    scheme: String,
    params: Vec<(String, String)>,
    raw: String,
}

/// Cursor-parses every challenge in a `WWW-Authenticate` header.
fn parse_all_challenges(header: &str) -> Vec<OAuthChallenge> {
    let bytes = header.as_bytes();
    let mut challenges = Vec::new();
    let mut i = skip_ws_and_commas(bytes, 0);
    while i < bytes.len() {
        let start = i;
        // Scheme token up to the first whitespace or comma (RFC 9110 `token`).
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b',' {
            i += 1;
        }
        let scheme = header[start..i].to_string();
        let mut raw_end = i;
        i = skip_ws(bytes, i);

        // Auth params until a bare token signals the next challenge.
        let mut params = Vec::new();
        loop {
            if i >= bytes.len() {
                break;
            }
            let key_start = i;
            // A param key ends at '='; a bare token (token68 / next scheme)
            // ends at ',' or the quote of a malformed value.
            while i < bytes.len() && bytes[i] != b'=' && bytes[i] != b',' && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'=' {
                if params.is_empty() {
                    // Trailing token68 belongs to this challenge's raw span.
                    raw_end = i;
                }
                i = skip_ws_and_commas(bytes, key_start);
                break;
            }
            let key = header[key_start..i].trim().to_string();
            i += 1; // consume '='
            let (value, next) = parse_value(bytes, i);
            i = next;
            if !key.is_empty() {
                params.push((key, value));
            }
            raw_end = i;
            i = skip_ws(bytes, i);
            if i >= bytes.len() || bytes[i] != b',' {
                break;
            }
            // Look past the comma: another `key=` stays in this challenge; a
            // bare token (or quoted token) starts the next challenge.
            let after = skip_ws(bytes, i + 1);
            let mut probe = after;
            while probe < bytes.len()
                && bytes[probe] != b'='
                && bytes[probe] != b','
                && bytes[probe] != b'"'
            {
                probe += 1;
            }
            if probe < bytes.len() && bytes[probe] == b'=' {
                i = skip_ws_and_commas(bytes, i);
            } else {
                i = after;
                break;
            }
        }

        challenges.push(RawChallenge {
            raw: header[start..raw_end].trim().to_string(),
            scheme,
            params,
        });
        i = skip_ws_and_commas(bytes, i);
    }
    challenges
        .into_iter()
        .map(|raw| {
            let get = |k: &str| -> Option<String> {
                raw.params
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(k))
                    .map(|(_, v)| v.clone())
            };
            OAuthChallenge {
                scheme: raw.scheme,
                realm: get("realm"),
                resource_metadata: get("resource_metadata"),
                scopes: get("scope").map(|s| s.split_whitespace().map(str::to_string).collect()),
                raw: raw.raw,
            }
        })
        .collect()
}

/// Parses one auth-param value (quoted-string with backslash escapes, or a
/// bare token), returning `(value, index-after-value)`.
fn parse_value(bytes: &[u8], start: usize) -> (String, usize) {
    if start < bytes.len() && bytes[start] == b'"' {
        let mut i = start + 1;
        let mut value = String::new();
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 1; // drop the backslash, keep the escaped char verbatim
            }
            value.push(bytes[i] as char);
            i += 1;
        }
        if i < bytes.len() {
            i += 1; // closing quote
        }
        (value, i)
    } else {
        let mut i = start;
        while i < bytes.len() && bytes[i] != b',' {
            i += 1;
        }
        (
            String::from_utf8_lossy(&bytes[start..i]).trim().to_string(),
            i,
        )
    }
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn skip_ws_and_commas(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
        i += 1;
    }
    i
}

/// RFC 9728 protected-resource metadata (the fields an MCP client uses).
#[derive(Debug, Clone, Deserialize)]
pub struct ProtectedResourceMetadata {
    /// Canonical resource identifier (the MCP endpoint URL).
    pub resource: String,
    /// Issuer identifiers of authorization servers acceptable to this resource.
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    /// Scopes the resource understands.
    #[serde(default, rename = "scopes_supported")]
    pub scopes_supported: Vec<String>,
    /// How the bearer token is accepted (RFC 9728: `"header"` / `"body"`).
    #[serde(default, rename = "bearer_methods_supported")]
    pub bearer_methods_supported: Vec<String>,
}

/// RFC 8414 OAuth 2.0 authorization-server metadata (OAuth 2.1 profile subset).
#[derive(Debug, Clone, Deserialize)]
pub struct AuthorizationServerMetadata {
    /// Issuer identifier; must match the URL the metadata was fetched from.
    pub issuer: String,
    /// Browser navigation target for the authorization request.
    pub authorization_endpoint: Option<String>,
    /// Token exchange endpoint used by [`OAuthTokenClient`].
    pub token_endpoint: Option<String>,
    /// Dynamic-client-registration endpoint (RFC 7591), when DCR is offered.
    pub registration_endpoint: Option<String>,
    /// e.g. `authorization_code`, `refresh_token`, `client_credentials`.
    #[serde(default, rename = "grant_types_supported")]
    pub grant_types_supported: Vec<String>,
    /// PKCE methods, normally `["S256"]` under OAuth 2.1.
    #[serde(default, rename = "code_challenge_methods_supported")]
    pub code_challenge_methods_supported: Vec<String>,
}

/// How strictly OAuth discovery validates metadata/token endpoint URLs (B8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryMode {
    /// Production default: endpoints must be `https`; `http` (including on
    /// loopback) is rejected. A `401 → metadata_url` downgrade would otherwise
    /// hand credentials to a plaintext channel.
    Public,
    /// Explicit developer opt-in: allows `http` on loopback hosts only
    /// (`127.0.0.1`, `[::1]`, `localhost`) for local fixture servers. Any
    /// non-loopback URL is still required to be `https`.
    Dev,
}

/// `true` when `host` is a loopback host (`localhost`, `127.0.0.1`, `[::1]`,
/// `[::ffff:127.0.0.1]`), case-insensitive.
fn is_loopback_host(host: Option<&str>) -> bool {
    match host {
        Some(h) => {
            let lower = h.trim_matches(['[', ']']).to_ascii_lowercase();
            lower == "localhost"
                || lower == "localhost."
                || lower == "127.0.0.1"
                || lower == "::1"
                || lower == "::ffff:127.0.0.1"
        }
        None => false,
    }
}

/// Validates a metadata/token endpoint URL before any request is sent (B8).
/// HTTPS is mandatory (the OAuth credential channel); loopback HTTP requires
/// an explicit [`DiscoveryMode::Dev`]. SSRF and redirect handling are enforced
/// separately at fetch time.
fn validate_discovery_url(url: &str, mode: DiscoveryMode) -> Result<reqwest::Url, MCPError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| MCPError::new(-32000, format!("invalid metadata URL: {e}")))?;
    let scheme = parsed.scheme();
    let loopback = is_loopback_host(parsed.host_str());
    let dev_http = scheme == "http" && loopback && mode == DiscoveryMode::Dev;
    if scheme != "https" && !dev_http {
        return Err(MCPError::new(
            -32000,
            format!(
                "metadata endpoint must use HTTPS (got scheme '{scheme}'{}); \
                 loopback HTTP requires explicit DiscoveryMode::Dev",
                if loopback { " on a loopback host" } else { "" }
            ),
        ));
    }
    Ok(parsed)
}

/// Rejects a redirect that leaves the originally-validated origin (B8): a
/// metadata server may `302` to a sibling path, but never to a different
/// scheme, host, or port — that is the classic SSRF/user-revocation pivot
/// (https→http downgrade on the same host included). SSRF per hop is already
/// handled by `lc_core::ssrf::guarded_get`.
fn ensure_same_host(original: &reqwest::Url, actual: &reqwest::Url) -> Result<(), MCPError> {
    let a_host = original.host_str().unwrap_or_default().to_ascii_lowercase();
    let a_scheme = original.scheme();
    let a_port = original.port_or_known_default();
    let b_host = actual.host_str().unwrap_or_default().to_ascii_lowercase();
    let b_scheme = actual.scheme();
    let b_port = actual.port_or_known_default();
    if a_host != b_host || a_scheme != b_scheme || a_port != b_port {
        return Err(MCPError::new(
            -32000,
            format!(
                "metadata discovery rejected cross-host redirect: \
                 {a_scheme}://{a_host}{} -> {b_scheme}://{b_host}{}",
                a_port.map(|p| format!(":{p}")).unwrap_or_default(),
                b_port.map(|p| format!(":{p}")).unwrap_or_default(),
            ),
        ));
    }
    Ok(())
}

/// Full discovery settings: URL strictness plus an optional egress allowlist
/// (B8).
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// URL strictness ([`DiscoveryMode::Public`] by default).
    pub mode: DiscoveryMode,
    /// Optional egress allowlist applied to the fetched metadata host. An empty
    /// policy blocks all egress; `None` skips the egress check.
    pub egress: Option<Arc<EgressPolicy>>,
}

impl DiscoveryConfig {
    /// Builds a config with a mode and no egress restriction.
    pub fn new(mode: DiscoveryMode) -> Self {
        Self { mode, egress: None }
    }

    /// Adds an egress allowlist (empty policy rejects all outbound).
    pub fn with_egress(mut self, policy: Arc<EgressPolicy>) -> Self {
        self.egress = Some(policy);
        self
    }
}

/// Fetches and parses an RFC 9728 protected-resource metadata document under
/// production rules ([`DiscoveryMode::Public`]: https only).
///
/// `metadata_url` is typically the `resource_metadata` value from the
/// [`OAuthChallenge`] (already a complete URL).
pub async fn discover_protected_resource(
    metadata_url: &str,
) -> Result<ProtectedResourceMetadata, MCPError> {
    discover_protected_resource_with(metadata_url, DiscoveryMode::Public).await
}

/// [`discover_protected_resource`] with an explicit [`DiscoveryMode`].
///
/// Fetched through [`lc_core::ssrf::guarded_get`] (per-hop SSRF + IP pinning),
/// then a same-host check rejects any cross-host redirect (B8).
pub async fn discover_protected_resource_with(
    metadata_url: &str,
    mode: DiscoveryMode,
) -> Result<ProtectedResourceMetadata, MCPError> {
    discover_protected_resource_config(metadata_url, DiscoveryConfig::new(mode)).await
}

/// [`discover_protected_resource_with`] with full [`DiscoveryConfig`] (mode +
/// optional egress allowlist).
pub async fn discover_protected_resource_config(
    metadata_url: &str,
    config: DiscoveryConfig,
) -> Result<ProtectedResourceMetadata, MCPError> {
    let validated = validate_discovery_url(metadata_url, config.mode)?;
    let resp = fetch_metadata(
        metadata_url,
        &validated,
        config.mode,
        config.egress.as_deref(),
        "resource metadata",
    )
    .await?;
    resp.json::<ProtectedResourceMetadata>()
        .await
        .map_err(|e| MCPError::new(-32700, format!("invalid protected-resource metadata: {e}")))
}

/// Fetches and parses an RFC 8414 authorization-server metadata document for
/// `issuer` (path `/.well-known/oauth-authorization-server` under the issuer)
/// under production rules ([`DiscoveryMode::Public`]).
pub async fn discover_authorization_server(
    issuer: &str,
) -> Result<AuthorizationServerMetadata, MCPError> {
    discover_authorization_server_with(issuer, DiscoveryMode::Public).await
}

/// [`discover_authorization_server`] with an explicit [`DiscoveryMode`].
///
/// In addition to https + same-host checks, the returned document's `issuer`
/// must equal the requested issuer (RFC 8414 §4.1) — mismatch is rejected
/// (B8), so a hijacked/wrong authorization server cannot impersonate the
/// intended one.
pub async fn discover_authorization_server_with(
    issuer: &str,
    mode: DiscoveryMode,
) -> Result<AuthorizationServerMetadata, MCPError> {
    discover_authorization_server_config(issuer, DiscoveryConfig::new(mode)).await
}

/// [`discover_authorization_server_with`] with full [`DiscoveryConfig`].
pub async fn discover_authorization_server_config(
    issuer: &str,
    config: DiscoveryConfig,
) -> Result<AuthorizationServerMetadata, MCPError> {
    let base = issuer.trim_end_matches('/');
    let url = format!("{base}/.well-known/oauth-authorization-server");
    let validated = validate_discovery_url(&url, config.mode)?;
    let resp = fetch_metadata(
        &url,
        &validated,
        config.mode,
        config.egress.as_deref(),
        "authorization-server metadata",
    )
    .await?;
    let meta: AuthorizationServerMetadata = resp.json().await.map_err(|e| {
        MCPError::new(
            -32700,
            format!("invalid authorization-server metadata: {e}"),
        )
    })?;
    if meta.issuer.trim_end_matches('/') != issuer.trim_end_matches('/') {
        return Err(MCPError::new(
            -32000,
            format!(
                "authorization-server issuer mismatch: requested {issuer:?}, \
                 metadata advertises {:?}",
                meta.issuer
            ),
        ));
    }
    Ok(meta)
}

/// Fetch a metadata document through `lc_core::ssrf` so every redirect hop is
/// re-validated (SSRF + IP pinning), then enforce the same-host rule and any
/// configured egress allowlist.
///
/// `egress`: an empty policy rejects **all** egress (fail-closed); a configured
/// policy allows only allowlisted hosts. `None` skips the egress check.
async fn fetch_metadata(
    url: &str,
    validated: &reqwest::Url,
    mode: DiscoveryMode,
    egress: Option<&EgressPolicy>,
    what: &str,
) -> Result<reqwest::Response, MCPError> {
    let host = validated
        .host_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if let Some(policy) = egress {
        if !policy.allows(&host) {
            return Err(MCPError::new(
                -32000,
                format!(
                    "{what} egress blocked: host '{host}' is not in the EgressPolicy allowlist"
                ),
            ));
        }
    }
    // check_ssrf=true under `Public` (rejects intranet). Under `Dev` the URL is
    // already a validated loopback host; per-hop SSRF is off but the same-host
    // check below still bounds the fetch to that one host.
    let check_ssrf = mode != DiscoveryMode::Dev;
    let resp = lc_core::ssrf::guarded_get(url, check_ssrf, Some(OAUTH_HTTP_TIMEOUT))
        .await
        .map_err(|e| MCPError::new(-32000, format!("{what} request failed: {e}")))?;
    // `guarded_get` stops following once non-redirect; reject any redirect that
    // escaped to a different host.
    ensure_same_host(validated, resp.url())?;
    if !resp.status().is_success() {
        return Err(MCPError::new(
            -32000,
            format!("{what} request failed: HTTP {}", resp.status()),
        ));
    }
    Ok(resp)
}

/// Successful token-endpoint response (RFC 6749 §5.1, subset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthTokenResponse {
    /// Access token to send as `Bearer`.
    pub access_token: String,
    /// Token type (OAuth 2.1 requires `"Bearer"`).
    #[serde(default)]
    pub token_type: String,
    /// Lifetime in seconds.
    pub expires_in: Option<u64>,
    /// Refresh token for renewing access without user interaction.
    pub refresh_token: Option<String>,
    /// Granted scope (may differ from the requested scope).
    pub scope: Option<String>,
}

/// Minimal OAuth 2.1 token-endpoint client.
///
/// Client credentials are sent in the form body, which works for both
/// confidential clients and public MCP clients obtained via DCR.
#[derive(Debug, Clone)]
pub struct OAuthTokenClient {
    endpoint: String,
    client_id: String,
    client_secret: Option<String>,
    mode: DiscoveryMode,
    egress: Option<Arc<EgressPolicy>>,
}

impl OAuthTokenClient {
    /// Creates a client for a token endpoint and client id, under production
    /// URL rules ([`DiscoveryMode::Public`]). The token endpoint is where the
    /// client presents its `client_secret`, so the channel must be https unless
    /// an explicit [`DiscoveryMode::Dev`] opts into loopback http.
    pub fn new(endpoint: impl Into<String>, client_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client_id: client_id.into(),
            client_secret: None,
            mode: DiscoveryMode::Public,
            egress: None,
        }
    }

    /// Sets the client secret (confidential clients).
    pub fn with_client_secret(mut self, secret: impl Into<String>) -> Self {
        self.client_secret = Some(secret.into());
        self
    }

    /// Sets the discovery URL strictness for the token endpoint (B8).
    /// Use [`DiscoveryMode::Dev`] for local fixture servers; production stays
    /// https-only by default.
    pub fn with_discovery_mode(mut self, mode: DiscoveryMode) -> Self {
        self.mode = mode;
        self
    }

    /// Requires the token-endpoint host to be in `policy` (an empty policy
    /// blocks all egress). B8: the token endpoint is attacker-adjacent (it is
    /// drawn from server metadata), so an allowlist keeps credentials on
    /// approved hosts.
    pub fn with_egress(mut self, policy: Arc<EgressPolicy>) -> Self {
        self.egress = Some(policy);
        self
    }

    /// Exchanges an authorization code (with the PKCE `code_verifier`) for
    /// tokens. `redirect_uri` must match the one used at the authorization
    /// endpoint.
    pub async fn exchange_authorization_code(
        &self,
        code: &str,
        redirect_uri: &str,
        code_verifier: Option<&str>,
        scope: Option<&str>,
    ) -> Result<OAuthTokenResponse, MCPError> {
        let mut form: Vec<(String, String)> = vec![
            ("grant_type".into(), "authorization_code".into()),
            ("code".into(), code.into()),
            ("redirect_uri".into(), redirect_uri.into()),
        ];
        if let Some(verifier) = code_verifier {
            form.push(("code_verifier".into(), verifier.into()));
        }
        if let Some(scope) = scope {
            form.push(("scope".into(), scope.into()));
        }
        self.token_request(form).await
    }

    /// Refreshes an access token.
    pub async fn refresh(
        &self,
        refresh_token: &str,
        scope: Option<&str>,
    ) -> Result<OAuthTokenResponse, MCPError> {
        let mut form: Vec<(String, String)> = vec![
            ("grant_type".into(), "refresh_token".into()),
            ("refresh_token".into(), refresh_token.into()),
        ];
        if let Some(scope) = scope {
            form.push(("scope".into(), scope.into()));
        }
        self.token_request(form).await
    }

    /// Client-credentials grant (machine-to-machine; no resource owner).
    pub async fn client_credentials(
        &self,
        scope: Option<&str>,
    ) -> Result<OAuthTokenResponse, MCPError> {
        let mut form: Vec<(String, String)> =
            vec![("grant_type".into(), "client_credentials".into())];
        if let Some(scope) = scope {
            form.push(("scope".into(), scope.into()));
        }
        self.token_request(form).await
    }

    async fn token_request(
        &self,
        mut form: Vec<(String, String)>,
    ) -> Result<OAuthTokenResponse, MCPError> {
        form.push(("client_id".into(), self.client_id.clone()));
        if let Some(secret) = &self.client_secret {
            form.push(("client_secret".into(), secret.clone()));
        }
        let endpoint_url = validate_discovery_url(&self.endpoint, self.mode)?;
        if let Some(policy) = &self.egress {
            let host = endpoint_url
                .host_str()
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !policy.allows(&host) {
                return Err(MCPError::new(
                    -32000,
                    format!(
                        "token endpoint egress blocked: host '{host}' is not in the EgressPolicy allowlist"
                    ),
                ));
            }
        }
        // F3: send the client secret through the guarded, resolve-once /
        // IP-pinned POST so a `token_endpoint` that resolves into an intranet
        // host is rejected *before* the secret leaves the process. Under
        // DiscoveryMode::Dev the loopback fixture endpoint is permitted; under
        // Public any private/internal address is blocked (SSRF).
        let check_ssrf = self.mode != DiscoveryMode::Dev;
        let resp = lc_core::ssrf::guarded_post_form(
            &self.endpoint,
            &form,
            check_ssrf,
            Some(OAUTH_HTTP_TIMEOUT),
        )
        .await
        .map_err(|e| MCPError::new(-32000, format!("token request failed: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| MCPError::new(-32000, format!("token response read failed: {e}")))?;
        if !status.is_success() {
            return Err(MCPError::new(
                MCP_ERROR_UNAUTHORIZED,
                format!("token endpoint returned HTTP {status}: {body}"),
            ));
        }
        serde_json::from_str::<OAuthTokenResponse>(&body)
            .map_err(|e| MCPError::new(-32700, format!("invalid token response: {e}: {body}")))
    }
}

/// Wraps a token provider in `Arc` (transport constructor convenience).
pub fn shared_provider<P>(provider: P) -> Arc<dyn BearerTokenProvider>
where
    P: BearerTokenProvider + 'static,
{
    Arc::new(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_bearer_challenge() {
        let header = r#"Bearer realm="MCP server",resource_metadata="https://host.test/.well-known/oauth-protected-resource",scope="mcp.read mcp.write""#;
        let challenge = OAuthChallenge::parse(header, Some("Bearer")).expect("challenge");
        assert_eq!(challenge.scheme, "Bearer");
        assert_eq!(challenge.realm.as_deref(), Some("MCP server"));
        assert_eq!(
            challenge.resource_metadata.as_deref(),
            Some("https://host.test/.well-known/oauth-protected-resource")
        );
        assert_eq!(
            challenge.scopes,
            Some(vec!["mcp.read".to_string(), "mcp.write".to_string()])
        );
    }

    #[test]
    fn parse_picks_bearer_among_multiple_challenges() {
        // Commas inside quoted values must not split the challenge.
        let header = r#"Basic realm="legacy, with comma",Negotiate, Bearer resource_metadata="https://rs/x""#;
        let challenge = OAuthChallenge::parse(header, Some("Bearer")).expect("bearer challenge");
        assert_eq!(challenge.resource_metadata.as_deref(), Some("https://rs/x"));
    }

    #[test]
    fn parse_all_challenges_keeps_bare_scheme_in_the_middle() {
        // Three challenges; `Negotiate` carries no params and sits between a
        // Basic challenge whose quoted realm itself contains a comma.
        let header = r#"Basic realm="legacy, with comma",Negotiate,Bearer realm="mcp",resource_metadata="http://127.0.0.1:9/x""#;
        let all = parse_all_challenges(header);
        assert_eq!(all.len(), 3, "expected three challenges");
        assert_eq!(all[0].scheme, "Basic");
        assert_eq!(all[0].realm.as_deref(), Some("legacy, with comma"));
        assert_eq!(all[1].scheme, "Negotiate");
        assert!(all[1].realm.is_none() && all[1].resource_metadata.is_none());
        assert_eq!(all[2].scheme, "Bearer");
        assert_eq!(all[2].realm.as_deref(), Some("mcp"));
        assert_eq!(
            all[2].resource_metadata.as_deref(),
            Some("http://127.0.0.1:9/x")
        );

        // The exact comma-joined-without-spaces shape the fixture emits.
        let compact = r#"Bearer realm="echo-streamable",resource_metadata="http://127.0.0.1:8254/.well-known/oauth-protected-resource",scope="mcp.read""#;
        let bearer = OAuthChallenge::parse(compact, Some("Bearer")).unwrap();
        assert_eq!(bearer.realm.as_deref(), Some("echo-streamable"));
        assert_eq!(
            bearer.resource_metadata.as_deref(),
            Some("http://127.0.0.1:8254/.well-known/oauth-protected-resource")
        );
        assert_eq!(bearer.scopes, Some(vec!["mcp.read".to_string()]));
        assert_eq!(bearer.raw, compact);
    }

    #[test]
    fn parse_bare_token68_challenge() {
        // RFC 9110 token68 form: no metadata, but scheme is still recognized.
        let challenge =
            OAuthChallenge::parse("Bearer abc123==", Some("Bearer")).expect("challenge");
        assert_eq!(challenge.scheme, "Bearer");
        assert!(challenge.resource_metadata.is_none());
        assert!(challenge.realm.is_none());
    }

    #[test]
    fn parse_returns_none_for_scheme_mismatch() {
        assert!(OAuthChallenge::parse("Basic realm=x", Some("Bearer")).is_none());
        assert!(OAuthChallenge::parse("", Some("Bearer")).is_none());
    }

    #[test]
    fn parse_escaped_quote_in_param_value() {
        let header = r#"Bearer realm="a\"b""#;
        let challenge = OAuthChallenge::parse(header, Some("Bearer")).unwrap();
        assert_eq!(challenge.realm.as_deref(), Some(r#"a"b"#));
    }

    #[test]
    fn token_response_requires_access_token_only() {
        let parsed: OAuthTokenResponse =
            serde_json::from_str(r#"{"access_token":"a","token_type":"Bearer"}"#).unwrap();
        assert_eq!(parsed.access_token, "a");
        assert_eq!(parsed.token_type, "Bearer");
        assert!(parsed.expires_in.is_none());
    }

    #[tokio::test]
    async fn static_provider_token_roundtrip() {
        let provider = StaticBearerToken("abc".into());
        assert_eq!(provider.token().await.unwrap(), "abc");
        provider.invalidate("abc").await; // default no-op must not panic
    }

    // -- B8 discovery hardening ----------------------------------------------

    #[test]
    fn discovery_url_requires_https_or_explicit_dev_loopback() {
        // https is always acceptable.
        assert!(validate_discovery_url(
            "https://example.com/.well-known/mcp",
            DiscoveryMode::Public
        )
        .is_ok());

        // Plain http on a non-loopback host is rejected in both modes.
        assert!(validate_discovery_url("http://example.com/x", DiscoveryMode::Public).is_err());
        assert!(validate_discovery_url("http://example.com/x", DiscoveryMode::Dev).is_err());

        // Loopback http is rejected in production, allowed only under Dev.
        assert!(validate_discovery_url("http://127.0.0.1:8254/x", DiscoveryMode::Public).is_err());
        assert!(validate_discovery_url("http://127.0.0.1:8254/x", DiscoveryMode::Dev).is_ok());
        assert!(validate_discovery_url("http://[::1]:8254/x", DiscoveryMode::Dev).is_ok());
        assert!(validate_discovery_url("http://localhost:9/x", DiscoveryMode::Dev).is_ok());

        // Garbage is not a URL.
        assert!(validate_discovery_url("not a url", DiscoveryMode::Public).is_err());
    }

    #[test]
    fn same_host_rejects_cross_host_redirect() {
        let a = reqwest::Url::parse("https://issuer.example/as/meta").unwrap();

        // Same host (case-insensitive), any path → allowed.
        let same = reqwest::Url::parse("https://issuer.example/as/.well-known").unwrap();
        assert!(ensure_same_host(&a, &same).is_ok());
        let same_case = reqwest::Url::parse("https://ISSUER.example/x").unwrap();
        assert!(ensure_same_host(&a, &same_case).is_ok());

        // Different host → rejected (the SSRF / wrong-issuer pivot).
        let cross = reqwest::Url::parse("https://evil.example/meta").unwrap();
        assert!(ensure_same_host(&a, &cross).is_err());
    }

    #[tokio::test]
    async fn discovery_rejects_non_https_before_network() {
        let err =
            discover_protected_resource("http://example.com/.well-known/oauth-protected-resource")
                .await
                .unwrap_err();
        assert!(err.to_string().contains("HTTPS"), "{err}");
    }

    #[tokio::test]
    async fn discovery_egress_empty_policy_blocks_before_network() {
        let config =
            DiscoveryConfig::new(DiscoveryMode::Public).with_egress(Arc::new(EgressPolicy::new()));
        let err = discover_protected_resource_config(
            "https://example.com/.well-known/oauth-protected-resource",
            config,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("egress"), "{err}");
    }

    #[tokio::test]
    async fn token_client_egress_empty_policy_blocks_before_network() {
        let client = OAuthTokenClient::new("https://token.example/token", "client-1")
            .with_egress(Arc::new(EgressPolicy::new()));
        let err = client.client_credentials(None).await.unwrap_err();
        assert!(err.to_string().contains("egress"), "{err}");
    }

    #[tokio::test]
    async fn discovery_rejects_cross_host_redirect() {
        // The server 302s from `http://127.0.0.1:PORT/origin` to
        // `http://127.0.0.2:PORT/final`. `127.0.0.2` is the standard loopback
        // alias: a *different host string* than `127.0.0.1` (so the same-host
        // guard fires), yet plain IPv4 loopback on every platform — unlike a
        // `localhost` target whose `::1`/`127.0.0.1` resolution order can fail
        // to connect in some environments *before* any response is returned,
        // which would skip the same-host guard entirely.
        let (origin_url, handle) = {
            use socket2::{Domain, Protocol, Socket as Sock, Type};
            // Bind a dual-stack (`::` / v6only=false) listener so the redirect
            // hop (127.0.0.2, v4-mapped) reaches this server.
            let sock = Sock::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP)).unwrap();
            sock.set_only_v6(false).unwrap();
            sock.set_reuse_address(true).unwrap();
            let addr: std::net::SocketAddr = "[::]:0".parse().unwrap();
            sock.bind(&addr.into()).unwrap();
            sock.listen(128).unwrap();
            let std_listener: std::net::TcpListener = sock.into();
            std_listener.set_nonblocking(true).unwrap();
            let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
            let port = listener.local_addr().unwrap().port();
            let local_final = format!("http://127.0.0.2:{port}/final");
            let task = tokio::spawn(async move {
                loop {
                    let (mut sock, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => break,
                    };
                    let final_url = local_final.clone();
                    tokio::spawn(async move {
                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                        let mut buf = [0u8; 2048];
                        let n = match sock.read(&mut buf).await {
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        let head_str = String::from_utf8_lossy(&buf[..n]).to_string();
                        let path = head_str
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("/")
                            .to_string();
                        eprintln!("XHOST-REQ path={path:?} head={head_str:?}");
                        let (status, body) = if path == "/origin" {
                            (302, String::new())
                        } else {
                            (
                                200,
                                r#"{"resource":"https://issuer.example/mcp"}"#.to_string(),
                            )
                        };
                        // Note the trailing blank line (`\r\n\r\n`) — without it hyper
                        // reports `IncompleteMessage` (the terminating empty line that
                        // separates headers from body never arrived).
                        let head = format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Length: {len}\r\n{extra}\r\n\r\n",
                            status = status,
                            reason = if status == 200 { "OK" } else { "Found" },
                            len = body.len(),
                            extra = if path == "/origin" {
                                format!("Location: {final_url}")
                            } else {
                                "Content-Type: application/json".to_string()
                            }
                        );
                        let _ = sock.write_all(head.as_bytes()).await;
                        let _ = sock.write_all(body.as_bytes()).await;
                        let _ = sock.flush().await;
                    });
                }
            });
            (format!("http://127.0.0.1:{port}/origin"), task)
        };

        let err = discover_protected_resource_with(&origin_url, DiscoveryMode::Dev)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("cross-host"), "{err}");
        handle.abort();
    }

    #[tokio::test]
    async fn authorization_server_rejects_issuer_mismatch() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (addr, handle) = {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let local_addr = addr.clone();
            let task = tokio::spawn(async move {
                loop {
                    let (mut sock, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => break,
                    };
                    let addr = local_addr.clone();
                    tokio::spawn(async move {
                        let mut buf = [0u8; 2048];
                        if sock.read(&mut buf).await.is_err() {
                            return;
                        }
                        // Advertise a mismatched issuer.
                        let body = serde_json::json!({
                            "issuer": format!("http://{addr}/as-wrong"),
                        })
                        .to_string();
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n",
                            body.len()
                        );
                        let _ = sock.write_all(head.as_bytes()).await;
                        let _ = sock.write_all(body.as_bytes()).await;
                        let _ = sock.flush().await;
                    });
                }
            });
            (addr, task)
        };

        let issuer = format!("http://{addr}/as");
        let err = discover_authorization_server_with(&issuer, DiscoveryMode::Dev)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("issuer mismatch"), "{err}");
        handle.abort();
    }
}
