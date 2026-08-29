// lc-tools/src/ssrf.rs
//! SSRF protection — the single shared implementation lives in `lc-core` (see
//! [`lc_core::ssrf`]). This module re-exports it so [`crate::url_fetch`] and
//! [`crate::extended::http`] keep a crate-local entry point. Any rule evolution
//! (CGNAT ranges, new IPv6 special addresses, ...) happens only in `lc-core`.

pub(crate) use lc_core::ssrf::{guarded_get, url_points_to_private_ip};

#[cfg(test)]
pub(crate) use lc_core::ssrf::is_private_ip;
