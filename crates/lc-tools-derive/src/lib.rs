#![warn(missing_docs)]
// lc-tools-derive/src/lib.rs
//! Procedural macro for deriving BaseTool implementations from functions.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_tools::{tool, BaseTool, Tool, ToolError};
//!
//! #[tool(description = "Useful for arithmetic calculations")]
//! fn calculator(
//!     #[param(desc = "The mathematical expression to evaluate")]
//!     expression: String,
//! ) -> Result<f64, ToolError> {
//!     expression
//!         .parse::<f64>()
//!         .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
//! }
//! ```
//!
//! This expands to:
//! - `CalculatorTool` struct
//! - `CalculatorInput` struct with `Deserialize` + `JsonSchema`
//! - `impl BaseTool for CalculatorTool`
//! - `impl Tool for CalculatorTool`
//! - The original `calculator` function is preserved

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Attribute, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Meta, MetaNameValue,
    Pat, PatType, Result, Signature, Type,
};

/// Attribute for individual parameters.
const PARAM_ATTR: &str = "param";

/// The `#[tool]` procedural macro.
///
/// Transforms a function into a full Tool implementation.
///
/// # Attributes
///
/// - `#[tool(description = "...")]` — Required. The tool description shown to the LLM.
/// - `#[param(desc = "...")]` — Optional per-parameter. Adds description to the JSON schema.
///
/// # Parameter Rules
///
/// - `String`, `i64`, `f64`, `bool` → required in schema
/// - `Option<T>` → optional in schema
/// - `Vec<T>` → array in schema
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Parse the attribute as `description = "..."`
    let description = match parse_tool_attr(attr) {
        Ok(d) => d,
        Err(err) => return err.to_compile_error().into(),
    };

    match tool_impl(description, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Parse `#[tool(description = "...")]` attribute tokens.
fn parse_tool_attr(attr: TokenStream) -> Result<String> {
    if attr.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[tool(description = \"...\")] is required",
        ));
    }

    // Parse as `description = "..."`
    let meta: Meta = syn::parse(attr)?;
    if let Meta::NameValue(MetaNameValue { path, value, .. }) = &meta {
        if path.is_ident("description") {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(lit), ..
            }) = value
            {
                return Ok(lit.value());
            }
        }
    }

    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "expected #[tool(description = \"...\")]",
    ))
}

fn tool_impl(description: String, mut func: ItemFn) -> Result<TokenStream2> {
    // 1. Extract all information from the function BEFORE mutating it
    let func_name_str = func.sig.ident.to_string();
    // J3:先校验 PascalCase 种子能作为合法标识符前缀,再交给 `format_ident!`。
    // 输入本是合法的 Rust fn 名(首字符必为字母/下划线),理论不会触发,但按防御性
    // 修正,让它成为可诊断的 `compile_error!` 而不是过程宏内部的 panic。
    let pascal = to_pascal_case(&func_name_str);
    if !is_valid_ident_seed(&pascal) {
        return Err(syn::Error::new_spanned(
            &func.sig.ident,
            format!(
                "cannot derive `{pascal}Tool`/`{pascal}Input` from function name `{func_name_str}`: \
                 generated identifiers must start with an alphabetic or underscore character"
            ),
        ));
    }
    let tool_struct_name = format_ident!("{}Tool", pascal);
    let input_struct_name = format_ident!("{}Input", pascal);
    let func_name = func.sig.ident.clone();

    // J7:self(`const`/`async`/`unsafe` etc.)为 async 时不支持——`invoke`/`run` 内
    // 以同步求值调用 `fn(...)`,直接展开会把 future 当同步值用而生成坏代码。给明确
    // 的「不支持」错误,而非静默展开成类型错误。
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &func.sig.ident,
            "async tool functions are not supported by #[tool]: make the function synchronous \
             (the derived Tool::invoke / BaseTool::run are already async)",
        ));
    }

    // 2. Extract parameters from function signature
    let params = extract_params(&func.sig)?;
    let field_names: Vec<Ident> = params.iter().map(|p| p.name.clone()).collect();

    // 3. Determine the output type from the function return type
    let output_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => {
            // If it's Result<T, E>, extract T; otherwise the type as-is.
            if let Some(inner) = extract_result_ok(&func.sig.output) {
                quote! { #inner }
            } else {
                quote! { #ty }
            }
        }
    };

    // 3b. F5:函数返回 `Result<_, ToolError>` 时,`invoke` 直接透传原错误
    // (参数错 / 业务错原样保留,不再统一压平成 `ExecutionFailed`);返回
    // 其他错误类型时才包 `ExecutionFailed`。此为 breaking:错误语义变化。
    let invoke_body = if return_type_is_tool_error(&func.sig.output) {
        quote! { #func_name(#(#field_names),*) }
    } else {
        quote! { #func_name(#(#field_names),*).map_err(|e| ::lc_core::tools::ToolError::ExecutionFailed(e.to_string())) }
    };

    // 4. Generate Input struct fields
    let input_fields = generate_input_fields(&params);

    // 5. Generate field-level schemars attributes for descriptions
    let input_field_attrs = generate_field_attrs(&params);

    // 6. Remove #[param] attributes from the original function so the compiler
    //    doesn't complain about unknown attributes
    strip_param_attrs(&mut func);

    // 7. Generate the full expanded code
    let expanded = quote! {
        // Preserve the original function (with #[param] attrs stripped)
        #func

        /// Auto-generated Tool struct.
        #[derive(Debug, Clone)]
        pub struct #tool_struct_name;

        impl ::std::default::Default for #tool_struct_name {
            fn default() -> Self {
                Self
            }
        }

        impl #tool_struct_name {
            pub fn new() -> Self {
                Self
            }
        }

        /// Auto-generated Input struct.
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        pub struct #input_struct_name {
            #(#input_field_attrs)*
            #(#input_fields)*
        }

        // Implement Tool trait (type-safe version)
        #[::async_trait::async_trait]
        impl ::lc_core::tools::Tool for #tool_struct_name {
            type Input = #input_struct_name;
            type Output = #output_type;

            async fn invoke(&self, input: Self::Input) -> ::std::result::Result<Self::Output, ::lc_core::tools::ToolError> {
                let #input_struct_name { #(#field_names),* } = input;
                #invoke_body
            }
        }

        // Implement BaseTool trait (string version, for Agent)
        #[::async_trait::async_trait]
        impl ::lc_core::tools::BaseTool for #tool_struct_name {
            fn name(&self) -> &str {
                #func_name_str
            }

            fn description(&self) -> &str {
                #description
            }

            async fn run(&self, input: ::std::string::String) -> ::std::result::Result<::std::string::String, ::lc_core::tools::ToolError> {
                let parsed: #input_struct_name = ::serde_json::from_str(&input)
                    .map_err(|e| ::lc_core::tools::ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;
                let #input_struct_name { #(#field_names),* } = parsed;
                let result = #func_name(#(#field_names),*)
                    .map_err(|e| ::lc_core::tools::ToolError::ExecutionFailed(e.to_string()))?;
                // F5:序列化失败不再静默回退 Debug 文本(会把 Rust 内部结构喂给 LLM),
                // 而是返回 ExecutionFailed 错误,让上层明确感知输出无法序列化。
                let serialized = ::serde_json::to_string(&result)
                    .map_err(|e| ::lc_core::tools::ToolError::ExecutionFailed(format!("Failed to serialize tool output: {}", e)))?;
                Ok(serialized)
            }

            fn args_schema(&self) -> ::std::option::Option<::serde_json::Value> {
                use ::schemars::schema_for;
                // J9:schema 序列化失败不再 `.ok()` 占位空 `None`——Input 必 derive
                // JsonSchema,schema 自描述必可序列化,真失败是内部错误,直接 panic 报出。
                Some(
                    ::serde_json::to_value(schema_for!(#input_struct_name)).expect(
                        "[lc-tools-derive] internal error: generated Input schema failed to serialize \
                         (Input must derive schemars::JsonSchema)",
                    ),
                )
            }
        }
    };

    Ok(expanded)
}

/// Parameter info extracted from function signature.
struct ParamInfo {
    name: Ident,
    ty: Type,
    desc: Option<String>,
}

/// Extract parameter info from the function signature.
fn extract_params(sig: &Signature) -> Result<Vec<ParamInfo>> {
    let mut params = Vec::new();

    for arg in &sig.inputs {
        // Skip self parameter
        if let FnArg::Receiver(_) = arg {
            continue;
        }

        if let FnArg::Typed(PatType { pat, ty, attrs, .. }) = arg {
            let name = match pat.as_ref() {
                Pat::Ident(ident) => ident.ident.clone(),
                // J8:非 ident 参数不再静默丢弃——丢弃会让生成代码静默少一个字段,用户
                // 无从得知宏为何没展开该参数,改为明确的宏错误(类型标注/tuple/wildcard
                // 等绑定模式均不支持)。
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "tool parameters must be plain identifiers \
                         (ascription / tuple / wildcard patterns are not supported)",
                    ));
                }
            };

            // Extract #[param(desc = "...")] attribute
            let desc = extract_param_desc(attrs)?;

            params.push(ParamInfo {
                name,
                ty: (*(*ty)).clone(),
                desc,
            });
        }
    }

    Ok(params)
}

/// 从 `Result<ok, err>` 提取两个泛型参数。
///
/// 唯一识别 Result 的入口:按路径 **最后一个** 段匹配 `Result`(J4),因此裸 `Result`
/// 与 `std::result::Result` 一致命中——消除旧 `extract_result_ok` 只认单段、而
/// `return_type_is_tool_error` 认末段,导致两者对 `std::result::Result` 判决不一致
/// 而产生双包装的问题。
fn result_generics(ret: &syn::ReturnType) -> Option<(Type, Type)> {
    let syn::ReturnType::Type(_, ty) = ret else {
        return None;
    };
    let Type::Path(type_path) = &**ty else {
        return None;
    };
    let seg = type_path.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut generic = args.args.iter().filter_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    });
    Some((generic.next()?.clone(), generic.next()?.clone()))
}

/// Extract `T` from `Result<T, E>`. Returns None if not a Result type.
fn extract_result_ok(ret: &syn::ReturnType) -> Option<Type> {
    result_generics(ret).map(|(ok, _)| ok)
}

/// 判断函数返回类型是否为 `Result<_, ToolError>`。F5:宏据此决定 `invoke` 是否直接
/// 透传原错误。
///
/// J4+J6:与 `extract_result_ok` 统一走 [`result_generics`] 识别 Result;错误类型不再
/// 宽泛匹配任意 `ToolError` 结尾,限定为裸 `ToolError`(len==1)或
/// `…::…::tools::ToolError` 路径(`lc_core::tools::ToolError` / `tools::ToolError`)。
/// 这样既命中真实用例(裸 `ToolError` 经 `use` 引入的常见写法),又不把 `MyToolError`、
/// `other::ToolError` 误判为库错误而透传。
fn return_type_is_tool_error(ret: &syn::ReturnType) -> bool {
    let Some((_, err)) = result_generics(ret) else {
        return false;
    };
    let Type::Path(err_path) = err else {
        return false;
    };
    let segs = err_path.path.segments;
    let Some(last) = segs.last() else {
        return false;
    };
    if last.ident != "ToolError" {
        return false;
    }
    // 裸 `ToolError`(len==1)直接放行;多段要求倒数第二段为 `tools`。
    segs.len() == 1
        || segs
            .get(segs.len().saturating_sub(2))
            .is_some_and(|s| s.ident == "tools")
}

/// J3:校验种子字符串能否作为合法标识符前缀(非空,首字符为字母或 `_`)。
fn is_valid_ident_seed(s: &str) -> bool {
    match s.chars().next() {
        Some(c) => c == '_' || c.is_ascii_alphabetic(),
        None => false,
    }
}

/// Extract `desc` from `#[param(desc = "...")]`.
///
/// J5:返回 `Result`,`#[param]` 解析/求值失败不再静默吞(原 `.ok()?`),改为向上抛
/// `syn::Error` 变成干净的 `compile_error!`,避免描述静默丢失。
fn extract_param_desc(attrs: &[Attribute]) -> Result<Option<String>> {
    for attr in attrs {
        if attr.path().is_ident(PARAM_ATTR) {
            let meta: Meta = attr.parse_args().map_err(|e| {
                syn::Error::new_spanned(
                    attr,
                    format!("failed to parse `#[{PARAM_ATTR}(...)]`: {e}"),
                )
            })?;
            if let Meta::NameValue(MetaNameValue { path, value, .. }) = &meta {
                if path.is_ident("desc") {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(lit), ..
                    }) = value
                    {
                        return Ok(Some(lit.value()));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Generate Input struct fields.
fn generate_input_fields(params: &[ParamInfo]) -> Vec<TokenStream2> {
    params
        .iter()
        .map(|p| {
            let name = &p.name;
            let ty = &p.ty;
            quote! {
                pub #name: #ty,
            }
        })
        .collect()
}

/// Generate schemars field attributes for descriptions.
///
/// Uses `#[doc = "..."]` instead of `#[schemars(description = "...")]` because
/// schemars 0.8 automatically extracts descriptions from doc comments, and using
/// `#[schemars(description)]` alongside the derive causes "duplicate attribute" errors
/// when there are multiple fields.
fn generate_field_attrs(params: &[ParamInfo]) -> Vec<TokenStream2> {
    params
        .iter()
        .map(|p| {
            if let Some(desc) = &p.desc {
                quote! {
                    #[doc = #desc]
                }
            } else {
                quote! {}
            }
        })
        .collect()
}

/// Convert snake_case to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// Remove all `#[param(...)]` attributes from function parameters.
/// This prevents the compiler from complaining about unknown attributes
/// when the original function is emitted.
fn strip_param_attrs(func: &mut ItemFn) {
    for arg in &mut func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            pat_type
                .attrs
                .retain(|attr| !attr.path().is_ident(PARAM_ATTR));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::ReturnType;

    fn rt(src: &str) -> ReturnType {
        syn::parse_str(src).unwrap()
    }

    fn ty_str(ret: &ReturnType) -> Option<String> {
        extract_result_ok(ret).map(|t| quote! { #t }.to_string())
    }

    /// J4:裸 `Result` 与 `std::result::Result` 一致识别为同一 ok 类型,消双包装。
    #[test]
    fn result_generics_bare_and_qualified_agree() {
        assert_eq!(ty_str(&rt("-> Result<f64, String>")), Some("f64".into()));
        assert_eq!(
            ty_str(&rt("-> std::result::Result<f64, String>")),
            Some("f64".into())
        );
        assert_eq!(ty_str(&rt("-> f64")), None);
        assert_eq!(ty_str(&rt("-> Result<f64>")), None); // 单泛型参数不是 Result<T,E>
    }

    /// J6:裸 `ToolError` 与 `…::tools::ToolError` 命中;`MyToolError`/`other::ToolError`/
    /// 非工具错误不命中,不再按名字宽泛过度匹配。
    #[test]
    fn return_type_is_tool_error_qualified_forms_only() {
        assert!(return_type_is_tool_error(&rt("-> Result<String, ToolError>")));
        assert!(return_type_is_tool_error(&rt("-> Result<String, lc_core::tools::ToolError>")));
        assert!(return_type_is_tool_error(&rt("-> Result<String, tools::ToolError>")));
        assert!(!return_type_is_tool_error(&rt("-> Result<String, MyToolError>")));
        assert!(!return_type_is_tool_error(&rt("-> Result<String, other::ToolError>")));
        assert!(!return_type_is_tool_error(&rt("-> Result<String, anyhow::Error>")));
        assert!(!return_type_is_tool_error(&rt("-> String")));
    }

    /// J3:标识符种子校验拒绝数字开头/空串。
    #[test]
    fn ident_seed_validation() {
        assert!(is_valid_ident_seed("Calculator"));
        assert!(is_valid_ident_seed("_private"));
        assert!(!is_valid_ident_seed("9lives"));
        assert!(!is_valid_ident_seed(""));
    }
}
