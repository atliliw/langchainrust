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
    let tool_struct_name = format_ident!("{}Tool", to_pascal_case(&func_name_str));
    let input_struct_name = format_ident!("{}Input", to_pascal_case(&func_name_str));
    let func_name = func.sig.ident.clone();

    // 2. Extract parameters from function signature
    let params = extract_params(&func.sig)?;
    let field_names: Vec<Ident> = params.iter().map(|p| p.name.clone()).collect();

    // 3. Determine the output type from the function return type
    let output_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => {
            // If it's Result<T, E>, extract T
            if let Some(inner) = extract_result_ok(ty) {
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
                ::serde_json::to_value(schema_for!(#input_struct_name)).ok()
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
                _ => continue,
            };

            // Extract #[param(desc = "...")] attribute
            let desc = extract_param_desc(attrs);

            params.push(ParamInfo {
                name,
                ty: (*(*ty)).clone(),
                desc,
            });
        }
    }

    Ok(params)
}

/// Extract `T` from `Result<T, E>`. Returns None if not a Result type.
fn extract_result_ok(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        if type_path.path.segments.len() == 1 {
            let segment = &type_path.path.segments[0];
            if segment.ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner.clone());
                    }
                }
            }
        }
    }
    None
}

/// 判断函数返回类型是否为 `Result<_, ToolError>`(错误类型最后一个路径段为
/// `ToolError`)。F5:宏据此决定 `invoke` 是否直接透传原错误。
///
/// 只能是语法级判断:任何以 `ToolError` 结尾的错误类型都被视为库的 `ToolError`。
/// 常见写法均命中——裸 `ToolError`、`lc_core::tools::ToolError`、`tools::ToolError`
/// 或用户 `use` 进来的别名;返回 `anyhow::Error` / `String` 等其他错误类型时不命中。
fn return_type_is_tool_error(ret: &syn::ReturnType) -> bool {
    let syn::ReturnType::Type(_, ty) = ret else {
        return false;
    };
    let Type::Path(type_path) = &**ty else {
        return false;
    };
    let Some(seg) = type_path.path.segments.last() else {
        return false;
    };
    if seg.ident != "Result" {
        return false;
    }
    // 取出 `Result<_, E>` 的第二个泛型参数作为错误类型
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return false;
    };
    let mut generic = args.args.iter().filter_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    });
    let _ok = generic.next();
    let Some(err) = generic.next() else {
        return false;
    };
    let Type::Path(err_path) = err else {
        return false;
    };
    err_path
        .path
        .segments
        .last()
        .is_some_and(|s| s.ident == "ToolError")
}

/// Extract `desc` from `#[param(desc = "...")]`.
fn extract_param_desc(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident(PARAM_ATTR) {
            let meta: Meta = attr.parse_args().ok()?;
            if let Meta::NameValue(MetaNameValue { path, value, .. }) = &meta {
                if path.is_ident("desc") {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(lit), ..
                    }) = value
                    {
                        return Some(lit.value());
                    }
                }
            }
        }
    }
    None
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
