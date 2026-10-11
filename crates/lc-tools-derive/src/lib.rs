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
use syn::parse::Parser;
use syn::{
    parse_macro_input, Attribute, Expr, ExprLit, ExprPath, FnArg, Ident, ItemFn, Lit, Meta,
    MetaNameValue, Pat, PatType, Result, Signature, Type,
};

/// Attribute for individual parameters.
const PARAM_ATTR: &str = "param";

/// How the generated `invoke`/`run` adapt the source function's error type.
///
/// A proc macro has no type information, so [`ErrorMode::Auto`] matches the *written* error
/// path against [`KNOWN_TOOL_ERROR_PATHS`]. The other two modes let the caller state the
/// truth instead of relying on that name match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorMode {
    /// Infer from the written return type (default).
    Auto,
    /// Pass the function's error through unchanged — requires the error type to *be*
    /// `lc_core::tools::ToolError`, so variants (`InvalidInput`, `Timeout`, …) survive.
    Passthrough,
    /// Wrap any error via `Display` into `ToolError::ExecutionFailed` — compiles for any
    /// `E: Display`, but flattens the variant.
    Wrap,
}

/// Parsed `#[tool(...)]` options.
struct ToolAttrs {
    /// Tool description shown to the LLM (required, non-empty).
    description: String,
    /// How the generated `invoke`/`run` adapt the function's error type.
    error: ErrorMode,
}

/// The `#[tool]` procedural macro.
///
/// Transforms a function into a full Tool implementation.
///
/// # Attributes
///
/// - `#[tool(description = "...")]` — Required. The tool description shown to the LLM.
/// - `#[param(desc = "...")]` — Optional per-parameter. Adds description to the JSON schema.
/// - `#[tool(error = auto | passthrough | wrap)]` — Optional. How the generated `invoke`/`run`
///   adapt the function's error type; defaults to `auto` (see `ErrorMode`). Needed only when
///   the written error path is not one of `KNOWN_TOOL_ERROR_PATHS`, or when a path that *is*
///   listed names something else in your crate.
///
/// # Parameter Rules
///
/// - `String`, `i64`, `f64`, `bool` → required in schema
/// - `Option<T>` → optional in schema
/// - `Vec<T>` → array in schema
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    let attrs = match parse_tool_attr(attr.into()) {
        Ok(a) => a,
        Err(err) => return err.to_compile_error().into(),
    };

    match tool_impl(attrs, func) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Parse `#[tool(description = "...", error = auto|passthrough|wrap)]`.
///
/// Takes a `proc_macro2` stream (not `proc_macro`) so the parser is unit-testable outside a
/// macro expansion.
fn parse_tool_attr(attr: TokenStream2) -> Result<ToolAttrs> {
    if attr.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[tool(description = \"...\")] is required",
        ));
    }

    let metas =
        syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated.parse2(attr)?;

    let mut description: Option<String> = None;
    let mut error = ErrorMode::Auto;

    for meta in &metas {
        let Meta::NameValue(MetaNameValue { path, value, .. }) = meta else {
            return Err(syn::Error::new_spanned(meta, UNKNOWN_TOOL_OPTION));
        };

        if path.is_ident("description") {
            let Expr::Lit(ExprLit {
                lit: Lit::Str(lit), ..
            }) = value
            else {
                return Err(syn::Error::new_spanned(
                    value,
                    "#[tool(description = \"...\")] must be a string literal",
                ));
            };
            let desc = lit.value();
            if desc.trim().is_empty() {
                return Err(syn::Error::new_spanned(
                    value,
                    "#[tool(description = \"...\")] must be a non-empty string",
                ));
            }
            // J5 风格:重复/冲突的写法不静默取其一,直接报出。
            if description.is_some() {
                return Err(syn::Error::new_spanned(path, "duplicate `description`"));
            }
            description = Some(desc);
        } else if path.is_ident("error") {
            let Expr::Path(ExprPath { path: mode, .. }) = value else {
                return Err(syn::Error::new_spanned(
                    value,
                    format!("`error` must be one of {ERROR_MODES}"),
                ));
            };
            let Some(ident) = mode.get_ident() else {
                return Err(syn::Error::new_spanned(
                    value,
                    format!("`error` must be one of {ERROR_MODES}"),
                ));
            };
            error = match ident.to_string().as_str() {
                "auto" => ErrorMode::Auto,
                "passthrough" => ErrorMode::Passthrough,
                "wrap" => ErrorMode::Wrap,
                other => {
                    return Err(syn::Error::new_spanned(
                        value,
                        format!("unknown `error` mode `{other}`; expected one of {ERROR_MODES}"),
                    ))
                }
            };
        } else {
            return Err(syn::Error::new_spanned(meta, UNKNOWN_TOOL_OPTION));
        }
    }

    let description = description.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[tool(description = \"...\")] is required",
        )
    })?;

    Ok(ToolAttrs { description, error })
}

/// Diagnostic shared by every unknown-option path in [`parse_tool_attr`].
const UNKNOWN_TOOL_OPTION: &str = "unknown #[tool(...)] option; expected `description = \"...\"` \
                                    and/or `error = auto|passthrough|wrap`";

/// The three accepted `#[tool(error = ...)]` modes, for diagnostics.
const ERROR_MODES: &str = "`auto`, `passthrough` or `wrap`";

fn tool_impl(attrs: ToolAttrs, mut func: ItemFn) -> Result<TokenStream2> {
    let ToolAttrs { description, error } = attrs;

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
    // T2 §4(符号碰撞检测):`to_pascal_case` 是**多对一**的(按 `_` 切分并丢弃空段),
    // `_greet` / `greet_` / `add__numbers` 与 `greet` / `add_numbers` 会生成同名类型,
    // 在同一模块里以 E0428 报在离病因很远的位置。过程宏只收到被标注的这一个 item,看不到
    // 同模块的兄弟项,因此能检测的只有派生自身的不确定性,这里把它挡在编译期。
    //
    // 另外两组名字不可能碰撞,故不设冗余守卫:`{P}Tool` 与 `{P}Input` 后缀不同;
    // `{P}Tool == fn_name` 会要求 `fn_name == to_pascal_case(fn_name) + "Tool"`,而
    // `to_pascal_case` 保首字符大写,该等式无解(生成名恒比原名长一个 `Tool`/`Input`)。
    assert_canonical_seed(&func_name_str, &func.sig.ident, &pascal)?;

    let tool_struct_name = format_ident!("{}Tool", pascal);
    let input_struct_name = format_ident!("{}Input", pascal);
    let func_name = func.sig.ident.clone();

    // 2. Extract parameters from function signature
    let params = extract_params(&func.sig)?;
    let field_names: Vec<Ident> = params.iter().map(|p| p.name.clone()).collect();

    // async 支持:派生出的 `invoke`/`run` 本身已是 async trait 方法,若源函数也是
    // `async fn`,则调用处需 `.await`。这里捕获该标志,两个调用体统一据此生成后缀。
    let is_async = func.sig.asyncness.is_some();

    // 3. Determine the output type from the function return type
    // D2:非 `Result` 返回类型必须干净拒绝。`invoke`/`run` 都以 `.map_err(...)` 联动函数
    // 返回值,只有 `Result<T, E>`(或无返回 `()`)才生成可编译代码;直接放行原类型会让
    // 用户看到一堆语义不明的类型错误,改为清晰的 `compile_error!` 并给 span 提示。
    let mut output_is_string = false;
    let output_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, _) => {
            // If it's Result<T, E>, extract T.
            if let Some(inner) = extract_result_ok(&func.sig.output) {
                output_is_string = is_string_type(&inner);
                quote! { #inner }
            } else {
                return Err(syn::Error::new_spanned(
                    &func.sig.output,
                    "tool functions must return `Result<T, E>` (or have no return type); \
                     non-Result return types are not supported by #[tool]",
                ));
            }
        }
    };

    // 源函数为 async 时,调用 `${fn(...).await}`;否则 `${fn(...)}`。放在这里统一算好,
    // 供 `invoke`/`run` 两个调用体复用,避免各写一份 `.await` 判断。
    let call = if is_async {
        quote! { #func_name(#(#field_names),*).await }
    } else {
        quote! { #func_name(#(#field_names),*) }
    };

    // 3b. F5 + D1 + T2:函数返回 `Result<_, ToolError>` 时,`invoke`/`run` 直接透传原错误
    // (参数错 / 业务错原样保留,不再统一压平成 `ExecutionFailed`);返回其他错误类型时才包
    // `ExecutionFailed`。`invoke` 与 `run` 共用同一份改写后的调用,两者的错误语义不会再
    // 各自漂移(D1 修的正是 F5 只做了 `invoke` 一半的问题)。
    //
    // 判定方式见 `ErrorMode`:默认按书写路径匹配 [`KNOWN_TOOL_ERROR_PATHS`],白名单外的
    // 写法(别名、自定义再导出)可用 `#[tool(error = passthrough)]` 显式声明,被白名单
    // 误命中的同名类型可用 `#[tool(error = wrap)]` 反向退出。
    let passthrough = match error {
        ErrorMode::Passthrough => {
            // `passthrough` 要求函数确实返回 `Result`;错误类型是不是库的 `ToolError`
            // 由编译器在 `impl` 处判定,宏只能先把「不是 Result」这种明确错误拦下。
            if result_generics(&func.sig.output).is_none() {
                return Err(syn::Error::new_spanned(
                    &func.sig.output,
                    "#[tool(error = passthrough)] requires the function to return `Result<T, E>`",
                ));
            }
            true
        }
        ErrorMode::Wrap => false,
        ErrorMode::Auto => return_type_is_tool_error(&func.sig.output),
    };
    let adapted_call = if passthrough {
        quote! { #call }
    } else {
        quote! { #call.map_err(|e| ::lc_core::tools::ToolError::ExecutionFailed(e.to_string())) }
    };

    // D3:输出已是 String 时直接返回该 String,不再 `serde_json::to_string` 多包一层
    // JSON 引号;否则才序列化(`run` 的返回类型固定为 String)。**此分支在宏阶段按
    // `output_is_string` 预选,不能用运行时 `if/else`**——运行时两臂都要通过类型检查,
    // 非 String 输出的 if 臂(`result` 是 i64/结构体等)与 else 臂(String)类型不兼容
    // 会报 E0308,导致所有非 String 输出的工具都编译不过。
    let serialized = if output_is_string {
        quote! { let serialized = result; }
    } else {
        // F5:序列化失败不再静默回退 Debug 文本(会把 Rust 内部结构喂给 LLM),
        // 而是返回 ExecutionFailed 错误,让上层明确感知输出无法序列化。
        quote! {
            let serialized = ::serde_json::to_string(&result)
                .map_err(|e| ::lc_core::tools::ToolError::ExecutionFailed(format!(
                    "Failed to serialize tool output: {}", e
                )))?;
        }
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

            /// OpenAI/Anthropic `tools[]` 风格的 `parameters` JSON Schema。
            ///
            /// 等价于 `BaseTool::args_schema`,但按 Chat API 的 `tools[].function.parameters`
            /// 语义直接给出 `{ "type": "object", "properties": {…}, "required": […] }` 形状,
            /// 便于工具定义直接喂给两家 Chat API。由 `schemars` 从 `#input_struct_name` 派生。
            pub fn parameters(&self) -> ::std::option::Option<::serde_json::Value> {
                use ::schemars::schema_for;
                ::serde_json::to_value(schema_for!(#input_struct_name)).ok()
            }

            /// `name + description + args_schema` 三元组,供 registry 自动化登记工具。
            ///
            /// `name`/`description` 在宏展开期已知,直接以字面量内联,不依赖 `BaseTool`
            /// trait 在调用方作用域可见;schema 与 `BaseTool::args_schema` 同源。
            pub fn describe(&self) -> (::std::string::String, ::std::string::String, ::std::option::Option<::serde_json::Value>) {
                use ::schemars::schema_for;
                (
                    ::std::string::String::from(#func_name_str),
                    ::std::string::String::from(#description),
                    ::serde_json::to_value(schema_for!(#input_struct_name)).ok(),
                )
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
                #adapted_call
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
                let result = #adapted_call?;
                #serialized
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
        // self:`#[tool]` 面向自由函数;带 receiver 的方法会被静默丢弃 `self` 状态,
        // 生成一个不引用 receiver 的 Tool——与 J8「非 ident 参数报错」一脉相承,
        // 显式拒绝而非静默吞。
        if let FnArg::Receiver(recv) = arg {
            let span = recv.self_token.span;
            return Err(syn::Error::new(
                span,
                "`#[tool]` only supports free functions; methods with a self receiver \
                 (fn foo(&self, ...)) are not supported — use a free function instead",
            ));
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

/// D3:判断 `Result<T, E>` 的 ok 类型 T 是否为 `String`。按路径末段匹配,`String` 与
/// `::std::string::String`(以及经 `use` 引入的别名)一致命中;`run` 据此决定输出是否
/// 还需 JSON 序列化(String 直接返回,避免多包一层 JSON 引号)。
fn is_string_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|s| s.ident == "String")
}

/// 判断函数返回类型是否为 `Result<_, ToolError>`。F5:宏据此决定 `invoke`/`run` 是否直接
/// 透传原错误(保留 `InvalidInput`/`Timeout` 等变体),而不是压平成 `ExecutionFailed`。
///
/// J4+J6:与 [`extract_result_ok`] 统一走 [`result_generics`] 识别 `Result`;错误类型按
/// [`KNOWN_TOOL_ERROR_PATHS`] 匹配**书写路径**。**漏判只降级错误语义,不会产生编译错误**
/// ——回退分支只要求 `E: Display`;会编译失败的是**误判**,白名单正是为消除误判而收窄的。
fn return_type_is_tool_error(ret: &syn::ReturnType) -> bool {
    let Some((_, err)) = result_generics(ret) else {
        return false;
    };
    let Type::Path(err_path) = err else {
        return false;
    };
    // `<X as Y>::ToolError` 这类限定路径指关联类型,不可能是库的那个枚举。
    if err_path.qself.is_some() {
        return false;
    }
    let written: Vec<String> = err_path
        .path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    KNOWN_TOOL_ERROR_PATHS.iter().any(|known| {
        known.len() == written.len()
            && known
                .iter()
                .zip(written.iter())
                .all(|(known_seg, seg)| *known_seg == seg.as_str())
    })
}

/// 本 workspace 真实暴露 `lc_core::tools::ToolError` 的书写路径(白名单)。
///
/// 过程宏拿不到类型信息,「这个 `ToolError` 是不是库的那个」只能按**书写形式**回答,所以
/// 这里枚举的是今天确实能解析到它的全部路径:定义处,加上下游拿得到的再导出(`lc-tools`
/// 与 `langchainrust` 门面),取代原先「末段是 `ToolError` 且倒数第二段是 `tools`」的启发式
/// ——那条启发式两个方向都错:漏判再导出路径(把真正的 `ToolError` 静默压平成
/// `ExecutionFailed`),又误判无关的 `other::tools::ToolError`(透传后在 `impl` 处类型不匹配,
/// 报错离病因很远)。
///
/// 白名单之外的写法不是死路:`#[tool(error = passthrough)]` 强制透传,
/// `#[tool(error = wrap)]` 强制压平。
const KNOWN_TOOL_ERROR_PATHS: &[&[&str]] = &[
    // `use lc_core::tools::ToolError;` / `use lc_tools::ToolError;` 之后裸写。
    &["ToolError"],
    // 或经作用域内的 `tools` 模块书写。
    &["tools", "ToolError"],
    // 定义处(`crates/lc-core/src/tools/base.rs`,并在 crate 根再导出)。
    &["lc_core", "ToolError"],
    &["lc_core", "tools", "ToolError"],
    // `lc-tools` 的再导出 —— `#[tool]` 自身的使用方就在这个 crate。
    &["lc_tools", "ToolError"],
    // `langchainrust` 门面:根、`tools` 模块,以及 `core`(即 `lc_core`)别名下。
    &["langchainrust", "ToolError"],
    &["langchainrust", "tools", "ToolError"],
    &["langchainrust", "core", "ToolError"],
    &["langchainrust", "core", "tools", "ToolError"],
];

/// 拒绝 Pascal 种子不可唯一还原的函数名(见 [`tool_impl`] 中的调用点)。
///
/// `snake(pascal(name)) == name` 恰好等价于「`name` 已是规范 snake_case」:不带前导/尾随/
/// 连续下划线,且没有 `_` 分隔的纯数字词(`greet_2` 与 `greet2` 同种子,只有后者是规范写法)。
fn assert_canonical_seed(func_name: &str, ident: &Ident, pascal: &str) -> Result<()> {
    let canonical = to_snake_case(pascal);
    if canonical == func_name {
        return Ok(());
    }
    Err(syn::Error::new_spanned(
        ident,
        format!(
            "`{func_name}` cannot be turned into collision-free tool type names: it maps to the \
             same PascalCase seed `{pascal}` as its canonical spelling `{canonical}`, so the two \
             would generate `{pascal}Tool` / `{pascal}Input` in the same module. Rename the \
             function to `{canonical}`."
        ),
    ))
}

/// [`to_pascal_case`] 在规范写法上的逆映射。
///
/// 正向按 `_` 切分、丢弃空段、把每词首字符大写;反向在每个非首位的大写字符前补一个 `_`。
/// 因此 `to_snake_case(to_pascal_case(x)) == x` 当且仅当 `x` 已是规范 snake_case。
fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.char_indices() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
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

    /// J6 + T2:白名单内的书写路径命中(含 `lc-tools` / `langchainrust` 再导出);
    /// `MyToolError`、无关 crate 的 `…::tools::ToolError`、限定路径、非 `Result` 不命中。
    #[test]
    fn return_type_is_tool_error_known_paths_only() {
        // 命中:定义处 + 作用域内 `tools` 模块 + 再导出。
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, tools::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, lc_core::tools::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, ::lc_core::tools::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, lc_core::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, lc_tools::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, langchainrust::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, langchainrust::tools::ToolError>"
        )));
        assert!(return_type_is_tool_error(&rt(
            "-> Result<String, langchainrust::core::tools::ToolError>"
        )));
        // 外层 `Result` 的书写形式不影响判定(J4)。
        assert!(return_type_is_tool_error(&rt(
            "-> std::result::Result<String, ToolError>"
        )));

        // 不命中:旧启发式在这里会误判成库错误并原样透传 → 在 `impl` 处类型不匹配。
        assert!(!return_type_is_tool_error(&rt(
            "-> Result<String, other::tools::ToolError>"
        )));
        assert!(!return_type_is_tool_error(&rt(
            "-> Result<String, MyToolError>"
        )));
        assert!(!return_type_is_tool_error(&rt(
            "-> Result<String, anyhow::Error>"
        )));
        assert!(!return_type_is_tool_error(&rt(
            "-> Result<String, <X as Y>::ToolError>"
        )));
        assert!(!return_type_is_tool_error(&rt("-> String")));
    }

    /// T2:`snake(pascal(x)) == x` 恰好刻画「x 已是规范 snake_case」。
    #[test]
    fn to_snake_case_inverts_pascal_case() {
        assert_eq!(to_snake_case("Greet"), "greet");
        assert_eq!(to_snake_case("AddNumbers"), "add_numbers");
        // 数字留在词内,不在数字边界拆词(`base64_encode` 不能被拆成 `base_64_encode`)。
        assert_eq!(to_snake_case("Base64Encode"), "base64_encode");
        assert_eq!(to_snake_case("Sha256Hash"), "sha256_hash");
        assert_eq!(to_snake_case(""), "");
    }

    /// T2(符号碰撞检测):只有规范 snake_case 的函数名才能生成唯一的类型名。
    #[test]
    fn canonical_seed_detects_colliding_names() {
        let check = |name: &str| {
            assert_canonical_seed(
                name,
                &Ident::new("f", proc_macro2::Span::call_site()),
                &to_pascal_case(name),
            )
        };

        // 规范写法:`pascal` 可逆,两个不同函数名不会生成同名类型。
        assert!(check("greet").is_ok());
        assert!(check("add_numbers").is_ok());
        assert!(check("url_fetch").is_ok());
        assert!(check("sha256_hash").is_ok());
        assert!(check("greet2").is_ok());

        // `_` 被 `split('_')` 折叠:`_greet` / `greet_` / `add__numbers` 分别与
        // `greet` / `greet` / `add_numbers` 同种子,必须报错而不是生成同名类型。
        assert!(check("_greet").is_err());
        assert!(check("greet_").is_err());
        assert!(check("add__numbers").is_err());
        // 纯数字词:`greet_2` 与 `greet2` 同种子,只有规范写法 `greet2` 放行。
        assert!(check("greet_2").is_err());
    }

    /// T2:`#[tool(error = ...)]` 解析出三种模式,`description` 仍是必填。
    #[test]
    fn parse_tool_attr_accepts_documented_forms() {
        let parse = |src: &str| parse_tool_attr(src.parse().unwrap());

        let a = parse(r#"description = "Adds two numbers""#).unwrap();
        assert_eq!(a.description, "Adds two numbers");
        assert_eq!(a.error, ErrorMode::Auto);

        let a = parse(r#"description = "x", error = passthrough"#).unwrap();
        assert_eq!(a.error, ErrorMode::Passthrough);

        // 选项顺序无关。
        let a = parse(r#"error = wrap, description = "x""#).unwrap();
        assert_eq!(a.description, "x");
        assert_eq!(a.error, ErrorMode::Wrap);
    }

    /// T2:畸形属性一律给出 `compile_error!`,不静默取默认值。
    #[test]
    fn parse_tool_attr_rejects_bad_forms() {
        let parse = |src: &str| parse_tool_attr(src.parse().unwrap());

        assert!(parse("").is_err()); // 空属性
        assert!(parse(r#"error = passthrough"#).is_err()); // 缺 description
        assert!(parse(r#"description = """#).is_err()); // 空 description
        assert!(parse(r#"description = 42"#).is_err()); // 非字符串字面量
        assert!(parse(r#"description = "x", error = nope"#).is_err()); // 未知模式
        assert!(parse(r#"description = "x", error = "wrap""#).is_err()); // 模式非 ident
        assert!(parse(r#"description = "x", description = "y""#).is_err()); // 重复
        assert!(parse(r#"description = "x", nope = 1"#).is_err()); // 未知选项
        assert!(parse(r#"description"#).is_err()); // 非 NameValue
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
