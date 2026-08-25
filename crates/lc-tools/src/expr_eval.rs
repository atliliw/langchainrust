// lc-tools/src/expr_eval.rs
//! A tiny all-`f64` math expression evaluator used by the `Calculator` tool.
//!
//! This replaces the `meval` crate, which pulled in an unmaintained `nom 1.x`
//! that will become a hard error on future Rust versions. The expression
//! language and semantics are kept identical to `meval`'s default context:
//!
//! - Arithmetic: `+`, `-`, `*`, `/`, `%`
//! - Power: `^` (right-associative, binds tighter than unary minus, so
//!   `-2^2 == -(2^2) == -4`)
//! - Unary `+` / `-`
//! - Parentheses
//! - Functions: `sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
//!   sqrt exp ln log abs floor ceil round signum` (unary), `atan2` (binary),
//!   `min` / `max` (variadic)
//! - Constants: `pi`, `e`
//! - Number literals including scientific notation (`1e3`, `2.5E-2`, `.5`)
//!
//! All arithmetic is floating point: `10 / 3` is `3.333…`, there is no
//! integer division. `log(x)` is the natural logarithm (as `log(e) == 1`).

/// Error type for the expression evaluator.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExprEvalError {
    /// The expression could not be evaluated (parse error, unknown
    /// function/identifier, or invalid number).
    #[error("{0}")]
    Msg(String),
}

/// Evaluate a math expression to a single `f64`.
pub fn eval(input: &str) -> Result<f64, ExprEvalError> {
    let mut p = Parser::new(input);
    p.skip_ws();
    if p.peek().is_none() {
        return Err(ExprEvalError::Msg("empty expression".to_string()));
    }
    let value = p.parse_expr()?;
    p.skip_ws();
    if let Some(c) = p.peek() {
        return Err(ExprEvalError::Msg(format!(
            "unexpected character '{}' at position {}",
            c, p.pos
        )));
    }
    Ok(value)
}

/// Recursive-descent parser. Grammar (loosely, by increasing precedence):
/// `expr  := term (('+' | '-') term)*`
/// `term  := unary (('*' | '/' | '%') unary)*`
/// `unary := ('+' | '-')* power`
/// `power := postfix ('^' unary)?`  (right-associative)
/// `postfix := number | constant | function '(' args ')' | '(' expr ')'`
struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_whitespace()) {
            self.bump();
        }
    }

    fn parse_expr(&mut self) -> Result<f64, ExprEvalError> {
        let mut value = self.parse_term()?;
        loop {
            self.skip_ws();
            if self.eat('+') {
                value += self.parse_term()?;
            } else if self.eat('-') {
                value -= self.parse_term()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_term(&mut self) -> Result<f64, ExprEvalError> {
        let mut value = self.parse_unary()?;
        loop {
            self.skip_ws();
            if self.eat('*') {
                value *= self.parse_unary()?;
            } else if self.eat('/') {
                value /= self.parse_unary()?;
            } else if self.eat('%') {
                value %= self.parse_unary()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_unary(&mut self) -> Result<f64, ExprEvalError> {
        self.skip_ws();
        let mut negate = false;
        loop {
            self.skip_ws();
            if self.eat('-') {
                negate = !negate;
            } else if self.eat('+') {
                // unary plus is a no-op
            } else {
                break;
            }
        }
        let value = self.parse_power()?;
        Ok(if negate { -value } else { value })
    }

    fn parse_power(&mut self) -> Result<f64, ExprEvalError> {
        let base = self.parse_postfix()?;
        self.skip_ws();
        if self.eat('^') {
            // Right-associative: the exponent is itself parsed at unary level,
            // so `2^3^2 == 2^(3^2)` and `2^-2 == 2^(-2)`.
            Ok(base.powf(self.parse_unary()?))
        } else {
            Ok(base)
        }
    }

    fn parse_postfix(&mut self) -> Result<f64, ExprEvalError> {
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<f64, ExprEvalError> {
        self.skip_ws();
        let Some(c) = self.peek() else {
            return Err(ExprEvalError::Msg(format!(
                "unexpected end of expression at position {}",
                self.pos
            )));
        };
        if c.is_ascii_digit() || c == '.' {
            self.parse_number()
        } else if c == '(' {
            self.bump();
            let value = self.parse_expr()?;
            self.skip_ws();
            if !self.eat(')') {
                return Err(ExprEvalError::Msg(format!(
                    "expected ')' at position {}",
                    self.pos
                )));
            }
            Ok(value)
        } else if c.is_ascii_alphabetic() || c == '_' {
            self.parse_identifier()
        } else {
            Err(ExprEvalError::Msg(format!(
                "unexpected character '{}' at position {}",
                c, self.pos
            )))
        }
    }

    fn parse_identifier(&mut self) -> Result<f64, ExprEvalError> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            self.bump();
        }
        let name = &self.input[start..self.pos];
        self.skip_ws();
        if self.eat('(') {
            let args = self.parse_args()?;
            call_function(name, args, start)
        } else {
            constant(name).ok_or_else(|| {
                ExprEvalError::Msg(format!(
                    "unknown identifier '{}' at position {}",
                    name, start
                ))
            })
        }
    }

    fn parse_args(&mut self) -> Result<Vec<f64>, ExprEvalError> {
        let mut args = Vec::new();
        self.skip_ws();
        if self.eat(')') {
            return Ok(args);
        }
        loop {
            args.push(self.parse_expr()?);
            self.skip_ws();
            if self.eat(',') {
                continue;
            }
            if self.eat(')') {
                return Ok(args);
            }
            return Err(ExprEvalError::Msg(format!(
                "expected ',' or ')' in argument list at position {}",
                self.pos
            )));
        }
    }

    fn parse_number(&mut self) -> Result<f64, ExprEvalError> {
        let start = self.pos;
        // Integer part
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.bump();
        }
        // Fractional part
        if self.peek() == Some('.') {
            self.bump();
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
        }
        // Exponent part (scientific notation)
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.bump();
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.bump();
            }
            let exp_digits = self.pos;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
            if self.pos == exp_digits {
                return Err(ExprEvalError::Msg(format!(
                    "invalid number '{}' at position {}",
                    &self.input[start..self.pos],
                    start
                )));
            }
        }
        let slice = &self.input[start..self.pos];
        slice.parse::<f64>().map_err(|_| {
            ExprEvalError::Msg(format!("invalid number '{}' at position {}", slice, start))
        })
    }
}

/// Look up a bare-identifier constant. Matches `meval`'s default context.
fn constant(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        _ => None,
    }
}

/// Dispatch a parsed function call. `start` is the position of the name for
/// error messages.
fn call_function(name: &str, args: Vec<f64>, start: usize) -> Result<f64, ExprEvalError> {
    let err_unknown =
        || ExprEvalError::Msg(format!("unknown function '{}' at position {}", name, start));
    match name {
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "asinh"
        | "acosh" | "atanh" | "sqrt" | "exp" | "ln" | "log" | "abs" | "floor" | "ceil"
        | "round" | "signum" => {
            let a = single_arg(name, args, start)?;
            Ok(match name {
                "sin" => a.sin(),
                "cos" => a.cos(),
                "tan" => a.tan(),
                "asin" => a.asin(),
                "acos" => a.acos(),
                "atan" => a.atan(),
                "sinh" => a.sinh(),
                "cosh" => a.cosh(),
                "tanh" => a.tanh(),
                "asinh" => a.asinh(),
                "acosh" => a.acosh(),
                "atanh" => a.atanh(),
                "sqrt" => a.sqrt(),
                "exp" => a.exp(),
                "ln" => a.ln(),
                // meval exposed no `log`; the calculator documents `log(e) == 1`
                // (natural log). Provide it rather than leaving it a parse error.
                "log" => a.ln(),
                "abs" => a.abs(),
                "floor" => a.floor(),
                "ceil" => a.ceil(),
                "round" => a.round(),
                "signum" => a.signum(),
                _ => return Err(err_unknown()),
            })
        }
        "atan2" => {
            let (a, b) = two_args(name, args, start)?;
            Ok(a.atan2(b))
        }
        "min" | "max" => {
            let Some(mut acc) = args.first().copied() else {
                return Err(ExprEvalError::Msg(format!(
                    "function '{}' requires at least 1 argument at position {}",
                    name, start
                )));
            };
            for v in args.into_iter().skip(1) {
                acc = if name == "min" {
                    acc.min(v)
                } else {
                    acc.max(v)
                };
            }
            Ok(acc)
        }
        _ => Err(err_unknown()),
    }
}

fn single_arg(name: &str, args: Vec<f64>, start: usize) -> Result<f64, ExprEvalError> {
    if args.len() == 1 {
        Ok(args[0])
    } else {
        Err(ExprEvalError::Msg(format!(
            "function '{}' requires exactly 1 argument, got {} at position {}",
            name,
            args.len(),
            start
        )))
    }
}

fn two_args(name: &str, args: Vec<f64>, start: usize) -> Result<(f64, f64), ExprEvalError> {
    if args.len() == 2 {
        Ok((args[0], args[1]))
    } else {
        Err(ExprEvalError::Msg(format!(
            "function '{}' requires exactly 2 arguments, got {} at position {}",
            name,
            args.len(),
            start
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::eval;

    fn assert_close(expr: &str, expected: f64) {
        let actual = eval(expr).unwrap_or_else(|e| panic!("`{}` failed: {}", expr, e));
        assert!(
            (actual - expected).abs() < 1e-10,
            "`{}` = {} (expected {})",
            expr,
            actual,
            expected
        );
    }

    #[test]
    fn basic_arithmetic() {
        assert_close("2 + 3", 5.0);
        assert_close("10 - 3", 7.0);
        assert_close("3.14 * 10", 31.4);
        assert_close("10 / 3", 10.0 / 3.0); // float division, no integer division
        assert_close("1 + 2 + 3", 6.0);
    }

    #[test]
    fn operator_precedence() {
        assert_close("2 + 3 * 4", 14.0); // not 20
        assert_close("2 * 3 + 4", 10.0);
        assert_close("20 - 10 / 2", 15.0);
    }

    #[test]
    fn power_operator() {
        assert_close("2^10", 1024.0);
        assert_close("2^3^2", 512.0); // right-associative
        assert_close("-2^2", -4.0); // -(2^2), power binds tighter than unary minus
        assert_close("(-2)^2", 4.0);
        assert_close("2^-2", 0.25);
        assert_close("2^0.5", std::f64::consts::SQRT_2);
    }

    #[test]
    fn remainder_operator() {
        assert_close("7 % 3", 1.0);
        assert_close("2 + 7 % 3 * 2", 4.0);
    }

    #[test]
    fn unary_minus_and_whitespace() {
        assert_close("-5", -5.0);
        assert_close(" 2   +   3 ", 5.0);
        assert_close("2 * -3", -6.0);
        assert_close("- -3", 3.0);
    }

    #[test]
    fn constants() {
        assert_close("pi", std::f64::consts::PI);
        assert_close("e", std::f64::consts::E);
        assert_close("2 * pi", 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn functions() {
        assert_close("sqrt(16)", 4.0);
        assert_close("sin(pi/2)", 1.0);
        assert_close("cos(0)", 1.0);
        assert_close("tan(0)", 0.0);
        assert_close("abs(-3)", 3.0);
        assert_close("exp(1)", std::f64::consts::E);
        assert_close("ln(e)", 1.0);
        assert_close("log(e)", 1.0); // documented behavior: log == natural log
        assert_close("floor(3.7)", 3.0);
        assert_close("ceil(3.2)", 4.0);
        assert_close("round(3.5)", 4.0);
        assert_close("signum(-2)", -1.0);
        assert_close("atan2(1, 1) * 4", std::f64::consts::PI); // pi/4 * 4
        assert_close("min(3, 1, 2)", 1.0);
        assert_close("max(3, 1, 2)", 3.0);
        assert_close("sqrt(sqrt(16))", 2.0); // nested calls
    }

    #[test]
    fn scientific_notation() {
        assert_close("2e3", 2000.0);
        assert_close("1.5E2", 150.0);
        assert_close("2e-2", 0.02);
        assert_close(".5", 0.5);
        assert_close("5.", 5.0);
    }

    #[test]
    fn plain_number() {
        assert_close("42", 42.0);
        assert_close("0", 0.0);
        assert_close("-2.5", -2.5);
    }

    #[test]
    fn invalid_expressions() {
        assert!(eval("hello").is_err()); // unknown identifier
        assert!(eval("").is_err()); // empty
        assert!(eval("   ").is_err()); // whitespace only
        assert!(eval("2 +").is_err()); // missing operand
        assert!(eval("2 3").is_err()); // no implicit multiplication
        assert!(eval("(2 + 3").is_err()); // unbalanced paren
        assert!(eval("2 + )").is_err()); // stray paren
        assert!(eval("foo(2)").is_err()); // unknown function
        assert!(eval("sin()").is_err()); // wrong arg count
        assert!(eval("sqrt(1, 2)").is_err()); // wrong arg count
        assert!(eval("atan2(1)").is_err()); // wrong arg count
        assert!(eval("min()").is_err()); // needs >= 1 arg
        assert!(eval("2e").is_err()); // exponent without digits
        assert!(eval("max(1,,2)").is_err()); // empty arg
    }
}
