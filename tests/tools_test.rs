#[cfg(test)]
mod tests {
    use langchainrust::tools::{Calculator, Tool, ToolInput, WeatherTool};
    use std::collections::HashMap;

    fn mock_tool_input(params: Vec<(&str, &str)>) -> ToolInput {
        let mut map = HashMap::new();
        for (k, v) in params {
            map.insert(k.to_string(), v.to_string());
        }
        ToolInput {
            tool_name: "".to_string(),
            parameters: map,
        }
    }

    #[tokio::test]
    async fn test_calculator_addition() {
        let calc = Calculator {};
        let input = mock_tool_input(vec![("expression", "3 + 4")]);
        let output = calc.invoke(input).await.unwrap();
        assert_eq!(output.result, "7");
        println!("{}", calc.name());
        println!("{}", calc.description());
        println!("{:?}", calc.parameters());
        println!("{}", calc.return_direct());
    }

    #[tokio::test]
    async fn test_weather_tool() {
        let weather = WeatherTool;
        let input = mock_tool_input(vec![("city", "杭州")]);
        let output = weather.invoke(input).await.unwrap();
        assert!(output.result.contains("杭州"));
        assert!(output.result.contains("晴天"));
        assert!(output.result.contains("气温25℃"));
    }

    #[tokio::test]
    async fn test_weather_tool_missing_city() {
        let weather = WeatherTool;
        let input = mock_tool_input(vec![]);
        let result = weather.invoke(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("缺少 city 参数"));
    }
}
