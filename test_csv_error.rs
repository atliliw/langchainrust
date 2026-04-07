use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 测试 CSV 错误类型的正确名称
    let result = csv::Reader::from_reader("a,b\n1,2".as_bytes()).headers();
    match result {
        Ok(_) => println!("Success"),
        Err(e) => println!("Error type: {}", e),
    }

    Ok(())
}
