use csv;
use std::error::Error;
use std::io::Cursor;
fn main() -> Result<(), Box<dyn Error>> {
    let data = "header1,header2\nvalue1,value2\n";
    let cursor = Cursor::new(data);
    let mut rdr = csv::Reader::from_reader(cursor);
    let headers = rdr.headers()?;
    println!("Headers: {:?}", headers);
    Ok(())
}
