pub fn wrap_with_braces(expr: String) -> String {
    match expr.parse::<i32>() {
        Ok(_) => expr.clone(),

        Err(_) => {
            if expr.starts_with('(') && expr.ends_with(')') {
                expr.clone()
            } else {
                format!("({})", expr)
            }
        }
    }
}
