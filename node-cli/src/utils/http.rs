use reqwest;
use serde_json;
use std::time::Instant;

pub struct HttpClient {
    client: reqwest::Client,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_json(
        &self,
        url: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let response = self.client.get(url).send().await?;

        if response.status().is_success() {
            let text = response.text().await?;
            let json: serde_json::Value = serde_json::from_str(&text)?;
            Ok(json)
        } else {
            Err(format!("HTTP {}: {}", response.status(), response.text().await?).into())
        }
    }

    pub async fn get_text(&self, url: &str) -> Result<String, Box<dyn std::error::Error>> {
        let response = self.client.get(url).send().await?;

        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(format!("HTTP {}: {}", response.status(), response.text().await?).into())
        }
    }

    pub async fn get_with_timing(
        &self,
        url: &str,
    ) -> Result<(serde_json::Value, std::time::Duration), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let result = self.get_json(url).await?;
        let duration = start_time.elapsed();
        Ok((result, duration))
    }
}

pub fn build_url(host: &str, port: u16, path: &str) -> String {
    format!("http://{}:{}{}", host, port, path)
}
