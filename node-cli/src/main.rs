use clap::Parser;
use node_cli::args::Cli;
use node_cli::dispatcher::Dispatcher;
use node_cli::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    Dispatcher::dispatch(&cli).await
}
