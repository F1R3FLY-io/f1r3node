#!/usr/bin/env python3
"""
RChain-Inspired Autopropose Service

Automated block proposing service implementing core RChain patterns:
- Time-synchronized proposing (RChain's sleep pattern)
- Round-robin validator rotation
- Deploy-before-propose pattern
- Basic health checking (gRPC connection only)
- Minimal configuration and logging

See https://github.com/rchain/rchain-testnet-node/blob/c7fcde9c7240cf5ecd2356f08b98a3809219a964/scripts/propose_in_turn.py
"""

import logging
import sys
import time
import yaml
import grpc
from argparse import ArgumentParser
from pathlib import Path

# RChain client imports 
try:
    from rchain.client import RClient, RClientException
    from rchain.crypto import PrivateKey
except ImportError as e:
    logging.error(f"Failed to import RChain libraries: {e}")
    logging.error("Make sure pyrchain is installed from git source (see requirements.txt)")
    sys.exit(1)


class SimpleAutoProposer:
    """
    Simplified RChain autopropose service focusing on core functionality.
    """
    
    def __init__(self, config_path):
        """Initialize the autoproposer with basic configuration."""
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.setup_parameters()
        self.current_validator_index = 0
        self.running = False
        
        # Simple failure tracking
        self.validator_failures = {v['name']: 0 for v in self.enabled_validators}
        
        logging.info("SimpleAutoProposer initialized")
        logging.info(f"Propose period: {self.period} seconds")
        logging.info(f"Validators: {[v['name'] for v in self.enabled_validators]}")
        logging.info(f"Rotation: {'Enabled' if len(self.enabled_validators) > 1 else 'Disabled'}")
    
    def setup_logging(self):
        """Configure basic logging."""
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    
    def load_config(self, config_path):
        """Load YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.error(f"Invalid YAML configuration: {e}")
            sys.exit(1)
    
    def setup_parameters(self):
        """Extract configuration parameters."""
        # Basic autopropose settings
        autopropose_config = self.config.get('autopropose', {})
        self.period = autopropose_config.get('period', 60)
        self.enabled = autopropose_config.get('enabled', True)
        
        # Validators configuration
        all_validators = self.config.get('validators', [])
        if not all_validators:
            logging.error("No validators configured")
            sys.exit(1)
        
        # Filter to only enabled validators
        self.enabled_validators = [v for v in all_validators if v.get('enabled', True)]
        if not self.enabled_validators:
            logging.error("No enabled validators found")
            sys.exit(1)
        
        # Rotation settings
        rotation_config = self.config.get('rotation', {})
        self.rotation_enabled = rotation_config.get('enabled', len(self.enabled_validators) > 1)
        self.rotate_on_success = rotation_config.get('rotate_on_success', True)
        self.rotate_on_failure = rotation_config.get('rotate_on_failure', False)
        self.max_failures_per_validator = rotation_config.get('max_failures_per_validator', 3)
        
        # Deploy configuration
        deploy_config = self.config.get('deploy', {})
        self.deploy_enabled = deploy_config.get('enabled', True)
        self.contract_path = deploy_config.get('contract', '/contracts/simpleInsertLookup.rho')
        self.phlo_limit = deploy_config.get('phlo_limit', 100000)
        self.phlo_price = deploy_config.get('phlo_price', 1)
        
        # Load private key
        try:
            deploy_key_hex = deploy_config.get('deploy_key')
            if not deploy_key_hex:
                logging.error("Deploy key not configured")
                sys.exit(1)
            self.deploy_key = PrivateKey.from_hex(deploy_key_hex)
            logging.info("Deploy private key loaded")
        except Exception as e:
            logging.error(f"Failed to load deploy private key: {e}")
            sys.exit(1)
        
        # Basic timing
        timing_config = self.config.get('timing', {})
        self.max_retries = timing_config.get('max_retries', 2)
        self.retry_delay = timing_config.get('retry_delay', 15)
        self.startup_delay = timing_config.get('startup_delay', 120)  # 2 minutes default wait for Casper
    
    def get_current_validator(self):
        """Get the currently selected validator."""
        if not self.enabled_validators:
            return None
        return self.enabled_validators[self.current_validator_index % len(self.enabled_validators)]
    
    def rotate_to_next_validator(self):
        """Rotate to the next validator in the sequence."""
        if not self.rotation_enabled or len(self.enabled_validators) <= 1:
            return
        
        previous_validator = self.get_current_validator()
        self.current_validator_index = (self.current_validator_index + 1) % len(self.enabled_validators)
        current_validator = self.get_current_validator()
        
        logging.info(f"Rotated from {previous_validator['name']} → {current_validator['name']}")
    
    def handle_validator_failure(self, validator_name, error):
        """Handle validator failures with simple tracking."""
        self.validator_failures[validator_name] += 1
        failure_count = self.validator_failures[validator_name]
        
        logging.warning(f"Validator {validator_name} failed (attempt {failure_count}): {error}")
        
        # Rotate if too many failures
        if failure_count >= self.max_failures_per_validator:
            logging.error(f"Validator {validator_name} has failed {failure_count} times, rotating")
            if self.rotate_on_failure and len(self.enabled_validators) > 1:
                self.rotate_to_next_validator()
                self.validator_failures[validator_name] = 0  # Reset after rotation
    
    def handle_validator_success(self, validator_name, block_hash):
        """Handle successful propose."""
        self.validator_failures[validator_name] = 0  # Reset failure count
        logging.info(f"Validator {validator_name} successfully proposed: {block_hash}")
        
        # Rotate on success if configured
        if self.rotate_on_success and len(self.enabled_validators) > 1:
            self.rotate_to_next_validator()
    
    def is_validator_ready(self, validator):
        """Simple validator readiness check - just test gRPC connection."""
        try:
            host = validator['host']
            grpc_port = validator['grpc_port']
            
            # Create RClient using PyPI pattern (host, port)
            client = RClient(host, grpc_port)
            # Test basic connectivity - if this works, we're ready
            return True
            
        except Exception as e:
            logging.warning(f"Validator {validator['name']} not ready: {e}")
            return False
    
    def wait_for_validators_ready(self):
        """Wait for current validator to be ready."""
        validator = self.get_current_validator()
        if not validator:
            logging.error("No validator available")
            return False
        
        max_attempts = 5
        for attempt in range(max_attempts):
            if self.is_validator_ready(validator):
                logging.info(f"Validator {validator['name']} is ready")
                return True
            
            if attempt < max_attempts - 1:
                logging.info(f"Waiting for validator {validator['name']} (attempt {attempt + 1}/{max_attempts})")
                time.sleep(10)
        
        logging.error(f"Validator {validator['name']} not ready after {max_attempts} attempts")
        return False
    
    def load_contract(self):
        """Load contract content from file."""
        try:
            contract_path = Path(self.contract_path)
            if not contract_path.exists():
                logging.error(f"Contract file not found: {self.contract_path}")
                return None
            
            with open(contract_path, 'r') as f:
                contract_content = f.read().strip()
                
            if not contract_content:
                logging.error(f"Contract file is empty: {self.contract_path}")
                return None
                
            logging.debug(f"Loaded contract from {self.contract_path}")
            return contract_content
            
        except Exception as e:
            logging.error(f"Failed to load contract: {e}")
            return None
    
    def deploy_contract(self, client, validator_name):
        """Deploy contract before proposing."""
        if not self.deploy_enabled:
            return True

        try:
            contract_content = self.load_contract()
            if not contract_content:
                return False

            timestamp = int(time.time() * 1000)

            logging.info(f"Deploying contract on {validator_name}")
            deploy_id = client.deploy_with_vabn_filled(
                self.deploy_key,
                contract_content,
                self.phlo_price,
                self.phlo_limit,
                timestamp,
                shard_id='root'  # Specify the root shard
            )

            logging.info(f"Contract deployed on {validator_name}: {deploy_id}")
            return True

        except Exception as e:
            logging.error(f"Contract deployment failed on {validator_name}: {e}")
            return False
    
    def propose_with_current_validator(self):
        """Propose a block using the current validator."""
        validator = self.get_current_validator()
        if not validator:
            logging.error("No validator available for proposing")
            return False
        
        host = validator['host']
        grpc_port = validator['grpc_port']
        validator_name = validator['name']
        
        logging.info(f"Proposing with validator: {validator_name}")
        
        # Use PyPI pyrchain pattern (host, port) instead of channel
        try:
            client = RClient(host, grpc_port)
            
            try:
                # Try to propose directly first
                block_hash = client.propose()
                self.handle_validator_success(validator_name, block_hash)
                return True
                
            except RClientException as e:
                error_message = str(e)
                
                # Handle "NoNewDeploys" case - deploy then propose
                if "NoNewDeploys" in error_message:
                    logging.info(f"No new deploys on {validator_name}, deploying contract first")
                    
                    if not self.deploy_contract(client, validator_name):
                        self.handle_validator_failure(validator_name, "Contract deployment failed")
                        return False
                    
                    # Wait briefly then propose
                    time.sleep(2)
                    
                    try:
                        block_hash = client.propose()
                        logging.info(f"Deploy-then-propose successful on {validator_name}: {block_hash}")
                        self.handle_validator_success(validator_name, block_hash)
                        return True
                    except Exception as propose_error:
                        self.handle_validator_failure(validator_name, f"Post-deploy propose failed: {propose_error}")
                        return False
                else:
                    # Other RChain errors
                    self.handle_validator_failure(validator_name, error_message)
                    return False

        except Exception as e:
            self.handle_validator_failure(validator_name, f"Connection error: {e}")
            return False
    
    def sleep_until_next_interval(self):
        """
        RChain-style time synchronization.
        Sleep until the next propose interval aligned to clock time.
        """
        if self.period <= 0:
            return

        current_time = int(time.time())
        seconds_into_period = current_time % self.period
        sleep_duration = self.period - seconds_into_period

        current_validator = self.get_current_validator()
        validator_name = current_validator['name'] if current_validator else "none"
        
        logging.info(f"Sleeping {sleep_duration}s until next interval (proposer: {validator_name})")
        time.sleep(sleep_duration)

    def run_forever(self):
        """
        Main autopropose loop - simplified version of RChain's pattern:

        startup_delay (wait for Casper)
        while :; do
            wait_for_current_validator_ready
            sleep_until_next_interval
            propose_with_current_validator
            rotate_based_on_result
        done
        """
        if not self.enabled:
            logging.info("Autopropose is disabled")
            return

        # Wait for Casper to be ready before starting
        if self.startup_delay > 0:
            logging.info(f"⏳ Waiting {self.startup_delay} seconds for Casper consensus to be ready...")
            logging.info("   (Validators need time to initialize and establish consensus)")
            time.sleep(self.startup_delay)
            logging.info("✓ Startup delay complete, beginning autopropose operations")

        logging.info("Starting autopropose main loop...")
        logging.info("Validator rotation order:")
        for i, validator in enumerate(self.enabled_validators):
            marker = "→" if i == 0 else " "
            logging.info(f"  {marker} {validator['name']} ({validator['host']}:{validator['grpc_port']})")

        self.running = True
        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1
                current_validator = self.get_current_validator()

                logging.info(f"=== Propose Cycle #{cycle_count} ===")
                if current_validator:
                    failures = self.validator_failures.get(current_validator['name'], 0)
                    logging.info(f"Current proposer: {current_validator['name']} (failures: {failures})")

                # Wait for current validator to be ready
                if not self.wait_for_validators_ready():
                    logging.error("Validator not ready, retrying...")
                    time.sleep(self.retry_delay)
                    continue

                # Time-synchronized sleep (RChain pattern)
                self.sleep_until_next_interval()

                # Propose with current validator
                success = self.propose_with_current_validator()

                if success:
                    logging.info("✓ Propose cycle completed successfully")
                else:
                    logging.warning("✗ Propose cycle failed, will retry next interval")

            except KeyboardInterrupt:
                logging.info("Shutdown requested by user")
                self.running = False
                break

            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
                logging.info(f"Sleeping {self.retry_delay} seconds before retry")
                time.sleep(self.retry_delay)

        logging.info("Autopropose main loop exited")

    def stop(self):
        """Stop the autopropose service."""
        logging.info("Stopping autopropose service...")
        self.running = False


def main():
    """Main entry point."""
    parser = ArgumentParser(description="RChain-inspired autopropose service (simplified)")
    parser.add_argument(
        "--config", 
        default="config.yml", 
        help="Path to configuration file (default: config.yml)"
    )
    
    args = parser.parse_args()
    
    try:
        autoproposer = SimpleAutoProposer(args.config)
        autoproposer.run_forever()
    except Exception as e:
        logging.error(f"Failed to start autopropose service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 