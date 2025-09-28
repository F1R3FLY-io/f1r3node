# SETUP

If you've run RNode previously, delete the pre-existing configuration files.

    rm -rf ~/.rnode/
    
From the `rchain` directory, build RNode.

    sbt node/universal:stage
    
From `node/target/universal/stage`, start RNode.

    ./bin/rnode run -s --wallets-file $HOME/IdeaProjects/rchain/rholang/examples/wallets.txt
    
This generates a random Secp256k1 private key corresponding to a validator. Next, terminate RNode and restart as one of 
the randomly generated validators.

    ./bin/rnode run -s --validator-private-key $(cat ~/.rnode/genesis/*.sk | tail -1) --wallets-file $HOME/IdeaProjects/rchain/rholang/examples/wallets.txt

Open a new terminal and navigate to `rholang/examples`, then add simulated user credentials to bash environment.

    . keys.env

# DEMO START

## Know your ASIAddress

Here's how Alice would check her ASI vault address:

    ./propose.sh $ALICE_PRV vault_demo/1.know_ones_revaddress.rho "-e s/%PUB_KEY/$ALICE_PUB/"

## Access your own vault

Here's how Alice would check her ASI vault balance:

    ./propose.sh $ALICE_PRV vault_demo/2.check_balance.rho "-e s/%ASI_ADDR/$ALICE_REV/"
        
Notice that anyone can check Alice's ASI vault balance.

    ./propose.sh $BOB_PRV vault_demo/2.check_balance.rho "-e s/%ASI_ADDR/$ALICE_REV/"

## Transfer to a ASIAddress

Suppose Alice wants to on-board Bob and that she knows his ASI address. Here's how she would transfer 100 ASI to Bob.

    ./propose.sh $ALICE_PRV vault_demo/3.transfer_funds.rho "-e s/%FROM/$ALICE_REV/ -e s/%TO/$BOB_REV/"
    ./propose.sh $ALICE_PRV vault_demo/2.check_balance.rho "-e s/%ASI_ADDR/$ALICE_REV/"
    
Notice the transfer hasn't been finished yet. Still, funds have been deducted from Alice's vault.

Now, let's have Bob check his own balance:

    ./propose.sh $BOB_PRV vault_demo/2.check_balance.rho "-e s/%ASI_ADDR/$BOB_REV/"

When Bob checks his balance for the first time, a ASI vault is created at the ASI address he provides. Once his vault is 
created, all previous transfers to his vault complete. In other words, the order in which one creates a vault and transfers
REV into that vault doesn't matter.

This means that the first access to one's vault needs to be done by a 3rd-party having the REV
to pay for it. So the exchanges should not only do a `transfer`, but also at `findOrCreate`
the destination vault. So should the Testnet operators distributing the funds.

Because the "transfer" method takes a ASIAddress (and not a ASIVault), transfers between different "kinds", or security 
schemes of ASIVaults are possible. For now, we only provide a simple ASIVault that only grants access to its designated 
user.

## Attempt a transfer despite insufficient funds

    ./propose.sh $ALICE_PRV vault_demo/3.transfer_funds.rho "-e s/%FROM/$ALICE_REV/ -e s/%TO/$BOB_REV/"

## Attempt a transfer despite invalid RevAddress

    ./propose.sh $ALICE_PRV vault_demo/3.transfer_funds.rho "-e s/%FROM/$ALICE_REV/ -e s/%TO/lala/"

Notice the platform only checks whether the address is syntactically correct. A typo means the funds are lost.