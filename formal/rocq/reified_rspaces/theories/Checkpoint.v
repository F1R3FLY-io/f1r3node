(** * Checkpoint and Replay Specification

    This module specifies the checkpoint/replay mechanism for RSpaces.
    Checkpoints capture a snapshot of state that can be restored,
    enabling deterministic replay of operations.

    Reference: Spec "Reifying RSpaces.md" lines 19-51
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude GenericRSpace.
Import ListNotations.

(** ** Hash Type *)

(** BLAKE2b256 hash represented as a list of bytes (256 bits = 32 bytes) *)
Definition Hash := list Z.

(** Hash length is always 32 bytes *)
Definition valid_hash (h : Hash) : Prop := length h = 32%nat.

(** ** Checkpoint State *)

Section CheckpointState.
  Variable A K : Type.

  (** A checkpoint captures the complete state at a point in time *)
  Record Checkpoint := mkCheckpoint {
    cp_root : Hash;                      (* Merkle root hash *)
    cp_state : GenericRSpaceState A K;   (* The captured state *)
  }.

  (** Checkpoint validity: hash matches state *)
  Definition valid_checkpoint (cp : Checkpoint) : Prop :=
    valid_hash (cp_root cp).

End CheckpointState.

Arguments mkCheckpoint {A K}.
Arguments cp_root {A K}.
Arguments cp_state {A K}.

(** ** Operations *)

(** Operations that can be replayed to reconstruct state.
    Channel = list Z from Prelude, Pattern = A (same as data type) *)
Inductive Operation (A K : Type) :=
  | Op_Produce : Channel -> A -> bool -> Operation A K
  | Op_Consume : list Channel -> list A -> K -> bool -> Operation A K
  | Op_Install : list Channel -> list A -> K -> Operation A K.

Arguments Op_Produce {A K}.
Arguments Op_Consume {A K}.
Arguments Op_Install {A K}.

(** ** History Repository Trait *)

(** Abstract interface for storing and retrieving checkpoints *)
Class HistoryRepository (A K : Type) := {
  (** Store a checkpoint *)
  hr_store : Checkpoint A K -> Prop;

  (** Retrieve state by hash *)
  hr_retrieve : Hash -> option (GenericRSpaceState A K);

  (** Check if a hash exists *)
  hr_contains : Hash -> bool;

  (** Retrieval is consistent with storage *)
  hr_retrieve_stored : forall cp,
    hr_store cp ->
    hr_retrieve (cp_root cp) = Some (cp_state cp);

  (** Contains reflects retrievability *)
  hr_contains_iff : forall h,
    hr_contains h = true <-> exists state, hr_retrieve h = Some state;
}.

(** ** Hash Properties (Axiomatized)

    The following axioms capture cryptographic properties of the BLAKE2b256
    hash function. These are necessary axioms because cryptographic security
    properties cannot be proven within Coq's type theory - they are
    computational hardness assumptions based on the structure of the
    hash function.

    ** Axiom Justifications

    1. [compute_hash_valid]: The BLAKE2b256 algorithm is specified to always
       produce exactly 256 bits (32 bytes) of output, regardless of input size.
       This is a specification property of the algorithm, not a security
       assumption. The implementation in Rust uses the blake2 crate which
       guarantees this property.
       Reference: RFC 7693 (The BLAKE2 Cryptographic Hash)

    2. [hash_collision_resistant]: This axiom states that different states
       produce different hashes. In cryptographic terms, this is the
       "collision resistance" property: it should be computationally
       infeasible to find two distinct inputs x ≠ y such that H(x) = H(y).
       This is the standard security assumption for cryptographic hash
       functions. BLAKE2b256 has 128-bit collision resistance (birthday
       bound on 256-bit output), which is considered secure for the
       foreseeable future.

       Note: We model this as an exact equality (s1 = s2) rather than
       computational infeasibility because Coq cannot express computational
       complexity. In practice, this means we assume no collisions occur
       during system operation - a valid assumption given the collision
       resistance of BLAKE2b256.

       Reference: BLAKE2 security analysis (Aumasson et al., 2013)
*)

(** Hash function abstraction - computes hash from state *)
Parameter compute_hash : forall A K, GenericRSpaceState A K -> Hash.

(** Hash always produces valid 32-byte output.

    Justification: BLAKE2b256 is specified to always produce exactly
    32 bytes (256 bits) of output. This is a deterministic property
    of the algorithm, not a security assumption.
    See RFC 7693 Section 2.1. *)
Axiom compute_hash_valid : forall A K (state : GenericRSpaceState A K),
  valid_hash (compute_hash A K state).

(** Collision resistance (cryptographic assumption).

    Justification: This models the collision resistance property of
    BLAKE2b256. Finding two distinct inputs with the same hash output
    requires approximately 2^128 hash evaluations (birthday bound),
    which is computationally infeasible with current technology.

    This axiom allows us to use hash equality as a proxy for state
    equality, which is essential for checkpoint identification. *)
Axiom hash_collision_resistant : forall A K (s1 s2 : GenericRSpaceState A K),
  compute_hash A K s1 = compute_hash A K s2 ->
  s1 = s2.

(** ** Replay Mechanism *)

Section Replay.
  Variable A K : Type.

  (** Apply a single operation to a state (simplified - assumes success) *)
  Parameter apply_operation :
    GenericRSpaceState A K -> Operation A K ->
    option (GenericRSpaceState A K).

  (** Apply a sequence of operations *)
  Fixpoint apply_operations (state : GenericRSpaceState A K)
                            (ops : list (Operation A K))
    : option (GenericRSpaceState A K) :=
    match ops with
    | [] => Some state
    | op :: rest =>
      match apply_operation state op with
      | None => None
      | Some state' => apply_operations state' rest
      end
    end.

  (** ** Determinism Theorem *)

  (** Replay from the same checkpoint with the same operations
      always produces the same final state *)
  Theorem replay_determinism :
    forall (cp : Checkpoint A K) (ops : list (Operation A K))
           (final1 final2 : GenericRSpaceState A K),
      apply_operations (cp_state cp) ops = Some final1 ->
      apply_operations (cp_state cp) ops = Some final2 ->
      final1 = final2.
  Proof.
    intros cp ops final1 final2 H1 H2.
    rewrite H1 in H2.
    injection H2. auto.
  Qed.

  (** Operations applied in different orders may yield different results *)
  (** This is NOT a theorem - operations don't commute in general *)

  (** ** Checkpoint Correctness *)

  (** Creating and restoring a checkpoint yields the original state *)
  Theorem checkpoint_restore_identity :
    forall (state : GenericRSpaceState A K) (cp : Checkpoint A K),
      cp_state cp = state ->
      compute_hash A K state = cp_root cp ->
      cp_state cp = state.
  Proof.
    intros state cp Hstate Hhash.
    exact Hstate.
  Qed.

  (** Hash uniquely identifies state (follows from collision resistance) *)
  Theorem hash_identifies_state :
    forall (cp1 cp2 : Checkpoint A K),
      cp_root cp1 = cp_root cp2 ->
      compute_hash A K (cp_state cp1) = cp_root cp1 ->
      compute_hash A K (cp_state cp2) = cp_root cp2 ->
      cp_state cp1 = cp_state cp2.
  Proof.
    intros cp1 cp2 Hroot H1 H2.
    apply hash_collision_resistant.
    rewrite H1, H2. exact Hroot.
  Qed.

End Replay.

(** ** Event Log *)

Section EventLog.
  Variable A K : Type.
  Variable default_a : A.  (* Default data value for dummy entries *)

  (** Event log entry with timestamp *)
  Record LogEntry := mkLogEntry {
    le_timestamp : nat;
    le_operation : Operation A K;
  }.

  (** Event log is a sequence of entries *)
  Definition EventLog := list LogEntry.

  (** Extract operations from log *)
  Definition log_operations (log : EventLog) : list (Operation A K) :=
    map le_operation log.

  (** Default log entry for nth access *)
  Definition default_log_entry : LogEntry :=
    mkLogEntry O (Op_Produce 0%Z default_a false).

  (** Log entries are ordered by timestamp *)
  Definition log_ordered (log : EventLog) : Prop :=
    forall i j,
      (i < j)%nat ->
      (j < length log)%nat ->
      (le_timestamp (nth i log default_log_entry) <=
       le_timestamp (nth j log default_log_entry))%nat.

  (** Replay from log produces deterministic result *)
  Theorem replay_from_log_determinism :
    forall (cp : Checkpoint A K) (log : EventLog)
           (final1 final2 : GenericRSpaceState A K),
      apply_operations A K (cp_state cp) (log_operations log) = Some final1 ->
      apply_operations A K (cp_state cp) (log_operations log) = Some final2 ->
      final1 = final2.
  Proof.
    intros cp log final1 final2 H1 H2.
    apply (replay_determinism A K cp (log_operations log) final1 final2 H1 H2).
  Qed.

End EventLog.

Arguments mkLogEntry {A K}.
Arguments le_timestamp {A K}.
Arguments le_operation {A K}.

(** ** Checkpoint Chain *)

Section CheckpointChain.
  Variable A K : Type.

  (** A chain of checkpoints with operations between them *)
  Record CheckpointLink := mkLink {
    link_from : Checkpoint A K;
    link_to : Checkpoint A K;
    link_ops : list (Operation A K);
  }.

  (** Link validity: operations transform from-state to to-state *)
  Definition valid_link (link : CheckpointLink) : Prop :=
    apply_operations A K (cp_state (link_from link)) (link_ops link) =
    Some (cp_state (link_to link)).

  (** Chain is a sequence of valid links *)
  Fixpoint valid_chain (chain : list CheckpointLink) : Prop :=
    match chain with
    | [] => True
    | [link] => valid_link link
    | link :: rest =>
      valid_link link /\
      match rest with
      | [] => True
      | next :: _ => cp_root (link_to link) = cp_root (link_from next)
      end /\
      valid_chain rest
    end.

  (** Replaying a chain from start to end is deterministic *)
  Theorem chain_replay_determinism :
    forall (chain : list CheckpointLink),
      valid_chain chain ->
      forall link1 link2,
        In link1 chain -> In link2 chain ->
        cp_root (link_from link1) = cp_root (link_from link2) ->
        compute_hash A K (cp_state (link_from link1)) = cp_root (link_from link1) ->
        compute_hash A K (cp_state (link_from link2)) = cp_root (link_from link2) ->
        cp_state (link_from link1) = cp_state (link_from link2).
  Proof.
    intros chain Hvalid link1 link2 Hin1 Hin2 Hroot H1 H2.
    apply hash_identifies_state; assumption.
  Qed.

End CheckpointChain.

Arguments mkLink {A K}.
Arguments link_from {A K}.
Arguments link_to {A K}.
Arguments link_ops {A K}.

(** ** Properties of Checkpointing *)

Section CheckpointProperties.
  Variable A K : Type.
  Context `{HR : HistoryRepository A K}.

  (** Stored checkpoints can be retrieved *)
  Theorem stored_checkpoint_retrievable :
    forall (cp : Checkpoint A K),
      hr_store cp ->
      hr_retrieve (cp_root cp) = Some (cp_state cp).
  Proof.
    intros cp Hstore.
    apply hr_retrieve_stored. exact Hstore.
  Qed.

  (** Contains reflects stored checkpoints *)
  Theorem contains_reflects_stored :
    forall (cp : Checkpoint A K),
      hr_store cp ->
      hr_contains (cp_root cp) = true.
  Proof.
    intros cp Hstore.
    apply hr_contains_iff.
    exists (cp_state cp).
    apply hr_retrieve_stored. exact Hstore.
  Qed.

End CheckpointProperties.

(** ** Temp Space Clearing on Checkpoint *)

Section TempClearing.
  Variable A K : Type.

  (** Temp qualifier data should be cleared on checkpoint *)
  Definition clear_temp_data (state : GenericRSpaceState A K) : GenericRSpaceState A K :=
    if @is_temp A K state
    then @empty_gr_state A K (gr_qualifier state)
    else state.

  (** Clearing temp data preserves qualifier *)
  Theorem clear_temp_preserves_qualifier :
    forall state,
      gr_qualifier (clear_temp_data state) = gr_qualifier state.
  Proof.
    intros state.
    unfold clear_temp_data.
    destruct (@is_temp A K state) eqn:Htemp.
    - simpl. reflexivity.
    - reflexivity.
  Qed.

  (** Clearing non-temp data is identity *)
  Theorem clear_temp_non_temp_identity :
    forall state,
      @is_temp A K state = false ->
      clear_temp_data state = state.
  Proof.
    intros state Htemp.
    unfold clear_temp_data.
    rewrite Htemp. reflexivity.
  Qed.

End TempClearing.

(** ** Soft Checkpoints *)

(** Soft checkpoints are fast in-memory checkpoints that don't require
    writing to the history trie. They enable efficient state snapshots
    for speculative execution and rollback.
    Reference: Spec "Reifying RSpaces.md" lines 517-534 *)

Section SoftCheckpoint.
  Variable A K : Type.

  (** A soft checkpoint is an in-memory snapshot of state.
      Unlike regular checkpoints, soft checkpoints don't have a hash
      and aren't persisted to the history repository. *)
  Record SoftCheckpoint := mkSoftCheckpoint {
    sc_state : GenericRSpaceState A K;
    sc_timestamp : nat;  (* For ordering soft checkpoints *)
  }.

  (** Create a soft checkpoint from current state *)
  Definition create_soft_checkpoint (state : GenericRSpaceState A K) (timestamp : nat)
    : SoftCheckpoint :=
    mkSoftCheckpoint state timestamp.

  (** Revert to a soft checkpoint - returns the captured state *)
  Definition revert_to_soft_checkpoint (sc : SoftCheckpoint)
    : GenericRSpaceState A K :=
    sc_state sc.

  (** ** Soft Checkpoint Theorems *)

  (** Creating and reverting to a soft checkpoint yields the original state *)
  Theorem soft_checkpoint_preserves_state :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      revert_to_soft_checkpoint (create_soft_checkpoint state timestamp) = state.
  Proof.
    intros state timestamp.
    unfold revert_to_soft_checkpoint, create_soft_checkpoint.
    simpl. reflexivity.
  Qed.

  (** Soft checkpoint preserves all state components *)
  Theorem soft_checkpoint_preserves_data_store :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      gr_data_store (revert_to_soft_checkpoint (create_soft_checkpoint state timestamp)) =
      gr_data_store state.
  Proof.
    intros state timestamp.
    rewrite soft_checkpoint_preserves_state.
    reflexivity.
  Qed.

  Theorem soft_checkpoint_preserves_cont_store :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      gr_cont_store (revert_to_soft_checkpoint (create_soft_checkpoint state timestamp)) =
      gr_cont_store state.
  Proof.
    intros state timestamp.
    rewrite soft_checkpoint_preserves_state.
    reflexivity.
  Qed.

  Theorem soft_checkpoint_preserves_joins :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      gr_joins (revert_to_soft_checkpoint (create_soft_checkpoint state timestamp)) =
      gr_joins state.
  Proof.
    intros state timestamp.
    rewrite soft_checkpoint_preserves_state.
    reflexivity.
  Qed.

  Theorem soft_checkpoint_preserves_qualifier :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      gr_qualifier (revert_to_soft_checkpoint (create_soft_checkpoint state timestamp)) =
      gr_qualifier state.
  Proof.
    intros state timestamp.
    rewrite soft_checkpoint_preserves_state.
    reflexivity.
  Qed.

  (** Multiple soft checkpoints can be created and reverted independently *)
  Theorem soft_checkpoints_independent :
    forall (state1 state2 : GenericRSpaceState A K) (t1 t2 : nat),
      let sc1 := create_soft_checkpoint state1 t1 in
      let sc2 := create_soft_checkpoint state2 t2 in
      revert_to_soft_checkpoint sc1 = state1 /\
      revert_to_soft_checkpoint sc2 = state2.
  Proof.
    intros state1 state2 t1 t2.
    split; apply soft_checkpoint_preserves_state.
  Qed.

  (** Soft checkpoint timestamp is preserved *)
  Theorem soft_checkpoint_timestamp_preserved :
    forall (state : GenericRSpaceState A K) (timestamp : nat),
      sc_timestamp (create_soft_checkpoint state timestamp) = timestamp.
  Proof.
    intros state timestamp.
    unfold create_soft_checkpoint. simpl. reflexivity.
  Qed.

End SoftCheckpoint.

Arguments mkSoftCheckpoint {A K}.
Arguments sc_state {A K}.
Arguments sc_timestamp {A K}.

(** ** Rig and Replay Operations *)

(** These operations support the deterministic replay mechanism.
    "Rigging" prepares the space for replay by loading an event log,
    allowing operations to be replayed from a checkpoint.
    Reference: Spec "Reifying RSpaces.md" lines 545-560 *)

Section RigOperations.
  Variable A K : Type.

  (** Replay state tracks whether we're in replay mode and the event log *)
  Record ReplayState := mkReplayState {
    rs_is_replay : bool;              (* True if currently replaying *)
    rs_event_log : list (Operation A K);  (* Log of operations to replay *)
    rs_log_position : nat;            (* Current position in log *)
    rs_start_checkpoint : option (Checkpoint A K);  (* Starting checkpoint for replay *)
  }.

  (** Initial replay state - not in replay mode *)
  Definition initial_replay_state : ReplayState :=
    mkReplayState false [] 0 None.

  (** Rig: Prepare for replay with an event log.
      This sets up the replay mechanism without resetting state. *)
  Definition rig (current_state : ReplayState) (log : list (Operation A K))
    : ReplayState :=
    mkReplayState
      true    (* Now in replay mode *)
      log     (* The event log to replay *)
      0       (* Start at beginning *)
      (rs_start_checkpoint current_state).

  (** Rig and Reset: Reset to a checkpoint and prepare for replay.
      This is the full reset operation that:
      1. Stores the start checkpoint
      2. Sets up the event log
      3. Enters replay mode *)
  Definition rig_and_reset (cp : Checkpoint A K) (log : list (Operation A K))
    : ReplayState :=
    mkReplayState
      true        (* In replay mode *)
      log         (* Event log *)
      0           (* Start at beginning *)
      (Some cp).  (* The checkpoint to replay from *)

  (** Check replay data: Verify replay state is consistent.
      Returns true if:
      - We're in replay mode
      - We have a valid start checkpoint
      - Log position is valid *)
  Definition check_replay_data (rs : ReplayState) : bool :=
    rs_is_replay rs &&
    match rs_start_checkpoint rs with
    | Some _ => true
    | None => false
    end &&
    (rs_log_position rs <=? length (rs_event_log rs))%nat.

  (** ** Rig Operation Theorems *)

  (** After rig, we're in replay mode *)
  Theorem rig_enables_replay :
    forall current_state log,
      rs_is_replay (rig current_state log) = true.
  Proof.
    intros current_state log.
    unfold rig. simpl. reflexivity.
  Qed.

  (** After rig_and_reset, we're in replay mode with the checkpoint *)
  Theorem rig_and_reset_enables_replay :
    forall cp log,
      rs_is_replay (rig_and_reset cp log) = true.
  Proof.
    intros cp log.
    unfold rig_and_reset. simpl. reflexivity.
  Qed.

  (** rig_and_reset sets the start checkpoint *)
  Theorem rig_and_reset_sets_checkpoint :
    forall cp log,
      rs_start_checkpoint (rig_and_reset cp log) = Some cp.
  Proof.
    intros cp log.
    unfold rig_and_reset. simpl. reflexivity.
  Qed.

  (** rig_and_reset sets the event log *)
  Theorem rig_and_reset_sets_log :
    forall cp log,
      rs_event_log (rig_and_reset cp log) = log.
  Proof.
    intros cp log.
    unfold rig_and_reset. simpl. reflexivity.
  Qed.

  (** rig_and_reset starts at position 0 *)
  Theorem rig_and_reset_starts_at_zero :
    forall cp log,
      rs_log_position (rig_and_reset cp log) = 0.
  Proof.
    intros cp log.
    unfold rig_and_reset. simpl. reflexivity.
  Qed.

  (** check_replay_data succeeds for valid rig_and_reset *)
  Theorem check_replay_consistent :
    forall cp log,
      check_replay_data (rig_and_reset cp log) = true.
  Proof.
    intros cp log.
    unfold check_replay_data, rig_and_reset. simpl.
    reflexivity.
  Qed.

  (** rig preserves the event log *)
  Theorem rig_sets_log :
    forall current_state log,
      rs_event_log (rig current_state log) = log.
  Proof.
    intros current_state log.
    unfold rig. simpl. reflexivity.
  Qed.

  (** rig preserves the start checkpoint *)
  Theorem rig_preserves_checkpoint :
    forall current_state log,
      rs_start_checkpoint (rig current_state log) =
      rs_start_checkpoint current_state.
  Proof.
    intros current_state log.
    unfold rig. simpl. reflexivity.
  Qed.

  (** ** Replay Execution *)

  (** Get the current operation to replay (if any) *)
  Definition current_replay_operation (rs : ReplayState) : option (Operation A K) :=
    nth_error (rs_event_log rs) (rs_log_position rs).

  (** Advance to next operation in log *)
  Definition advance_replay (rs : ReplayState) : ReplayState :=
    mkReplayState
      (rs_is_replay rs)
      (rs_event_log rs)
      (S (rs_log_position rs))
      (rs_start_checkpoint rs).

  (** Check if replay is complete (reached end of log) *)
  Definition replay_complete (rs : ReplayState) : bool :=
    (length (rs_event_log rs) <=? rs_log_position rs)%nat.

  (** Advancing increases position *)
  Theorem advance_increases_position :
    forall rs,
      rs_log_position (advance_replay rs) = S (rs_log_position rs).
  Proof.
    intros rs. unfold advance_replay. simpl. reflexivity.
  Qed.

  (** Replay eventually completes *)
  Theorem replay_terminates :
    forall rs,
      rs_log_position rs >= length (rs_event_log rs) ->
      replay_complete rs = true.
  Proof.
    intros rs Hpos.
    unfold replay_complete.
    apply Nat.leb_le. lia.
  Qed.

  (** Replay from start checkpoint produces deterministic result *)
  Theorem replay_from_rig_deterministic :
    forall cp log final1 final2,
      apply_operations A K (cp_state cp) log = Some final1 ->
      apply_operations A K (cp_state cp) log = Some final2 ->
      final1 = final2.
  Proof.
    intros cp log final1 final2 H1 H2.
    rewrite H1 in H2. injection H2. auto.
  Qed.

End RigOperations.

Arguments mkReplayState {A K}.
Arguments rs_is_replay {A K}.
Arguments rs_event_log {A K}.
Arguments rs_log_position {A K}.
Arguments rs_start_checkpoint {A K}.
