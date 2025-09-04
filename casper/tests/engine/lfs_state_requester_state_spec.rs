// See casper/src/test/scala/coop/rchain/casper/engine/LfsStateRequesterStateSpec.scala

use casper::rust::engine::lfs_tuple_space_requester::ST;
use std::collections::HashSet;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_next_should_return_empty_list_when_called_again() {
        let st = ST::new(vec![10]);

        // Calling next should produce initial set
        let (st1, ids1) = st.get_next(false);

        assert_eq!(ids1, vec![10]);

        // Calling next again should NOT return new items
        let (st2, ids2) = st1.get_next(false);

        assert_eq!(ids2, Vec::<i32>::new());

        assert_eq!(st1, st2);
    }

    #[test]
    fn get_next_should_return_new_items_after_add() {
        let st = ST::new(vec![10]);

        // Add new items
        let st2 = st.add({
            let mut set = HashSet::new();
            set.insert(9);
            set.insert(8);
            set
        });

        // Calling next should return new items
        let (_, ids2) = st2.get_next(false);

        // Note: Order may differ since we use HashMap, so we'll check contents
        let mut expected_ids = vec![10, 9, 8];
        let mut actual_ids = ids2.clone();
        expected_ids.sort();
        actual_ids.sort();
        assert_eq!(actual_ids, expected_ids);
    }

    #[test]
    fn get_next_should_return_requested_items_on_resend() {
        let st = ST::new(vec![10]);

        // Calling next should return new items
        let (st1, ids1) = st.get_next(false);

        assert_eq!(ids1, vec![10]);

        // Calling next with resend should return already requested
        let (_, ids2) = st1.get_next(true);

        assert_eq!(ids2, vec![10]);
    }

    #[test]
    fn received_should_return_true_for_requested_and_false_for_unknown() {
        let st = ST::new(vec![10]);

        // Mark next as requested
        let (st1, _) = st.get_next(false);

        // Received requested item
        let (_, is_received_true) = st1.received(10);

        assert_eq!(is_received_true, true);

        // Received unknown item
        let (_, is_received_false) = st1.received(100);

        assert_eq!(is_received_false, false);
    }

    #[test]
    fn done_should_make_state_finished() {
        let st = ST::new(vec![10]);

        // If item is not received, it should stay unfinished
        let st1 = st.done(10);

        assert_eq!(st1.is_finished(), false);

        // Mark next as requested ...
        let (st2, _) = st1.get_next(false);

        // ... and received
        let (st3, _) = st2.received(10);

        let st4 = st3.done(10);

        assert_eq!(st4.is_finished(), true);
    }

    #[test]
    fn from_start_to_finish_should_receive_one_item() {
        let st = ST::new(vec![10]);

        // Calling next should produce initial set
        let (st1, ids1) = st.get_next(false);

        assert_eq!(ids1, vec![10]);

        // Calling next again should NOT return new items
        let (st2, ids2) = st1.get_next(false);

        assert_eq!(ids2, Vec::<i32>::new());

        // It should not be finished until all items are Done
        assert_eq!(st2.is_finished(), false);

        // Received first item
        let (st3, is_received) = st2.received(10);

        assert_eq!(is_received, true);

        let st4 = st3.done(10);

        // Return finished when all items as Done
        assert_eq!(st4.is_finished(), true);
    }
}
