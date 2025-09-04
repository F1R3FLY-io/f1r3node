// See casper/src/test/scala/coop/rchain/casper/engine/LfsBlockRequesterStateSpec.scala

use casper::rust::engine::lfs_block_requester::ST;
use std::collections::HashSet;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_next_should_return_empty_list_when_called_again() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            None,
            None,
        );

        // Calling next should produce initial set
        let (st1, ids1) = st.get_next(false);

        let mut expected_ids1 = HashSet::new();
        expected_ids1.insert(10);
        assert_eq!(ids1, expected_ids1);

        // Calling next again should NOT return new items
        let (st2, ids2) = st1.get_next(false);

        assert_eq!(ids2, HashSet::new());

        // States should be equal (st1 == st2)
        assert_eq!(st1.d, st2.d);
        assert_eq!(st1.latest, st2.latest);
        assert_eq!(st1.lower_bound, st2.lower_bound);
        assert_eq!(st1.height_map, st2.height_map);
        assert_eq!(st1.finished, st2.finished);
    }

    #[test]
    fn get_next_should_return_new_items_after_add() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            None,
            None,
        );

        // Add new items
        let st2 = st.add({
            let mut set = HashSet::new();
            set.insert(9);
            set.insert(8);
            set
        });

        // Calling next should return new items
        let (_, ids2) = st2.get_next(false);

        let mut expected_ids = HashSet::new();
        expected_ids.insert(10);
        expected_ids.insert(9);
        expected_ids.insert(8);
        assert_eq!(ids2, expected_ids);
    }

    #[test]
    fn get_next_should_return_requested_items_on_resend() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            None,
            None,
        );

        // Calling next should return new items
        let (st1, ids1) = st.get_next(false);

        let mut expected_ids1 = HashSet::new();
        expected_ids1.insert(10);
        assert_eq!(ids1, expected_ids1);

        // Calling next with resend should return already requested
        let (_, ids2) = st1.get_next(true);

        let mut expected_ids2 = HashSet::new();
        expected_ids2.insert(10);
        assert_eq!(ids2, expected_ids2);
    }

    #[test]
    fn received_should_return_true_for_requested_and_false_for_unknown() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            None,
            None,
        );

        // Mark next as requested
        let (st1, _) = st.get_next(false);

        // Received requested item
        let (_, receive_info) = st1.received(10, 100, None);
        assert_eq!(receive_info.requested, true);

        // Received unknown item
        let (_, receive_info1) = st1.received(100, 200, None);
        assert_eq!(receive_info1.requested, false);
    }

    #[test]
    fn received_should_return_flag_based_on_calculated_height() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set.insert(11);
                set
            },
            Some({
                let mut set = HashSet::new();
                set.insert(10);
                set
            }),
            Some(200),
        );

        // Mark next as requested
        let (st1, _) = st.get_next(false);

        // Received the last latest item (sets minimum height)
        let (st2, receive_info1) = st1.received(10, 100, None);
        assert_eq!(receive_info1.requested, true);
        assert_eq!(receive_info1.latest, true);
        assert_eq!(receive_info1.lastlatest, true);

        // Minimum height should be recalculated based on the last latest item (-1)
        assert_eq!(st2.lower_bound, 99);

        // Mark next as requested
        let (st3, ids2) = st2.get_next(false);
        let mut expected_ids2 = HashSet::new();
        expected_ids2.insert(11);
        assert_eq!(ids2, expected_ids2);

        // Received higher height should be accepted
        let (st4, receive_info3) = st3.received(11, 50, None);
        assert_eq!(receive_info3.requested, true);

        // Minimum height should stay the same after all latest items received
        assert_eq!(st4.lower_bound, 99);
    }

    #[test]
    fn received_should_return_next_only_after_latest_are_received() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set.insert(11);
                set.insert(12);
                set
            },
            Some({
                let mut set = HashSet::new();
                set.insert(10);
                set.insert(11);
                set
            }),
            None,
        );

        // Mark next as requested
        let (st1, _) = st.get_next(false);

        // Received latest item
        let (st2, receive_info) = st1.received(10, 100, None);
        assert_eq!(receive_info.requested, true);
        assert_eq!(receive_info.latest, true);
        assert_eq!(receive_info.lastlatest, false);

        // Before all latest received, next should be empty
        let (st3, ids1) = st2.get_next(false);
        assert_eq!(ids1, HashSet::new());

        // Received latest item (the last one)
        let (st4, receive_info1) = st3.received(11, 110, None);
        assert_eq!(receive_info1.requested, true);
        assert_eq!(receive_info1.latest, true);
        assert_eq!(receive_info1.lastlatest, true);

        // After the last of latest received, the rest of items should be requested
        let (_, ids4) = st4.get_next(false);
        let mut expected_ids4 = HashSet::new();
        expected_ids4.insert(12);
        assert_eq!(ids4, expected_ids4);
    }

    #[test]
    fn done_should_make_state_finished() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            None,
            None,
        );

        // If item is not received, it should stay unfinished
        let st1 = st.done(10);
        assert_eq!(st1.is_finished(), false);

        // Mark next as requested ...
        let (st2, _) = st1.get_next(false);
        // ... and received
        let (st3, _) = st2.received(10, 100, None);

        let st4 = st3.done(10);
        assert_eq!(st4.is_finished(), true);
    }

    #[test]
    fn from_start_to_finish_should_receive_one_item() {
        let st = ST::new(
            {
                let mut set = HashSet::new();
                set.insert(10);
                set
            },
            Some({
                let mut set = HashSet::new();
                set.insert(10);
                set
            }),
            None,
        );

        // Calling next should produce initial set
        let (st1, ids1) = st.get_next(false);
        let mut expected_ids1 = HashSet::new();
        expected_ids1.insert(10);
        assert_eq!(ids1, expected_ids1);

        // Calling next again should NOT return new items
        let (st2, ids2) = st1.get_next(false);
        assert_eq!(ids2, HashSet::new());

        // It should not be finished until all items are Done
        assert_eq!(st2.is_finished(), false);

        // Received first item
        let (st3, receive_info) = st2.received(10, 100, None);
        assert_eq!(receive_info.requested, true);

        let st4 = st3.done(10);

        // Return finished when all items as Done
        assert_eq!(st4.is_finished(), true);
    }
}
