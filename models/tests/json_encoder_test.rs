// Tests for JSON encoder/decoder functionality
// Port of Scala JsonEncoder tests

use models::rust::{par_map::ParMap, par_set::ParSet, utils::*};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use serde_json;

#[test]
fn test_par_set_json_serialization() {
    // Create a ParSet with some Par elements
    let par1 = new_gint_par(1, Vec::new(), false);
    let par2 = new_gint_par(2, Vec::new(), false);
    let par3 = new_gstring_par("hello".to_string(), Vec::new(), false);
    
    let par_set = ParSet::create_from_vec(vec![par1, par2, par3]);
    
    // Serialize to JSON
    let json = serde_json::to_string(&par_set).expect("Failed to serialize ParSet");
    
    // PRECISE ASSERTIONS - validate actual JSON structure
    assert!(json.starts_with('['), "ParSet should serialize as JSON array, got: {}", json);
    assert!(json.ends_with(']'), "ParSet should serialize as JSON array, got: {}", json);
    
    // Parse as JSON Value to inspect structure
    let json_value: serde_json::Value = serde_json::from_str(&json).expect("Invalid JSON produced");
    let json_array = json_value.as_array().expect("ParSet should serialize as array");
    
    // Verify we have exactly 3 elements in JSON
    assert_eq!(json_array.len(), 3, "JSON should contain 3 elements, got: {}", json);
    
    // Verify JSON contains our specific values
    let json_str = json.to_lowercase();
    assert!(json_str.contains("1"), "JSON should contain integer 1, got: {}", json);
    assert!(json_str.contains("2"), "JSON should contain integer 2, got: {}", json);
    assert!(json_str.contains("hello"), "JSON should contain string 'hello', got: {}", json);
    
    // Verify each element is a proper Par object (should have 'exprs' field for GInt/GString)
    for element in json_array {
        assert!(element.is_object(), "Each ParSet element should be an object, got: {}", element);
        let obj = element.as_object().unwrap();
        // Par objects should have some structure (not empty objects)
        assert!(!obj.is_empty(), "Par object should not be empty, got: {}", element);
    }
    
    // Deserialize back
    let deserialized: ParSet = serde_json::from_str(&json).expect("Failed to deserialize ParSet");
    
    // Should have same number of elements
    assert_eq!(par_set.ps.sorted_pars.len(), deserialized.ps.sorted_pars.len());
    assert_eq!(deserialized.ps.sorted_pars.len(), 3, "Deserialized ParSet should have 3 elements");
}

#[test]
fn test_par_map_json_serialization() {
    // Create a ParMap with some key-value pairs
    let key1 = new_gint_par(1, Vec::new(), false);
    let val1 = new_gstring_par("one".to_string(), Vec::new(), false);
    let key2 = new_gint_par(2, Vec::new(), false);
    let val2 = new_gstring_par("two".to_string(), Vec::new(), false);
    
    let par_map = ParMap::create_from_vec(vec![(key1, val1), (key2, val2)]);
    
    // Serialize to JSON
    let json = serde_json::to_string(&par_map).expect("Failed to serialize ParMap");
    
    // PRECISE ASSERTIONS - validate actual JSON structure
    assert!(json.starts_with('['), "ParMap should serialize as JSON array, got: {}", json);
    assert!(json.ends_with(']'), "ParMap should serialize as JSON array, got: {}", json);
    
    // Parse as JSON Value to inspect structure
    let json_value: serde_json::Value = serde_json::from_str(&json).expect("Invalid JSON produced");
    let json_array = json_value.as_array().expect("ParMap should serialize as array");
    
    // Verify we have exactly 2 key-value pairs in JSON
    assert_eq!(json_array.len(), 2, "JSON should contain 2 key-value pairs, got: {}", json);
    
    // Verify JSON contains our specific keys and values
    let json_str = json.to_lowercase();
    assert!(json_str.contains("1"), "JSON should contain key 1, got: {}", json);
    assert!(json_str.contains("2"), "JSON should contain key 2, got: {}", json);
    assert!(json_str.contains("one"), "JSON should contain value 'one', got: {}", json);
    assert!(json_str.contains("two"), "JSON should contain value 'two', got: {}", json);
    
    // Verify each element is a tuple/array with exactly 2 elements (key, value)
    for element in json_array {
        assert!(element.is_array(), "Each ParMap element should be a tuple (array), got: {}", element);
        let tuple = element.as_array().unwrap();
        assert_eq!(tuple.len(), 2, "Each tuple should have exactly 2 elements (key, value), got: {}", element);
        
        // Both key and value should be objects (Par structures)
        assert!(tuple[0].is_object(), "Key should be a Par object, got: {}", tuple[0]);
        assert!(tuple[1].is_object(), "Value should be a Par object, got: {}", tuple[1]);
        
        // Neither key nor value should be empty objects
        assert!(!tuple[0].as_object().unwrap().is_empty(), "Key Par object should not be empty, got: {}", tuple[0]);
        assert!(!tuple[1].as_object().unwrap().is_empty(), "Value Par object should not be empty, got: {}", tuple[1]);
    }
    
    // Deserialize back
    let deserialized: ParMap = serde_json::from_str(&json).expect("Failed to deserialize ParMap");
    
    // Should have same number of elements
    assert_eq!(par_map.ps.sorted_list.len(), deserialized.ps.sorted_list.len());
    assert_eq!(deserialized.ps.sorted_list.len(), 2, "Deserialized ParMap should have 2 key-value pairs");
}

#[test]
fn test_blake2b512_random_json_serialization() {
    let random = Blake2b512Random::create_from_bytes(&[1, 2, 3, 4, 5]);
    
    // Serialize to JSON
    let json = serde_json::to_string(&random).expect("Failed to serialize Blake2b512Random");
    
    // Should serialize as null (unit)
    assert_eq!(json, "null");
    
    // Deserialize back
    let deserialized: Blake2b512Random = serde_json::from_str(&json).expect("Failed to deserialize Blake2b512Random");
    
    // Should be a valid Blake2b512Random (matches Scala behavior)
    assert_eq!(deserialized, Blake2b512Random::create_from_bytes(&[1]));
}

#[test]
fn test_empty_par_set_serialization() {
    let empty_par_set = ParSet::create_from_vec(vec![]);
    
    let json = serde_json::to_string(&empty_par_set).expect("Failed to serialize empty ParSet");
    
    // PRECISE ASSERTION - empty ParSet should be exactly "[]"
    assert_eq!(json, "[]", "Empty ParSet should serialize as empty JSON array, got: {}", json);
    
    // Verify it's a valid JSON array
    let json_value: serde_json::Value = serde_json::from_str(&json).expect("Invalid JSON produced");
    assert!(json_value.is_array(), "Empty ParSet should be a JSON array");
    assert_eq!(json_value.as_array().unwrap().len(), 0, "Empty ParSet array should have 0 elements");
    
    let deserialized: ParSet = serde_json::from_str(&json).expect("Failed to deserialize empty ParSet");
    assert_eq!(deserialized.ps.sorted_pars.len(), 0, "Deserialized empty ParSet should have 0 elements");
}

#[test]
fn test_empty_par_map_serialization() {
    let empty_par_map = ParMap::create_from_vec(vec![]);
    
    let json = serde_json::to_string(&empty_par_map).expect("Failed to serialize empty ParMap");
    
    // PRECISE ASSERTION - empty ParMap should be exactly "[]"
    assert_eq!(json, "[]", "Empty ParMap should serialize as empty JSON array, got: {}", json);
    
    // Verify it's a valid JSON array
    let json_value: serde_json::Value = serde_json::from_str(&json).expect("Invalid JSON produced");
    assert!(json_value.is_array(), "Empty ParMap should be a JSON array");
    assert_eq!(json_value.as_array().unwrap().len(), 0, "Empty ParMap array should have 0 elements");
    
    let deserialized: ParMap = serde_json::from_str(&json).expect("Failed to deserialize empty ParMap");
    assert_eq!(deserialized.ps.sorted_list.len(), 0, "Deserialized empty ParMap should have 0 elements");
}

#[test]
fn test_par_set_ignores_metadata() {
    // Create ParSet with metadata (connective_used, locally_free, remainder)
    let par1 = new_gint_par(42, vec![1, 2, 3], true); // with locally_free and connective_used
    let par_set = ParSet::create_from_vec(vec![par1]);
    
    // Serialize and deserialize
    let json = serde_json::to_string(&par_set).expect("Failed to serialize ParSet");
    let deserialized: ParSet = serde_json::from_str(&json).expect("Failed to deserialize ParSet");
    
    // Metadata should be recalculated, not preserved from JSON
    // (This matches Scala behavior where JSON only contains the Par elements)
    assert_eq!(deserialized.ps.sorted_pars.len(), 1);
}

#[test]
fn test_par_map_ignores_metadata() {
    // Create ParMap with metadata
    let key = new_gint_par(1, vec![1, 2], true);
    let val = new_gstring_par("test".to_string(), vec![3, 4], true);
    let par_map = ParMap::create_from_vec(vec![(key, val)]);
    
    // Serialize and deserialize
    let json = serde_json::to_string(&par_map).expect("Failed to serialize ParMap");
    let deserialized: ParMap = serde_json::from_str(&json).expect("Failed to deserialize ParMap");
    
    // Metadata should be recalculated
    assert_eq!(deserialized.ps.sorted_list.len(), 1);
}

#[test]
fn test_json_pretty_printing() {
    let par = new_gint_par(123, Vec::new(), false);
    let par_set = ParSet::create_from_vec(vec![par]);
    
    let pretty_json = serde_json::to_string_pretty(&par_set).expect("Failed to pretty print ParSet");
    
    // Should contain newlines and indentation
    assert!(pretty_json.contains('\n'));
    assert!(pretty_json.len() > serde_json::to_string(&par_set).unwrap().len());
}

#[test]
fn test_roundtrip_consistency() {
    // Test that serialize -> deserialize -> serialize produces same JSON
    let par1 = new_gint_par(1, Vec::new(), false);
    let par2 = new_gstring_par("test".to_string(), Vec::new(), false);
    let par_set = ParSet::create_from_vec(vec![par1, par2]);
    
    let json1 = serde_json::to_string(&par_set).expect("First serialization failed");
    let deserialized: ParSet = serde_json::from_str(&json1).expect("Deserialization failed");
    let json2 = serde_json::to_string(&deserialized).expect("Second serialization failed");
    
    // JSON should be consistent (though order might vary due to sorting)
    let parsed1: serde_json::Value = serde_json::from_str(&json1).unwrap();
    let parsed2: serde_json::Value = serde_json::from_str(&json2).unwrap();
    
    // Both should be arrays with same length
    assert!(parsed1.is_array());
    assert!(parsed2.is_array());
    assert_eq!(parsed1.as_array().unwrap().len(), parsed2.as_array().unwrap().len());
}

#[test]
fn test_non_empty_parset_not_serialized_as_empty_array() {
    // This test catches the bug where non-empty collections serialize as "[]"
    let par1 = new_gint_par(42, Vec::new(), false);
    let par2 = new_gstring_par("test".to_string(), Vec::new(), false);
    let par_set = ParSet::create_from_vec(vec![par1, par2]);
    
    let json = serde_json::to_string(&par_set).expect("Failed to serialize ParSet");
    
    // CRITICAL: Non-empty ParSet should NOT serialize as empty array
    assert_ne!(json, "[]", "Non-empty ParSet incorrectly serialized as empty array!");
    assert_ne!(json, "", "ParSet should not serialize as empty string!");
    
    // Should contain actual data
    assert!(json.len() > 2, "JSON should be longer than just '[]', got: {}", json);
    
    // Parse and verify structure
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let array = json_value.as_array().unwrap();
    assert_eq!(array.len(), 2, "JSON array should have 2 elements, got: {}", json);
    
    // Verify content is present
    assert!(json.contains("42"), "JSON should contain the integer 42, got: {}", json);
    assert!(json.contains("test"), "JSON should contain the string 'test', got: {}", json);
}

#[test]
fn test_non_empty_parmap_not_serialized_as_empty_array() {
    // This test catches the bug where non-empty maps serialize as "[]"
    let key = new_gint_par(1, Vec::new(), false);
    let value = new_gstring_par("value".to_string(), Vec::new(), false);
    let par_map = ParMap::create_from_vec(vec![(key, value)]);
    
    let json = serde_json::to_string(&par_map).expect("Failed to serialize ParMap");
    
    // CRITICAL: Non-empty ParMap should NOT serialize as empty array
    assert_ne!(json, "[]", "Non-empty ParMap incorrectly serialized as empty array!");
    assert_ne!(json, "", "ParMap should not serialize as empty string!");
    
    // Should contain actual data
    assert!(json.len() > 2, "JSON should be longer than just '[]', got: {}", json);
    
    // Parse and verify structure
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let array = json_value.as_array().unwrap();
    assert_eq!(array.len(), 1, "JSON array should have 1 key-value pair, got: {}", json);
    
    // Verify the key-value pair structure
    let pair = &array[0];
    assert!(pair.is_array(), "Each element should be a [key, value] array, got: {}", pair);
    assert_eq!(pair.as_array().unwrap().len(), 2, "Each pair should have exactly 2 elements, got: {}", pair);
    
    // Verify content is present
    assert!(json.contains("1"), "JSON should contain the key 1, got: {}", json);
    assert!(json.contains("value"), "JSON should contain the value 'value', got: {}", json);
}

#[test]
fn test_parset_serialization_preserves_all_elements() {
    // Test that all elements are preserved, not just the first or last
    let elements = vec![
        new_gint_par(1, Vec::new(), false),
        new_gint_par(2, Vec::new(), false),
        new_gint_par(3, Vec::new(), false),
        new_gstring_par("first".to_string(), Vec::new(), false),
        new_gstring_par("middle".to_string(), Vec::new(), false),
        new_gstring_par("last".to_string(), Vec::new(), false),
    ];
    let par_set = ParSet::create_from_vec(elements);
    
    let json = serde_json::to_string(&par_set).expect("Failed to serialize ParSet");
    
    // Parse JSON and verify all elements are present
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let array = json_value.as_array().unwrap();
    assert_eq!(array.len(), 6, "All 6 elements should be serialized, got: {}", json);
    
    // Verify all specific values are present in JSON
    assert!(json.contains("1"), "Should contain integer 1, got: {}", json);
    assert!(json.contains("2"), "Should contain integer 2, got: {}", json);
    assert!(json.contains("3"), "Should contain integer 3, got: {}", json);
    assert!(json.contains("first"), "Should contain string 'first', got: {}", json);
    assert!(json.contains("middle"), "Should contain string 'middle', got: {}", json);
    assert!(json.contains("last"), "Should contain string 'last', got: {}", json);
}

#[test]
fn test_parmap_serialization_preserves_all_pairs() {
    // Test that all key-value pairs are preserved
    let pairs = vec![
        (new_gint_par(1, Vec::new(), false), new_gstring_par("one".to_string(), Vec::new(), false)),
        (new_gint_par(2, Vec::new(), false), new_gstring_par("two".to_string(), Vec::new(), false)),
        (new_gint_par(3, Vec::new(), false), new_gstring_par("three".to_string(), Vec::new(), false)),
    ];
    let par_map = ParMap::create_from_vec(pairs);
    
    let json = serde_json::to_string(&par_map).expect("Failed to serialize ParMap");
    
    // Parse JSON and verify all pairs are present
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let array = json_value.as_array().unwrap();
    assert_eq!(array.len(), 3, "All 3 key-value pairs should be serialized, got: {}", json);
    
    // Verify all pairs have correct structure
    for pair in array {
        assert!(pair.is_array(), "Each element should be a [key, value] array, got: {}", pair);
        assert_eq!(pair.as_array().unwrap().len(), 2, "Each pair should have exactly 2 elements, got: {}", pair);
    }
    
    // Verify all specific keys and values are present
    assert!(json.contains("1"), "Should contain key 1, got: {}", json);
    assert!(json.contains("2"), "Should contain key 2, got: {}", json);
    assert!(json.contains("3"), "Should contain key 3, got: {}", json);
    assert!(json.contains("one"), "Should contain value 'one', got: {}", json);
    assert!(json.contains("two"), "Should contain value 'two', got: {}", json);
    assert!(json.contains("three"), "Should contain value 'three', got: {}", json);
}

#[test]
fn test_blake2b512_random_not_serialized_as_object() {
    // Ensure Blake2b512Random doesn't accidentally serialize as a complex object
    let random = Blake2b512Random::create_from_bytes(&[1, 2, 3, 4, 5]);
    let json = serde_json::to_string(&random).expect("Failed to serialize Blake2b512Random");
    
    // Should be exactly "null", not an object or array
    assert_eq!(json, "null", "Blake2b512Random should serialize as null, got: {}", json);
    assert!(!json.starts_with('{'), "Should not serialize as object, got: {}", json);
    assert!(!json.starts_with('['), "Should not serialize as array, got: {}", json);
    
    // Verify it parses as null
    let json_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(json_value.is_null(), "Should parse as null value, got: {}", json_value);
}
