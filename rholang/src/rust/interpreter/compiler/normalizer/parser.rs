use tree_sitter::{Node, Parser, Tree};
use rholang_parser::RholangParser;
use validated::Validated;

use crate::rust::interpreter::{
    compiler::rholang_ast::{
        Block, Branch, BundleType, Case, Collection, Conjunction, Decl, Decls, DeclsChoice,
        Disjunction, Eval, KeyValuePair, LinearBind, Name, NameDecl, Names, Negation, PeekBind,
        Proc, ProcList, Quote, Receipt, Receipts, RepeatedBind, SendType, SimpleType, Source,
        SyncSendCont, UriLiteral, Var, VarRef, VarRefKind,
    },
    errors::InterpreterError,
};

pub fn parse_rholang_code(code: &str) -> Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rholang::LANGUAGE.into())
        .expect("Error loading Rholang grammar");
    parser.parse(code, None).expect("Failed to parse code")
}

pub fn parse_rholang_code_new(code: &str) -> Result<String, String> {
    let parser = RholangParser::new();
    let result = parser.parse(code);
    
    match result {
        Validated::Good(procs) => {
            let summary = format!("Successfully parsed {} processes", procs.len());
            Ok(summary)
        },
        Validated::Fail(failures) => {
            let error_msg = format!("Parse failed: {:?}", failures);
            Err(error_msg)
        }
    }
}

pub fn parse_rholang_code_to_proc(code: &str) -> Result<Proc, InterpreterError> {
    // println!("\nhit parse_rholang_code_to_proc");
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rholang::LANGUAGE.into())
        .expect("Error loading Rholang grammar");

    let tree = match parser.parse(code, None) {
        Some(tree) => tree,
        None => {
            return Err(InterpreterError::SyntaxError(
                "Failed to parse code".to_string(),
            ))
        }
    };

    // println!("\nTree: {:#?} \n", tree.root_node().to_sexp());

    // Check if there were syntax errors during parsing
    if tree.root_node().has_error() {
        return Err(InterpreterError::SyntaxError(format!(
            "Syntax error in code: {}",
            code
        )));
    }

    let root_node = tree.root_node();
    let start_node = match root_node.named_child(0) {
        Some(node) => Ok(node),
        None => Err(InterpreterError::ParserError(
            "The code does not contain any valid Rholang process. Expected a child node"
                .to_string(),
        )),
    }?;

    parse_proc(&start_node, code)
}

fn parse_proc(node: &Node, source: &str) -> Result<Proc, InterpreterError> {
    // println!("\nparse_proc, node: {:?}", node.to_sexp());
    let line_num = node.start_position().row;
    let col_num = node.start_position().column;

    match node.kind() {
        "par" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Par {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "send_sync" => {
            let name_node = get_child_by_field_name(node, "name")?;
            let name_proc = parse_name(&name_node, source)?;

            let messages_node = get_child_by_field_name(node, "messages")?;
            let messages_proc = parse_proc_list(&messages_node, source)?;

            let cont: SyncSendCont = {
                let sync_send_cont_node = get_child_by_field_name(node, "cont")?;
                match sync_send_cont_node.named_child(0) {
                    Some(choice_node) => match choice_node.kind() {
                        "empty_cont" => SyncSendCont::Empty {
                            line_num: choice_node.start_position().row,
                            col_num: choice_node.start_position().column,
                        },

                        "non_empty_cont" => {
                            let proc_node = match choice_node.named_child(0) {
                                Some(_proc_node) => Ok(_proc_node),
                                None => Err(InterpreterError::ParserError(
                                    "Expected a _proc node in non_empty_cont at index 1"
                                        .to_string(),
                                )),
                            }?;

                            SyncSendCont::NonEmpty {
                                proc: Box::new(parse_proc(&proc_node, source)?),
                                line_num: choice_node.start_position().row,
                                col_num: choice_node.start_position().column,
                            }
                        }

                        _ => {
                            return Err(InterpreterError::ParserError(format!(
                                "Unexpected choice node kind: {:?} of sync_send_cont",
                                choice_node.kind(),
                            )))
                        }
                    },
                    None => {
                        return Err(InterpreterError::ParserError(format!(
                            "No child choice node at index 0 for sync_send_cont",
                        )))
                    }
                }
            };

            Ok(Proc::SendSync {
                name: name_proc,
                messages: messages_proc,
                cont,
                line_num,
                col_num,
            })
        }

        "new" => {
            let decls_node = get_child_by_field_name(node, "decls")?;
            let mut name_decls = Vec::new();

            for child in decls_node.named_children(&mut decls_node.walk()) {
                let var_node = child
                    .named_child(0)
                    .filter(|n| n.kind() == "var")
                    .ok_or_else(|| {
                        InterpreterError::ParserError(
                            "Expected a var node in name_decl at index 0".to_string(),
                        )
                    })?;
                let var_proc = parse_var(&var_node, source)?;

                let uri = match child.child_by_field_name("uri") {
                    Some(uri_literal_node) => {
                        match uri_literal_node.next_named_sibling() {
                            // 'uri_literal_node' is '(' so it's sibling is the actual rule we want
                            Some(rule_node) => Some(parse_uri_literal(&rule_node, source)?),
                            None => {
                                return Err(InterpreterError::ParserError(
                                    "Expected rule on uri_literal node".to_string(),
                                ))
                            }
                        }
                    }
                    None => None,
                };

                name_decls.push(NameDecl {
                    var: var_proc,
                    uri,
                    line_num: child.start_position().row,
                    col_num: child.start_position().column,
                });
            }

            let decls_proc = Decls {
                decls: name_decls,
                line_num: decls_node.start_position().row,
                col_num: decls_node.start_position().column,
            };

            let proc_node = get_child_by_field_name(node, "proc")?;
            let proc = parse_proc(&proc_node, source)?;

            Ok(Proc::New {
                decls: decls_proc,
                proc: Box::new(proc),
                line_num,
                col_num,
            })
        }

        "ifElse" => {
            let condition_node = get_child_by_field_name(node, "condition")?;
            let condition_proc = parse_proc(&condition_node, source)?;

            let if_true_node = get_child_by_field_name(node, "ifTrue")?;
            let if_true_proc = parse_proc(&if_true_node, source)?;

            let alternative_proc = match node.named_child(2) {
                Some(alternative_proc_node) => {
                    Some(Box::new(parse_proc(&alternative_proc_node, source)?))
                }
                None => None,
            };

            Ok(Proc::IfElse {
                condition: Box::new(condition_proc),
                if_true: Box::new(if_true_proc),
                alternative: alternative_proc,
                line_num,
                col_num,
            })
        }

        "let" => {
            fn parse_decls(node: &Node, source: &str) -> Result<Vec<Decl>, InterpreterError> {
                let mut decls = Vec::new();

                for child in node.named_children(&mut node.walk()) {
                    let names_node = get_child_by_field_name(&child, "names")?;
                    let names_proc = parse_names(&names_node, source)?;

                    let procs_node = get_child_by_field_name(&child, "procs")?;
                    let procs = parse_comma_sep_procs(&procs_node, source)?;

                    if names_proc.names.len() != procs.len() {
                        return Err(InterpreterError::ParserError(
                            "Names must have the same length as Procs".to_string(),
                        ));
                    }

                    decls.push(Decl {
                        names: names_proc,
                        procs,
                        line_num: child.start_position().row,
                        col_num: child.start_position().column,
                    });
                }

                Ok(decls)
            }

            let decls_node = get_child_by_field_name(node, "decls")?;
            let decls_row = decls_node.start_position().row;
            let decls_col = decls_node.start_position().column;

            let decls_proc = match decls_node.kind() {
                "linear_decls" => Ok(DeclsChoice::LinearDecls {
                    decls: parse_decls(&decls_node, source)?,
                    line_num: decls_row,
                    col_num: decls_col,
                }),

                "conc_decls" => Ok(DeclsChoice::ConcDecls {
                    decls: parse_decls(&decls_node, source)?,
                    line_num: decls_row,
                    col_num: decls_col,
                }),

                _ => Err(InterpreterError::ParserError(format!(
                    "Unexpected choice node kind: {:?} of _decls",
                    node.kind(),
                ))),
            }?;

            let body_node = get_child_by_field_name(node, "body")?;
            let body_proc = parse_block(&body_node, source)?;

            Ok(Proc::Let {
                decls: decls_proc,
                body: Box::new(body_proc),
                line_num,
                col_num,
            })
        }

        "bundle" => {
            let bundle_node = get_child_by_field_name(node, "bundle_type")?;
            let line_num = node.start_position().row;
            let col_num = node.start_position().column;

            let bundle = match bundle_node.kind() {
                "bundle_write" => BundleType::BundleWrite { line_num, col_num },
                "bundle_read" => BundleType::BundleRead { line_num, col_num },
                "bundle_equiv" => BundleType::BundleEquiv { line_num, col_num },
                "bundle_read_write" => BundleType::BundleReadWrite { line_num, col_num },

                _ => {
                    return Err(InterpreterError::ParserError(format!(
                        "Unexpected choice node kind: {:?} of _bundle",
                        bundle_node.kind(),
                    )))
                }
            };

            let proc_node = get_child_by_field_name(node, "proc")?;
            let proc = parse_block(&proc_node, source)?;

            Ok(Proc::Bundle {
                bundle_type: bundle,
                proc: Box::new(proc),
                line_num,
                col_num,
            })
        }

        "match" => {
            let mut expression_node = get_child_by_field_name(node, "expression")?;
            if expression_node.kind() == "(" {
                expression_node = expression_node.next_sibling().unwrap();
            };

            let expression_proc = parse_proc(&expression_node, source)?;

            let cases_node = get_child_by_field_name(node, "cases")?;
            let mut cases = Vec::new();

            for child in cases_node.named_children(&mut cases_node.walk()) {
                let pattern_node = get_child_by_field_name(&child, "pattern")?;
                let pattern_proc = parse_proc(&pattern_node, source)?;

                let proc_node = get_child_by_field_name(&child, "proc")?;
                let proc = parse_proc(&proc_node, source)?;

                cases.push(Case {
                    pattern: pattern_proc,
                    proc,
                    line_num: child.start_position().row,
                    col_num: child.start_position().column,
                });
            }

            Ok(Proc::Match {
                expression: Box::new(expression_proc),
                cases,
                line_num,
                col_num,
            })
        }

        "choice" => {
            let branches_node = get_child_by_field_name(node, "branches")?;
            let mut branches = Vec::new();

            for branch_node in branches_node.named_children(&mut branches_node.walk()) {
                let mut linear_binds = Vec::new();
                for pattern_node in branch_node.named_children(&mut branch_node.walk()) {
                    if pattern_node.kind() == "linear_bind" {
                        linear_binds.push(parse_linear_bind(&pattern_node, source)?);
                    }
                }

                let proc_node = get_child_by_field_name(&branch_node, "proc")?;
                let proc = parse_proc(&proc_node, source)?;

                branches.push(Branch {
                    pattern: linear_binds,
                    proc,
                    line_num: branch_node.start_position().row,
                    col_num: branch_node.start_position().column,
                });
            }

            Ok(Proc::Choice {
                branches,
                line_num,
                col_num,
            })
        }

        "contract" => {
            let name_node = get_child_by_field_name(node, "name")?;
            let name_proc = parse_name(&name_node, source)?;

            let formals_node = get_child_by_field_name(node, "formals")?;
            let formals_proc = parse_names(&formals_node, source)?;

            let proc_node = get_child_by_field_name(node, "proc")?;
            let proc = parse_block(&proc_node, source)?;

            Ok(Proc::Contract {
                name: name_proc,
                formals: formals_proc,
                proc: Box::new(proc),
                line_num,
                col_num,
            })
        }

        "input" => {
            fn parse_receipt(
                receipt_node: &Node,
                source: &str,
            ) -> Result<Receipt, InterpreterError> {
                // Check which '_receipt' we are dealing with and then iterate through
                match receipt_node.kind() {
                    "linear_bind" => Ok(Receipt::LinearBinds(parse_linear_bind(
                        &receipt_node,
                        source,
                    )?)),

                    "repeated_bind" => {
                        let names_node = get_child_by_field_name(&receipt_node, "names")?;
                        let names_proc = parse_names(&names_node, source)?;

                        let input_node = get_child_by_field_name(&receipt_node, "input")?;
                        let input_proc = parse_name(&input_node, source)?;

                        Ok(Receipt::RepeatedBinds(RepeatedBind {
                            names: names_proc,
                            input: input_proc,
                            line_num: receipt_node.start_position().row,
                            col_num: receipt_node.start_position().column,
                        }))
                    }

                    "peek_bind" => {
                        let names_node = get_child_by_field_name(&receipt_node, "names")?;
                        let names_proc = parse_names(&names_node, source)?;

                        let input_node = get_child_by_field_name(&receipt_node, "input")?;
                        let input_proc = parse_name(&input_node, source)?;

                        Ok(Receipt::PeekBinds(PeekBind {
                            names: names_proc,
                            input: input_proc,
                            line_num: receipt_node.start_position().row,
                            col_num: receipt_node.start_position().column,
                        }))
                    }

                    _ => {
                        return Err(InterpreterError::ParserError(format!(
                            "Unexpected choice node kind: {:?} of _receipt",
                            receipt_node.kind(),
                        )))
                    }
                }
            }

            let receipts_node = get_child_by_field_name(node, "formals")?;
            let mut receipts = Vec::new();

            for child in receipts_node.named_children(&mut receipts_node.walk()) {
                receipts.push(parse_receipt(&child, source)?);
            }

            let proc_node = get_child_by_field_name(node, "proc")?;
            let proc = parse_block(&proc_node, source)?;

            Ok(Proc::Input {
                formals: Receipts {
                    receipts,
                    line_num: receipts_node.start_position().row,
                    col_num: receipts_node.start_position().column,
                },
                proc: Box::new(proc),
                line_num,
                col_num,
            })
        }

        "send" => {
            let name_node = get_child_by_field_name(node, "name")?;
            let name_proc = parse_name(&name_node, source)?;

            let send_type_node = get_child_by_field_name(node, "send_type")?;
            let send_type_proc = match send_type_node.kind() {
                "send_single" => SendType::Single { line_num, col_num },

                "send_multiple" => SendType::Multiple { line_num, col_num },

                _ => {
                    return Err(InterpreterError::ParserError(format!(
                        "Unexpected choice node kind: {:?} of send_type",
                        node.kind(),
                    )))
                }
            };

            let inputs_node = get_child_by_field_name(node, "inputs")?;
            let inputs_proc = parse_proc_list(&inputs_node, source)?;

            Ok(Proc::Send {
                name: name_proc,
                send_type: send_type_proc,
                inputs: inputs_proc,
                line_num,
                col_num,
            })
        }

        // _proc_expression
        "or" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Or {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "and" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::And {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "matches" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Matches {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "eq" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Eq {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "neq" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Neq {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "lt" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Lt {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "lte" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Lte {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "gt" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Gt {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "gte" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Gte {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "concat" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Concat {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "minus_minus" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::MinusMinus {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "minus" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Minus {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "add" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Add {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "percent_percent" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::PercentPercent {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "mult" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Mult {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "div" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Div {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "mod" => {
            let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

            Ok(Proc::Mod {
                left: Box::new(left_proc),
                right: Box::new(right_proc),
                line_num,
                col_num,
            })
        }

        "not" => {
            let proc_node = get_child_by_field_name(node, "proc")?;
            let proc = parse_proc(&proc_node, source)?;

            Ok(Proc::Not {
                proc: Box::new(proc),
                line_num,
                col_num,
            })
        }

        "neg" => {
            let proc_node = get_child_by_field_name(node, "proc")?;
            if proc_node.is_named() {
                let proc = parse_proc(&proc_node, source)?;

                Ok(Proc::Neg {
                    proc: Box::new(proc),
                    line_num,
                    col_num,
                })
            } else {
                match proc_node.next_named_sibling() {
                    // At this point, the value is wrapped in parentheses
                    Some(value_node) => Ok(Proc::Neg {
                        proc: Box::new(parse_proc(&value_node, source)?),
                        line_num,
                        col_num,
                    }),
                    None => {
                        return Err(InterpreterError::ParserError(
                            "Expected named node inside neg parentheses".to_string(),
                        ))
                    }
                }
            }
        }

        "method" => {
            let receiver_node = get_child_by_field_name(node, "receiver")?;
            let receiver_proc = parse_proc(&receiver_node, source)?;

            let name_node = get_child_by_field_name(node, "name")?;
            let name_proc = parse_var(&name_node, source)?;

            let args_node = get_child_by_field_name(node, "args")?;
            let args_proc = parse_proc_list(&args_node, source)?;

            Ok(Proc::Method {
                receiver: Box::new(receiver_proc),
                name: name_proc,
                args: args_proc,
                line_num,
                col_num,
            })
        }

        "eval" => Ok(Proc::Eval(parse_eval(node, source)?)),

        "quote" => Ok(Proc::Quote(parse_quote(node, source)?)),

        "disjunction" => Ok(Proc::Disjunction(parse_disjunction(node, source)?)),

        "conjunction" => Ok(Proc::Conjunction(parse_conjunction(node, source)?)),

        "negation" => Ok(Proc::Negation(parse_negation(node, source)?)),

        // _ground_expression
        "block" => Ok(Proc::Block(Box::new(parse_block(node, source)?))),

        "collection" => {
            // println!("\ncollection node: {:?}", node.to_sexp());
            fn get_cont_id(node: &Node) -> Result<i32, InterpreterError> {
                match node.child_by_field_name("cont") {
                    Some(cont_node) => match cont_node.next_named_sibling() {
                        // 'cont_node' is '...' so it's sibling is the actual rule we want
                        Some(rule_node) => Ok(rule_node.id() as i32),
                        None => {
                            return Err(InterpreterError::ParserError(
                                "Expected rule on cont node".to_string(),
                            ))
                        }
                    },
                    None => Ok(-1),
                }
            }

            fn get_cont_proc(
                node: &Node,
                source: &str,
            ) -> Result<Option<Box<Proc>>, InterpreterError> {
                match node.child_by_field_name("cont") {
                    Some(cont_node) => match cont_node.next_named_sibling() {
                        // 'cont_node' is '...' so it's sibling is the actual rule we want
                        Some(rule_node) => Ok(Some(Box::new(parse_proc(&rule_node, source)?))),
                        None => {
                            return Err(InterpreterError::ParserError(
                                "Expected rule on cont node".to_string(),
                            ))
                        }
                    },
                    None => Ok(None),
                }
            }

            fn parse_comma_collection(
                node: &Node,
                source: &str,
            ) -> Result<Vec<Proc>, InterpreterError> {
                let mut procs = Vec::new();

                for child in node.named_children(&mut node.walk()) {
                    if child.id() as i32 == get_cont_id(node)? {
                        continue;
                    }
                    procs.push(parse_proc(&child, source)?)
                }

                Ok(procs)
            }

            let line_num = node.start_position().row;
            let col_num = node.start_position().column;

            let mut cursor = node.walk();
            cursor.goto_first_child();
            let current_node = cursor.node();

            let collection = match current_node.kind() {
                "list" => Ok(Collection::List {
                    elements: parse_comma_collection(&current_node, source)?,
                    cont: get_cont_proc(&current_node, source)?,
                    line_num,
                    col_num,
                }),

                "set" => Ok(Collection::Set {
                    elements: { parse_comma_collection(&current_node, source)? },
                    cont: get_cont_proc(&current_node, source)?,
                    line_num,
                    col_num,
                }),

                "map" => Ok(Collection::Map {
                    pairs: {
                        let mut kvs = Vec::new();

                        for child in current_node.named_children(&mut current_node.walk()) {
                            if child.id() as i32 == get_cont_id(&current_node)? {
                                continue;
                            }
                            let key_node = get_child_by_field_name(&child, "key")?;
                            let key_proc = parse_proc(&key_node, source)?;

                            let value_node = get_child_by_field_name(&child, "value")?;
                            let value_proc = parse_proc(&value_node, source)?;

                            kvs.push(KeyValuePair {
                                key: key_proc,
                                value: value_proc,
                                line_num: child.start_position().row,
                                col_num: child.start_position().column,
                            });
                        }

                        kvs
                    },
                    cont: get_cont_proc(&current_node, source)?,
                    line_num,
                    col_num,
                }),

                "tuple" => Ok(Collection::Tuple {
                    elements: { parse_comma_collection(&current_node, source)? },
                    line_num,
                    col_num,
                }),

                _ => Err(InterpreterError::ParserError(format!(
                    "Unexpected choice node kind: {:?} of collection",
                    node.kind(),
                ))),
            }?;

            Ok(Proc::Collection(collection))
        }

        "simple_type" => {
            let simple_type_value = get_node_value(node, source.as_bytes())?;
            match simple_type_value.as_str() {
                "Bool" => Ok(Proc::SimpleType(SimpleType::Bool { line_num, col_num })),

                "Int" => Ok(Proc::SimpleType(SimpleType::Int { line_num, col_num })),

                "String" => Ok(Proc::SimpleType(SimpleType::String { line_num, col_num })),

                "Uri" => Ok(Proc::SimpleType(SimpleType::Uri { line_num, col_num })),

                "ByteArray" => Ok(Proc::SimpleType(SimpleType::ByteArray {
                    line_num,
                    col_num,
                })),

                _ => Err(InterpreterError::ParserError(format!(
                    "Unexpected value: {:?} of node simple_type",
                    simple_type_value,
                ))),
            }
        }

        //_ground
        "bool_literal" => Ok(Proc::BoolLiteral {
            value: {
                let bool_value = get_node_value(node, source.as_bytes())?;
                match bool_value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(InterpreterError::ParserError(format!(
                            "Invalid bool literal value: {}",
                            bool_value
                        )));
                    }
                }
            },
            line_num,
            col_num,
        }),

        "long_literal" => Ok(Proc::LongLiteral {
            value: {
                let i64_string = get_node_value(node, source.as_bytes())?;
                match i64_string.parse::<i128>() {
                    Ok(long) => {
                        // Manually wrap around if it exceeds i64::MAX
                        if long > i64::MAX as i128 {
                            (long - (i64::MAX as i128 + 1)) as i64
                        } else {
                            long as i64
                        }
                    }
                    Err(e) => {
                        return Err(InterpreterError::ParserError(format!(
                            "Failed to convert long_literal into i64. String: {:?}; Error: {:?}",
                            i64_string, e
                        )))
                    }
                }
            },
            line_num,
            col_num,
        }),

        "string_literal" => Ok(Proc::StringLiteral {
            value: {
                let value_with_quotes = get_node_value(node, source.as_bytes())?;
                let value = value_with_quotes.trim_matches('"').to_string();
                value
            },
            line_num,
            col_num,
        }),

        "uri_literal" => Ok(Proc::UriLiteral(parse_uri_literal(node, source)?)),

        "nil" => Ok(Proc::Nil { line_num, col_num }),

        // _proc_var
        "wildcard" => Ok(Proc::Wildcard {
            line_num: node.start_position().row,
            col_num: node.start_position().column,
        }),

        "var" => Ok(Proc::Var(parse_var(node, source)?)),

        // var_ref
        "var_ref" => {
            let var_ref_kind_node = node.named_child(0).ok_or_else(|| {
                InterpreterError::ParserError(
                    "Expected a var_ref_kind node in var_ref at index 0".to_string(),
                )
            })?;

            let var_ref_kind = match var_ref_kind_node
                .child(0)
                .ok_or_else(|| {
                    InterpreterError::ParserError(
                        "Expected a var_ref_kind in var_ref_kind_node at index 0".to_string(),
                    )
                })?
                .kind()
            {
                "=" => VarRefKind::Proc,
                "=*" => VarRefKind::Name,
                _ => {
                    return Err(InterpreterError::ParserError(format!(
                        "Unexpected choice node kind: {:?} of var_ref_kind.",
                        var_ref_kind_node.kind(),
                    )))
                }
            };

            let var_node = node.named_child(1).ok_or_else(|| {
                InterpreterError::ParserError(
                    "Expected a var node in var_ref at index 1".to_string(),
                )
            })?;

            let var = parse_var(&var_node, source)?;

            Ok(Proc::VarRef(VarRef {
                var_ref_kind,
                var,
                line_num: node.start_position().row,
                col_num: node.start_position().column,
            }))
        }

        _ => Err(InterpreterError::ParserError(format!(
            "Unrecognizable process. Node: {:?}",
            node.kind()
        ))),
    }
}

fn parse_name(node: &Node, source: &str) -> Result<Name, InterpreterError> {
    match node.kind() {
        "quote" => Ok(Name::Quote(Box::new(parse_quote(node, source)?))),

        _ => {
            let proc_result = parse_proc(node, source);
            match proc_result {
                Ok(proc) => Ok(Name::ProcVar(Box::new(proc))),
                Err(proc_err) => Err(InterpreterError::ParserError(format!(
                    "{:?}. Unexpected choice node kind: {:?} of name.",
                    proc_err,
                    node.kind(),
                ))),
            }
        }
    }
}

fn parse_var(node: &Node, source: &str) -> Result<Var, InterpreterError> {
    let var_value = get_node_value(node, source.as_bytes())?;
    Ok(Var {
        name: var_value,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_quote(node: &Node, source: &str) -> Result<Quote, InterpreterError> {
    let quotable_node = node.named_child(0).ok_or_else(|| {
        InterpreterError::ParserError("Expected a quotable node in quote at index 1".to_string())
    })?;

    Ok(Quote {
        quotable: Box::new(parse_proc(&quotable_node, source)?),
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_eval(node: &Node, source: &str) -> Result<Eval, InterpreterError> {
    let name_node = node.named_child(0).ok_or_else(|| {
        InterpreterError::ParserError("Expected a name node in eval at index 1".to_string())
    })?;

    Ok(Eval {
        name: parse_name(&name_node, source)?,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_disjunction(node: &Node, source: &str) -> Result<Disjunction, InterpreterError> {
    let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

    Ok(Disjunction {
        left: Box::new(left_proc),
        right: Box::new(right_proc),
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_conjunction(node: &Node, source: &str) -> Result<Conjunction, InterpreterError> {
    let (left_proc, right_proc) = parse_left_and_right_nodes(node, source)?;

    Ok(Conjunction {
        left: Box::new(left_proc),
        right: Box::new(right_proc),
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_negation(node: &Node, source: &str) -> Result<Negation, InterpreterError> {
    let proc_node = get_child_by_field_name(node, "proc")?;
    let proc = parse_proc(&proc_node, source)?;

    Ok(Negation {
        proc: Box::new(proc),
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_block(node: &Node, source: &str) -> Result<Block, InterpreterError> {
    let body_node = get_child_by_field_name(node, "body")?;
    let body_proc = parse_proc(&body_node, source)?;

    Ok(Block {
        proc: body_proc,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_uri_literal(node: &Node, source: &str) -> Result<UriLiteral, InterpreterError> {
    Ok(UriLiteral {
        value: {
            let value_with_quotes = get_node_value(node, source.as_bytes())?;
            let value_with_ticks = value_with_quotes.trim_matches('"').to_string();
            let value = value_with_ticks.trim_matches('`').to_string();
            value
        },
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_comma_sep_procs(node: &Node, source: &str) -> Result<Vec<Proc>, InterpreterError> {
    let mut procs = Vec::new();
    for child in node.named_children(&mut node.walk()) {
        procs.push(parse_proc(&child, source)?);
    }
    Ok(procs)
}

fn parse_proc_list(node: &Node, source: &str) -> Result<ProcList, InterpreterError> {
    let procs = parse_comma_sep_procs(&node, source)?;
    Ok(ProcList {
        procs,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_linear_bind(node: &Node, source: &str) -> Result<LinearBind, InterpreterError> {
    let names_proc = if let Ok(names_node) = get_child_by_field_name(&node, "names") {
        parse_names(&names_node, source)?
    } else {
        Names {
            names: Vec::new(),
            cont: None,
            line_num: node.start_position().row,
            col_num: node.start_position().column,
        }
    };

    let input_node = get_child_by_field_name(&node, "input")?;
    let input_node_line_num = input_node.start_position().row;
    let input_node_line_col = input_node.start_position().column;

    let input_proc = match input_node.kind() {
        "simple_source" => Source::Simple {
            name: {
                let name_node = input_node
                    .named_child(0)
                    .ok_or(InterpreterError::ParserError(
                        "Expected name node in simple_source at index 0".to_string(),
                    ))?;
                parse_name(&name_node, source)?
            },
            line_num: input_node_line_num,
            col_num: input_node_line_col,
        },

        "receive_send_source" => Source::ReceiveSend {
            name: {
                let name_node = input_node
                    .named_child(0)
                    .ok_or(InterpreterError::ParserError(
                        "Expected name node in receive_send_source at index 0".to_string(),
                    ))?;
                parse_name(&name_node, source)?
            },
            line_num: input_node_line_num,
            col_num: input_node_line_col,
        },

        "send_receive_source" => Source::SendReceive {
            name: {
                let name_node = input_node
                    .named_child(0)
                    .ok_or(InterpreterError::ParserError(
                        "Expected name node in send_receive_source at index 0".to_string(),
                    ))?;
                parse_name(&name_node, source)?
            },
            inputs: {
                let inputs_node = get_child_by_field_name(&input_node, "inputs")?;
                parse_proc_list(&inputs_node, source)?
            },
            line_num: input_node_line_num,
            col_num: input_node_line_col,
        },

        _ => {
            return Err(InterpreterError::ParserError(format!(
                "Unexpected choice node kind: {:?} of _source",
                input_node.kind(),
            )))
        }
    };

    Ok(LinearBind {
        names: names_proc,
        input: input_proc,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_names(node: &Node, source: &str) -> Result<Names, InterpreterError> {
    fn get_cont_id(node: &Node) -> Result<i32, InterpreterError> {
        match node.child_by_field_name("cont") {
            Some(cont_node) => {
                // 'cont_node' is '...@' so it's subsequent sibling is the actual rule we want
                let sibling_node = match cont_node.next_named_sibling() {
                    Some(node) => node,
                    None => {
                        return Err(InterpreterError::ParserError(
                            "Expected sibling node not found".to_string(),
                        ))
                    }
                };

                Ok(sibling_node.id() as i32)
            }
            None => Ok(-1),
        }
    }

    fn get_cont_proc(node: &Node, source: &str) -> Result<Option<Box<Proc>>, InterpreterError> {
        match node.child_by_field_name("cont") {
            Some(cont_node) => {
                // 'cont_node' is '...@' so it's subsequent sibling is the actual rule we want
                let sibling_node = match cont_node.next_named_sibling() {
                    Some(node) => node,
                    None => {
                        return Err(InterpreterError::ParserError(
                            "Expected sibling node not found".to_string(),
                        ))
                    }
                };

                Ok(Some(Box::new(parse_proc(&sibling_node, source)?)))
            }
            None => Ok(None),
        }
    }

    let mut names = Vec::new();
    for child in node.named_children(&mut node.walk()) {
        if child.id() as i32 == get_cont_id(node)? {
            continue;
        }
        names.push(parse_name(&child, source)?);
    }

    Ok(Names {
        names,
        cont: get_cont_proc(node, source)?,
        line_num: node.start_position().row,
        col_num: node.start_position().column,
    })
}

fn parse_left_and_right_nodes(node: &Node, source: &str) -> Result<(Proc, Proc), InterpreterError> {
    let left_node = get_named_child(node, 0)?;
    let left_proc = parse_proc(&left_node, source)?;

    let right_node = get_named_child(node, 1)?;
    let right_proc = parse_proc(&right_node, source)?;

    Ok((left_proc, right_proc))
}

fn get_child_by_field_name<'a>(
    node: &'a Node<'a>,
    name: &'a str,
) -> Result<Node<'a>, InterpreterError> {
    match node.child_by_field_name(name) {
        Some(child_node) => Ok(child_node),
        None => Err(InterpreterError::ParserError(format!(
            "Error: did not find expected field: {:?}, on node {:?}. Line: {:?}, Col: {:?}",
            name,
            node.kind(),
            node.start_position().row,
            node.start_position().column,
        ))),
    }
}

fn get_named_child<'a>(node: &'a Node<'a>, index: usize) -> Result<Node<'a>, InterpreterError> {
    match node.named_child(index) {
        Some(child_node) => Ok(child_node),
        None => Err(InterpreterError::ParserError(format!(
            "Error: did not find named child at index: {:?}, on node {:?}. Line: {:?}, Col: {:?}",
            index,
            node.kind(),
            node.start_position().row,
            node.start_position().column,
        ))),
    }
}

fn get_node_value(node: &Node, bytes: &[u8]) -> Result<String, InterpreterError> {
    match node.utf8_text(bytes) {
        Ok(str) => Ok(str.to_owned()),
        Err(e) => Err(InterpreterError::ParserError(format!(
            "Failed to get node value. Error: {:?}",
            e,
        ))),
    }
}
